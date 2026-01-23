import os
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

import google.auth
import httplib2
from googleapiclient.discovery import build
from google_auth_httplib2 import AuthorizedHttp

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


from google.cloud import storage
import joblib


# --------------------------
# utilities
# --------------------------
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return v if v is not None else default


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _upload_file_to_gcs(bucket_name: str, local_path: str, blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def _sheets_service(creds):
    """
    重要:
      build() に credentials=creds と http=... を同時に渡すと
      'Arguments http and credentials are mutually exclusive' で落ちます。
    対策:
      AuthorizedHttp(creds, http=httplib2.Http(...)) を http= に渡す。
    """
    http = httplib2.Http(timeout=120)
    authed_http = AuthorizedHttp(creds, http=http)
    return build("sheets", "v4", http=authed_http, cache_discovery=False)


def _get_sheet_values(svc, spreadsheet_id: str, range_a1: str) -> List[List[str]]:
    return (
        svc.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_a1)
        .execute()
        .get("values", [])
    )


def _guess_label_column(cols: List[str]) -> Optional[str]:
    # learn_log の典型列名から推測
    candidates = [
        "Win/Lose",
        "WinLose",
        "win_lose",
        "EvalStatus",
        "eval_status",
        "Label",
        "label",
        "y",
    ]
    for c in candidates:
        if c in cols:
            return c
    return None


def _label_to_binary(series: pd.Series) -> pd.Series:
    # "Win" / "Lose" / "1" / "0" / True/False などを 1/0 に寄せる
    s = series.astype(str).str.strip().str.lower()

    win_tokens = {"win", "w", "1", "true", "yes", "y", "勝ち", "勝", "takeprofit", "tp"}
    lose_tokens = {"lose", "l", "0", "false", "no", "n", "負け", "負", "stoploss", "sl"}

    out: List[float] = []
    for v in s.tolist():
        if v in win_tokens:
            out.append(1.0)
        elif v in lose_tokens:
            out.append(0.0)
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index)


def _is_leak_feature(col: str) -> bool:
    """
    実運用で当たるモデルにするため、リーク（結果を含む列）を除外します。
    代表例:
      ExitPrice / ExitTime / ExitReason / PnL_Pct / HoldMin / Status など
    """
    leak_exact = {
        "ExitTime",
        "ExitPrice",
        "ExitReason",
        "PnL_Pct",
        "PnL",
        "HoldMin",
        "Win/Lose",
        "EvalStatus",
        "Status",
        "Note",
    }
    if col in leak_exact:
        return True

    # “Exit...” のような列が増えても自動で弾く
    if col.lower().startswith("exit"):
        return True

    # “pnl” を含む列名も弾く
    if "pnl" in col.lower():
        return True

    # “hold” を含む列名も弾く
    if "hold" in col.lower():
        return True

    return False


def _to_numeric_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    全列を数値化トライ（失敗は NaN）
    ・空文字/空白/null 系は NaN
    ・カンマ区切りを除去

    重要:
      学習の特徴量数を固定したいので「全部NaNの列を drop しない」。
      NaN は後段の SimpleImputer(strategy="median") が基本的に埋める。
    戻り:
      (数値化DataFrame, すべてNaNだった列名)
    """
    x_num = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
        x_num[c] = pd.to_numeric(s, errors="coerce")

    all_nan_cols = [c for c in x_num.columns if x_num[c].isna().all()]
    return x_num, all_nan_cols



def main() -> int:
    print(f"[TRAIN] start {_now_utc_str()}", flush=True)

    spreadsheet_id = _env("SPREADSHEET_ID", "")
    learn_sheet = _env("LEARN_SHEET_NAME", "learn_log")

    bucket_name = _env("BUCKET_NAME", "")
    model_version = _env("MODEL_VERSION", "")
    out_dir_prefix = _env("OUT_DIR_PREFIX", "models")

    out_pkl_name = _env("OUT_PKL_NAME", "trade_ai_model.pkl")
    out_report_name = _env("OUT_REPORT_NAME", "train_report.json")

    print(f"[TRAIN] SPREADSHEET_ID_SET={bool(spreadsheet_id)}", flush=True)
    print(f"[TRAIN] LEARN_SHEET_NAME={learn_sheet}", flush=True)
    print(f"[TRAIN] BUCKET_NAME={bucket_name}", flush=True)
    print(f"[TRAIN] MODEL_VERSION={model_version}", flush=True)
    print(f"[TRAIN] OUT_DIR_PREFIX={out_dir_prefix}", flush=True)

    if not spreadsheet_id:
        print("[TRAIN][ERROR] SPREADSHEET_ID is empty. Set Job env var SPREADSHEET_ID.", flush=True)
        return 2
    if not bucket_name:
        print("[TRAIN][ERROR] BUCKET_NAME is empty. Set Job env var BUCKET_NAME.", flush=True)
        return 3
    if not model_version:
        print("[TRAIN][ERROR] MODEL_VERSION is empty. Set Job env var MODEL_VERSION.", flush=True)
        return 4

    # ---- auth & sheets
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds, project = google.auth.default(scopes=scopes)
    print(f"[TRAIN] ADC_PROJECT={project}", flush=True)

    svc = _sheets_service(creds)

    # ---- read header + rows
    header_vals = _get_sheet_values(svc, spreadsheet_id, f"{learn_sheet}!1:1")
    header = header_vals[0] if header_vals and header_vals[0] else []
    header_cols = len(header)
    print(f"[TRAIN] learn_log header cols={header_cols}", flush=True)

    def _col_letter(n: int) -> str:
        # 1 -> A, 26 -> Z, 27 -> AA ...
        if n <= 0:
            return "A"
        s = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(ord("A") + r) + s
        return s

    # ヘッダ列数に合わせて最終列まで読む（列が増えても追従）
    end_col = _col_letter(header_cols)  # header_cols=40 なら AN
    range_a1 = f"{learn_sheet}!A2:{end_col}"
    all_vals = _get_sheet_values(svc, spreadsheet_id, range_a1)
    rows_total = len(all_vals)
    print(f"[TRAIN] read_range={range_a1} rows_total={rows_total}", flush=True)


    if header_cols == 0 or rows_total == 0:
        print("[TRAIN][ERROR] learn_log is empty (header or rows).", flush=True)
        return 5

    # 行長がバラけても落ちないように揃える
    max_cols = max(header_cols, max((len(r) for r in all_vals), default=0))
    cols = header + [f"col_{i+1}" for i in range(header_cols, max_cols)]

    fixed_rows: List[List[str]] = []
    for r in all_vals:
        rr = r[:max_cols] + [""] * max(0, max_cols - len(r))
        fixed_rows.append(rr)

    df = pd.DataFrame(fixed_rows, columns=cols)

    # ---- label
    label_col = _guess_label_column(df.columns.tolist())
    if label_col is None:
        print("[TRAIN][ERROR] label column not found. Need one of: Win/Lose, EvalStatus, Label, y ...", flush=True)
        return 6

    y = _label_to_binary(df[label_col])

    # label が解釈できない行は落とす
    ok_idx = y.notna()
    df = df.loc[ok_idx].copy()
    y = y.loc[ok_idx].astype(int)

    rows_labeled = int(len(df))
    print(f"[TRAIN] rows_labeled(after label clean)={rows_labeled}", flush=True)

    if rows_labeled < 200:
        print(f"[TRAIN][ERROR] not enough labeled rows after cleaning: {rows_labeled}", flush=True)
        return 7

    # ---- features: learn_log から学習に使う列を定義（数値 + カテゴリ + 時刻特徴）
    # 形式上、既存の変数も残す（レポート用途・互換）
    candidate_cols = [c for c in df.columns if c != label_col]
    leak_cols = [c for c in candidate_cols if _is_leak_feature(c)]

    NUMERIC_FEATURE_COLUMNS: List[str] = [
        # エントリー時点で分かる数値
        "EntryPrice",
        "ScoreSigma",
        "VolSigma",
        "Sigma",
        "TP",
        "SL",
        "TP_Pct",
        "SL_Pct",
        "Leverage",
        "Reserved1",
        "Reserved2",
        "Reserved3",
        "Reserved4",
        "BTC_1h_Change",
        "BTC_Calm",
        "RSI",
        "BandWidth",
        "BW_Change",
        "Vol_Change",
        "Rise_Score",
        "Drop_Score",
        "BTC_Ret",
        "BTC_Vol",
        # （あれば）Market AI のスコアも数値として使える
        "market_ai_score",
    ]

    CATEGORICAL_FEATURE_COLUMNS: List[str] = [
        # エントリー時点で分かるカテゴリ
        "Symbol",
        "Side",
        "SignalType",
        "MarketTag",
        "BTC_Mode",
        "Version",
    ]

    # Datetime 列は “そのまま文字列” では使わず、下で hour/dow/month 等へ変換する
    DATETIME_COL_CANDIDATES: List[str] = [
        "Datetime(SymbolTime_JST)",
        "SymbolTime_JST",
        "Datetime",
        "Time",
    ]

    def _pick_existing_col(df_: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df_.columns:
                return c
        return None

    # Sigma が無いが VolSigma があるケースを吸収（既存コードの安全策を維持）
    if "Sigma" not in df.columns and "VolSigma" in df.columns:
        df["Sigma"] = df["VolSigma"]

    dt_col = _pick_existing_col(df, DATETIME_COL_CANDIDATES)
    TIME_FEATURE_COLUMNS: List[str] = []
    if dt_col:
        dt = pd.to_datetime(df[dt_col], errors="coerce")

        df["time_hour"] = dt.dt.hour.astype("Int64")
        df["time_day_of_week"] = dt.dt.dayofweek.astype("Int64")  # 0=Mon
        df["time_month"] = dt.dt.month.astype("Int64")

        # 周期性（任意だが効きやすい）
        hour = df["time_hour"].fillna(0).astype(float)
        df["time_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["time_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

        TIME_FEATURE_COLUMNS = [
            "time_hour",
            "time_day_of_week",
            "time_month",
            "time_hour_sin",
            "time_hour_cos",
        ]

    # 欠けている列は “学習で落とさない” ために作って埋める
    for c in NUMERIC_FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    for c in CATEGORICAL_FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    feature_cols_all: List[str] = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS

    # X: 欠損や型の整形は前処理 Pipeline 側で吸収（ただし numeric は先に数値化しておく）
    X = df[feature_cols_all].copy()

    # BTC_Calm は TRUE/FALSE 文字列が混ざりやすいので 0/1 に寄せる（安全策）
    if "BTC_Calm" in X.columns:
        s = X["BTC_Calm"].astype(str).str.strip().str.lower()
        X["BTC_Calm"] = s.map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}).astype("float")

    # bool -> int（念のため）
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    # 数値列は「文字列のまま」だと median が落ちるので、ここで数値化して NaN に寄せる
    for c in (NUMERIC_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS):
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # カテゴリ列は string に寄せる（OneHotEncoder に安全に渡す）
    for c in CATEGORICAL_FEATURE_COLUMNS:
        if c in X.columns:
            X[c] = X[c].astype(str)

    # y を X に揃えた Series として明示
    y_series = pd.Series(y.values, index=X.index)

    # クラスが両方あるかチェック
    unique_classes = np.unique(y_series.values)
    if unique_classes.size < 2:
        print(f"[TRAIN][ERROR] only one class present after cleaning: classes={unique_classes.tolist()}", flush=True)
        return 10

    counts = {int(k): int((y_series.values == k).sum()) for k in unique_classes.tolist()}

    stratify_arg = y_series
    if min(counts.values()) < 10:
        stratify_arg = None
        print(f"[TRAIN][WARN] too few samples in a class -> stratify disabled. class_counts={counts}", flush=True)

    # ---- split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_series,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    # ---- preprocessing + model
    numeric_cols = [c for c in (NUMERIC_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS) if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in X.columns]

    # 学習不能（特徴量ゼロ）を早期検知
    if (len(numeric_cols) + len(cat_cols)) == 0:
        print("[TRAIN][ERROR] no usable feature columns (numeric+categorical=0).", flush=True)
        return 9

    # numeric が “全て欠損” の列は SimpleImputer(median) が落ちるので除外
    all_nan_cols = [c for c in numeric_cols if X_train[c].isna().all()]
    if all_nan_cols:
        print(f"[WARN] drop all-NaN numeric cols: {all_nan_cols}", flush=True)
    numeric_cols = [c for c in numeric_cols if c not in set(all_nan_cols)]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    
    model.fit(X_train, y_train)

    # --------------------------
    # Evaluation (probability-aware)
    # --------------------------
    pred_default = model.predict(X_test)

    acc = float(accuracy_score(y_test, pred_default))
    report_txt = classification_report(y_test, pred_default, digits=4)
    cm_default = confusion_matrix(y_test, pred_default).tolist()

    # predict_proba の「陽性(=Win=1)」列を取り出す（classes_ を見て安全に列選択）
    proba = None
    pos_idx = None
    try:
        proba_all = model.predict_proba(X_test)
        # 2値分類を想定（念のため形状チェック）
        if hasattr(proba_all, "shape") and len(proba_all.shape) == 2 and proba_all.shape[1] >= 2:
            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes_list = [int(x) for x in list(classes)]
                if 1 in classes_list:
                    pos_idx = classes_list.index(1)
                else:
                    pos_idx = 1  # 従来互換（不明なら 2列目）
            else:
                pos_idx = 1

            proba = proba_all[:, pos_idx]
    except Exception:
        proba = None
        pos_idx = None


    # proba 分布と “推奨しきい値(F1最大)” を計算
    best_thr = None
    best_f1 = None
    cm_best = None
    proba_summary = {}

    if proba is not None and len(proba) > 0:
        # 分位点（運用のAI_TH調整の根拠）
        qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        proba_summary = {f"p{q}": float(np.percentile(proba, q)) for q in qs}

        # F1 最大の閾値探索（0.01刻み）
        # ※ sklearn を増やさず自前で計算
        y_true = np.asarray(y_test, dtype=int)
        best_f1 = -1.0
        best_thr = 0.5
        best_pred = None

        for t in np.linspace(0.0, 1.0, 101):
            y_hat = (proba >= t).astype(int)
            tp = int(((y_true == 1) & (y_hat == 1)).sum())
            fp = int(((y_true == 0) & (y_hat == 1)).sum())
            fn = int(((y_true == 1) & (y_hat == 0)).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)
                best_pred = y_hat

        if best_pred is not None:
            cm_best = confusion_matrix(y_true, best_pred).tolist()

    print(f"[TRAIN] trained. features={X.shape[1]} rows_used={len(X)} acc={acc:.4f}", flush=True)
    print("[TRAIN] classification_report:\n" + report_txt, flush=True)
    print(f"[TRAIN] confusion_matrix(default_predict)={cm_default}", flush=True)

    if proba_summary:
        print(f"[TRAIN] proba_percentiles={json.dumps(proba_summary, ensure_ascii=False)}", flush=True)
    if best_thr is not None:
        print(f"[TRAIN] best_threshold_f1={best_thr:.2f} best_f1={best_f1:.4f} cm_at_best={cm_best}", flush=True)

    # 既存の変数名 cm は downstream で使うので合わせる
    cm = cm_default


    # ---- outputs
    local_pkl_path = "/tmp/" + out_pkl_name
    local_report_path = "/tmp/" + out_report_name

    payload = {
        "pipeline": model,
        # 推論側は「この入力列名（X.columns）」を揃えると精度が出やすい
        "feature_columns": list(X.columns),
        "label_column": label_col,
        "trained_at_utc": _now_utc_str(),
        "rows_used": int(len(X)),
        "leak_removed_columns": leak_cols,
        "all_nan_dropped_columns": all_nan_cols,
    }

    # trade_ai_model.pkl は「モデル直」で保存（dictは禁止）
    joblib.dump(payload["pipeline"], local_pkl_path)

    print("[SAVE] pkl_path:", local_pkl_path, flush=True)
    print("[SAVE] saved_object_type:", type(payload["pipeline"]).__name__, flush=True)
    print("[SAVE] input_feature_cols:", int(X.shape[1]), flush=True)
    print("[SAVE] n_features_in:", int(getattr(payload["pipeline"], "n_features_in_", 0) or 0), flush=True)

    report_obj = {
        "trained_at_utc": _now_utc_str(),
        "model_version": model_version,
        "spreadsheet_id_set": bool(spreadsheet_id),
        "learn_sheet": learn_sheet,
        "header_cols": int(header_cols),
        "rows_total": int(rows_total),
        "rows_labeled": int(rows_labeled),
        "rows_used": int(len(X)),
        "label_column": label_col,
        "class_counts": counts,
        "feature_count": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "leak_removed_columns": leak_cols,
        "all_nan_dropped_columns": all_nan_cols,

        # --- metrics（既存）
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report_txt,

        # --- probability-aware metrics（追加）
        "positive_class_index": (int(pos_idx) if "pos_idx" in locals() and pos_idx is not None else None),
        "proba_percentiles": (proba_summary if isinstance(proba_summary, dict) else {}),
        "best_threshold_f1": (float(best_thr) if best_thr is not None else None),
        "best_f1": (float(best_f1) if best_f1 is not None else None),
        "confusion_matrix_default": cm_default,
        "confusion_matrix_best_f1": cm_best,

        "note": "Leakage removed (Exit/PnL/Hold/Status etc) + OneHot(categorical) + time features + NaN-safe preprocessing + LogisticRegression(liblinear, balanced).",
    }

    _write_json(local_report_path, report_obj)


    # ---- write to /gcs mount first, fallback to API
    gcs_dir = f"{out_dir_prefix}/{model_version}"
    gcs_mount_dir = f"/gcs/{gcs_dir}"
    gcs_pkl_path = f"{gcs_mount_dir}/{out_pkl_name}"
    gcs_report_path = f"{gcs_mount_dir}/{out_report_name}"

    wrote_by_mount = False
    try:
        _safe_mkdir(gcs_mount_dir)
        with open(local_pkl_path, "rb") as rf, open(gcs_pkl_path, "wb") as wf:
            wf.write(rf.read())
        with open(local_report_path, "rb") as rf, open(gcs_report_path, "wb") as wf:
            wf.write(rf.read())
        wrote_by_mount = True
        print(f"[OUT] OUT_PKL=gs://{bucket_name}/{gcs_dir}/{out_pkl_name}", flush=True)
        print(f"[OUT] OUT_REPORT=gs://{bucket_name}/{gcs_dir}/{out_report_name}", flush=True)
    except Exception as e:
        print(f"[TRAIN][WARN] write via /gcs failed -> fallback to GCS API. err={e}", flush=True)

    if not wrote_by_mount:
        blob_pkl = f"{gcs_dir}/{out_pkl_name}"
        blob_report = f"{gcs_dir}/{out_report_name}"
        out_pkl_uri = _upload_file_to_gcs(bucket_name, local_pkl_path, blob_pkl)
        out_report_uri = _upload_file_to_gcs(bucket_name, local_report_path, blob_report)
        print(f"[OUT] OUT_PKL={out_pkl_uri}", flush=True)
        print(f"[OUT] OUT_REPORT={out_report_uri}", flush=True)

    print(f"[TRAIN] done {_now_utc_str()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("[TRAIN][FATAL] exception", flush=True)
        traceback.print_exc()
        sys.exit(1)
