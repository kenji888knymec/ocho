import os
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict

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

    # 安全に広めに取得（列増でも落ちないように）
    all_vals = _get_sheet_values(svc, spreadsheet_id, f"{learn_sheet}!A2:AF")
    rows_total = len(all_vals)
    print(f"[TRAIN] learn_log rows_total={rows_total}", flush=True)

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

    # ---- features: 本番互換の 9特徴量に固定（推論側 expected_cols と一致させる）
    FEATURE_COLUMNS = [
        "Sigma",
        "BandWidth",
        "BW_Change",
        "RSI",
        "Vol_Change",
        "Rise_Score",
        "Drop_Score",
        "BTC_Ret",
        "BTC_Vol",
    ]

    # 形式上、既存の変数も残す（レポート用途・互換）
    candidate_cols = [c for c in df.columns if c != label_col]
    leak_cols = [c for c in candidate_cols if _is_leak_feature(c)]
    feat_cols = FEATURE_COLUMNS[:]  # 9列固定

    # Sigma が無いが VolSigma があるケースを吸収（main 側の alias と整合）
    if "Sigma" not in df.columns and "VolSigma" in df.columns:
        df["Sigma"] = df["VolSigma"]

    # 欠損列があっても学習が落ちないように 0.0 で補完
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    for c in missing_cols:
        df[c] = 0.0

    X_raw = df[FEATURE_COLUMNS].copy()


    # 数値化（失敗は NaN）
    X_num, all_nan_cols = _to_numeric_frame(X_raw)
    
    # 重要：全行NaNの列は SimpleImputer(median) が詰む可能性があるので 0.0 で埋める
    for c in all_nan_cols:
        X_num[c] = 0.0
    
    # 任意：ログ（事故検知が早くなる）
    print(f"[TRAIN] FEATURE_COLUMNS={FEATURE_COLUMNS}", flush=True)
    print(f"[TRAIN] all_nan_cols={all_nan_cols}", flush=True)


    if X_num.shape[1] < 5:
        print(f"[TRAIN][ERROR] too few numeric feature columns: {X_num.shape[1]}", flush=True)
        print(f"[TRAIN] hint: maybe most columns are non-numeric or empty.", flush=True)
        return 9

    # クラスが両方あるかチェック（片寄り過ぎで stratify が落ちるのを防ぐ）
    # y も index を保ったまま扱う（X_num とズレにくくする）
    y_series = pd.Series(y.values, index=X_num.index)
    
    unique_classes = np.unique(y_series.values)
    if unique_classes.size < 2:
        print(f"[TRAIN][ERROR] only one class present after cleaning: classes={unique_classes.tolist()}", flush=True)
        return 10
    
    stratify_arg = y_series
    # 少なすぎる場合は stratify 無しにする（落ちないための保険）
    counts = {int(k): int((y_series.values == k).sum()) for k in unique_classes.tolist()}
    if min(counts.values()) < 10:
        stratify_arg = None
        print(f"[TRAIN][WARN] too few samples in a class -> stratify disabled. class_counts={counts}", flush=True)
    
    # ---- split & train (NaN は SimpleImputer で埋める)
    # DataFrameのまま学習（列順事故を減らす）
    X_train, X_test, y_train, y_test = train_test_split(
        X_num,
        y_series,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
        ]
    )
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)


    acc = float(accuracy_score(y_test, pred))
    report_txt = classification_report(y_test, pred, digits=4)
    cm = confusion_matrix(y_test, pred).tolist()

    print(f"[TRAIN] trained. features={X_num.shape[1]} rows_used={len(X_num)} acc={acc:.4f}", flush=True)
    print("[TRAIN] classification_report:\n" + report_txt, flush=True)
    print(f"[TRAIN] confusion_matrix={cm}", flush=True)

    # ---- outputs
    local_pkl_path = "/tmp/" + out_pkl_name
    local_report_path = "/tmp/" + out_report_name

    payload = {
        "pipeline": model,
        "feature_columns": X_num.columns.tolist(),  # 推論側はこの順序に揃える
        "label_column": label_col,
        "trained_at_utc": _now_utc_str(),
        "rows_used": int(len(X_num)),
        "leak_removed_columns": leak_cols,
        "all_nan_dropped_columns": all_nan_cols,
    }

    # trade_ai_model.pkl は「モデル直」で保存（dictは禁止）
    joblib.dump(payload["pipeline"], local_pkl_path)

    print("[SAVE] pkl_path:", local_pkl_path, flush=True)
    print("[SAVE] saved_object_type:", type(payload["pipeline"]).__name__, flush=True)
    print("[SAVE] n_features:", int(X_num.shape[1]), flush=True)


    report_obj = {
        "trained_at_utc": _now_utc_str(),
        "model_version": model_version,
        "spreadsheet_id_set": bool(spreadsheet_id),
        "learn_sheet": learn_sheet,
        "header_cols": int(header_cols),
        "rows_total": int(rows_total),
        "rows_labeled": int(rows_labeled),
        "rows_used": int(len(X_num)),
        "label_column": label_col,
        "class_counts": counts,
        "feature_count": int(X_num.shape[1]),
        "feature_columns": X_num.columns.tolist(),
        "leak_removed_columns": leak_cols,
        "all_nan_dropped_columns": all_nan_cols,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report_txt,
        "note": "Leakage removed (Exit/PnL/Hold/Status etc) + NaN-safe training (SimpleImputer median) + LogisticRegression(liblinear).",
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
