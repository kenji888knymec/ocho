import os
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import google.auth
from googleapiclient.discovery import build

import joblib
from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return v if v is not None else default


def _upload_file_to_gcs(bucket_name: str, src_path: str, dst_blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dst_blob_name)
    blob.upload_from_filename(src_path)
    return f"gs://{bucket_name}/{dst_blob_name}"


def _read_sheet_as_df(svc, spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    # 52列分くらいまで一気に取る（A:AZ）
    rng = f"{sheet_name}!A:AZ"
    values = (
        svc.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=rng)
        .execute()
        .get("values", [])
    )
    if not values or len(values) < 2:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]

    # 行ごとに列数がバラつくことがあるので、最大列数に揃える
    max_cols = len(header)
    norm_rows = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]
        norm_rows.append(r)

    df = pd.DataFrame(norm_rows, columns=header)
    return df


def _detect_label(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series], str]:
    """
    返り値: (label_col_name, y_series, label_rule_description)
    """
    candidates = [
        "Win/Lose", "WinLose", "win_lose", "result", "label", "y",
        "EvalStatus", "eval_status",
    ]

    for col in candidates:
        if col in df.columns:
            s = df[col].astype(str).str.strip()

            # "Win"/"Lose" パターン
            if s.str.contains("Win", case=False, na=False).any() or s.str.contains("Lose", case=False, na=False).any():
                y = s.map(lambda x: 1 if str(x).lower() == "win" else (0 if str(x).lower() == "lose" else np.nan))
                return col, y, f"{col}: Win->1, Lose->0"

            # 数値っぽい場合
            y_num = pd.to_numeric(s, errors="coerce")
            if y_num.notna().sum() > 0:
                # 0/1 or -1/1 なども許容
                return col, y_num, f"{col}: numeric"

    # 最後の手段：PnL_Pct があれば >0 を勝ちにする
    if "PnL_Pct" in df.columns:
        pnl = pd.to_numeric(df["PnL_Pct"], errors="coerce")
        y = (pnl > 0).astype(float)
        y[pnl.isna()] = np.nan
        return "PnL_Pct", y, "PnL_Pct > 0 -> 1 else 0"

    return None, None, "no label detected"


def _make_features(df: pd.DataFrame, label_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = set()
    if label_col:
        drop_cols.add(label_col)

    # まずID/文字列系っぽい列は落とす候補にする（残しても numeric 化で全部NaNになりやすい）
    id_like_keywords = ["Time", "Datetime", "Symbol", "Side", "Version", "Note", "Reason", "Status", "Tag", "Mode"]
    for c in df.columns:
        for kw in id_like_keywords:
            if kw.lower() in str(c).lower():
                drop_cols.add(c)
                break

    feat_cols = [c for c in df.columns if c not in drop_cols]

    # 数値化（できないものは NaN になる）
    X_num = pd.DataFrame()
    kept_cols: List[str] = []
    for c in feat_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        # ある程度値が入ってる列だけ残す（全部NaN列は捨てる）
        if s.notna().sum() >= max(10, int(len(df) * 0.05)):
            X_num[c] = s
            kept_cols.append(c)

    return X_num, kept_cols


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

    # ADC（Cloud Run Job の実行SAで認証）
    creds, project = google.auth.default()
    print(f"[TRAIN] ADC_PROJECT={project}", flush=True)

    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)

    df = _read_sheet_as_df(svc, spreadsheet_id, learn_sheet)
    if df.empty:
        print("[TRAIN][ERROR] learn_log is empty or cannot be read.", flush=True)
        return 5

    print(f"[TRAIN] learn_log header cols={len(df.columns)}", flush=True)
    print(f"[TRAIN] learn_log rows={len(df)}", flush=True)

    label_col, y_raw, label_rule = _detect_label(df)
    print(f"[TRAIN] label_detected={label_col} rule={label_rule}", flush=True)
    if y_raw is None:
        print("[TRAIN][ERROR] No label column detected. Need Win/Lose or PnL_Pct etc.", flush=True)
        return 6

    # y を数値に寄せる（float）
    y = pd.to_numeric(y_raw, errors="coerce")

    # 特徴量を作る（数値化できる列だけ）
    X, feature_cols = _make_features(df, label_col)
    print(f"[TRAIN] feature_cols={len(feature_cols)}", flush=True)

    # y が無い行は落とす
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # 特徴量が全てNaNの行も落とす（全部欠損だと imputer でも情報ゼロ）
    non_empty = X.notna().sum(axis=1) > 0
    X = X.loc[non_empty].copy()
    y = y.loc[non_empty].copy()

    n = len(y)
    print(f"[TRAIN] usable_rows={n}", flush=True)
    if n < 50:
        print("[TRAIN][ERROR] Too few usable rows for training (<50).", flush=True)
        return 7

    # クラス数チェック（1クラスしか無いと学習不可）
    classes = sorted(pd.unique(y))
    if len(classes) < 2:
        print(f"[TRAIN][ERROR] Only one class in label: {classes}", flush=True)
        return 8

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(pd.unique(y)) > 1 else None
    )

    # 欠損対策：SimpleImputer で中央値補完 → 標準化 → ロジスティック回帰
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    pipe.fit(X_train, y_train)

    # 評価
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(pd.unique(y_test)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = None

    print(f"[TRAIN] metrics={metrics}", flush=True)

    # 成果物保存
    local_pkl_path = "/tmp/" + out_pkl_name
    local_report_path = "/tmp/" + out_report_name

    model_obj = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "label_rule": label_rule,
        "trained_at_utc": _now_utc_str(),
        "model_version": model_version,
    }
    joblib.dump(model_obj, local_pkl_path)

    report_obj = {
        "created_utc": _now_utc_str(),
        "project": project,
        "bucket": bucket_name,
        "prefix": f"{out_dir_prefix}/{model_version}",
        "out_pkl": f"gs://{bucket_name}/{out_dir_prefix}/{model_version}/{out_pkl_name}",
        "out_report": f"gs://{bucket_name}/{out_dir_prefix}/{model_version}/{out_report_name}",
        "learn_log": {
            "sheet": learn_sheet,
            "header_cols": int(len(df.columns)),
            "row_count": int(len(df)),
            "usable_rows": int(n),
            "label_detected": label_col,
            "label_rule": label_rule,
            "feature_cols": int(len(feature_cols)),
        },
        "metrics": metrics,
        "note": "real training with NaN handling (SimpleImputer + LogisticRegression)",
    }

    with open(local_report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    # GCSへアップロード
    gcs_dir = f"{out_dir_prefix}/{model_version}"
    gcs_pkl_blob = f"{gcs_dir}/{out_pkl_name}"
    gcs_report_blob = f"{gcs_dir}/{out_report_name}"

    out_pkl_uri = _upload_file_to_gcs(bucket_name, local_pkl_path, gcs_pkl_blob)
    out_report_uri = _upload_file_to_gcs(bucket_name, local_report_path, gcs_report_blob)

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
