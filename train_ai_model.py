import os
import sys
import json
import traceback
from datetime import datetime, timezone

import google.auth
from googleapiclient.discovery import build

import joblib
from google.cloud import storage


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


def main() -> int:
    print(f"[TRAIN] start {_now_utc_str()}", flush=True)

    spreadsheet_id = _env("SPREADSHEET_ID", "")
    learn_sheet = _env("LEARN_SHEET_NAME", "learn_log")

    bucket_name = _env("BUCKET_NAME", "")
    model_version = _env("MODEL_VERSION", "")
    out_dir_prefix = _env("OUT_DIR_PREFIX", "models")  # 例: models

    out_pkl_name = _env("OUT_PKL_NAME", "trade_ai_model.pkl")
    out_report_name = _env("OUT_REPORT_NAME", "train_report.json")

    print(f"[TRAIN] SPREADSHEET_ID_SET={bool(spreadsheet_id)}", flush=True)
    print(f"[TRAIN] LEARN_SHEET_NAME={learn_sheet}", flush=True)
    print(f"[TRAIN] BUCKET_NAME={bucket_name}", flush=True)
    print(f"[TRAIN] MODEL_VERSION={model_version}", flush=True)

    if not spreadsheet_id:
        print("[TRAIN][ERROR] SPREADSHEET_ID is empty. Set Job env var SPREADSHEET_ID.", flush=True)
        return 2

    if not bucket_name:
        print("[TRAIN][ERROR] BUCKET_NAME is empty. Set Job env var BUCKET_NAME.", flush=True)
        return 3

    if not model_version:
        print("[TRAIN][ERROR] MODEL_VERSION is empty. Set Job env var MODEL_VERSION.", flush=True)
        return 4

    # Cloud Run Job の実行サービスアカウントで ADC（Application Default Credentials）を使う
    creds, project = google.auth.default()
    print(f"[TRAIN] ADC_PROJECT={project}", flush=True)

    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)

    # learn_log のヘッダー（1行目）を取得して、列数を見る
    header_range = f"{learn_sheet}!1:1"
    header = (
        svc.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=header_range)
        .execute()
        .get("values", [])
    )
    header_cols = len(header[0]) if header and header[0] else 0
    print(f"[TRAIN] learn_log header cols={header_cols}", flush=True)

    # A列の行数をざっくり取得（存在確認＆アクセス確認）
    a_col_range = f"{learn_sheet}!A:A"
    a_vals = (
        svc.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=a_col_range)
        .execute()
        .get("values", [])
    )
    row_count = len(a_vals)
    print(f"[TRAIN] learn_log row_count(A)={row_count}", flush=True)

    # ==========================================================
    # ここから「実学習版」
    #  - learn_log を読み込んで学習し、欠損(NaN)を必ず処理して落ちないようにする
    #  - 学習済み Pipeline を trade_ai_model.pkl として GCS に保存する
    # ==========================================================
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # --- learn_log をシート丸ごと取得（A:ZZで広めに取る）---
    data_range = f"{learn_sheet}!A:ZZ"
    values = (
        svc.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=data_range)
        .execute()
        .get("values", [])
    )

    if not values or len(values) < 2:
        print("[TRAIN][ERROR] learn_log has no data rows.", flush=True)
        return 5

    header_row = values[0]
    rows = values[1:]

    df = pd.DataFrame(rows, columns=header_row)

    # 空文字を NaN に寄せる（欠損処理しやすくする）
    df = df.replace("", np.nan)

    print(f"[TRAIN] df_shape={df.shape}", flush=True)
    print(f"[TRAIN] df_cols={len(df.columns)}", flush=True)

    # --- 目的変数(y)を決める（優先順位：Win/Lose -> PnL_Pct）---
    y_col = None
    if "Win/Lose" in df.columns:
        y_col = "Win/Lose"

        def _to_label(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            s = str(v).strip().lower()
            if s in ["win", "w", "1", "true", "yes"]:
                return 1
            if s in ["lose", "loss", "l", "0", "false", "no"]:
                return 0
            return np.nan

        y = df[y_col].map(_to_label)
    elif "PnL_Pct" in df.columns:
        y_col = "PnL_Pct"
        pnl = pd.to_numeric(df[y_col], errors="coerce")
        y = (pnl > 0).astype("float")
    else:
        print("[TRAIN][ERROR] No label column found. Need 'Win/Lose' or 'PnL_Pct' in learn_log.", flush=True)
        return 6

    # --- 特徴量X：まずは y_col を除いた全列から作る ---
    X = df.drop(columns=[y_col], errors="ignore").copy()

    # Google Sheets 由来で全部文字列になりがちなので、数値にできる列は数値へ寄せる
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="ignore")

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"[TRAIN] label_col={y_col}", flush=True)
    print(f"[TRAIN] numeric_cols={len(numeric_cols)} categorical_cols={len(categorical_cols)}", flush=True)

    # --- y の欠損行を落とす（学習できないため）---
    ok_idx = ~y.isna()
    X = X.loc[ok_idx].copy()
    y = y.loc[ok_idx].astype(int).copy()

    if len(y) < 50:
        print(f"[TRAIN][ERROR] too few labeled rows after filtering: {len(y)}", flush=True)
        return 7

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    print(f"[TRAIN] train_rows={len(y_train)} test_rows={len(y_test)}", flush=True)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    print(f"[TRAIN] accuracy={acc:.4f}", flush=True)

    # --- 成果物保存 ---
    local_pkl_path = "/tmp/" + out_pkl_name
    local_report_path = "/tmp/" + out_report_name

    joblib.dump(clf, local_pkl_path)

    report_obj = {
        "created_utc": _now_utc_str(),
        "project": project,
        "bucket": bucket_name,
        "prefix": f"{out_dir_prefix}/{model_version}",
        "out_pkl": f"gs://{bucket_name}/{out_dir_prefix}/{model_version}/{out_pkl_name}",
        "out_report": f"gs://{bucket_name}/{out_dir_prefix}/{model_version}/{out_dir_prefix}/{out_report_name}".replace(f"/{out_dir_prefix}/{out_dir_prefix}/", f"/{out_dir_prefix}/"),
        "learn_log": {
            "sheet": learn_sheet,
            "header_cols": header_cols,
            "row_count": int(len(values) - 1),
            "labeled_rows": int(len(y)),
        },
        "label_col": y_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "metrics": {
            "accuracy": acc,
        },
        "note": "real training (Pipeline with imputers to avoid NaN crash)",
    }

    with open(local_report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    gcs_dir = f"{out_dir_prefix}/{model_version}"
    gcs_pkl_blob = f"{gcs_dir}/{out_pkl_name}"
    gcs_report_blob = f"{gcs_dir}/{out_report_name}"

    try:
        out_pkl_uri = _upload_file_to_gcs(bucket_name, local_pkl_path, gcs_pkl_blob)
        out_report_uri = _upload_file_to_gcs(bucket_name, local_report_path, gcs_report_blob)
    except Exception as e:
        print("[TRAIN][ERROR] failed to upload artifacts to GCS", flush=True)
        print(f"[TRAIN][ERROR] {type(e).__name__}: {e}", flush=True)
        raise

    print(f"[OUT] OUT_PKL={out_pkl_uri}", flush=True)
    print(f"[OUT] OUT_REPORT={out_report_uri}", flush=True)

    print(f"[TRAIN] done {_now_utc_str()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception:
        print("[TRAIN][FATAL] exception", flush=True)
        traceback.print_exc()
        sys.exit(1)
