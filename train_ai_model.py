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
    # ここから「最小の学習成果物」を作る（まずは動線を完成させる）
    #  - 本格学習は後でOK。まずは pkl/report が GCS に出る状態を作る。
    # ==========================================================
    model_obj = {
        "trained_at_utc": _now_utc_str(),
        "learn_sheet": learn_sheet,
        "header_cols": header_cols,
        "row_count_a": row_count,
        "note": "minimal artifact to verify train -> upload pipeline",
    }

    report_obj = {
        "trained_at_utc": _now_utc_str(),
        "spreadsheet_id_set": bool(spreadsheet_id),
        "learn_sheet": learn_sheet,
        "header_cols": header_cols,
        "row_count_a": row_count,
        "model_version": model_version,
    }

    local_pkl_path = "/tmp/" + out_pkl_name
    local_report_path = "/tmp/" + out_report_name

    joblib.dump(model_obj, local_pkl_path)
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
