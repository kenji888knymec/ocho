import os
import sys
import json
import traceback
from datetime import datetime, timezone

import google.auth
from googleapiclient.discovery import build
from google.cloud import storage


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return v.strip() if isinstance(v, str) else default


def main() -> int:
    print(f"[TRAIN] start {_now()}", flush=True)

    # ===== Job env（入力）=====
    spreadsheet_id = _env("SPREADSHEET_ID", "")
    learn_sheet = _env("LEARN_SHEET_NAME", "learn_log")

    # ===== Job env（出力先）=====
    bucket_name = _env("BUCKET_NAME", "")
    model_version = _env("MODEL_VERSION", "")

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

    out_prefix = f"models/{model_version}/"
    out_pkl = f"{out_prefix}trade_ai_model.pkl"
    out_report = f"{out_prefix}train_report.json"

    print(f"[OUT] OUT_DIR=gs://{bucket_name}/{out_prefix}", flush=True)
    print(f"[OUT] OUT_PKL=gs://{bucket_name}/{out_pkl}", flush=True)
    print(f"[OUT] OUT_REPORT=gs://{bucket_name}/{out_report}", flush=True)

    # ===== Sheets 読み取り（あなたの既存スモークテストは残す）=====
    creds, project = google.auth.default()
    print(f"[TRAIN] ADC_PROJECT={project}", flush=True)

    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)

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

    # ===== ここが “本題”：モデルとレポートをGCSへ書き出す =====
    # いまは学習ロジックが未実装なので、ダミーの成果物を生成して
    # 「GCSに書ける」ことを先に確定します（最短で前進するため）。
    dummy_model_bytes = b"DUMMY_MODEL_PLACEHOLDER\n"
    report_obj = {
        "ts_utc": _now(),
        "project": project,
        "spreadsheet_id_set": bool(spreadsheet_id),
        "learn_sheet": learn_sheet,
        "learn_log_header_cols": header_cols,
        "learn_log_row_count_a": row_count,
        "note": "smoke test output; replace dummy_model_bytes with real model later",
    }
    report_bytes = (json.dumps(report_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blob_pkl = bucket.blob(out_pkl)
        blob_pkl.upload_from_string(dummy_model_bytes, content_type="application/octet-stream")

        blob_report = bucket.blob(out_report)
        blob_report.upload_from_string(report_bytes, content_type="application/json")

        print("[OUT] upload ok", flush=True)

    except Exception as e:
        print("[TRAIN][ERROR] failed to write outputs to GCS", flush=True)
        print(f"[TRAIN][ERROR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 10

    print(f"[TRAIN] done {_now()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception:
        print("[TRAIN][FATAL] exception", flush=True)
        traceback.print_exc()
        sys.exit(1)
