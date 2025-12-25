import os
import sys
import traceback
from datetime import datetime, timezone

import google.auth
from googleapiclient.discovery import build


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> int:
    print(f"[TRAIN] start {_now()}", flush=True)

    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    learn_sheet = os.environ.get("LEARN_SHEET_NAME", "learn_log")

    print(f"[TRAIN] SPREADSHEET_ID_SET={bool(spreadsheet_id)}", flush=True)
    print(f"[TRAIN] LEARN_SHEET_NAME={learn_sheet}", flush=True)

    if not spreadsheet_id:
        print("[TRAIN][ERROR] SPREADSHEET_ID is empty. Set Job env var SPREADSHEET_ID.", flush=True)
        return 2

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

    print(f"[TRAIN] done {_now()} (this is a smoke test; training logic will be added next)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception:
        print("[TRAIN][FATAL] exception", flush=True)
        traceback.print_exc()
        sys.exit(1)
