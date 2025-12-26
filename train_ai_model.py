import os
import sys
import json
import pickle
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

import google.auth
from googleapiclient.discovery import build


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def _utc_model_version_default() -> str:
    return datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H")


def main() -> int:
    print(f"[TRAIN] start {_now()}", flush=True)

    # ===== 入力（環境変数）=====
    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    learn_sheet = os.environ.get("LEARN_SHEET_NAME", "learn_log")

    # ===== 出力（環境変数）=====
    # Cloud Storage を Cloud Run Job に volume mount している想定:
    #   mount path: /gcs
    #   -> /gcs/models/<MODEL_VERSION>/... に成果物を出す
    gcs_mount_dir = os.environ.get("GCS_MOUNT_DIR", "/gcs")
    model_version = os.environ.get("MODEL_VERSION", _utc_model_version_default())

    out_dir = os.path.join(gcs_mount_dir, "models", model_version)
    out_pkl = os.path.join(out_dir, "trade_ai_model.pkl")
    out_report = os.path.join(out_dir, "train_report.json")

    print(f"[TRAIN] SPREADSHEET_ID_SET={bool(spreadsheet_id)}", flush=True)
    print(f"[TRAIN] LEARN_SHEET_NAME={learn_sheet}", flush=True)

    print(f"[OUT] MODEL_VERSION={model_version}", flush=True)
    print(f"[OUT] OUT_DIR={out_dir}", flush=True)
    print(f"[OUT] OUT_PKL={out_pkl}", flush=True)
    print(f"[OUT] OUT_REPORT={out_report}", flush=True)

    if not spreadsheet_id:
        print("[TRAIN][ERROR] SPREADSHEET_ID is empty. Set Job env var SPREADSHEET_ID.", flush=True)
        return 2

    # 出力先ディレクトリ作成
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"[TRAIN][ERROR] failed to create out_dir: {out_dir} err={e}", flush=True)
        return 3

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

    # ===== ここから「成果物出力」（学習ロジックはまだ無いのでダミーを出す）=====
    report: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "spreadsheet_id_set": bool(spreadsheet_id),
        "learn_sheet": learn_sheet,
        "header_cols": header_cols,
        "row_count_a": row_count,
        "note": "smoke test output. training logic will be added next.",
    }

    try:
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OUT] wrote report: {out_report}", flush=True)
    except Exception as e:
        print(f"[TRAIN][ERROR] failed to write report: {out_report} err={e}", flush=True)
        return 4

    # pkl は「正しいpickle」として最小限の辞書を保存しておく（後で本物のモデルに置換）
    dummy_model: Dict[str, Any] = {
        "type": "dummy_model",
        "model_version": model_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "header_cols": header_cols,
        "row_count_a": row_count,
    }

    try:
        with open(out_pkl, "wb") as f:
            pickle.dump(dummy_model, f)
        print(f"[OUT] wrote pkl: {out_pkl}", flush=True)
    except Exception as e:
        print(f"[TRAIN][ERROR] failed to write pkl: {out_pkl} err={e}", flush=True)
        return 5

    print(f"[TRAIN] done {_now()} (smoke test artifacts created)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception:
        print("[TRAIN][FATAL] exception", flush=True)
        traceback.print_exc()
        sys.exit(1)
