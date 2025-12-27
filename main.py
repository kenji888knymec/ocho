import os
import time
import threading
import random
import json
import traceback
from typing import Optional, Dict, Any, List, Tuple, Set

import joblib
import numpy as np
import pandas as pd
import ccxt
import requests
import gc
import google.auth

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None

from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify

# ★設定は config.py から読む
from config import (
    VERSION,
    SPREADSHEET_ID, MAIN_SHEET_NAME, LEARN_SHEET_NAME,
    CAND_SIGMA, ALERT_SIGMA, AI_TH, DEFAULT_LEV,
    ENABLE_JUDGE, AUTO_JUDGE_AFTER_RUN, MIN_BARS,
    HEADER_COL_END, HEADER_LEN_TABLE, HEADER_LEN_LEARN,
    AUTO_FIX_HEADERS, AUTO_CREATE_SHEETS, STRICT_HEADER_CHECK,
    HEADER_TTL_SEC, SVC_TTL_SEC, COLCOUNT_TTL_SEC,
    DEDUP_LOOKBACK_ROWS, DEDUP_TTL_SEC,
    FETCH_RETRY, FETCH_RETRY_SLEEP_SEC,
    JUDGE_LOOKBACK_ROWS, EXCHANGE_TTL_SEC, OKX_DEFAULT_TYPE,
    RUN_MUTEX_ENABLED, RUN_MUTEX_SHEET, RUN_MUTEX_CELL, RUN_MUTEX_TTL_SEC,
    EXPECTED_HEADERS_LEARN, TABLE_FIELDS, FIELD_ALIASES, TABLE_REQUIRED_FIELDS,
    JST,
    RUN_MUTEX,
)

from discord_util import send_discord_message

# ★ここに以下の1行を追加してください
NOTIFY_CANDIDATES = (os.environ.get("NOTIFY_CANDIDATES", "0").strip() == "1")

# ==========================================================
# AIモデル運用の堅牢化
# ==========================================================
MODEL_VERSION = os.environ.get("MODEL_VERSION", "").strip() or VERSION
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "").strip()
MODEL_LOCAL_FALLBACK = os.environ.get("MODEL_LOCAL_FALLBACK", "trade_ai_model.pkl").strip()
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "").strip()

_model_lock = threading.Lock()
_model_obj = None
_model_info = {
    "enabled": bool(ENABLE_JUDGE),
    "loaded": False,
    "model_version": MODEL_VERSION,
    "source": "",
    "local_path": "",
    "sha256": "",
    "loaded_at": "",
    "error": "",
}
_startup_model_notify_done = False


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"MODEL_GCS_URI must start with gs:// got={gs_uri}")
    no_scheme = gs_uri[len("gs://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    if not bucket or not blob:
        raise ValueError(f"Invalid gs uri: {gs_uri}")
    return bucket, blob


def _download_model_from_gcs(gs_uri: str, dst_path: str) -> None:
    if storage is None:
        raise ImportError("google-cloud-storage is not installed. Add it to requirements.txt")

    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    blob.download_to_filename(dst_path)



def _resolve_model_path() -> Tuple[str, str]:
    # 明示指定があれば最優先（ローカルの絶対パス等）
    if MODEL_LOCAL_PATH:
        return MODEL_LOCAL_PATH, "local"

    # GCS指定があれば /tmp/models/{MODEL_VERSION}/trade_ai_model.pkl に落とす
    if MODEL_GCS_URI:
        dst_dir = f"/tmp/models/{MODEL_VERSION}"
        dst_path = os.path.join(dst_dir, "trade_ai_model.pkl")
        return dst_path, "gcs"

    # それ以外は repo 同梱のファイル（または Cloud Run の作業ディレクトリにある想定）
    return MODEL_LOCAL_FALLBACK, "local"


def load_ai_model_if_needed(force: bool = False) -> bool:
    global _model_obj
    with _model_lock:
        if not ENABLE_JUDGE:
            _model_info["enabled"] = False
            _model_info["loaded"] = False
            _model_info["error"] = "ENABLE_JUDGE is False"
            _model_obj = None
            return False

        if _model_obj is not None and _model_info.get("loaded") and not force:
            return True

        path, source = _resolve_model_path()
        try:
            if source == "gcs":
                _download_model_from_gcs(MODEL_GCS_URI, path)

            if not os.path.exists(path):
                raise FileNotFoundError(f"model file not found: {path}")

            obj = joblib.load(path)
            sha = _sha256_file(path)

            _model_obj = obj
            _model_info.update({
                "enabled": True, "loaded": True, "source": source,
                "local_path": path, "sha256": sha,
                "loaded_at": datetime.now(timezone.utc).isoformat(), "error": ""
            })
            return True
        except Exception as e:
            _model_obj = None
            _model_info.update({
                "loaded": False, "source": source, "local_path": path,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(e).__name__}: {e}"
            })
            return False


def get_ai_model() -> Optional[object]:
    """
    ロード済みAIモデルを返す。
    - グローバル変数名が環境や版で違っても落ちないように探索する
    - GCS由来で dict に包まれている場合は中身（pipeline等）を取り出す
    """
    g = globals()

    # まず「あり得る変数名」を順に探す（無ければ None）
    m = None
    for name in ("_ai_model", "AI_MODEL", "ai_model", "MODEL", "_MODEL", "model"):
        if name in g:
            m = g.get(name)
            break

    if m is None:
        return None

    # GCS の pkl が dict で {'pipeline': Pipeline, ...} のケースに対応
    if isinstance(m, dict):
        for k in ("pipeline", "model", "clf", "estimator", "sk_model"):
            v = m.get(k)
            if v is not None:
                return v
        return None

    return m



def _boot_marker_once_if_possible(ok: bool, err: str) -> None:
    try:
        if not MODEL_GCS_URI:
            return
        bucket_name, _ = _parse_gs_uri(MODEL_GCS_URI)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        key = os.environ.get("K_REVISION", "").strip() or MODEL_VERSION or "unknown"
        marker_name = f"markers/crypto-alert/boot/{key}.json"
        blob = bucket.blob(marker_name)
        body = {
            "ok": bool(ok),
            "service": os.environ.get("K_SERVICE", ""),
            "revision": os.environ.get("K_REVISION", ""),
            "model_version": _model_info.get("model_version", ""),
            "model_uri": MODEL_GCS_URI,
            "model_local_path": _model_info.get("local_path", ""),
            "loaded_at": _model_info.get("loaded_at", ""),
            "last_error": ("" if ok else (err or ""))[:500],
        }
        blob.upload_from_string(
            data=json.dumps(body, ensure_ascii=False),
            content_type="application/json",
            if_generation_match=0
        )
    except Exception as e:
        print(f"[WARN] boot marker skipped/failed: {e}")


def _startup_notify_model_status_once() -> None:
    """
    起動時のAIモデル通知を「必要な時だけ」に抑制する。
    - FAIL は常に通知（重要）
    - OK は「モデルが変わった時だけ」通知（再起動ノイズを消す）
    さらに、GCSへ状態を書き込むときに ifGenerationMatch を使わず上書きするため
    412 (conditionNotMet / If-Match) のノイズも消える。
    """
    global _startup_model_notify_done
    if _startup_model_notify_done:
        return

    ok = load_ai_model_if_needed(force=False)
    current_sha = str(_model_info.get("sha256") or "").strip()
    current_ver = str(MODEL_VERSION or "").strip()
    current_src = str(_model_info.get("source") or "").strip()
    current_err = str(_model_info.get("error") or "").strip()

    def _get_bucket_and_state_blob() -> Tuple[Optional[str], Optional[str]]:
        # MODEL_GCS_URI がない場合は永続抑制できないため、None を返す
        if not MODEL_GCS_URI:
            return None, None
        try:
            bucket_name, _ = _parse_gs_uri(MODEL_GCS_URI)
            # “前回通知状態”を固定ファイルに保存（上書き）
            blob_name = "markers/crypto-alert/boot_state.json"
            return bucket_name, blob_name
        except Exception:
            return None, None

    def _read_last_state(bucket_name: str, blob_name: str) -> Dict[str, Any]:
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if not blob.exists():
                return {}
            txt = blob.download_as_text()
            return json.loads(txt) if txt else {}
        except Exception:
            return {}

    def _write_state(bucket_name: str, blob_name: str, body: Dict[str, Any]) -> None:
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            # if_generation_match を使わず上書きする（→ 412を出さない）
            blob.upload_from_string(
                data=json.dumps(body, ensure_ascii=False),
                content_type="application/json",
            )
        except Exception as e:
            print(f"[WARN] boot state write skipped/failed: {e}")

    # ---- 通知要否判定（GCSに前回状態があるなら、それと比較してOKを抑制） ----
    bucket_name, state_blob = _get_bucket_and_state_blob()
    last = _read_last_state(bucket_name, state_blob) if (bucket_name and state_blob) else {}

    last_ok = bool(last.get("ok"))
    last_ver = str(last.get("model_version") or "").strip()
    last_sha = str(last.get("sha256") or "").strip()

    send_needed = False
    reason = ""

    if not ok:
        # FAILは必ず通知
        send_needed = True
        reason = "fail"
    else:
        # OKは「モデルが変わったら」通知
        if (not last_ok) or (last_ver != current_ver) or (last_sha != current_sha):
            send_needed = True
            reason = "changed"
        else:
            send_needed = False
            reason = "same"

    try:
        if send_needed:
            if ok:
                msg = (
                    f"[BOOT] AI model loaded: OK\n"
                    f"- model_version: {current_ver}\n"
                    f"- source: {current_src}\n"
                    f"- sha256: {current_sha[:12]}..."
                )
            else:
                msg = (
                    f"[BOOT] AI model loaded: FAIL\n"
                    f"- model_version: {current_ver}\n"
                    f"- error: {current_err}"
                )
            send_discord_message(msg)

        # ---- 状態は毎回GCSへ保存（次回の抑制判定に使う） ----
        if bucket_name and state_blob:
            body = {
                "ok": bool(ok),
                "reason": reason,
                "service": os.environ.get("K_SERVICE", ""),
                "revision": os.environ.get("K_REVISION", ""),
                "model_version": current_ver,
                "source": current_src,
                "sha256": current_sha,
                "loaded_at": _model_info.get("loaded_at", ""),
                "last_error": ("" if ok else current_err)[:500],
                "ts_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_state(bucket_name, state_blob, body)

        _startup_model_notify_done = True
    except Exception:
        pass



# ==========================================
# Flask設定（起動を軽くする）
# ==========================================
_boot_args = os.environ.get("GUNICORN_CMD_ARGS", "")
print(f"[BOOT] K_REVISION={os.environ.get('K_REVISION','')!r} PORT={os.environ.get('PORT','')!r}")
print(f"[BOOT] GUNICORN_CMD_ARGS={_boot_args!r}")
print(f"[BOOT] GUNICORN_CMD_ARGS_LEN={len(_boot_args)}")

app = Flask(__name__)

_startup_kick_done = False
_startup_kick_lock = threading.Lock()

def kickoff_startup_model_notify_once():
    """最初のリクエスト時に1回だけ、モデルロード＋通知を別スレッドで実行する。"""
    global _startup_kick_done
    with _startup_kick_lock:
        if _startup_kick_done:
            return
        _startup_kick_done = True

    def _task():
        try:
            _startup_notify_model_status_once()
        except Exception as e:
            print(f"[WARN] startup model notify failed: {e}")

    threading.Thread(target=_task, daemon=True).start()

@app.before_request
def _kickoff_on_first_request():
    kickoff_startup_model_notify_once()



# ==========================================
# グローバル & ヘルパー
# ==========================================
last_alert_records, last_candidate_records = {}, {}
_sheet_header_cache = {}
_sheet_service_cache = {"svc": None, "ts": 0.0}
_sheet_colcount_cache = {}
_dedup_cache = {}
_row_count_cache = {}
ROWCOUNT_TTL_SEC = 180

_exchange_cache, _symbol_resolve_cache = {"ex": None, "ts": 0.0}, {}
_run_lock = RUN_MUTEX
_INSTANCE_ID = "|".join([
    os.environ.get("K_SERVICE", "svc"),
    os.environ.get("K_REVISION", "rev"),
    os.environ.get("HOSTNAME", "host"),
    str(os.getpid())
])

_DT_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]


def parse_dt_any(s: str) -> Optional[datetime]:
    s = str(s or "").strip()
    if not s:
        return None
    if s.startswith("'"):
        s = s[1:].strip()
    s = s.replace("JST", "").strip()

    dt = None
    for fmt in _DT_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            break
        except Exception:
            pass

    if dt is None:
        try:
            ts = pd.to_datetime(s, errors="coerce")
            dt = None if pd.isna(ts) else ts.to_pydatetime()
        except Exception:
            dt = None

    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=JST)
    else:
        dt = dt.astimezone(JST)

    return dt


def normalize_dt_str(s: str) -> str:
    dt = parse_dt_any(s)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else str(s or "").strip()


def col_to_a1(idx0: int) -> str:
    idx, s = int(idx0), ""
    while True:
        idx, r = divmod(idx, 26)
        s = chr(65 + r) + s
        if idx == 0:
            break
        idx -= 1
    return s


def _invalidate_sheet_caches(sheet_name: str):
    for cache in [_row_count_cache, _dedup_cache, _sheet_header_cache]:
        cache.pop(sheet_name, None)


# ==========================================
# Sheets API 実行
# ==========================================
def sheets_execute(req, desc: str = ""):
    for attempt in range(7):
        try:
            return req.execute()
        except HttpError as e:
            st = int(getattr(e.resp, "status", 0) or 0)
            if st in (429, 500, 503) and attempt < 6:
                time.sleep(min(1.0 * (2 ** attempt), 20.0) + random.random())
                continue
            raise
        except Exception:
            if attempt < 6:
                time.sleep(min(1.0 * (2 ** attempt), 20.0) + random.random())
                continue
            raise

def sheets_append_or_raise(
    svc,
    spreadsheet_id: str,
    a1_range: str,
    rows: list,
    label: str,
):
    """
    Google Sheets append を実行し、成功/失敗を必ずログに出す。
    失敗時は握りつぶさず raise して Cloud Run 側にエラーを残す。
    """
    try:
        row_count = len(rows) if rows is not None else 0
        print(f"[SHEETS] append start label={label} range={a1_range} rows={row_count}")

        if row_count == 0:
            print(f"[SHEETS] skip append (0 rows) label={label}")
            return None

        body = {"values": rows}
        resp = (
            svc.spreadsheets()
            .values()
            .append(
                spreadsheetId=spreadsheet_id,
                range=a1_range,
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body=body,
            )
            .execute()
        )

        updates = resp.get("updates", {})
        print(
            "[SHEETS] append ok "
            f"label={label} updatedRows={updates.get('updatedRows')} "
            f"updatedCells={updates.get('updatedCells')} "
            f"updatedRange={updates.get('updatedRange')}"
        )
        return resp

    except Exception as e:
        print(f"[SHEETS][ERROR] append failed label={label} range={a1_range} err={e}")
        print("[SHEETS][ERROR] traceback:\n" + traceback.format_exc())
        raise






def get_sheet_service():
    now = time.time()
    if _sheet_service_cache["svc"] and (now - _sheet_service_cache["ts"]) <= SVC_TTL_SEC:
        return _sheet_service_cache["svc"]
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/spreadsheets"])
    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)
    _sheet_service_cache.update({"svc": svc, "ts": now})
    return svc


# ==========================================
# シート/ヘッダー管理
# ==========================================
def _get_sheets_meta():
    return sheets_execute(
        get_sheet_service().spreadsheets().get(
            spreadsheetId=SPREADSHEET_ID,
            fields="sheets(properties(sheetId,title,gridProperties(rowCount,columnCount)))"
        )
    )


def _resize_sheet_if_needed(sheet_name: str, min_rows: int, min_cols: int):
    try:
        meta = _get_sheets_meta()
        target = next((s["properties"] for s in meta.get("sheets", []) if s["properties"]["title"] == sheet_name), None)
        if not target:
            return False
        cur_r = int(target["gridProperties"]["rowCount"])
        cur_c = int(target["gridProperties"]["columnCount"])
        if cur_r >= min_rows and cur_c >= min_cols:
            return True
        req = {
            "requests": [
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": target["sheetId"],
                            "gridProperties": {
                                "rowCount": max(min_rows, cur_r),
                                "columnCount": max(min_cols, cur_c)
                            }
                        },
                        "fields": "gridProperties(rowCount,columnCount)"
                    }
                }
            ]
        }
        sheets_execute(get_sheet_service().spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=req))
        _invalidate_sheet_caches(sheet_name)
        return True
    except Exception:
        return False


def ensure_sheet_exists(sheet_name: str, min_rows: int = 1000, min_cols: int = 26):
    if not AUTO_CREATE_SHEETS:
        return True
    try:
        meta = _get_sheets_meta()
        if any(s["properties"]["title"] == sheet_name for s in meta.get("sheets", [])):
            return _resize_sheet_if_needed(sheet_name, min_rows, min_cols)
        req = {"requests": [{"addSheet": {"properties": {"title": sheet_name, "gridProperties": {"rowCount": min_rows, "columnCount": min_cols}}}}]}
        sheets_execute(get_sheet_service().spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=req))
        return _resize_sheet_if_needed(sheet_name, min_rows, min_cols)
    except Exception:
        return False


def get_sheet_colcount(sheet_name: str):
    now = time.time()
    if sheet_name in _sheet_colcount_cache and (now - _sheet_colcount_cache[sheet_name]["ts"]) <= COLCOUNT_TTL_SEC:
        return _sheet_colcount_cache[sheet_name]["n"]
    try:
        meta = _get_sheets_meta()
        target = next((s["properties"] for s in meta.get("sheets", []) if s["properties"]["title"] == sheet_name), None)
        if target:
            cc = int(target["gridProperties"]["columnCount"])
            _sheet_colcount_cache[sheet_name] = {"ts": now, "n": cc}
            return cc
    except Exception:
        pass
    return None


def read_header_row(sheet_name: str):
    try:
        res = sheets_execute(
            get_sheet_service().spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{sheet_name}!A1:{HEADER_COL_END}1"
            )
        )
        hs = [str(h or "").strip() for h in (res.get("values", [[]])[0])]
        while hs and not hs[-1]:
            hs.pop()
        if hs:
            return hs
    except Exception:
        pass
    return _sheet_header_cache.get(sheet_name, {}).get("headers", [])


def write_header_row(sheet_name: str, headers: List[str]):
    try:
        rng = f"{sheet_name}!A1:{col_to_a1(len(headers) - 1)}1"
        sheets_execute(
            get_sheet_service().spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=rng,
                valueInputOption="RAW",
                body={"values": [headers]}
            )
        )
        _invalidate_sheet_caches(sheet_name)
        return True
    except Exception:
        return False


def update_single_cell(sheet_name: str, col0: int, row1: int, value: Any):
    try:
        sheets_execute(
            get_sheet_service().spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{sheet_name}!{col_to_a1(col0)}{row1}",
                valueInputOption="RAW",
                body={"values": [[value]]}
            )
        )
        _invalidate_sheet_caches(sheet_name)
        return True
    except Exception:
        return False


def ensure_learn_headers():
    ensure_sheet_exists(LEARN_SHEET_NAME, 5000, 40)
    hs = read_header_row(LEARN_SHEET_NAME)
    if hs[:len(EXPECTED_HEADERS_LEARN)] == EXPECTED_HEADERS_LEARN:
        return True
    return write_header_row(LEARN_SHEET_NAME, EXPECTED_HEADERS_LEARN) if AUTO_FIX_HEADERS else (not STRICT_HEADER_CHECK)


def ensure_table_headers():
    ensure_sheet_exists(MAIN_SHEET_NAME, 20000, 40)
    hs = read_header_row(MAIN_SHEET_NAME)
    hm = {str(h).strip(): i for i, h in enumerate(hs) if str(h).strip()}
    if AUTO_FIX_HEADERS:
        for f in TABLE_REQUIRED_FIELDS:
            if f not in hm:
                for alias in FIELD_ALIASES.get(f, [f]):
                    if alias in hm:
                        update_single_cell(MAIN_SHEET_NAME, hm[alias], 1, f)
                        break
        hs = read_header_row(MAIN_SHEET_NAME)
        hm = {str(h).strip(): i for i, h in enumerate(hs) if str(h).strip()}
        for f in TABLE_REQUIRED_FIELDS:
            if f not in hm:
                idx = next((i for i, h in enumerate(hs + [""] * 10) if not h), -1)
                if idx == -1:
                    return False
                update_single_cell(MAIN_SHEET_NAME, idx, 1, f)
                hs = read_header_row(MAIN_SHEET_NAME)
                hm = {str(h).strip(): i for i, h in enumerate(hs) if str(h).strip()}
    return all(f in hm for f in TABLE_REQUIRED_FIELDS) if STRICT_HEADER_CHECK else True


def get_headers_and_len(sheet_name: str):
    now = time.time()
    if sheet_name in _sheet_header_cache and (now - _sheet_header_cache[sheet_name]["ts"]) <= HEADER_TTL_SEC:
        return _sheet_header_cache[sheet_name]["headers"], get_sheet_colcount(sheet_name), _sheet_header_cache[sheet_name]["ok"]

    hs = read_header_row(sheet_name)
    ok = True
    if STRICT_HEADER_CHECK:
        if sheet_name == LEARN_SHEET_NAME:
            ok = hs[:len(EXPECTED_HEADERS_LEARN)] == EXPECTED_HEADERS_LEARN
        else:
            hm = {str(h).strip(): i for i, h in enumerate(hs)}
            ok = all(any(a in hm for a in FIELD_ALIASES.get(f, [f])) for f in TABLE_REQUIRED_FIELDS)

    if hs:
        _sheet_header_cache[sheet_name] = {"headers": hs, "ts": now, "ok": ok}
    return hs, get_sheet_colcount(sheet_name), ok


def append_rows_to_sheet(sheet_name: str, rows_values: List[List[Any]], fields: List[str]):
    headers, cc, ok = get_headers_and_len(sheet_name)
    if not headers:
        if sheet_name == LEARN_SHEET_NAME:
            ensure_learn_headers()
            headers, cc, ok = list(EXPECTED_HEADERS_LEARN), get_sheet_colcount(sheet_name), True
        else:
            return
    if STRICT_HEADER_CHECK and not ok:
        return

    hm = {str(h).strip(): i for i, h in enumerate(headers)}
    idxs = [next((hm[a] for a in FIELD_ALIASES.get(f, [f]) if a in hm), -1) for f in fields]
    out_len = max(len(headers), (max(idxs) + 1) if idxs else 0)
    if cc and out_len > cc:
        return

    try:
        adj = []
        for rv in rows_values:
            row = [""] * out_len
            for f_idx, val in enumerate(rv):
                if idxs[f_idx] >= 0:
                    row[idxs[f_idx]] = val
            adj.append(row)
    
        sheets_execute(
            get_sheet_service().spreadsheets().values().append(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{sheet_name}!A1",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": adj},
            )
        )
    
        _invalidate_sheet_caches(sheet_name)
    
    except Exception as e:
        # 1) Cloud Run ログに「本当の原因」を必ず出す（ここが重要）
        print(f"[SHEETS][ERROR] append failed sheet={sheet_name} out_len={out_len} rows={len(rows_values)} err={repr(e)}")
    
        # 2) Discord にも短く出す（既存運用を維持）
        send_discord_message(f"[WARN] Append error {sheet_name}: {str(e)[:180]}")
    
        # 3) 原因特定を最短化するなら、失敗は成功扱いにせず落とす
        raise




# ==========================================
# 数値 / 取引所
# ==========================================
def to_float(val, default=None):
    try:
        s = str(val or "").strip().replace(",", "")
        if s.startswith("'"):
            s = s[1:].strip()
        v = float(s)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean().replace(0, np.nan)
    rsi = 100 - (100 / (1 + (gain / loss)))
    return rsi.replace([np.inf, -np.inf], np.nan)


# --- OKX REST 用の HTTP セッション（接続再利用で高速化） ---
_OKX_HTTP = requests.Session()
_OKX_HTTP.headers.update({
    "User-Agent": "CryptoAlertBot/1.0"
})

def build_exchange():
    """
    取引所オブジェクトは「REST直叩きの設定」を返すだけにする。
    ccxt.load_markets() を踏むと TypeError 起点で長引き、WORKER TIMEOUT / SIGKILL の原因になり得るため。
    """
    now = time.time()
    if _exchange_cache["ex"] and (now - _exchange_cache["ts"]) <= EXCHANGE_TTL_SEC:
        return _exchange_cache["ex"]

    default_type = (OKX_DEFAULT_TYPE or "").strip() or "swap"
    ex = {"defaultType": default_type}  # dictで十分
    _exchange_cache.update({"ex": ex, "ts": now})
    _symbol_resolve_cache.clear()
    return ex


def _normalize_symbol_to_base_quote(symbol: str) -> Tuple[str, str]:
    """
    - "BTC/USDT" -> ("BTC","USDT")
    - "BTC"      -> ("BTC","USDT")  ※シートは "BTC" なのでUSDTを既定に
    - "BTC/USDT:USDT" -> ("BTC","USDT")  ※suffix除去
    """
    s = str(symbol or "").strip()
    if not s:
        return "", "USDT"

    if ":" in s:
        s = s.split(":", 1)[0].strip()

    if "/" in s:
        base, quote = s.split("/", 1)
        return base.strip(), quote.strip()

    return s, "USDT"


def _okx_bar_from_timeframe(timeframe: str) -> str:
    tf = str(timeframe or "").strip()
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "1d": "1D",
    }
    return mapping.get(tf, "15m")


def _okx_inst_id(symbol: str, default_type: str) -> str:
    base, quote = _normalize_symbol_to_base_quote(symbol)
    if not base:
        return ""

    dt = (default_type or "").strip() or "swap"
    if dt == "spot":
        return f"{base}-{quote}"
    # 永続SWAP（あなたの想定）
    return f"{base}-{quote}-SWAP"


def _fetch_ohlcv_okx_rest(exchange_cfg: dict, symbol: str, timeframe: str, limit: int, since=None) -> Optional[List[List[Any]]]:
    """
    OKX REST: /api/v5/market/candles
    戻り: [[ts_ms, open, high, low, close, volume], ...]（昇順）
    """
    default_type = str(exchange_cfg.get("defaultType") or "").strip() or "swap"
    inst_id = _okx_inst_id(symbol, default_type)
    if not inst_id:
        return None

    bar = _okx_bar_from_timeframe(timeframe)
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": str(int(limit or 60)),
    }

    since_ms = int(since) if since is not None else None

    r = _OKX_HTTP.get(url, params=params, timeout=8)
    r.raise_for_status()
    js = r.json()

    data = js.get("data", [])
    if not data:
        return None

    out = []
    for row in data:
        if not row or len(row) < 6:
            continue
        ts = int(row[0])
        if since_ms is not None and ts < since_ms:
            continue
        out.append([ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])

    out.sort(key=lambda x: x[0])
    return out if out else None


def _resolve_okx_symbol(exchange, symbol: str):
    """
    互換のため残すが、REST直叩きでは markets 解決は不要。
    ここでは正規化だけ返す。
    """
    base, quote = _normalize_symbol_to_base_quote(symbol)
    if not base:
        return symbol
    return f"{base}/{quote}"


def fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, limit: int, since=None, retries=FETCH_RETRY):
    """
    最優先: OKX REST 直叩き
    （ccxt/load_markets 起点の TypeError と、長時間化→WORKER TIMEOUT を断つ）
    """
    last_err = None
    max_try = int(retries or 0) + 1

    for k in range(max_try):
        try:
            data = _fetch_ohlcv_okx_rest(exchange, symbol, timeframe, limit, since=since)
            return data
        except Exception as e:
            last_err = e
            if k < max_try - 1:
                time.sleep(float(FETCH_RETRY_SLEEP_SEC or 1) * (k + 1))
                continue

    print(f"[WARN] fetch_ohlcv_safe failed symbol={symbol} tf={timeframe} limit={limit} err={type(last_err).__name__}: {last_err}")
    return None



def _get_row_count_cached(sheet_name: str):
    now = time.time()
    if sheet_name in _row_count_cache and (now - _row_count_cache[sheet_name]["ts"]) <= ROWCOUNT_TTL_SEC:
        return _row_count_cache[sheet_name]["n"]
    try:
        hs, _, _ = get_headers_and_len(sheet_name)
        hm = {str(h).strip(): i for i, h in enumerate(hs)}
        col = col_to_a1(hm.get("Time", 0))
        res = sheets_execute(
            get_sheet_service().spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{sheet_name}!{col}:{col}"
            )
        )
        n = len(res.get("values", []))
        _row_count_cache[sheet_name] = {"ts": now, "n": n}
        return n
    except Exception:
        return 0


def _get_recent_dedup_keys(sheet_name: str):
    now = time.time()
    if sheet_name in _dedup_cache and (now - _dedup_cache[sheet_name]["ts"]) <= DEDUP_TTL_SEC:
        return _dedup_cache[sheet_name]["keys"]

    keys = set()
    try:
        last = _get_row_count_cached(sheet_name)
        if last >= 2:
            start = max(2, last - DEDUP_LOOKBACK_ROWS + 1)
            hs, _, _ = get_headers_and_len(sheet_name)
            hm = {str(h).strip(): i for i, h in enumerate(hs)}
            c_t, c_s = hm.get("Time", 0), hm.get("Symbol", 1)
            c1, c2 = min(c_t, c_s), max(c_t, c_s)
            res = sheets_execute(
                get_sheet_service().spreadsheets().values().get(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f"{sheet_name}!{col_to_a1(c1)}{start}:{col_to_a1(c2)}{last}"
                )
            )
            for r in res.get("values", []):
                t = str(r[c_t - c1]).strip() if len(r) > (c_t - c1) else ""
                s = str(r[c_s - c1]).strip() if len(r) > (c_s - c1) else ""
                if t and s:
                    keys.add(f"{s}|{normalize_dt_str(t)}")
    except Exception:
        pass

    _dedup_cache[sheet_name] = {"ts": now, "keys": keys}
    return keys


# ==========================================
# 分散ロック / 自己修復
# ==========================================
def _mutex_read():
    ensure_sheet_exists(RUN_MUTEX_SHEET, 50, 5)
    res = sheets_execute(
        get_sheet_service().spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{RUN_MUTEX_SHEET}!{RUN_MUTEX_CELL}"
        )
    )
    v = res.get("values", [[]])[0]
    return str(v[0]).strip() if v else ""


def _mutex_write(value: str):
    sheets_execute(
        get_sheet_service().spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{RUN_MUTEX_SHEET}!{RUN_MUTEX_CELL}",
            valueInputOption="RAW",
            body={"values": [[value]]}
        )
    )


def acquire_run_mutex():
    if not RUN_MUTEX_ENABLED:
        return True, ""
    token = f"{int(time.time())}|{_INSTANCE_ID}"
    try:
        cur = _mutex_read()
        if cur and (time.time() - int(float(cur.split("|")[0]))) <= RUN_MUTEX_TTL_SEC:
            return False, ""
        _mutex_write(token)
        time.sleep(0.25)
        return (True, token) if _mutex_read() == token else (False, "")
    except Exception:
        return True, ""


def release_run_mutex(token: str):
    if RUN_MUTEX_ENABLED and token:
        try:
            if _mutex_read() == token:
                _mutex_write("")
        except Exception:
            pass


def self_heal_prerequisites():
    try:
        ok_l = ensure_sheet_exists(RUN_MUTEX_SHEET, 50, 5)
        ok_ln = ensure_learn_headers()
        ok_t = ensure_table_headers()
        return (ok_l and ok_ln and ok_t), f"lock={ok_l} learn={ok_ln} table={ok_t}"
    except Exception as e:
        return False, str(e)


# ==========================================
# ロジック本体（あなたの版を維持）
# ==========================================
def logic_main():
    now_jst = datetime.now(JST)
    # ===== Debug（/run戻り値に原因を出す）=====
    RUN_DEBUG = str(os.environ.get("RUN_DEBUG", "0")).strip() in ("1", "true", "True", "yes", "on")
    dbg = {"btc_ok": 0, "sym_ok": 0, "fetch_none": 0, "too_short": 0, "loop_err": 0}
    # ========================================

    ok, msg = self_heal_prerequisites()
    if not ok:
        send_discord_message(f"[WARN] self_heal failed: {msg}")
        return f"Fail: {msg}"

    if (request.args.get("force") != "1") and (now_jst.minute % 15 >= 10):
        return "Waiting..."

    load_ai_model_if_needed(force=False)
    ai_model = get_ai_model()

    ex = build_exchange()

    btc_ok, btc_mode, median_sigma, btc_1h_change, btc_ret, btc_vol = False, "Range", 0.01, 0.0, 0.0, 0.0
    try:
        ohlcv = fetch_ohlcv_safe(ex, "BTC/USDT", "15m", 60)
        if ohlcv and len(ohlcv) >= MIN_BARS:
            df = pd.DataFrame(ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
            df["Pct"] = df["Close"].pct_change()
            df["Sig"] = df["Pct"].rolling(20).std().fillna(0.01).clip(lower=1e-4)
            median_sigma = float(df["Sig"].tail(20).median())
            c_now, c_1h = float(df.iloc[-2]["Close"]), float(df.iloc[-6]["Close"])
            btc_1h_change = (c_now - c_1h) / c_1h
            btc_ret, btc_vol, btc_ok = float(df.iloc[-2]["Pct"]), abs(float(df.iloc[-2]["Pct"])), True
            dbg["btc_ok"] += 1

            btc_mode = "Up" if btc_1h_change > 0.001 else "Down" if btc_1h_change < -0.001 else "Range"
    except Exception as e:
        print(f"[WARN] BTC fetch fail: {e}")

    BTC_CALM = bool(btc_ok and median_sigma < 0.005)

    alert_sigma_eff = float(ALERT_SIGMA) + (0.4 if not BTC_CALM else -0.2)
    if btc_mode == "Down":
        alert_sigma_eff += 0.1
    elif btc_mode == "Up":
        alert_sigma_eff += 0.05
    alert_sigma_eff = max(alert_sigma_eff, 1.0)

    symbols = [
        "BTC/USDT", "DOT/USDT", "BONK/USDT", "DOGE/USDT", "LINK/USDT", "ETH/USDT",
        "SUI/USDT", "BNB/USDT", "UNI/USDT", "ADA/USDT", "ATOM/USDT", "XRP/USDT",
        "NEAR/USDT", "LTC/USDT", "TRX/USDT", "SHIB/USDT", "HBAR/USDT", "SEI/USDT",
        "SOL/USDT", "AAVE/USDT", "AVAX/USDT", "APT/USDT", "FET/USDT", "ARB/USDT",
        "INJ/USDT", "POL/USDT", "STX/USDT", "XLM/USDT"
    ]

    pending_c, pending_a = [], []

    def calc_tp_sl(item):
        cp, sig = item["close"], item["sigma"]
        tp = cp * (1 + sig * 3.8) if item["is_buy"] else cp * (1 - sig * 3.8)
        sl = cp * (1 - sig * 1.5) if item["is_buy"] else cp * (1 + sig * 1.5)
        return tp, sl, abs(tp / cp - 1) * 100, abs(sl / cp - 1) * 100

    for sym in symbols:
        try:
            ohlcv = fetch_ohlcv_safe(ex, sym, "15m", 60)

            if not ohlcv:
                dbg["fetch_none"] += 1
                continue

            if len(ohlcv) < MIN_BARS:
                dbg["too_short"] += 1
                continue

            dbg["sym_ok"] += 1


            df = pd.DataFrame(ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
            df["Pct"] = df["Close"].pct_change()
            df["Sig"] = df["Pct"].rolling(20).std().fillna(0.01).clip(lower=1e-4)
            df["MA20"] = df["Close"].rolling(20).mean()

            df["BW"] = (df["MA20"] + 2 * df["Close"] * df["Sig"] - (df["MA20"] - 2 * df["Close"] * df["Sig"])) / df["MA20"]
            df["BWC"] = df["BW"].pct_change().fillna(0)

            df["RSI"] = calculate_rsi(df["Close"]).fillna(50)
            df["VolC"] = pd.to_numeric(df["Volume"]).pct_change().fillna(0)
            df["DS"] = df["Pct"].apply(lambda x: abs(x) if x < 0 else 0) / df["Sig"]
            df["RS"] = df["Pct"].apply(lambda x: abs(x) if x > 0 else 0) / df["Sig"]
            df["VMA"] = pd.to_numeric(df["Volume"]).rolling(20).mean()

            r = df.iloc[-2]
            vol_r = float(r["Volume"]) / float(r["VMA"]) if float(r["VMA"]) else 0.0

            is_b = (r["RS"] >= CAND_SIGMA and btc_mode != "Down" and r["Close"] > df.iloc[-6]["Close"])
            is_s = (r["DS"] >= CAND_SIGMA and btc_mode != "Up")
            if not (is_b or is_s):
                continue

            ai_s = None
            ai_err = ""
            if ai_model is not None:
                feats = pd.DataFrame([{
                    "Sigma": float(r["Sig"]),
                    "BandWidth": float(r["BW"]),
                    "BW_Change": float(r["BWC"]),
                    "RSI": float(r["RSI"]),
                    "Vol_Change": float(r["VolC"]),
                    "Rise_Score": float(r["RS"]),
                    "Drop_Score": float(r["DS"]),
                    "BTC_Ret": float(btc_ret),
                    "BTC_Vol": float(btc_vol),
                }])

                try:
                    # --- 重要：学習済みモデルが期待する列順に合わせる（列名ズレ/順番ズレ対策） ---
                    need_cols = getattr(ai_model, "feature_names_in_", None)
                    if need_cols is not None:
                        feats = feats.reindex(columns=list(need_cols), fill_value=0.0)

                    proba = ai_model.predict_proba(feats)
                    ai_s = float(proba[0][1])

                except Exception as e:
                    ai_err = f"{type(e).__name__}: {e}"
                    print(f"[AI][ERROR] predict_proba failed sym={sym} err={ai_err}")
                    print(f"[AI][ERROR] feats_cols={list(feats.columns)} feats={feats.to_dict(orient='records')}")
                    ai_s = None


            item = {
                "symbol": sym.replace("/USDT", ""),
                "time": int(r["Time"]),
                "is_buy": bool(is_b),
                "is_sell": bool(is_s),
                "close": float(r["Close"]),
                "score": float(max(r["DS"], r["RS"])),
                "sigma": float(r["Sig"]),
                "rsi": float(r["RSI"]),
                "type": "LONG" if is_b else "SHORT",
                "dt": datetime.fromtimestamp(int(r["Time"]) / 1000, JST),

                # AI
                "ai_score": ai_s,
                "ai_error": ai_err,
                # AIが有効なのに推論できていない場合は pass にしない（安全側）
                "ai_pass": (True if (ai_model is None) else (ai_s is not None)),

                "chg_pct": float(r["Pct"]) * 100,
                "vol_ratio": float(vol_r),
            }


            tp, sl, tp_p, sl_p = calc_tp_sl(item)
            item["E"] = (ai_s * tp_p - (1 - ai_s) * sl_p) if ai_s is not None else None

            if ai_model is not None:
                if ai_s is None:
                    # AIモデルがあるのに推論できていない → 明確に落とす
                    item["ai_pass"] = False
                else:
                    base_e_th = to_float(os.environ.get("E_TH", ""), default=0.0)
                    ath = float(AI_TH) + (0.07 if not BTC_CALM else -0.03)
                    eth = float(base_e_th) + (0.03 if not BTC_CALM else -0.02)

                    if (is_b and btc_mode == "Down") or (is_s and btc_mode == "Up"):
                        ath += 0.05
                        eth += 0.02

                    th = max(0.05, min(0.95, ath))

                    # デバッグ用
                    item["ai_th_used"] = th
                    item["e_th_used"] = eth
                    item["e_th_base"] = base_e_th

                    item["ai_pass"] = (
                        (item["E"] is not None) and
                        (ai_s >= th) and
                        (item["E"] > eth)
                    )



            pending_c.append(item)
            if item["ai_pass"] and item["score"] >= alert_sigma_eff:
                pending_a.append(item)

        except Exception as e:
            dbg["loop_err"] += 1
            print(f"[WARN] loop fail {sym}: {e}")
            continue


    l_keys, t_keys = _get_recent_dedup_keys(LEARN_SHEET_NAME), _get_recent_dedup_keys(MAIN_SHEET_NAME)

    c_rows = []
    for it in pending_c:
        dt_s = normalize_dt_str(it["dt"].strftime("%Y-%m-%d %H:%M:%S"))
        if it["symbol"] in last_candidate_records and last_candidate_records[it["symbol"]] == it["time"]:
            continue
        if f"{it['symbol']}|{dt_s}" in l_keys:
            continue

        last_candidate_records[it["symbol"]] = it["time"]
        tp, sl, tp_p, sl_p = calc_tp_sl(it)
        ai_has = (ai_model is not None)
        ai_err = str(it.get("ai_error") or "").strip()

        if ai_has and (it.get("ai_score") is None) and ai_err:
            status = "AI_ERROR"
        elif ai_has and (not it.get("ai_pass", True)):
            status = "AI_REJECT"
        else:
            status = "CANDIDATE"

        ai_str = f"{float(it['ai_score']) * 100:.1f}%" if it.get("ai_score") is not None else "N/A"
        e_str = f"{float(it['E']):+.2f}%" if it.get("E") is not None else ""
        err_str = f" err:{ai_err[:120]}" if ai_err else ""

        if it.get("ai_th_used") is not None and it.get("e_th_used") is not None:
            note = (
                f"AI:{ai_str} E:{e_str} "
                f"th:{float(it['ai_th_used']):.2f} "
                f"eth:{float(it['e_th_used']):.2f} "
                f"baseE:{float(it.get('e_th_base', 0.0)):.2f} "
                f"Pass:{it['ai_pass']} BTC:{btc_mode}{err_str}"
            )
        else:
            note = f"AI:{ai_str} E:{e_str} Pass:{it['ai_pass']} BTC:{btc_mode}{err_str}"



        row = [
            "'" + dt_s, it["symbol"], it["type"], it["close"], it["score"], it["sigma"],
            status, tp, sl, tp_p, sl_p, DEFAULT_LEV,
            it.get("ai_score") if it.get("ai_score") is not None else "",
            it.get("E") if it.get("E") is not None else "",
            it["ai_pass"], BTC_CALM, VERSION,
            it.get("signal_type", it["type"]),
            0, 0,
            "CALM" if BTC_CALM else "STORM", btc_mode, btc_1h_change, it["rsi"], note
        ] + [""] * 7
        c_rows.append(row[:len(EXPECTED_HEADERS_LEARN)])



    if c_rows:
        append_rows_to_sheet(LEARN_SHEET_NAME, c_rows, EXPECTED_HEADERS_LEARN)
        if NOTIFY_CANDIDATES:
            try:
                lines = []
                lines.append(
                    f"[CANDIDATES] n={len(pending_c)} BTC:{btc_mode} CALM:{BTC_CALM} Ver:{VERSION}"
                )
                for it2 in pending_c[:10]:
                    sym = str(it2.get("symbol", ""))
                    side = str(it2.get("type", ""))
                    sc = it2.get("score", None)
                    sg = it2.get("sigma", None)
                    rsi = it2.get("rsi", None)
                    ai_val = it2.get("ai_score", None)
                    e_val = it2.get("E", None)
    
                    sc_s = f"{float(sc):.2f}" if sc is not None and sc != "" else ""
                    sg_s = f"{float(sg):.2f}" if sg is not None and sg != "" else ""
                    rsi_s = f"{float(rsi):.1f}" if rsi is not None and rsi != "" else ""
                    ai_s = f"{float(ai_val) * 100:.1f}%" if ai_val is not None and ai_val != "" else "N/A"
                    e_s = f"{float(e_val):+.2f}%" if e_val is not None and e_val != "" else ""
    
                    lines.append(f"- {sym} {side} score={sc_s} sigma={sg_s} ai={ai_s} E={e_s} rsi={rsi_s}")
    
                send_discord_message("\n".join(lines))
            except Exception:
                pass
    
    a_rows = []
    for it in sorted(pending_a, key=lambda x: x["score"], reverse=True)[:3]:
        dt_s = normalize_dt_str(it["dt"].strftime("%Y-%m-%d %H:%M:%S"))
        if it["symbol"] in last_alert_records and last_alert_records[it["symbol"]] == it["time"]:
            continue
        if f"{it['symbol']}|{dt_s}" in t_keys:
            continue

        last_alert_records[it["symbol"]] = it["time"]
        tp, sl, tp_p, sl_p = calc_tp_sl(it)
        ai_str = f"{float(it['ai_score']) * 100:.1f}%" if it.get("ai_score") is not None else "N/A"
        side_str = "🚀 LONG" if it["is_buy"] else "☄️ SHORT"

        msg = (
            f"{side_str} {it['symbol']}\n"
            f"Score:{it['score']:.2f} AI:{ai_str}\n"
            f"BTC:{btc_mode} Calm:{BTC_CALM}\n"
            f"Price:{it['close']:.4f}\n"
            f"TP:{tp:.4f} ({tp_p:.2f}%)\n"
            f"SL:{sl:.4f} ({sl_p:.2f}%)"
        )
        send_discord_message(msg)

        note = f"{it['type']} | AI:{ai_str} | BTC:{btc_mode}"
        a_rows.append([
            "'" + dt_s, it["symbol"], it["type"], it["close"], it["score"], it["sigma"],
            "AI_PASS", tp, sl, tp_p, sl_p, DEFAULT_LEV, tp_p * DEFAULT_LEV, sl_p * DEFAULT_LEV,
            BTC_CALM, True, VERSION, note, it["chg_pct"], it["vol_ratio"], "MARKET"
        ])

    if a_rows:
        append_rows_to_sheet(MAIN_SHEET_NAME, a_rows, TABLE_FIELDS)

    res = f"{VERSION} | Alerts:{len(a_rows)} Candidates:{len(c_rows)}"
    if RUN_DEBUG:
        res += (
            f" | DBG btc_ok={dbg['btc_ok']} sym_ok={dbg['sym_ok']}"
            f" fetch_none={dbg['fetch_none']} too_short={dbg['too_short']} loop_err={dbg['loop_err']}"
        )
    return res




# ==========================================
# 判定係 / 分析
# ==========================================
def _hm_pick_index(hm: Dict[str, int], names: List[str]) -> Optional[int]:
    for n in names:
        if n in hm:
            return hm[n]
    return None


def judge_sheet(sheet_name: str, lookback=JUDGE_LOOKBACK_ROWS, max_j=30):
    svc, ex = get_sheet_service(), build_exchange()
    hs, _, okh = get_headers_and_len(sheet_name)
    if STRICT_HEADER_CHECK and not okh:
        return 0

    last = _get_row_count_cached(sheet_name)
    if last < 2:
        return 0

    start = max(2, last - lookback + 1)
    res = sheets_execute(
        svc.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!A{start}:{HEADER_COL_END}{last}"
        )
    )
    rows = res.get("values", [])
    if not rows:
        return 0

    hm = {str(h).strip(): i for i, h in enumerate(hs) if str(h).strip()}

    # EvalStatus 列は固定名が多いが、念のため候補で拾う
    c_ev = _hm_pick_index(hm, ["EvalStatus", "Eval", "Status"])
    if c_ev is None:
        return 0

    # 列名揺れを吸収（Time / Direction など）
    idx_symbol = _hm_pick_index(hm, ["Symbol"])
    idx_time = _hm_pick_index(hm, ["Time", "Datetime(SymbolTime_JST)", "Datetime", "SymbolTime_JST", "DateTime", "Timestamp"])
    idx_entry = _hm_pick_index(hm, ["EntryPrice", "Entry", "Entry_Price", "Price"])
    idx_tp = _hm_pick_index(hm, ["TP_Price", "TP", "TPPrice"])
    idx_sl = _hm_pick_index(hm, ["SL_Price", "SL", "SLPrice"])
    idx_dir = _hm_pick_index(hm, ["Direction", "Side", "SignalType", "Type"])

    if any(x is None for x in [idx_symbol, idx_time, idx_entry, idx_tp, idx_sl, idx_dir]):
        return 0

    updates, judged = [], 0
    for i, row in enumerate(reversed(rows)):
        if judged >= max_j:
            break

        # EvalStatus列が存在し、かつ値が入っている行はスキップ
        if len(row) > c_ev and str(row[c_ev]).strip():
            continue

        max_idx = max(idx_symbol, idx_time, idx_entry, idx_tp, idx_sl, idx_dir)
        if len(row) <= max_idx:
            continue

        sym = str(row[idx_symbol]).strip()
        t_raw = row[idx_time]
        entry = to_float(row[idx_entry])
        tp = to_float(row[idx_tp])
        sl = to_float(row[idx_sl])

        if not (sym and entry and tp and sl):
            continue

        dt = parse_dt_any(t_raw)
        if not dt:
            continue

        since = int((dt + timedelta(minutes=15)).timestamp() * 1000)
        side = "LONG" if "LONG" in str(row[idx_dir]).upper() else "SHORT"

        candles = fetch_ohlcv_safe(ex, sym, "15m", 100, since=since)
        if not candles:
            continue

        res_s, res_w, res_r, res_p = "PENDING", "", "", None
        for ts, o, h, l, c, v in candles:
            tp_h, sl_h = (h >= tp, l <= sl) if side == "LONG" else (l <= tp, h >= sl)
            if tp_h and sl_h:
                res_s, res_r, res_p = "AMBIGUOUS", "Both", entry
                break
            if tp_h:
                res_s, res_w, res_r, res_p = "DONE", "Win", "TP", tp
                break
            if sl_h:
                res_s, res_w, res_r, res_p = "DONE", "Lose", "SL", sl
                break

        if res_s == "PENDING" and len(candles) >= 96:
            res_s, res_r = "EXPIRED", "TimeOver"

        if res_s in ("DONE", "AMBIGUOUS", "EXPIRED"):
            ridx = start + len(rows) - 1 - i
            pnl = (res_p / entry - 1) * 100 if (entry and res_p and res_s == "DONE") else 0
            if side == "SHORT":
                pnl = -pnl

            for col_n, val in [
                ("EvalStatus", res_s),
                ("Win/Lose", res_w),
                ("ExitPrice", res_p),
                ("PnL_Pct", pnl),
                ("ExitReason", res_r),
            ]:
                c_idx = _hm_pick_index(hm, [col_n])
                if c_idx is not None:
                    updates.append({"range": f"{sheet_name}!{col_to_a1(c_idx)}{ridx}", "values": [[val]]})

            judged += 1

    if updates:
        sheets_execute(
            svc.spreadsheets().values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"valueInputOption": "USER_ENTERED", "data": updates}
            )
        )
    return judged


def judge_main():
    return f"Judged:{judge_sheet(MAIN_SHEET_NAME) + judge_sheet(LEARN_SHEET_NAME)}"


def _e_report(days=30):
    try:
        creds = google.auth.default(scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])[0]
        svc = build("sheets", "v4", credentials=creds, cache_discovery=False)
        rows = svc.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{LEARN_SHEET_NAME}!A:ZZ"
        ).execute().get("values", [])

        if len(rows) < 2:
            return "Empty"

        headers = [str(h or "").strip() for h in rows[0]]
        hm = {h: i for i, h in enumerate(headers) if h}

        for c in ["EvalStatus", "Reserved2", "PnL_Pct"]:
            if c not in hm:
                return f"Missing column: {c}"

        time_candidates = ["Time", "Datetime(SymbolTime_JST)", "Datetime", "SymbolTime_JST", "DateTime", "Timestamp"]
        time_col = next((hm[name] for name in time_candidates if name in hm), 0)

        cutoff_utc = datetime.now(timezone.utc) - timedelta(days=int(days))

        data = []
        for r in rows[1:]:
            max_need = max(hm["Reserved2"], hm["PnL_Pct"], hm["EvalStatus"], time_col)
            if len(r) <= max_need:
                continue

            if str(r[hm["EvalStatus"]]).strip().upper() != "DONE":
                continue

            dt = parse_dt_any(r[time_col])
            if not dt:
                continue
            dt_utc = dt.astimezone(timezone.utc)
            if dt_utc < cutoff_utc:
                continue

            e = to_float(r[hm["Reserved2"]])
            pnl = to_float(r[hm["PnL_Pct"]])
            if e is not None and pnl is not None:
                data.append((e, pnl))

        if not data:
            return "No Data in target period"

        es = sorted([x[0] for x in data])
        q33 = es[int(len(es) * 0.33)]
        q67 = es[int(len(es) * 0.67)]

        stats = {"LOW": [0, 0, 0.0], "MID": [0, 0, 0.0], "HIGH": [0, 0, 0.0]}
        for e, pnl in data:
            b = "LOW" if e <= q33 else "HIGH" if e > q67 else "MID"
            stats[b][0] += 1
            stats[b][1] += 1 if pnl > 0 else 0
            stats[b][2] += pnl

        res = f"[E_REPORT] {int(days)} days\n"
        for k, v in stats.items():
            if v[0]:
                res += f"{k}: n={v[0]} WR:{v[1]/v[0]:.2f} Avg:{v[2]/v[0]:.2f}%\n"
            else:
                res += f"{k}: n=0\n"
        return res

    except Exception as e:
        return str(e)


# ==========================================
# ルーティング
# ==========================================
@app.get("/ai_health")
def ai_health():
    load_ai_model_if_needed(force=False)
    m = get_ai_model()
    ok = (m is not None)
    payload = {
        "ok": bool(ok), "service": os.environ.get("K_SERVICE", ""),
        "revision": os.environ.get("K_REVISION", ""), "model_version": _model_info.get("model_version", ""),
        "model_uri": MODEL_GCS_URI, "model_local_path": _model_info.get("local_path", ""),
        "loaded_at": _model_info.get("loaded_at", ""), "last_error": _model_info.get("error", ""),
    }
    return jsonify(payload), (200 if ok else 503)
@app.get("/ai_smoke")
def ai_smoke():
    load_ai_model_if_needed(force=False)
    m = get_ai_model()

    # GCSのモデルは dict ラップになっている（例: {"pipeline": Pipeline, ...}）
    if isinstance(m, dict):
        for k in ("pipeline", "model", "clf", "estimator", "sk_model"):
            if k in m:
                m = m[k]
                break

    if (m is None) or (not hasattr(m, "predict_proba")):
        return jsonify({
            "ok": False,
            "error": f"model invalid type={type(m).__name__} (need predict_proba)",
            "model_version": _model_info.get("model_version", ""),
            "last_error": _model_info.get("error", ""),
        }), 503


    feats = pd.DataFrame([{
        "Sigma": 0.0020,
        "BandWidth": 0.0300,
        "BW_Change": 0.0010,
        "RSI": 55.0,
        "Vol_Change": 0.10,
        "Rise_Score": 1.20,
        "Drop_Score": 0.80,
        "BTC_Ret": 0.0010,
        "BTC_Vol": 0.0010,
    }])

    try:
        # --- 重要: 学習済みモデルが期待する列順に合わせる（列名ズレ/順番ズレ対策） ---
        need_cols = getattr(m, "feature_names_in_", None)
        if need_cols is not None:
            feats = feats.reindex(columns=list(need_cols), fill_value=0.0)

        proba = m.predict_proba(feats)
        ai_s = float(proba[0][1])

        return jsonify({
            "ok": True,
            "ai_score": ai_s,
            "model_version": _model_info.get("model_version", ""),
            "source": _model_info.get("source", ""),
            "sha256": _model_info.get("sha256", ""),
        }), 200

    except Exception as e:
        ai_err = f"{type(e).__name__}: {e}"
        print(f"[AI][ERROR] predict_proba failed err={ai_err}")
        print(f"[AI][ERROR] feats_cols={list(feats.columns)} feats={feats.to_dict(orient='records')}")

        return jsonify({
            "ok": False,
            "error": ai_err,
            "model_version": _model_info.get("model_version", ""),
            "source": _model_info.get("source", ""),
            "sha256": _model_info.get("sha256", ""),
        }), 500




@app.route("/run")
def run_route():
    if not _run_lock.acquire(blocking=False):
        return "Busy", 429
    token = ""
    try:
        ok, t = acquire_run_mutex()
        if not ok:
            return "Busy Mutex", 429
        token = t
        res = str(logic_main())
        if AUTO_JUDGE_AFTER_RUN:
            res += " | " + str(judge_main())
        return res, 200
    finally:
        release_run_mutex(token)
        _run_lock.release()


@app.route("/judge")
def judge_route():
    if not _run_lock.acquire(blocking=False):
        return "Busy", 429
    token = ""
    try:
        ok, t = acquire_run_mutex()
        if not ok:
            return "Busy Mutex", 429
        token = t
        return str(judge_main()), 200
    finally:
        release_run_mutex(token)
        _run_lock.release()


@app.route("/e_report")
def e_report_route():
    return _e_report(int(request.args.get("days", 30))), 200


@app.route("/health")
@app.route("/healthz")
def health():
    return "ok", 200


@app.route("/")
def home():
    return f"{VERSION} Active", 200


@app.route("/preflight")
def preflight():
    ok, msg = self_heal_prerequisites()
    return (f"OK: {msg}", 200) if ok else (f"NG: {msg}", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))






















