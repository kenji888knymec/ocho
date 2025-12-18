import os
import time
import threading
import random
import json
from typing import Optional, Dict, Any, List, Tuple, Set

import joblib
import numpy as np
import pandas as pd
import ccxt
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import storage
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify


# ★追加：設定は config.py から読む（値は同じ）
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


# ★追加：Discord送信は discord_util.py に分離（中身は同じ）
from discord_util import send_discord_message

# ==========================================
# Flask設定（Buildpacks標準：main.py の app を起動）
# ==========================================
app = Flask(__name__)
AI_MODEL = None

_AI_LOADED_AT = ""
_AI_LAST_ERROR = ""


@app.get("/ai_health")
def ai_health():
    global ai_model
    ok = ai_model is not None

    payload = {
        "ok": bool(ok),
        "service": os.environ.get("K_SERVICE", ""),
        "revision": os.environ.get("K_REVISION", ""),
        "model_version": os.environ.get("MODEL_VERSION", "").strip(),
        "model_uri": os.environ.get("MODEL_GCS_URI", "").strip(),
        "model_local_path": os.environ.get("MODEL_LOCAL_PATH", "/tmp/trade_ai_model.pkl").strip(),
        "loaded_at": _AI_LOADED_AT,
        "last_error": _AI_LAST_ERROR,
    }
    status = 200 if ok else 503
    return jsonify(payload), status



# ==========================================
# グローバル
# ==========================================
ai_model = None

last_alert_records: Dict[str, int] = {}
last_candidate_records: Dict[str, int] = {}

_sheet_header_cache: Dict[str, Dict[str, Any]] = {}
_sheet_service_cache: Dict[str, Any] = {"svc": None, "ts": 0.0}
_sheet_colcount_cache: Dict[str, Dict[str, Any]] = {}
_dedup_cache: Dict[str, Dict[str, Any]] = {}
_row_count_cache: Dict[str, Dict[str, Any]] = {}
ROWCOUNT_TTL_SEC = 180


_exchange_cache: Dict[str, Any] = {"ex": None, "ts": 0.0}
_symbol_resolve_cache: Dict[str, str] = {}

_run_lock = RUN_MUTEX

_INSTANCE_ID = "|".join(
    [
        os.environ.get("K_SERVICE", "svc"),
        os.environ.get("K_REVISION", "rev"),
        os.environ.get("HOSTNAME", "host"),
        str(os.getpid()),
    ]
)

# ==========================================
# 日時の正規化
# ==========================================
_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
]

def parse_dt_any(s: str) -> Optional[datetime]:
    s = ("" if s is None else str(s)).strip()
    if not s:
        return None
    if s.startswith("'"):
        s = s[1:].strip()
    s = s.replace("JST", "").strip()

    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None

def normalize_dt_str(s: str) -> str:
    dt = parse_dt_any(s)
    if dt is None:
        return ("" if s is None else str(s)).strip()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ===========================
# A1列変換（0-based）
# ===========================
def col_to_a1(idx0: int) -> str:
    idx = int(idx0)
    if idx < 0:
        return "A"
    s = ""
    while True:
        idx, r = divmod(idx, 26)
        s = chr(65 + r) + s
        if idx == 0:
            break
        idx -= 1
    return s

def _invalidate_sheet_caches(sheet_name: str):
    _row_count_cache.pop(sheet_name, None)
    _dedup_cache.pop(sheet_name, None)
    _sheet_header_cache.pop(sheet_name, None)

# ==========================================
# Sheets execute wrapper（429/5xx対策）
# ==========================================
SHEETS_RETRY_MAX = 6
SHEETS_RETRY_BASE_SEC = 1.0
SHEETS_RETRY_MAX_SLEEP_SEC = 20.0

def _http_status(err: Exception) -> int:
    try:
        resp = getattr(err, "resp", None)
        st = getattr(resp, "status", 0)
        return int(st) if st is not None else 0
    except Exception:
        return 0

def sheets_execute(req, desc: str = ""):
    last = None
    for attempt in range(SHEETS_RETRY_MAX + 1):
        try:
            return req.execute()
        except HttpError as e:
            last = e
            st = _http_status(e)
            retryable = st in (429, 500, 503)
            if retryable and attempt < SHEETS_RETRY_MAX:
                sleep = min(SHEETS_RETRY_BASE_SEC * (2 ** attempt), SHEETS_RETRY_MAX_SLEEP_SEC)
                sleep += random.random()
                print(f"[WARN] Sheets retry {attempt+1}/{SHEETS_RETRY_MAX} sleep={sleep:.1f}s status={st} {desc}")
                time.sleep(sleep)
                continue
            raise
        except Exception as e:
            last = e
            if attempt < SHEETS_RETRY_MAX:
                sleep = min(SHEETS_RETRY_BASE_SEC * (2 ** attempt), SHEETS_RETRY_MAX_SLEEP_SEC)
                sleep += random.random()
                print(f"[WARN] Sheets retry {attempt+1}/{SHEETS_RETRY_MAX} sleep={sleep:.1f}s {desc} err={e}")
                time.sleep(sleep)
                continue
            raise last

# ==========================================
# Sheets: service
# ==========================================
def get_sheet_service():
    now = time.time()
    svc = _sheet_service_cache.get("svc")
    ts = float(_sheet_service_cache.get("ts", 0))
    if svc is not None and (now - ts) <= SVC_TTL_SEC:
        return svc
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/spreadsheets"])
    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)
    _sheet_service_cache["svc"] = svc
    _sheet_service_cache["ts"] = now
    return svc

def _normalize_headers(headers: List[Any]) -> List[str]:
    hs = [("" if h is None else str(h)).strip() for h in headers]
    while hs and hs[-1] == "":
        hs.pop()
    return hs

def _build_headers_map(headers: List[str]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for i, h in enumerate(headers):
        key = ("" if h is None else str(h)).strip()
        if key != "":
            m[key] = i
    return m

def _resolve_col_idx(headers_map: Dict[str, int], field: str) -> int:
    for cand in FIELD_ALIASES.get(field, [field]):
        if cand in headers_map:
            return int(headers_map[cand])
    return -1

def _expected_header_len(sheet_name: str) -> int:
    if sheet_name == MAIN_SHEET_NAME:
        return HEADER_LEN_TABLE
    if sheet_name == LEARN_SHEET_NAME:
        return HEADER_LEN_LEARN
    return 0

# ==========================================
# シート存在確認・作成（_lock等）
# ==========================================
def _get_sheets_meta() -> Dict[str, Any]:
    service = get_sheet_service()
    return sheets_execute(
        service.spreadsheets().get(
            spreadsheetId=SPREADSHEET_ID,
            fields="sheets(properties(sheetId,title,gridProperties(rowCount,columnCount)))",
        ),
        desc="_get_sheets_meta",
    )

def ensure_sheet_exists(sheet_name: str, min_rows: int = 1000, min_cols: int = 26) -> bool:
    if not AUTO_CREATE_SHEETS:
        return True

    try:
        meta = _get_sheets_meta()
        for sh in meta.get("sheets", []) or []:
            p = (sh or {}).get("properties", {}) or {}
            if p.get("title") == sheet_name:
                return True

        service = get_sheet_service()
        req = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheet_name,
                            "gridProperties": {"rowCount": int(min_rows), "columnCount": int(min_cols)},
                        }
                    }
                }
            ]
        }
        sheets_execute(
            service.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=req),
            desc=f"ensure_sheet_exists addSheet={sheet_name}",
        )
        print(f"[CFG] created sheet: {sheet_name}")
        return True
    except Exception as e:
        print(f"[WARN] ensure_sheet_exists failed: sheet={sheet_name} err={e}")
        return False

def get_sheet_colcount(sheet_name: str) -> Optional[int]:
    now = time.time()
    c = _sheet_colcount_cache.get(sheet_name)
    if c and (now - float(c.get("ts", 0))) <= COLCOUNT_TTL_SEC:
        n = c.get("n")
        return int(n) if isinstance(n, int) and n > 0 else None

    try:
        meta = _get_sheets_meta()
        for sh in meta.get("sheets", []) or []:
            p = (sh or {}).get("properties", {}) or {}
            if p.get("title") == sheet_name:
                cc = int(((p.get("gridProperties") or {}).get("columnCount") or 0))
                if cc > 0:
                    _sheet_colcount_cache[sheet_name] = {"ts": now, "n": cc}
                    return cc
                break
    except Exception as e:
        print(f"[WARN] get_sheet_colcount failed: sheet={sheet_name} err={e}")

    fs = _expected_header_len(sheet_name)
    return fs if fs > 0 else None

def read_header_row(sheet_name: str) -> List[str]:
    try:
        service = get_sheet_service()
        rng = f"{sheet_name}!A1:{HEADER_COL_END}1"
        res = sheets_execute(
            service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=rng),
            desc=f"read_header_row sheet={sheet_name}",
        )
        raw = (res.get("values", [[]]) or [[]])[0]
        hs = _normalize_headers(raw)
        if hs:
            return hs

        # ここ重要：一時的に空が返ってもキャッシュを使って壊しにくくする
        c = _sheet_header_cache.get(sheet_name)
        prev = (c or {}).get("headers", [])
        return prev if prev else []

    except Exception as e:
        print(f"[WARN] read_header_row failed: sheet={sheet_name} err={e}")
        c = _sheet_header_cache.get(sheet_name)
        prev = (c or {}).get("headers", [])
        return prev if prev else []

def write_header_row(sheet_name: str, headers: List[str]) -> bool:
    try:
        service = get_sheet_service()
        end_col = col_to_a1(max(len(headers) - 1, 0))
        rng = f"{sheet_name}!A1:{end_col}1"
        sheets_execute(
            service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=rng,
                valueInputOption="RAW",
                body={"values": [headers]},
            ),
            desc=f"write_header_row sheet={sheet_name}",
        )
        _invalidate_sheet_caches(sheet_name)
        return True
    except Exception as e:
        print(f"[WARN] write_header_row failed: sheet={sheet_name} err={e}")
        return False

def update_single_cell(sheet_name: str, col0: int, row1: int, value: str) -> bool:
    try:
        service = get_sheet_service()
        a1 = col_to_a1(col0)
        rng = f"{sheet_name}!{a1}{row1}"
        sheets_execute(
            service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=rng,
                valueInputOption="RAW",
                body={"values": [[value]]},
            ),
            desc=f"update_single_cell sheet={sheet_name} cell={rng}",
        )
        _invalidate_sheet_caches(sheet_name)
        return True
    except Exception as e:
        print(f"[WARN] update_single_cell failed: sheet={sheet_name} err={e}")
        return False

# ==========================================
# ヘッダー自動修復
# ==========================================
def ensure_learn_headers() -> bool:
    ensure_sheet_exists(LEARN_SHEET_NAME, min_rows=5000, min_cols=max(32, HEADER_LEN_LEARN, 40))

    headers = read_header_row(LEARN_SHEET_NAME)
    if headers[:len(EXPECTED_HEADERS_LEARN)] == EXPECTED_HEADERS_LEARN:
        return True

    if not AUTO_FIX_HEADERS:
        return not STRICT_HEADER_CHECK

    ok = write_header_row(LEARN_SHEET_NAME, EXPECTED_HEADERS_LEARN)
    if ok:
        print("[CFG] learn_log headers fixed.")
    return ok

def _find_first_blank_index(headers: List[str], limit: Optional[int]) -> int:
    max_i = len(headers) if limit is None else min(len(headers), limit)
    for i in range(max_i):
        if str(headers[i]).strip() == "":
            return i
    return -1

def ensure_table_headers() -> bool:
    ensure_sheet_exists(MAIN_SHEET_NAME, min_rows=20000, min_cols=max(HEADER_LEN_TABLE, 40))

    headers = read_header_row(MAIN_SHEET_NAME)
    colcount = get_sheet_colcount(MAIN_SHEET_NAME)
    hm = _build_headers_map(headers)

    if AUTO_FIX_HEADERS:
        for canonical in TABLE_REQUIRED_FIELDS:
            if canonical in hm:
                continue
            for alias in FIELD_ALIASES.get(canonical, [canonical]):
                if alias in hm:
                    idx = int(hm[alias])
                    if str(alias).strip() != canonical:
                        update_single_cell(MAIN_SHEET_NAME, idx, 1, canonical)
                    break

        headers = read_header_row(MAIN_SHEET_NAME)
        hm = _build_headers_map(headers)
        for canonical in TABLE_REQUIRED_FIELDS:
            if canonical in hm:
                continue
            limit = int(colcount) if isinstance(colcount, int) and colcount > 0 else len(headers)
            blank_idx = _find_first_blank_index(headers + [""] * 5, limit)
            if blank_idx == -1:
                msg = f"[WARN] table missing required col '{canonical}' and no blank header cell. Please add a blank column."
                print(msg)
                send_discord_message(msg)
                return False
            update_single_cell(MAIN_SHEET_NAME, blank_idx, 1, canonical)
            headers = read_header_row(MAIN_SHEET_NAME)
            hm = _build_headers_map(headers)

    headers = read_header_row(MAIN_SHEET_NAME)
    hm = _build_headers_map(headers)
    missing = [f for f in TABLE_REQUIRED_FIELDS if _resolve_col_idx(hm, f) == -1]
    if missing:
        msg = f"[WARN] table headers still missing: {missing}"
        print(msg)
        send_discord_message(msg)
        return (not STRICT_HEADER_CHECK)

    return True

def get_headers_and_len(sheet_name: str) -> Tuple[List[str], Optional[int], bool]:
    now = time.time()

    if sheet_name in _sheet_header_cache:
        c = _sheet_header_cache[sheet_name]
        ts = float(c.get("ts", 0))
        if (now - ts) <= HEADER_TTL_SEC:
            headers = c.get("headers", [])
            ok = bool(c.get("ok", True))
            colcount = get_sheet_colcount(sheet_name)
            return headers, colcount, ok

    headers = read_header_row(sheet_name)
    colcount = get_sheet_colcount(sheet_name)

    ok = True
    if STRICT_HEADER_CHECK:
        if sheet_name == LEARN_SHEET_NAME:
            ok = (headers[:len(EXPECTED_HEADERS_LEARN)] == EXPECTED_HEADERS_LEARN)
        elif sheet_name == MAIN_SHEET_NAME:
            hm = _build_headers_map(headers)
            ok = all(_resolve_col_idx(hm, f) != -1 for f in TABLE_REQUIRED_FIELDS)

    # ここ重要：headersが空の時にキャッシュを空で上書きしない（壊れにくくする）
    if headers:
        _sheet_header_cache[sheet_name] = {"headers": headers, "ts": now, "ok": ok}
    else:
        if sheet_name not in _sheet_header_cache:
            _sheet_header_cache[sheet_name] = {"headers": [], "ts": now, "ok": ok}

    return headers, colcount, ok

# ==========================================
# append（列拡張防止）
# ==========================================
def _compute_out_len(sheet_name: str, headers: List[str], colcount: Optional[int], fields: List[str]) -> Optional[int]:
    hm = _build_headers_map(headers)
    idxs = []
    for f in fields:
        col = _resolve_col_idx(hm, f)
        if col >= 0:
            idxs.append(col)

    base_len = max(len(headers), _expected_header_len(sheet_name), 1)
    need_len = max(base_len, (max(idxs) + 1) if idxs else base_len)

    if colcount is not None and need_len > colcount:
        return None
    return need_len

def _make_aligned_row(headers: List[str], out_len: int, fields: List[str], row_values: List[Any]) -> List[Any]:
    hm = _build_headers_map(headers)
    out = [""] * out_len
    for f, v in zip(fields, row_values):
        col = _resolve_col_idx(hm, f)
        if col == -1:
            continue
        if 0 <= col < out_len:
            out[col] = v
    return out

def append_rows_to_sheet(sheet_name: str, rows_values: List[List[Any]], fields: List[str]):
    headers, colcount, ok = get_headers_and_len(sheet_name)

    if not headers:
        msg = f"[WARN] Headers empty. Skip append to prevent corruption. sheet={sheet_name}"
        print(msg)
        send_discord_message(msg)
        return

    if STRICT_HEADER_CHECK and not ok:
        msg = f"[WARN] Header check failed. Skip append to prevent corruption. sheet={sheet_name}"
        print(msg)
        send_discord_message(msg)
        return

    if colcount is None:
        out_len = len(headers)
    else:
        out_len = _compute_out_len(sheet_name, headers, colcount, fields)

    if not out_len or out_len <= 0:
        msg = f"[WARN] Output length invalid. Skip append. sheet={sheet_name}"
        print(msg)
        send_discord_message(msg)
        return

    if colcount is not None and out_len > colcount:
        msg = f"[WARN] Output length cannot fit sheet columnCount. Skip append. sheet={sheet_name}"
        print(msg)
        send_discord_message(msg)
        return

    try:
        adjusted_rows: List[List[Any]] = []
        for r in rows_values:
            adjusted_rows.append(_make_aligned_row(headers, out_len, fields, r))

        service = get_sheet_service()
        sheets_execute(
            service.spreadsheets().values().append(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{sheet_name}!A1",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": adjusted_rows},
            ),
            desc=f"append_rows_to_sheet sheet={sheet_name} rows={len(adjusted_rows)}",
        )

        _invalidate_sheet_caches(sheet_name)

    except Exception as e:
        print(f"[WARN] Sheet append error ({sheet_name}): {e}")
        send_discord_message(f"[WARN] Sheet append error: sheet={sheet_name} err={str(e)[:180]}")

# ==========================================
# 数値系
# ==========================================
def safe_float(x, default=""):
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default

def to_float(val, default=None):
    try:
        if val is None:
            return default
        s = str(val).strip()
        if s == "":
            return default
        if s.startswith("'"):
            s = s[1:].strip()
        v = float(s.replace(",", ""))
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    return rsi

def find_cols(headers, name: str) -> List[int]:
    return [i for i, h in enumerate(headers) if str(h).strip() == name]

def fmt_opt(label: str, v, suffix=""):
    if v == "" or v is None:
        return ""
    return f"{label}{v}{suffix}"

# ==========================================
# OKX
# ==========================================
def build_exchange() -> ccxt.Exchange:
    now = time.time()
    ex = _exchange_cache.get("ex")
    ts = float(_exchange_cache.get("ts", 0.0))
    if ex is not None and (now - ts) <= EXCHANGE_TTL_SEC:
        return ex

    exchange = ccxt.okx({
        "enableRateLimit": True,
        "timeout": 10000,
        "options": {"defaultType": OKX_DEFAULT_TYPE},
    })
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"[WARN] okx.load_markets failed: {e}")

    _exchange_cache["ex"] = exchange
    _exchange_cache["ts"] = now
    _symbol_resolve_cache.clear()
    return exchange

def _resolve_okx_symbol(exchange: ccxt.Exchange, symbol: str) -> str:
    if not symbol:
        return symbol
    if symbol in _symbol_resolve_cache:
        return _symbol_resolve_cache[symbol]

    mk = getattr(exchange, "markets", None)
    if not isinstance(mk, dict) or not mk:
        _symbol_resolve_cache[symbol] = symbol
        return symbol

    if symbol in mk:
        _symbol_resolve_cache[symbol] = symbol
        return symbol

    if "/" in symbol and ":" not in symbol:
        try:
            base, quote = symbol.split("/")
            cand = f"{base}/{quote}:{quote}"
            if cand in mk:
                _symbol_resolve_cache[symbol] = cand
                return cand
        except Exception:
            pass

    if "/" not in symbol:
        base = symbol
        quote = "USDT"
        for cand in (f"{base}/{quote}", f"{base}/{quote}:{quote}"):
            if cand in mk:
                _symbol_resolve_cache[symbol] = cand
                return cand

    _symbol_resolve_cache[symbol] = symbol
    return symbol

def fetch_ohlcv_safe(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int,
                     since: Optional[int] = None, retries: int = FETCH_RETRY) -> Optional[List[List[Any]]]:
    last_err = None
    sym = _resolve_okx_symbol(exchange, symbol)

    for k in range(retries + 1):
        try:
            if since is None:
                return exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            return exchange.fetch_ohlcv(sym, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            last_err = e
            if k < retries:
                time.sleep(FETCH_RETRY_SLEEP_SEC * (k + 1))
                continue
            break

    print(f"[WARN] fetch_ohlcv_safe failed: {symbol} (resolved={sym}) err={last_err}")
    return None

# ==========================================
# 行数/重複キー
# ==========================================
def _get_row_count_cached(sheet_name: str) -> int:
    now = time.time()
    c = _row_count_cache.get(sheet_name)
    if c and (now - float(c.get("ts", 0))) <= ROWCOUNT_TTL_SEC:
        return int(c.get("n", 0))

    try:
        headers, _, _ = get_headers_and_len(sheet_name)
        hm = _build_headers_map(headers)
        col_time = _resolve_col_idx(hm, "Time")
        col_letter = "A" if col_time < 0 else col_to_a1(col_time)

        service = get_sheet_service()
        res = sheets_execute(
            service.spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{sheet_name}!{col_letter}:{col_letter}",
            ),
            desc=f"row_count sheet={sheet_name} col={col_letter}",
        )
        vals = res.get("values", [])
        n = len(vals)
        _row_count_cache[sheet_name] = {"ts": now, "n": n}
        return n
    except Exception as e:
        print(f"[WARN] row_count fetch failed: sheet={sheet_name} err={e}")
        return int(c.get("n", 0)) if c else 0

def _get_recent_dedup_keys(sheet_name: str) -> Set[str]:
    now = time.time()
    c = _dedup_cache.get(sheet_name)
    if c and (now - float(c.get("ts", 0))) <= DEDUP_TTL_SEC:
        return set(c.get("keys", set()))

    keys: Set[str] = set()
    try:
        last_row = _get_row_count_cached(sheet_name)
        if last_row < 2:
            _dedup_cache[sheet_name] = {"ts": now, "keys": keys}
            return keys

        start = max(2, last_row - DEDUP_LOOKBACK_ROWS + 1)

        headers, _, _ = get_headers_and_len(sheet_name)
        hm = _build_headers_map(headers)
        col_time = _resolve_col_idx(hm, "Time")
        col_sym = _resolve_col_idx(hm, "Symbol")

        if col_time == -1 or col_sym == -1:
            col_time, col_sym = 0, 1

        c1 = min(col_time, col_sym)
        c2 = max(col_time, col_sym)
        off_time = col_time - c1
        off_sym = col_sym - c1

        a1 = col_to_a1(c1)
        a2 = col_to_a1(c2)

        service = get_sheet_service()
        res = sheets_execute(
            service.spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{sheet_name}!{a1}{start}:{a2}{last_row}",
            ),
            desc=f"dedup_scan sheet={sheet_name} range={a1}{start}:{a2}{last_row}",
        )

        rows = res.get("values", []) or []
        for r in rows:
            t_raw = str(r[off_time]).strip() if len(r) > off_time else ""
            sym = str(r[off_sym]).strip() if len(r) > off_sym else ""
            if not t_raw or not sym:
                continue
            t_key = normalize_dt_str(t_raw)
            keys.add(f"{sym}|{t_key}")

    except Exception as e:
        print(f"[WARN] dedup scan failed: sheet={sheet_name} err={e}")

    _dedup_cache[sheet_name] = {"ts": now, "keys": keys}
    return keys

# ==========================================
# 簡易分散ロック（_lock自動作成）
# ==========================================
def _ensure_mutex_sheet():
    ensure_sheet_exists(RUN_MUTEX_SHEET, min_rows=50, min_cols=5)

def _mutex_read() -> str:
    _ensure_mutex_sheet()
    service = get_sheet_service()
    rng = f"{RUN_MUTEX_SHEET}!{RUN_MUTEX_CELL}"
    res = sheets_execute(
        service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=rng),
        desc=f"mutex_read {rng}",
    )
    v = (res.get("values", [[]]) or [[]])[0]
    return "" if not v else str(v[0]).strip()

def _mutex_write(value: str):
    _ensure_mutex_sheet()
    service = get_sheet_service()
    rng = f"{RUN_MUTEX_SHEET}!{RUN_MUTEX_CELL}"
    sheets_execute(
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=rng,
            valueInputOption="RAW",
            body={"values": [[value]]},
        ),
        desc=f"mutex_write {rng}",
    )

def acquire_run_mutex() -> Tuple[bool, str]:
    if not RUN_MUTEX_ENABLED:
        return True, ""

    token = f"{int(time.time())}|{_INSTANCE_ID}"
    try:
        cur = _mutex_read()
        if cur:
            try:
                ts_str = cur.split("|", 1)[0].strip()
                ts = int(float(ts_str))
            except Exception:
                ts = 0

            if ts > 0 and (time.time() - ts) <= RUN_MUTEX_TTL_SEC:
                return False, ""

        _mutex_write(token)
        time.sleep(0.25)
        after = _mutex_read()
        if after == token:
            return True, token
        return False, ""
    except Exception as e:
        print(f"[WARN] acquire_run_mutex failed (fallback allow): {e}")
        return True, ""

def release_run_mutex(token: str):
    if not RUN_MUTEX_ENABLED:
        return
    if not token:
        return
    try:
        cur = _mutex_read()
        if cur == token:
            _mutex_write("")
    except Exception as e:
        print(f"[WARN] release_run_mutex failed: {e}")

# ==========================================
# モデル読み込み
# ==========================================
def _boot_notify_model_status_once(bucket, gcs_uri: str, local_path: str, ver: str, ok: bool, err: str):
    key = os.environ.get("K_REVISION", "").strip() or ver or "unknown"
    marker_name = f"markers/crypto-alert/boot/{key}.json"
    blob = bucket.blob(marker_name)

    body = {
        "ok": bool(ok),
        "service": os.environ.get("K_SERVICE", ""),
        "revision": os.environ.get("K_REVISION", ""),
        "model_version": ver,
        "model_uri": gcs_uri,
        "model_local_path": local_path,
        "loaded_at": _AI_LOADED_AT,
        "last_error": ("" if ok else (err or ""))[:500],
    }

    try:
        blob.upload_from_string(
            data=json.dumps(body, ensure_ascii=False),
            content_type="application/json",
            if_generation_match=0,
        )
    except Exception as e:
        s = str(e)
        if "412" in s or "Precondition" in s:
            print(f"[AI] boot notify skipped (marker exists): {marker_name}")
            return
        print(f"[WARN] boot marker create failed: {e}")
        return

    msg = (
        f"[AI] Boot Model Status\n"
        f"ok={ok}\n"
        f"ver={ver}\n"
        f"uri={gcs_uri}\n"
        f"path={local_path}\n"
        f"rev={os.environ.get('K_REVISION','')}\n"
        f"err={(err or '')[:180]}"
    )
    send_discord_message(msg)


def load_ai_model():
    """
    優先順位:
    1) MODEL_GCS_URI が gs://... の場合 -> GCS から MODEL_LOCAL_PATH にダウンロードして joblib.load
    2) それ以外 -> ローカルの trade_ai_model.pkl を joblib.load
    """
    global _AI_LOADED_AT, _AI_LAST_ERROR

    gcs_uri = os.environ.get("MODEL_GCS_URI", "").strip()
    local_path = os.environ.get("MODEL_LOCAL_PATH", "/tmp/trade_ai_model.pkl").strip()
    ver = os.environ.get("MODEL_VERSION", "").strip()

    try:
        if gcs_uri.startswith("gs://"):
            tmp = gcs_uri[5:]
            bucket_name, blob_name = tmp.split("/", 1)

            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)

            m = joblib.load(local_path)
            _AI_LOADED_AT = datetime.now(JST).isoformat()
            _AI_LAST_ERROR = ""
            print(f"[AI] Model Loaded Successfully uri={gcs_uri} path={local_path} ver={ver}")
            _boot_notify_model_status_once(bucket, gcs_uri, local_path, ver, True, "")
            return m

        if os.path.exists("trade_ai_model.pkl"):
            m = joblib.load("trade_ai_model.pkl")
            _AI_LOADED_AT = datetime.now(JST).isoformat()
            _AI_LAST_ERROR = ""
            print(f"[AI] Model Loaded Successfully uri=LOCAL path=trade_ai_model.pkl ver={ver}")
            return m

        print("[AI] trade_ai_model.pkl not found -> AI gate is bypassed (ai_pass=True).")
        _AI_LOADED_AT = ""
        _AI_LAST_ERROR = "model file not found (AI bypassed)"
        return None

    except Exception as e:
        _AI_LOADED_AT = ""
        _AI_LAST_ERROR = str(e)
        print(f"[AI] Load Failed: {e}")
        try:
            if gcs_uri.startswith("gs://"):
                client = storage.Client()
                tmp = gcs_uri[5:]
                bucket_name, _ = tmp.split("/", 1)
                bucket = client.bucket(bucket_name)
                _boot_notify_model_status_once(bucket, gcs_uri, local_path, ver, False, str(e))
        except Exception:
            pass
        return None


ai_model = load_ai_model()

# ==========================================
# ★起動時に「必要シート/ヘッダー」を自己修復
# ==========================================
def self_heal_prerequisites() -> Tuple[bool, str]:
    try:
        ok_lock = ensure_sheet_exists(RUN_MUTEX_SHEET, min_rows=50, min_cols=5)
        ok_learn = ensure_learn_headers()
        ok_table = ensure_table_headers()
        ok_all = bool(ok_lock and ok_learn and ok_table)
        return ok_all, f"lock={ok_lock} learn={ok_learn} table={ok_table}"
    except Exception as e:
        return False, f"self_heal failed: {e}"

# ==========================================
# ロジック本体 (/run)
# ==========================================
def logic_main():
    global last_alert_records, last_candidate_records, ai_model

    start = time.time()
    now_jst = datetime.now(JST)
    print(f"[RUN] start {now_jst.isoformat()}  VERSION={VERSION}")

    ok, msg = self_heal_prerequisites()
    if not ok:
        send_discord_message(f"[WARN] self_heal_prerequisites failed: {msg}")
        return f"SelfHealFailed: {msg}"

    if (now_jst.minute % 15) >= 10:
        print(f"[RUN] skip (waiting window) minute={now_jst.minute}")
        return "Waiting..."

    exchange = build_exchange()

    btc_mode = "Range"
    btc_1h_change = 0.0
    median_sigma = 0.0
    btc_ret = 0.0
    btc_vol = 0.0
    btc_ok = False

    try:
        btc_ohlcv = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe="15m", limit=60)
        if not btc_ohlcv or len(btc_ohlcv) < MIN_BARS:
            raise ValueError(f"BTC ohlcv bars不足: {0 if not btc_ohlcv else len(btc_ohlcv)}")

        btc_df = pd.DataFrame(btc_ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
        btc_df["Pct_Change"] = btc_df["Close"].pct_change(fill_method=None)
        btc_df["Dynamic_Sigma"] = btc_df["Pct_Change"].rolling(20).std().fillna(0.01).clip(lower=1e-4)

        median_sigma = float(btc_df["Dynamic_Sigma"].tail(20).median())
        btc_current = float(btc_df.iloc[-2]["Close"])
        btc_1h_ago = float(btc_df.iloc[-6]["Close"])
        btc_1h_change = (btc_current - btc_1h_ago) / btc_1h_ago
        btc_ret = float(btc_df.iloc[-2]["Pct_Change"])
        btc_vol = abs(btc_ret)

        if btc_1h_change > 0.001:
            btc_mode = "Up"
        elif btc_1h_change < -0.001:
            btc_mode = "Down"

        btc_ok = True
    except Exception as e:
        print(f"[WARN] BTC fetch failed: {e}")

    BTC_CALM = btc_ok and (median_sigma < 0.005)
    ALLOW_LONG = (btc_mode != "Down")
    ALLOW_SHORT = (btc_mode != "Up")

    symbols = [
        "BTC/USDT", "DOT/USDT", "BONK/USDT", "DOGE/USDT", "LINK/USDT", "ETH/USDT", "SUI/USDT", "BNB/USDT", "UNI/USDT",
        "ADA/USDT", "ATOM/USDT", "XRP/USDT", "NEAR/USDT", "LTC/USDT", "TRX/USDT", "SHIB/USDT", "HBAR/USDT", "SEI/USDT",
        "SOL/USDT", "AAVE/USDT", "AVAX/USDT", "APT/USDT", "FET/USDT", "ARB/USDT", "INJ/USDT", "POL/USDT",
        "STX/USDT", "XLM/USDT"
    ]

    pending_candidates: List[Dict[str, Any]] = []
    pending_alerts: List[Dict[str, Any]] = []

    for symbol in symbols:
        try:
            ohlcv = fetch_ohlcv_safe(exchange, symbol, timeframe="15m", limit=60)
            if not ohlcv or len(ohlcv) < MIN_BARS:
                continue

            df = pd.DataFrame(ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
            df["Pct_Change"] = df["Close"].pct_change(fill_method=None)
            df["Dynamic_Sigma"] = df["Pct_Change"].rolling(20).std().fillna(0.01).clip(lower=1e-4)

            df["MA20"] = df["Close"].rolling(20).mean()
            df["Upper2"] = df["MA20"] + (2 * df["Close"] * df["Dynamic_Sigma"])
            df["Lower2"] = df["MA20"] - (2 * df["Close"] * df["Dynamic_Sigma"])
            df["BandWidth"] = np.where(df["MA20"] != 0, (df["Upper2"] - df["Lower2"]) / df["MA20"], 0)

            df["BW_Change"] = df["BandWidth"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
            df["RSI"] = calculate_rsi(df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(50)

            v = pd.to_numeric(df["Volume"], errors="coerce").replace(0, np.nan)
            df["Vol_Change"] = v.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)

            df["Drop_Score"] = df["Pct_Change"].apply(lambda x: abs(x) if x < 0 else 0) / df["Dynamic_Sigma"]
            df["Rise_Score"] = df["Pct_Change"].apply(lambda x: abs(x) if x > 0 else 0) / df["Dynamic_Sigma"]

            df["Vol_MA20"] = pd.to_numeric(df["Volume"], errors="coerce").rolling(20).mean()

            row = df.iloc[-2]
            chg = safe_float(row.get("Pct_Change", None), default="")
            chg_pct_val = "" if chg == "" else (chg * 100.0)

            vol_ma20 = safe_float(row.get("Vol_MA20", None), default="")
            vol_now = safe_float(row.get("Volume", None), default="")
            if vol_ma20 == "" or vol_now == "" or vol_ma20 == 0:
                vol_ratio_val = ""
            else:
                vol_ratio_val = vol_now / vol_ma20

            is_buy = False
            is_sell = False
            signal_type = ""

            if (row["Drop_Score"] >= CAND_SIGMA) and ALLOW_SHORT:
                is_sell = True
                signal_type = "SHORT"
            elif (row["Rise_Score"] >= CAND_SIGMA) and ALLOW_LONG:
                if row["Close"] > df.iloc[-6]["Close"]:
                    is_buy = True
                    signal_type = "LONG"

            if not (is_buy or is_sell):
                continue


            # ===== AI判定（E値も算出）=====
            ai_score = None
            E = None  # 期待値（%）

            if ai_model is None:
                # モデルが無い時は従来通り「バイパス」
                ai_pass = True
            else:
                ai_pass = False  # 成功したらTrueにする

                feats = pd.DataFrame([{
                    "Sigma": float(row["Dynamic_Sigma"]),
                    "BandWidth": float(row["BandWidth"]),
                    "BW_Change": float(row["BW_Change"]),
                    "RSI": float(row["RSI"]),
                    "Vol_Change": float(row["Vol_Change"]),
                    "Rise_Score": float(row["Rise_Score"]),
                    "Drop_Score": float(row["Drop_Score"]),
                    "BTC_Ret": float(btc_ret),
                    "BTC_Vol": float(btc_vol),
                }])

                try:
                    ai_score = float(ai_model.predict_proba(feats)[0][1])  # P(win)っぽいもの

                    # TP/SL幅（%）を簡易計算（列ズレ回避のため配列には触らない）
                    tp_mult = 3.8
                    sl_mult = 1.5
                    sig = float(row["Dynamic_Sigma"])

                    tp_pct = abs(sig * tp_mult) * 100.0
                    sl_pct = abs(sig * sl_mult) * 100.0

                    p_tp = ai_score
                    p_sl = 1.0 - ai_score

                    # 期待値（%ベース）
                    E = (p_tp * tp_pct) - (p_sl * sl_pct)

                    # 通す条件：従来の閾値 + 期待値がプラス
                    ai_pass = (ai_score >= AI_TH) and (E > 0)

                except Exception as e:
                    print(f"[AI] predict_proba failed for {symbol}: {e}")
                    ai_score = None
                    E = None
                    ai_pass = False


            item = {
                "symbol": symbol.replace("/USDT", ""),
                "time": int(row["Time"]),
                "is_buy": bool(is_buy),
                "is_sell": bool(is_sell),
                "close": float(row["Close"]),
                "score": float(max(row["Drop_Score"], row["Rise_Score"])),
                "sigma": float(row["Dynamic_Sigma"]),
                "rsi": float(row["RSI"]),
                "type": signal_type,
                "dt": datetime.fromtimestamp(int(row["Time"]) / 1000, JST),
                "ai_score": ai_score,
                "ai_pass": bool(ai_pass),
                "E": E,
                "chg_pct": chg_pct_val,
                "vol_ratio": vol_ratio_val,
            }

            pending_candidates.append(item)

            if ai_pass and BTC_CALM and item["score"] >= ALERT_SIGMA:
                pending_alerts.append(item)

        except Exception as e:
            print(f"[ERR] {symbol} fetch/compute: {e}")
            continue

    def calc_tp_sl(item):
        tp_mult = 3.8
        sl_mult = 1.5
        cp = float(item["close"])
        sig = float(item["sigma"])
        if item["is_buy"]:
            tp = cp * (1 + sig * tp_mult)
            sl = cp * (1 - sig * sl_mult)
        else:
            tp = cp * (1 - sig * tp_mult)
            sl = cp * (1 + sig * sl_mult)
        tp_pct = abs((tp - cp) / cp) * 100.0
        sl_pct = abs((sl - cp) / cp) * 100.0
        return tp, sl, tp_pct, sl_pct

    learn_keys = _get_recent_dedup_keys(LEARN_SHEET_NAME)
    table_keys = _get_recent_dedup_keys(MAIN_SHEET_NAME)

    candidate_rows: List[List[Any]] = []
    for item in pending_candidates:
        sym = item["symbol"]
        ts_ms = item["time"]

        dt_str = normalize_dt_str(item["dt"].strftime("%Y-%m-%d %H:%M:%S"))
        dt_cell = "'" + dt_str

        if sym in last_candidate_records and last_candidate_records[sym] == ts_ms:
            continue
        if (now_jst - item["dt"]).total_seconds() > 3000:
            continue

        k = f"{sym}|{dt_str}"
        if k in learn_keys:
            continue

        last_candidate_records[sym] = ts_ms

        tp, sl, tp_pct, sl_pct = calc_tp_sl(item)

        ai_disp = "N/A" if item["ai_score"] is None else f"{float(item['ai_score']):.1%}"
        note_str = (
            f"AI:{ai_disp} Pass:{item['ai_pass']} "
            f"Calm:{BTC_CALM} SigmaMed:{median_sigma:.4f} BTC_OK:{btc_ok} "
            f"BTC:{btc_mode} 1h:{btc_1h_change:.2%}"
        )

        # ★追加：learn_logのStatusを分ける（後で抽出しやすくする）
        # - AIモデルがあるのに ai_pass=False → AI_REJECT
        # - それ以外 → CANDIDATE
        status = "AI_REJECT" if (ai_model is not None and (not item["ai_pass"])) else "CANDIDATE"

        # ★追加：Reserved1/Reserved2に数値を入れておく（列追加不要）
        reserved1 = "" if item["ai_score"] is None else float(item["ai_score"])
        reserved2 = float(AI_TH)

        candidate_rows.append([
            dt_cell, sym, "LONG" if item["is_buy"] else "SHORT",
            float(item["close"]), float(item["score"]), float(item["sigma"]), status,
            float(tp), float(sl), float(tp_pct), float(sl_pct),
            DEFAULT_LEV, reserved1, reserved2, bool(item["ai_pass"]), bool(BTC_CALM),
            VERSION, item["type"], 0, 0,
            ("STORM" if not BTC_CALM else "CALM"), btc_mode, float(btc_1h_change),
            float(item["rsi"]), note_str,
            "", "", "", "", "", "", ""
        ])


        learn_keys.add(k)

    if candidate_rows:
        append_rows_to_sheet(LEARN_SHEET_NAME, candidate_rows, EXPECTED_HEADERS_LEARN)

    filtered = sorted(pending_alerts, key=lambda x: x["score"], reverse=True)[:3]
    count = 0
    alert_rows: List[List[Any]] = []

    for item in filtered:
        sym = item["symbol"]
        ts_ms = item["time"]

        dt_str = normalize_dt_str(item["dt"].strftime("%Y-%m-%d %H:%M:%S"))
        dt_cell = "'" + dt_str

        if sym in last_alert_records and last_alert_records[sym] == ts_ms:
            continue
        if (now_jst - item["dt"]).total_seconds() > 3000:
            continue

        k = f"{sym}|{dt_str}"
        if k in table_keys:
            continue

        last_alert_records[sym] = ts_ms

        tp, sl, tp_pct, sl_pct = calc_tp_sl(item)
        cp = float(item["close"])
        lev = DEFAULT_LEV

        if item["is_buy"]:
            icon = "🚀"
            d_str = "買い(LONG)"
        else:
            icon = "☄️"
            d_str = "売り(SHORT)"

        ai_disp = "N/A" if item["ai_score"] is None else f"{float(item['ai_score']):.1%}"

        msg = (
            f"{icon} **{d_str}** {icon}\n"
            f"{VERSION}\n"
            f"💎 {sym} ({item['type']})\n"
            f"📈 Score:{item['score']:.2f}σ  AI:{ai_disp}\n"
            f"🟦 BTC:{btc_mode} 1h:{btc_1h_change:.2%}  Calm:{BTC_CALM}  BTC_OK:{btc_ok}\n"
            f"💰 {cp:.4f}\n"
            f"🎯 TP: {tp:.4f} ({tp_pct:.2f}%)\n"
            f"🛑 SL: {sl:.4f} ({sl_pct:.2f}%)"
        )
        send_discord_message(msg)
        count += 1

        parts = [
            f"{item['type']}",
            f"AI:{ai_disp}",
            f"RSI:{item['rsi']:.1f}",
            fmt_opt("Chg:", item["chg_pct"], "%"),
            fmt_opt("VolR:", item["vol_ratio"]),
            f"BTC:{btc_mode}",
            f"1h:{btc_1h_change:.2%}",
            f"BTC_OK:{btc_ok}",
        ]
        note_compact = " | ".join([p for p in parts if p])

        alert_rows.append([
            dt_cell, sym, "LONG" if item["is_buy"] else "SHORT",
            float(cp), float(item["score"]), float(item["sigma"]), "AI_PASS",
            float(tp), float(sl), float(tp_pct), float(sl_pct),
            lev, float(tp_pct * lev), float(sl_pct * lev),
            bool(BTC_CALM), True, VERSION, note_compact,
            item["chg_pct"], item["vol_ratio"], "MARKET",
        ])

        table_keys.add(k)

    if alert_rows:
        append_rows_to_sheet(MAIN_SHEET_NAME, alert_rows, TABLE_FIELDS)

    elapsed = time.time() - start
    print(f"[RUN] done alerts={count} candidates={len(candidate_rows)} elapsed={elapsed:.2f}s")
    print(f"[DBG] pending_candidates={len(pending_candidates)} pending_alerts={len(pending_alerts)} "
          f"BTC_CALM={BTC_CALM} btc_mode={btc_mode} median_sigma={median_sigma} btc_ok={btc_ok}")
    return f"Sent {count} alerts, Logged {len(candidate_rows)} candidates"

# ==========================================
# 判定係 (/judge)
# ==========================================
def judge_sheet(sheet_name: str, lookback_rows: int = JUDGE_LOOKBACK_ROWS, max_judge: int = 30) -> int:
    service = get_sheet_service()
    exchange = build_exchange()

    ok, msg = self_heal_prerequisites()
    if not ok:
        print(f"[WARN] self_heal_prerequisites failed (judge): {msg}")
        return 0

    headers, _, okh = get_headers_and_len(sheet_name)
    if STRICT_HEADER_CHECK and not okh:
        print(f"[WARN] judge_sheet header check failed -> skip. sheet={sheet_name}")
        return 0

    last_row = _get_row_count_cached(sheet_name)
    if last_row < 2:
        return 0

    start_row = max(2, last_row - lookback_rows + 1)

    res = sheets_execute(
        service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!A{start_row}:{HEADER_COL_END}{last_row}",
        ),
        desc=f"judge_read sheet={sheet_name} A{start_row}:{HEADER_COL_END}{last_row}",
    )

    rows = res.get("values", []) or []
    if not rows:
        return 0

    hm = _build_headers_map(headers)

    col_symbol = _resolve_col_idx(hm, "Symbol")
    col_time = _resolve_col_idx(hm, "Time")
    col_entry = _resolve_col_idx(hm, "EntryPrice")
    col_dir = _resolve_col_idx(hm, "Direction")
    col_tp = _resolve_col_idx(hm, "TP_Price")
    col_sl = _resolve_col_idx(hm, "SL_Price")
    col_lev = _resolve_col_idx(hm, "Lev")
    col_slpct = _resolve_col_idx(hm, "SL_%")

    cols_eval = find_cols(headers, "EvalStatus")
    cols_exit_time = find_cols(headers, "ExitTime")
    cols_exit_price = find_cols(headers, "ExitPrice")
    cols_reason_old = find_cols(headers, "Reason")
    cols_exit_reason = find_cols(headers, "ExitReason")
    cols_pnl_old = find_cols(headers, "PnL%")
    cols_pnl_new = find_cols(headers, "PnL_Pct")
    cols_winlose = find_cols(headers, "Win/Lose")
    cols_holdmin = find_cols(headers, "HoldMin")

    cols_levpnl = find_cols(headers, "LevPnL%")
    cols_rmult = find_cols(headers, "R-Mult")

    must = [col_symbol, col_time, col_entry, col_tp, col_sl]
    if any(c == -1 for c in must):
        print(f"[WARN] judge_sheet missing required columns in {sheet_name}")
        return 0

    if not cols_eval:
        print(f"[WARN] judge_sheet: EvalStatus not found in {sheet_name}")
        return 0

    updates: List[Dict[str, Any]] = []
    judged = 0

    def get_cell(row: List[Any], idx: int) -> str:
        if idx < 0:
            return ""
        return str(row[idx]).strip() if idx < len(row) else ""

    def put_many(row_idx_1based: int, col_indices: list, value):
        for col_idx in col_indices:
            if col_idx < 0:
                continue
            a1 = col_to_a1(col_idx)
            updates.append({"range": f"{sheet_name}!{a1}{row_idx_1based}", "values": [[value]]})

    for offset in range(len(rows) - 1, -1, -1):
        if judged >= max_judge:
            break

        row = rows[offset]
        sheet_row_idx = start_row + offset

        col_eval_last = cols_eval[-1]
        status = get_cell(row, col_eval_last)
        if status != "":
            continue

        sym = get_cell(row, col_symbol)
        tstr_raw = get_cell(row, col_time)
        tstr = normalize_dt_str(tstr_raw)

        entry = to_float(get_cell(row, col_entry), default=None)
        tp = to_float(get_cell(row, col_tp), default=None)
        sl = to_float(get_cell(row, col_sl), default=None)

        direction = get_cell(row, col_dir).upper()

        if (not sym) or (not tstr) or entry is None or tp is None or sl is None or tp == 0 or sl == 0:
            continue

        market = sym if "/USDT" in sym else f"{sym}/USDT"

        dt0 = parse_dt_any(tstr)
        if dt0 is None:
            continue

        if getattr(dt0, "tzinfo", None) is None:
            dt_jst = dt0.replace(tzinfo=JST)
        else:
            dt_jst = dt0.astimezone(JST)

        actual_entry_jst = dt_jst + timedelta(minutes=15)
        since_ms = int(actual_entry_jst.astimezone(timezone.utc).timestamp() * 1000)

        if "SHORT" in direction:
            side = "SHORT"
        elif "LONG" in direction:
            side = "LONG"
        else:
            side = "LONG" if tp > entry else "SHORT"

        candles = fetch_ohlcv_safe(exchange, market, timeframe="15m", since=since_ms, limit=100)
        if not candles:
            continue

        res_status = "PENDING"
        res_win = ""
        res_reason = ""
        res_exit_price = None
        res_exit_time_ms = None

        for ts, o, h, l, c, v in candles:
            if ts < since_ms:
                continue

            if side == "LONG":
                tp_hit = (h >= tp)
                sl_hit = (l <= sl)
            else:
                tp_hit = (l <= tp)
                sl_hit = (h >= sl)

            if tp_hit and sl_hit:
                res_status = "AMBIGUOUS"
                res_reason = "Both"
                res_exit_time_ms = ts
                res_exit_price = entry
                break

            if tp_hit:
                res_status = "DONE"
                res_reason = "TP"
                res_exit_time_ms = ts
                res_exit_price = tp
                res_win = "Win"
                break

            if sl_hit:
                res_status = "DONE"
                res_reason = "SL"
                res_exit_time_ms = ts
                res_exit_price = sl
                res_win = "Lose"
                break

        if res_status == "PENDING" and len(candles) >= 96:
            res_status = "EXPIRED"
            res_reason = "TimeOver"

        if res_status not in {"DONE", "AMBIGUOUS", "EXPIRED"}:
            continue

        if res_exit_time_ms is not None:
            exit_dt_jst = datetime.fromtimestamp(res_exit_time_ms / 1000, JST)
            exit_time_str = exit_dt_jst.strftime("%Y-%m-%d %H:%M:%S")
            hold_min = int((exit_dt_jst - actual_entry_jst).total_seconds() / 60)
        else:
            exit_time_str = ""
            hold_min = ""

        if res_exit_price is not None and entry is not None and res_status == "DONE":
            if side == "LONG":
                pnl_pct = (res_exit_price / entry - 1.0) * 100.0
            else:
                pnl_pct = (entry / res_exit_price - 1.0) * 100.0
        else:
            pnl_pct = ""

        lev = to_float(get_cell(row, col_lev), default=DEFAULT_LEV) if col_lev != -1 else DEFAULT_LEV
        levpnl = (pnl_pct * lev) if (pnl_pct != "" and lev is not None) else ""

        sl_pct_val = to_float(get_cell(row, col_slpct), default=None) if col_slpct != -1 else None
        if sl_pct_val not in (None, 0) and pnl_pct != "":
            r_mult = float(pnl_pct) / float(sl_pct_val)
        else:
            r_mult = ""

        put_many(sheet_row_idx, cols_eval, res_status)
        put_many(sheet_row_idx, cols_winlose, res_win)
        put_many(sheet_row_idx, cols_exit_time, exit_time_str)
        put_many(sheet_row_idx, cols_exit_price, "" if res_exit_price is None else res_exit_price)
        put_many(sheet_row_idx, cols_reason_old, res_reason)
        put_many(sheet_row_idx, cols_exit_reason, res_reason)
        put_many(sheet_row_idx, cols_pnl_old, pnl_pct)
        put_many(sheet_row_idx, cols_pnl_new, pnl_pct)
        put_many(sheet_row_idx, cols_holdmin, hold_min)
        if cols_levpnl:
            put_many(sheet_row_idx, cols_levpnl, levpnl)
        if cols_rmult:
            put_many(sheet_row_idx, cols_rmult, r_mult)

        judged += 1
        time.sleep(0.12)

    if updates:
        sheets_execute(
            service.spreadsheets().values().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"valueInputOption": "USER_ENTERED", "data": updates},
            ),
            desc=f"judge_batchUpdate sheet={sheet_name} updates={len(updates)}",
        )
        _invalidate_sheet_caches(sheet_name)

    return judged

def judge_main():
    total = 0
    total += judge_sheet(MAIN_SHEET_NAME)
    total += judge_sheet(LEARN_SHEET_NAME)
    return f"Judged {total} rows (table + learn_log)"

# ==========================================
# /preflight
# ==========================================
def preflight_check() -> Tuple[bool, str]:
    try:
        _ = _get_sheets_meta()

        ok, msg = self_heal_prerequisites()
        if not ok:
            return False, f"self_heal_ng: {msg}"

        if RUN_MUTEX_ENABLED:
            cur = _mutex_read()
            _mutex_write(f"preflight|{int(time.time())}|{_INSTANCE_ID}")
            time.sleep(0.1)
            _mutex_write(cur)

        return True, "ok"
    except HttpError as e:
        return False, f"google_api_http_error: {str(e)}"
    except Exception as e:
        return False, f"preflight_error: {e}"

# ==========================================
# ルーティング
# ==========================================
@app.route("/", methods=["GET"])
def home():
    return f"{VERSION} is Active", 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route("/health", methods=["GET", "HEAD"])
def health():
    print(f"[HEALTH] {request.method} {request.path}")
    return "ok", 200

@app.route("/preflight", methods=["GET"])
def preflight():
    ok, msg = preflight_check()
    return (f"OK: {msg}", 200) if ok else (f"NG: {msg}", 500)

@app.route("/run", methods=["GET", "POST"])
def run_process():
    if not _run_lock.acquire(blocking=False):
        return "Busy (run/judge already in progress).", 429

    mutex_token = ""
    try:
        ok, msg = preflight_check()
        if not ok:
            return f"Preflight NG: {msg}", 500

        okm, token = acquire_run_mutex()
        if not okm:
            return "Busy (distributed mutex).", 429
        mutex_token = token

        res_run = str(logic_main())

        if AUTO_JUDGE_AFTER_RUN:
            if not ENABLE_JUDGE:
                return res_run + " / Judge disabled", 200
            res_j = str(judge_main())
            return res_run + " / " + res_j, 200

        return res_run, 200

    finally:
        release_run_mutex(mutex_token)
        _run_lock.release()

@app.route("/judge", methods=["GET", "POST"])
def judge_process():
    if not ENABLE_JUDGE:
        return "Judge is disabled (set ENABLE_JUDGE=1 to enable).", 403

    if not _run_lock.acquire(blocking=False):
        return "Busy (run/judge already in progress).", 429

    mutex_token = ""
    try:
        ok, msg = preflight_check()
        if not ok:
            return f"Preflight NG: {msg}", 500

        okm, token = acquire_run_mutex()
        if not okm:
            return "Busy (distributed mutex).", 429
        mutex_token = token

        return str(judge_main()), 200

    finally:
        release_run_mutex(mutex_token)
        _run_lock.release()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


