import os
import time
import threading
import hashlib
import subprocess
import re
from typing import Optional, Dict, Any, List, Tuple, Set

import joblib
import numpy as np
import pandas as pd
import requests
import ccxt
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import storage
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request

# ==============================
# 学習用（scikit-learn）
# 環境に無い場合でも /run が落ちないように try にする
# ==============================
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    SKLEARN_OK = True
except Exception as _e:
    LogisticRegression = None
    train_test_split = None
    roc_auc_score = None
    accuracy_score = None
    SKLEARN_OK = False
    print(f"[WARN] scikit-learn import failed (train disabled): {_e}")


# ==========================================
# Flask設定（Buildpacks標準：main.py の app を起動）
# ==========================================
app = Flask(__name__)
app.url_map.strict_slashes = False  # /train と /train/ を両方受ける

# --- ここから追記（診断用：必ず app 定義の直後） ---
def _route_exists(path: str, method: str = "GET") -> bool:
    m = method.upper()
    for rule in app.url_map.iter_rules():
        if rule.rule == path and m in (rule.methods or set()):
            return True
    return False

# 現在このリビジョンで生きているルート一覧（診断用）
if not _route_exists("/__routes", "GET"):
    @app.get("/__routes")
    def __routes():
        rules = []
        for r in app.url_map.iter_rules():
            methods = sorted([m for m in (r.methods or set()) if m not in ("HEAD", "OPTIONS")])
            rules.append({"rule": r.rule, "methods": methods, "endpoint": r.endpoint})
        rules.sort(key=lambda x: x["rule"])
        return jsonify({
            "k_service": os.environ.get("K_SERVICE", ""),
            "k_revision": os.environ.get("K_REVISION", ""),
            "routes": rules,
        })

# 診断用：/train の代わりに /train_ping を用意（/train と衝突させない）
if not _route_exists("/train_ping", "GET"):
    @app.get("/train_ping")
    def train_ping():
        return jsonify({
            "ok": True,
            "msg": "/train_ping is alive",
            "k_service": os.environ.get("K_SERVICE", ""),
            "k_revision": os.environ.get("K_REVISION", ""),
        }), 200
# --- ここまで追記 ---

# （変更箇所：ここに以前あった train_endpoint は削除しました）

# ==========================================
# ==========================================
# 設定エリア（環境変数）
# ==========================================
discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
print(f"[CFG] DISCORD_WEBHOOK_URL_LEN={len(discord_webhook_url)}")

# Hyperliquid: 最大5倍銘柄（この銘柄だけ HL 表示を x5 にする）
# ※環境変数 MAX_LEV_5X_SYMBOLS が未設定（空）の場合は、このデフォルトセットを使う
HL_MAX5_SYMBOLS = {"STX", "XLM", "FET", "HBAR", "POL"}

SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1XwWkzijIwRlafg2zDgPHQ4tgjYModapFI3T_wbYS9_8")
MAIN_SHEET_NAME = os.environ.get("MAIN_SHEET_NAME", "table")
LEARN_SHEET_NAME = os.environ.get("LEARN_SHEET_NAME", "learn_log")

VERSION = "Ver7.7 ModelReloadFix (Code v3.5.5)"

# --- Thresholds (env configurable) ---
CAND_SIGMA = float(os.environ.get("CAND_SIGMA", "1.2"))
ALERT_SIGMA = float(os.environ.get("ALERT_SIGMA", "2.0"))
AI_TH = float(os.environ.get("AI_TH", "0.55"))

# Hyperliquid: 通常銘柄の表示レバ（基本10倍）
# ※DEFAULT_LEV を正として一本化（環境変数 DEFAULT_LEV で変更可能）
DEFAULT_LEV = int(float(os.environ.get("DEFAULT_LEV", "10")))

# "5倍までしか掛けられない銘柄" を環境変数で指定（例: "STX,XLM,FET,HBAR,POL"）
# ※環境変数が空なら HL_MAX5_SYMBOLS を使う
_env_max5 = os.environ.get("MAX_LEV_5X_SYMBOLS", "").strip()
if _env_max5:
    MAX_LEV_5X_SYMBOLS = {s.strip().upper() for s in _env_max5.split(",") if s.strip()}
else:
    MAX_LEV_5X_SYMBOLS = set(HL_MAX5_SYMBOLS)


# --- Guardrails (env configurable) ---
def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if v == "":
        return default
    try:
        return float(v)
    except Exception:
        print(f"[CFG] invalid {name}={v} -> default={default}")
        return default

def _parse_symbol_set(env_name: str) -> set:
    raw = os.environ.get(env_name, "").strip()
    if raw == "":
        return set()
    # ',' と '|' 両対応（空白も許容）
    raw = raw.replace("|", ",").replace(" ", ",")
    return {s.strip().upper() for s in raw.split(",") if s.strip()}

def _parse_float_range(env_name: str):
    raw = os.environ.get(env_name, "").strip()
    if raw == "":
        return None, None
    # "0.005,0.006" / "0.005-0.006" / "0.005〜0.006" などを許容
    s = raw.replace("〜", ",").replace("~", ",").replace("-", ",").replace("|", ",").replace(" ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        print(f"[CFG] invalid {env_name}={raw} -> disabled")
        return None, None
    try:
        a = float(parts[0])
        b = float(parts[1])
        lo, hi = (a, b) if a <= b else (b, a)
        return lo, hi
    except Exception:
        print(f"[CFG] invalid {env_name}={raw} -> disabled")
        return None, None

# 数値系（未設定ならフィルター無効：inf/-inf になる）
VOLRATIO_MAX = _env_float("VOLRATIO_MAX", float("inf"))           # 例: 2.0
RSI_SHORT_MIN = _env_float("RSI_SHORT_MIN", float("-inf"))        # 例: 20
RSI_LONG_MAX = _env_float("RSI_LONG_MAX", float("inf"))           # 例: 45

AI_PCT_MIN = _env_float("AI_PCT_MIN", float("-inf"))              # 未設定なら下限チェック無し（例: 62.7）
AI_PCT_MAX = _env_float("AI_PCT_MAX", float("inf"))               # 例: 69.4（%）

# 慎重銘柄だけ閾値を厳しく（未設定なら通常閾値を使う）
VOLRATIO_MAX_CAUTION = _env_float("VOLRATIO_MAX_CAUTION", VOLRATIO_MAX)  # 例: 1.6
AI_PCT_MAX_CAUTION = _env_float("AI_PCT_MAX_CAUTION", AI_PCT_MAX)        # 例: 67.5

# --- Side選択/荒れ相場ガード（最小差分） ---
# flip（逆張り側）を採用するには、Baseより一定以上「確率が上」になっていることを要求（ノイズ反転を減らす）
AI_SIDE_MARGIN = _env_float("AI_SIDE_MARGIN", 0.03)   # 例: 0.03 (=3%)

# 荒れ相場（強トレンド/高ボラ）扱いの閾値（score=σ倍率、sigma=Dynamic_Sigma）
# - この条件に入ったら flip（逆張り側）の採点をスキップ（=順張り側だけ評価）
CRASH_SIGMA = _env_float("CRASH_SIGMA", 2.0)          # 例: 2.0
CRASH_VOLSIGMA = _env_float("CRASH_VOLSIGMA", 0.0030) # 例: 0.0030

# リスト系（',' と '|' 両対応）
SYMBOL_BLOCKLIST = _parse_symbol_set("SYMBOL_BLOCKLIST")
SYMBOL_CAUTIONLIST = _parse_symbol_set("SYMBOL_CAUTIONLIST")

# 追加: VolSigma 禁止レンジ（sigma がこの範囲なら alert 採用しない）
# 例: VOLSIGMA_BAN_RANGE="0.005,0.006" （空なら無効）
VOLSIGMA_BAN_MIN, VOLSIGMA_BAN_MAX = _parse_float_range("VOLSIGMA_BAN_RANGE")

print(
    "[CFG] "
    f"GUARDRAILS VOLRATIO_MAX={VOLRATIO_MAX} VOLRATIO_MAX_CAUTION={VOLRATIO_MAX_CAUTION} "
    f"RSI_SHORT_MIN={RSI_SHORT_MIN} RSI_LONG_MAX={RSI_LONG_MAX} "
    f"AI_PCT_MIN={AI_PCT_MIN} AI_PCT_MAX={AI_PCT_MAX} AI_PCT_MAX_CAUTION={AI_PCT_MAX_CAUTION} "
    f"AI_SIDE_MARGIN={AI_SIDE_MARGIN} "
    f"CRASH_SIGMA={CRASH_SIGMA} CRASH_VOLSIGMA={CRASH_VOLSIGMA} "
    f"SYMBOL_BLOCKLIST={sorted(list(SYMBOL_BLOCKLIST))} "
    f"SYMBOL_CAUTIONLIST={sorted(list(SYMBOL_CAUTIONLIST))} "
    f"VOLSIGMA_BAN_RANGE={VOLSIGMA_BAN_MIN}-{VOLSIGMA_BAN_MAX}"
)




# --- Advanced toggles (SAFE DEFAULT: OFF) ---
ENABLE_E_FILTER = os.environ.get("ENABLE_E_FILTER", "0") == "1"
E_TH = float(os.environ.get("E_TH", "0.0"))



# --- Advanced toggles (SAFE DEFAULT: OFF) ---
ENABLE_E_FILTER = os.environ.get("ENABLE_E_FILTER", "0") == "1"
E_TH = float(os.environ.get("E_TH", "0.0"))


DYNAMIC_AI_TH = os.environ.get("DYNAMIC_AI_TH", "0") == "1"
AI_TH_MIN = float(os.environ.get("AI_TH_MIN", "0.45"))
AI_TH_MAX = float(os.environ.get("AI_TH_MAX", "0.75"))
AI_TH_UP_ADD = float(os.environ.get("AI_TH_UP_ADD", "0.00"))
AI_TH_DOWN_ADD = float(os.environ.get("AI_TH_DOWN_ADD", "0.02"))
AI_TH_STORM_ADD = float(os.environ.get("AI_TH_STORM_ADD", "0.05"))

ENABLE_MULTI_MODEL = os.environ.get("ENABLE_MULTI_MODEL", "0") == "1"
# 例: "BTC=gs://bucket/models/btc.pkl;ETH=gs://bucket/models/eth.pkl"
MODEL_MAP = os.environ.get("MODEL_MAP", "")
MODEL_VERSION_MAP = os.environ.get("MODEL_VERSION_MAP", "")
MODEL_CACHE_TTL_SEC = int(float(os.environ.get("MODEL_CACHE_TTL_SEC", "3600")))

ENABLE_JUDGE = os.environ.get("ENABLE_JUDGE", "1") == "1"
AUTO_JUDGE_AFTER_RUN = os.environ.get("AUTO_JUDGE_AFTER_RUN", "0") == "1"
FAIL_CLOSED_ON_AI_BYPASS = os.environ.get("FAIL_CLOSED_ON_AI_BYPASS", "1") == "1"

# 60本取れないケースを安定運用で捌くための最低本数（rolling20 + 参照(-6) を考慮）
MIN_BARS = int(float(os.environ.get("MIN_BARS", "30")))

# ---- ヘッダー取得レンジの上限（列） ----
HEADER_COL_END = os.environ.get("HEADER_COL_END", "ZZ")

# ---- 期待最低列数（ヘッダー取得失敗時のフェイルセーフ）----
HEADER_LEN_TABLE = int(float(os.environ.get("HEADER_LEN_TABLE", "37")))
HEADER_LEN_LEARN = int(float(os.environ.get("HEADER_LEN_LEARN", "34")))

# ---- Self-Heal 設定（重要：手作業を減らす）----
AUTO_FIX_HEADERS = os.environ.get("AUTO_FIX_HEADERS", "1") == "1"
AUTO_CREATE_SHEETS = os.environ.get("AUTO_CREATE_SHEETS", "1") == "1"
STRICT_HEADER_CHECK = os.environ.get("STRICT_HEADER_CHECK", "0") == "1"

# TTL
HEADER_TTL_SEC = int(float(os.environ.get("HEADER_TTL_SEC", "600")))
SVC_TTL_SEC = int(float(os.environ.get("SVC_TTL_SEC", "1800")))
COLCOUNT_TTL_SEC = int(float(os.environ.get("COLCOUNT_TTL_SEC", "3600")))

# Sheets側の重複防止
DEDUP_LOOKBACK_ROWS = int(float(os.environ.get("DEDUP_LOOKBACK_ROWS", "500")))
DEDUP_TTL_SEC = int(float(os.environ.get("DEDUP_TTL_SEC", "120")))

# ccxt fetch retry
FETCH_RETRY = int(float(os.environ.get("FETCH_RETRY", "2")))
FETCH_RETRY_SLEEP_SEC = float(os.environ.get("FETCH_RETRY_SLEEP_SEC", "0.8"))

# judge が参照する「最新側の行数ウィンドウ」
JUDGE_LOOKBACK_ROWS = int(float(os.environ.get("JUDGE_LOOKBACK_ROWS", "2500")))

# ccxt exchange の軽いキャッシュ
EXCHANGE_TTL_SEC = int(float(os.environ.get("EXCHANGE_TTL_SEC", "600")))

# OKX デフォルト種別
OKX_DEFAULT_TYPE = os.environ.get("OKX_DEFAULT_TYPE", "swap")

# ============================================================
# 多重実行抑止：簡易分散ロック
# ============================================================
RUN_MUTEX_ENABLED = os.environ.get("RUN_MUTEX_ENABLED", "1") == "1"
RUN_MUTEX_SHEET = os.environ.get("RUN_MUTEX_SHEET", "_lock")
RUN_MUTEX_CELL = os.environ.get("RUN_MUTEX_CELL", "A1")
RUN_MUTEX_TTL_SEC = int(float(os.environ.get("RUN_MUTEX_TTL_SEC", "900")))

_INSTANCE_ID = "|".join(
    [
        os.environ.get("K_SERVICE", "svc"),
        os.environ.get("K_REVISION", "rev"),
        os.environ.get("HOSTNAME", "host"),
        str(os.getpid()),
    ]
)

print(
    "[CFG] "
    f"CAND_SIGMA={CAND_SIGMA} ALERT_SIGMA={ALERT_SIGMA} AI_TH={AI_TH} DEFAULT_LEV={DEFAULT_LEV} "
    f"ENABLE_JUDGE={ENABLE_JUDGE} AUTO_JUDGE_AFTER_RUN={AUTO_JUDGE_AFTER_RUN} MIN_BARS={MIN_BARS} "
    f"FAIL_CLOSED_ON_AI_BYPASS={FAIL_CLOSED_ON_AI_BYPASS} "
    f"HEADER_COL_END={HEADER_COL_END} HEADER_LEN_TABLE={HEADER_LEN_TABLE} HEADER_LEN_LEARN={HEADER_LEN_LEARN} "
    f"AUTO_FIX_HEADERS={AUTO_FIX_HEADERS} AUTO_CREATE_SHEETS={AUTO_CREATE_SHEETS} STRICT_HEADER_CHECK={STRICT_HEADER_CHECK} "
    f"DEDUP_LOOKBACK_ROWS={DEDUP_LOOKBACK_ROWS} JUDGE_LOOKBACK_ROWS={JUDGE_LOOKBACK_ROWS} EXCHANGE_TTL_SEC={EXCHANGE_TTL_SEC} "
    f"COLCOUNT_TTL_SEC={COLCOUNT_TTL_SEC} OKX_DEFAULT_TYPE={OKX_DEFAULT_TYPE} "
    f"RUN_MUTEX_ENABLED={RUN_MUTEX_ENABLED} RUN_MUTEX_SHEET={RUN_MUTEX_SHEET} RUN_MUTEX_TTL_SEC={RUN_MUTEX_TTL_SEC}"
)

# ==========================================
# 期待ヘッダー
# ==========================================
EXPECTED_HEADERS_LEARN = [
    "Datetime(SymbolTime_JST)", "Symbol", "Side", "EntryPrice", "ScoreSigma", "VolSigma", "Status",
    "TP", "SL", "TP_Pct", "SL_Pct", "Leverage", "Reserved1", "Reserved2",
    "AI_Pass", "BTC_Calm", "Version", "SignalType", "Reserved3", "Reserved4",
    "MarketTag", "BTC_Mode", "BTC_1h_Change", "RSI", "Note",
    "EvalStatus", "ExitTime", "ExitPrice", "ExitReason", "PnL_Pct", "Win/Lose", "HoldMin",
    # --- optional-but-we-want-to-always-write (for analysis/debug) ---
    "BandWidth", "BW_Change", "Vol_Change", "BTC_Ret", "BTC_Vol",
    "market_ai_score", "market_ai_pass", "market_ai_debug",
]


TABLE_FIELDS = [
    "Time", "Symbol", "Direction", "EntryPrice", "Score", "Sigma", "Group",
    "TP_Price", "SL_Price", "TP_%", "SL_%", "Lev", "TP_Lev%", "SL_Lev%",
    "BTC_Calm", "ReverseOK", "Strategy", "Note", "Chg%", "VolRatio", "OrderType",
]

FIELD_ALIASES: Dict[str, List[str]] = {
    "Time": ["Time", "Datetime(SymbolTime_JST)", "Datetime", "Time_JST", "Datetime_JST"],
    "Symbol": ["Symbol"],
    "Direction": ["Direction", "Side"],
    "EntryPrice": ["EntryPrice"],
    "Score": ["Score", "ScoreSigma", "Score_Sigma"],
    "Sigma": ["Sigma", "VolSigma", "Dynamic_Sigma"],
    "Group": ["Group", "Status"],
    "TP_Price": ["TP_Price", "TP"],
    "SL_Price": ["SL_Price", "SL"],
    "TP_%": ["TP_%", "TP_Pct", "TP%"],
    "SL_%": ["SL_%", "SL_Pct", "SL%"],
    "Lev": ["Lev", "Leverage"],
    "TP_Lev%": ["TP_Lev%", "TP_LevPct", "TP_Lev_Pct"],
    "SL_Lev%": ["SL_Lev%", "SL_LevPct", "SL_Lev_Pct"],
    "BTC_Calm": ["BTC_Calm"],
    "ReverseOK": ["ReverseOK"],
    "Strategy": ["Strategy", "SignalType", "Version"],
    "Note": ["Note"],
    "Chg%": ["Chg%", "Chg_Pct", "Change%", "ChgPct"],
    "VolRatio": ["VolRatio", "Vol_Ratio", "VolR", "VolRatio20"],
    "OrderType": ["OrderType", "Order_Type", "Type"],
    "EvalStatus": ["EvalStatus", "Eval"],
    "ExitTime": ["ExitTime", "Exit_Time"],
    "ExitPrice": ["ExitPrice", "Exit_Price"],
    "ExitReason": ["ExitReason", "Reason", "Exit_Reason"],
    "PnL_Pct": ["PnL_Pct", "PnL%", "PnL"],
    "Win/Lose": ["Win/Lose", "WinLose", "Result"],
    "HoldMin": ["HoldMin", "Hold_Min"],
}

TABLE_REQUIRED_FIELDS = [
    "Time", "Symbol", "Direction", "EntryPrice", "Score", "Sigma", "TP_Price", "SL_Price", "Lev", "Note",
    "EvalStatus", "ExitTime", "ExitPrice", "ExitReason", "PnL_Pct", "Win/Lose", "HoldMin"
]

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

JST = timezone(timedelta(hours=9))

http = requests.Session()
http.headers.update({"User-Agent": "spidey-bot/v3.4.9"})

_run_lock = threading.Lock()

_exchange_cache: Dict[str, Any] = {"ex": None, "ts": 0.0}
_symbol_resolve_cache: Dict[str, str] = {}

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
# 便利関数
# ==========================================
def send_discord_message(text: str):
    if (not discord_webhook_url) or ("ここに" in discord_webhook_url):
        print("[DBG] discord webhook url empty or placeholder")
        return

    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)] or [text]
    for chunk in chunks:
        try:
            r = http.post(discord_webhook_url, json={"content": chunk}, timeout=10)
            print(f"[DBG] discord status={r.status_code}")
            if r.status_code >= 300:
                print(f"[DBG] discord body={r.text[:200]}")
        except Exception as e:
            print(f"[ERR] discord webhook post: {e}")

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
    return service.spreadsheets().get(
        spreadsheetId=SPREADSHEET_ID,
        fields="sheets(properties(sheetId,title,gridProperties(rowCount,columnCount)))",
    ).execute(num_retries=5)

def ensure_sheet_exists(sheet_name: str, min_rows: int = 1000, min_cols: int = 26) -> bool:
    if not AUTO_CREATE_SHEETS:
        return True

    try:
        meta = _get_sheets_meta()
        for sh in meta.get("sheets", []) or []:
            p = (sh or {}).get("properties", {}) or {}
            if p.get("title") == sheet_name:
                return True

        # create
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
        service.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=req).execute()
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
        res = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=rng).execute()
        raw = (res.get("values", [[]]) or [[]])[0]
        return _normalize_headers(raw)
    except Exception as e:
        print(f"[WARN] read_header_row failed: sheet={sheet_name} err={e}")
        return []

def write_header_row(sheet_name: str, headers: List[str]) -> bool:
    try:
        service = get_sheet_service()
        end_col = col_to_a1(max(len(headers) - 1, 0))
        rng = f"{sheet_name}!A1:{end_col}1"
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=rng,
            valueInputOption="RAW",
            body={"values": [headers]},
        ).execute()
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
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=rng,
            valueInputOption="RAW",
            body={"values": [[value]]},
        ).execute()
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
    current = read_header_row(LEARN_SHEET_NAME)

    if current[:len(EXPECTED_HEADERS_LEARN)] == EXPECTED_HEADERS_LEARN:
        return True

    if not AUTO_FIX_HEADERS:
        return not STRICT_HEADER_CHECK

    trailing = []
    if len(current) > len(EXPECTED_HEADERS_LEARN):
        trailing = current[len(EXPECTED_HEADERS_LEARN):]

    new_headers = list(EXPECTED_HEADERS_LEARN) + trailing
    ok = write_header_row(LEARN_SHEET_NAME, new_headers)
    if ok:
        print("[CFG] learn_log headers fixed (preserve trailing headers).")
    return ok

def _find_first_blank_index(headers: List[str], limit: Optional[int]) -> int:
    max_i = len(headers) if limit is None else int(limit)
    scan_i = min(len(headers), max_i)
    for i in range(scan_i):
        if str(headers[i]).strip() == "":
            return i
    if max_i > len(headers):
        return len(headers)
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
            blank_idx = _find_first_blank_index(headers, limit)
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

    _sheet_header_cache[sheet_name] = {"headers": headers, "ts": now, "ok": ok}
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
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": adjusted_rows},
        ).execute()

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
        "options": {"defaultType": (OKX_DEFAULT_TYPE or "swap")},
    })

    # === 反映確認用の目印（これがログに出ない＝このコードがCloud Runで動いていない） ===
    print("[DBG] build_exchange:v2_urls_harden_traceback")

    # ---- urls の None/空を潰す（apiがdict前提の実装にも寄せる）----
    try:
        base = "https://www.okx.com"

        urls = getattr(exchange, "urls", None)
        if not isinstance(urls, dict):
            exchange.urls = {}
            urls = exchange.urls

        api = urls.get("api")
        print(f"[DBG] okx.urls(before)={repr(urls)}")

        # api を dict に寄せる（okx実装の多くはdict前提）
        if not isinstance(api, dict):
            api = {}
        urls["api"] = api

        # NoneType + str の主因になりやすいキーを確実に埋める
        for k in ("rest", "public", "private", "ws"):
            v = api.get(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                api[k] = base

        # 念のため：urls直下の参照され得るキーも埋める（実装差異対策）
        for k in ("www", "doc"):
            v = urls.get(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                urls[k] = base

        print(f"[DBG] okx.urls(after)={repr(urls)}")

    except Exception as e:
        import traceback
        print(f"[WARN] okx urls harden failed: {e}")
        print(traceback.format_exc())

    # ---- 重要：初期化に失敗した exchange をキャッシュしない（TTL中ずっと死ぬのを防ぐ）----
    load_ok = True
    try:
        exchange.load_markets()
        mk = getattr(exchange, "markets", None) or {}
        print(f"[OKX] load_markets ok: markets={len(mk)}")
    except Exception as e:
        load_ok = False
        import traceback
        print(f"[WARN] okx.load_markets failed: {e}")
        print(traceback.format_exc())
        print(f"[WARN] okx.urls(dump)={repr(getattr(exchange, 'urls', None))}")
        print(f"[WARN] okx.options(dump)={repr(getattr(exchange, 'options', None))}")

    if load_ok:
        _exchange_cache["ex"] = exchange
        _exchange_cache["ts"] = now
    else:
        print("[WARN] build_exchange: not caching exchange due to init failure")

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
        res = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!{col_letter}:{col_letter}",
        ).execute()
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
        res = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!{a1}{start}:{a2}{last_row}",
        ).execute()

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
    res = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=rng).execute()
    v = (res.get("values", [[]]) or [[]])[0]
    return "" if not v else str(v[0]).strip()

def _mutex_write(value: str):
    _ensure_mutex_sheet()
    service = get_sheet_service()
    rng = f"{RUN_MUTEX_SHEET}!{RUN_MUTEX_CELL}"
    service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=rng,
        valueInputOption="RAW",
        body={"values": [[value]]},
    ).execute()

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
# AI 推論の安全化（feature mismatch 対策）
# - 特徴量名が取れるなら列を揃えて再試行
# - それでも無理なら例外で落とさず bypass する
# - 戻り値: (proba, bypassed, debug_dict)
# ==========================================
def _extract_feature_names(model) -> Optional[List[str]]:
    if model is None:
        return None

    # 1) estimator 直下
    try:
        fni = getattr(model, "feature_names_in_", None)
        if fni is not None:
            return [str(x) for x in list(fni)]
    except Exception:
        pass

    # 2) Pipeline / named_steps
    try:
        named_steps = getattr(model, "named_steps", None)
        if isinstance(named_steps, dict):
            for st in reversed(list(named_steps.values())):
                fni = getattr(st, "feature_names_in_", None)
                if fni is not None:
                    return [str(x) for x in list(fni)]
    except Exception:
        pass

    # 3) Pipeline / steps
    try:
        steps = getattr(model, "steps", None)
        if isinstance(steps, list):
            for _, st in reversed(steps):
                fni = getattr(st, "feature_names_in_", None)
                if fni is not None:
                    return [str(x) for x in list(fni)]
    except Exception:
        pass

    return None

def _infer_expected_n_features(model) -> int:
    try:
        if model is None:
            return -1

        if hasattr(model, "n_features_in_"):
            return int(getattr(model, "n_features_in_"))

        if hasattr(model, "steps"):
            for _, step in getattr(model, "steps", []):
                if hasattr(step, "n_features_in_"):
                    return int(getattr(step, "n_features_in_"))

        if hasattr(model, "named_steps"):
            for step in getattr(model, "named_steps", {}).values():
                if hasattr(step, "n_features_in_"):
                    return int(getattr(step, "n_features_in_"))

        return -1
    except Exception:
        return -1

def _align_by_feature_names(feats: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """
    方針：
      - 欠損や変換不能は埋めない（NaNのまま）
      - 期待列に無い列は作るが、中身は NaN（固定値0で埋めない）
    """
    aligned = pd.DataFrame(index=feats.index)
    for col in expected_cols:
        if col in feats.columns:
            aligned[col] = pd.to_numeric(feats[col], errors="coerce")
        else:
            aligned[col] = np.nan
    return aligned


def safe_predict_proba(model, feats: pd.DataFrame) -> Tuple[Optional[np.ndarray], bool, Dict[str, Any]]:
    """
    安全な predict_proba ラッパー。
    方針:
      - 予測不能/不整合/NaN混入など「信用できない入力」は固定値で埋めずに bypass（予測しない）
      - 返り値: (proba or None, bypassed, debug)
    """
    debug: Dict[str, Any] = {
        "expected_cols": None,
        "expected_n_features": -1,
        "input_n_features": -1,
        "action": "none",
        "error": "",
        # NaN原因特定用（常にキーを持たせる）
        "nan_cols": None,
        "nan_n_cols": None,
        "nan_cnt": None,
        "nan_filled": 0,
        "alias_renamed": None,
    }

    try:
        # 0) model が None の場合は bypass（推測値で埋めない）
        if model is None:
            debug["action"] = "model_none_bypass"
            return None, True, debug

        # 1) model が dict/tuple/list の wrapper の場合は中身(estimator)を取り出す
        if isinstance(model, dict):
            for k in ("model", "estimator", "clf", "pipeline", "sk_model"):
                if k in model:
                    model = model[k]
                    debug["action"] = "unwrapped_dict_model"
                    break

        if isinstance(model, (tuple, list)) and len(model) >= 1:
            model = model[0]
            debug["action"] = "unwrapped_list_model"

        # unwrap した結果でも predict_proba が無いなら bypass
        if not hasattr(model, "predict_proba"):
            debug["action"] = "no_predict_proba_bypass"
            debug["error"] = f"model_type={type(model)} has no predict_proba"
            print(f"[AI] safe_predict_proba bypass: {debug['error']}")
            return None, True, debug

        # 2) feats の整形
        if feats is None:
            feats = pd.DataFrame([{}])
        elif not isinstance(feats, pd.DataFrame):
            feats = pd.DataFrame(feats)

        # inf は NaN にする（埋めずに bypass 判定へ）
        feats = feats.replace([np.inf, -np.inf], np.nan)
        debug["input_n_features"] = int(feats.shape[1])

        # 14列モデルに対して列順だけ固定したい場合
        FEATURE_COLUMNS_14 = [
            "EntryPrice",
            "ScoreSigma",
            "VolSigma",
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
            "RSI",
        ]
        try:
            if isinstance(feats, pd.DataFrame) and hasattr(model, "n_features_in_"):
                expected_n = int(getattr(model, "n_features_in_", 0) or 0)
                if expected_n == 14:
                    feats = feats.reindex(columns=FEATURE_COLUMNS_14)
                    debug["input_n_features"] = int(feats.shape[1])
        except Exception:
            pass

        # 3) 特徴量の整合（feature_names_in_ に追従）
        expected_cols = _extract_feature_names(model)
        if expected_cols:
            expected_cols = [str(c) for c in list(expected_cols)]
            debug["expected_cols"] = list(expected_cols)
            debug["expected_n_features"] = int(len(expected_cols))

            # 列名揺れ吸収（スペース/アンダースコアの違い等）
            try:
                rename_map = {}
                if isinstance(feats, pd.DataFrame):
                    feats_cols = set(str(c) for c in feats.columns)

                    for c in expected_cols:
                        if c in feats_cols:
                            continue

                        candidates = [
                            c.replace("_", " "),
                            c.replace(" ", "_"),
                        ]

                        if c == "BandWidth":
                            candidates += ["Bandwidth", "Band_Width", "Band Width"]
                        elif c == "Bandwidth":
                            candidates += ["BandWidth", "Band_Width", "Band Width"]

                        for cand in candidates:
                            if cand in feats_cols:
                                rename_map[cand] = c
                                feats_cols.remove(cand)
                                feats_cols.add(c)
                                break

                    if rename_map:
                        feats = feats.rename(columns=rename_map)
                        debug["alias_renamed"] = dict(rename_map)

            except Exception:
                pass

            feats = _align_by_feature_names(feats, expected_cols)
            debug["action"] = "aligned_by_feature_names"
        else:
            expected_n = _infer_expected_n_features(model)
            debug["expected_n_features"] = int(expected_n)
            if expected_n > 0 and int(feats.shape[1]) != int(expected_n):
                debug["action"] = "feature_count_mismatch_bypass"
                debug["error"] = f"feature mismatch: X={feats.shape[1]} expected={expected_n}"
                return None, True, debug

        # 4) 方針：NaN が残っていたら「固定値で埋めずに予測しない（bypass）」
        if isinstance(feats, pd.DataFrame):
            if feats.shape[1] == 0:
                debug["action"] = "empty_features_bypass"
                debug["error"] = "no features after alignment"
                return None, True, debug

            feats = feats.replace([np.inf, -np.inf], np.nan)

            mask = feats.isna()
            if mask.any().any():
                nan_cols = [c for c in feats.columns if mask[c].any()]
                nan_cnt = int(mask.sum().sum())

                debug["nan_cols"] = nan_cols
                debug["nan_n_cols"] = int(len(nan_cols))
                debug["nan_cnt"] = nan_cnt
                debug["nan_filled"] = 0
                debug["action"] = "nan_input_bypass"
                debug["error"] = "input contains NaN; skip predict_proba"
                return None, True, debug

        # 5) predict_proba 実行
        proba = np.asarray(model.predict_proba(feats), dtype=float)
        if proba.ndim == 1:
            proba = np.vstack([1.0 - proba, proba]).T
        if proba.shape[1] == 1:
            proba = np.hstack([1.0 - proba, proba])

        debug["action"] = "predicted"
        return proba, False, debug

    except Exception as e:
        debug["action"] = "exception_bypass"
        debug["error"] = f"{type(e).__name__}: {e}"
        print(f"[AI] safe_predict_proba bypass: {debug['error']}")
        return None, True, debug


def derive_ai_debug(btc_mode: str, signal_type: str, side: str) -> str:
    """
    learn_log の ai_debug に入れる値を決める
      - DIRECT  : 順張り
      - REVERSE : 逆張り
      - RANGE   : レンジ
      - UNKNOWN : 判定不能（入力が想定外）
    ルール:
      BTC_Mode=Up   かつ LONG  -> DIRECT
      BTC_Mode=Down かつ SHORT -> DIRECT
      BTC_Mode=Up/Down で逆方向 -> REVERSE
      BTC_Mode=Range -> RANGE
    """
    bm = (btc_mode or "").strip().lower()
    st = (signal_type or "").strip().upper()
    sd = (side or "").strip().upper()

    # SignalType が空のときは Side を使う
    direction = st if st else sd

    # direction を LONG/SHORT に正規化（想定外はそのまま）
    if direction in ("BUY", "BULL", "UP", "L"):
        direction = "LONG"
    if direction in ("SELL", "BEAR", "DOWN", "S"):
        direction = "SHORT"

    if bm == "range":
        return "RANGE"

    if bm == "up":
        if direction == "LONG":
            return "DIRECT"
        if direction == "SHORT":
            return "REVERSE"
        return "UNKNOWN"

    if bm == "down":
        if direction == "SHORT":
            return "DIRECT"
        if direction == "LONG":
            return "REVERSE"
        return "UNKNOWN"

    return "UNKNOWN"

def compute_dynamic_ai_th(base_th: float, btc_mode: str, median_sigma: float, btc_ok: bool, btc_calm: bool) -> float:
    """
    DYNAMIC_AI_TH=1 のときだけ使う想定。
    事故らないように「少しだけ」動かし、上下限でクランプする。
    """
    th = float(base_th)

    if not btc_ok:
        th += float(AI_TH_STORM_ADD)

    if not btc_calm:
        th += float(AI_TH_STORM_ADD)

    mode = ("" if btc_mode is None else str(btc_mode)).strip().upper()
    if mode == "UP":
        th += float(AI_TH_UP_ADD)
    elif mode == "DOWN":
        th += float(AI_TH_DOWN_ADD)

    th = max(float(AI_TH_MIN), min(float(AI_TH_MAX), th))
    return th

# ==========================================
# モデル読み込み（GCS対応） + 銘柄別モデル（任意） + リロード
# ==========================================
MODEL_LOCAL_PATH = "trade_ai_model.pkl"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "")
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "")

AI_MODEL_VERSION_RUNTIME = MODEL_VERSION
AI_MODEL_SOURCE_RUNTIME = "none"

_model_cache: Dict[str, Dict[str, Any]] = {}  # key: uri -> {model, ts, version, source}

# ==========================================
# /train（学習→GCS保存）用 設定
# ==========================================
TRAIN_ENABLED = os.environ.get("TRAIN_ENABLED", "1") == "1"
TRAIN_LOOKBACK_ROWS = int(float(os.environ.get("TRAIN_LOOKBACK_ROWS", "2500")))  # learn_logから読む最大行数
TRAIN_MIN_SAMPLES = int(float(os.environ.get("TRAIN_MIN_SAMPLES", "120")))        # Win/Lose がこれ未満なら学習しない
TRAIN_TEST_SIZE = float(os.environ.get("TRAIN_TEST_SIZE", "0.2"))
TRAIN_RANDOM_STATE = int(float(os.environ.get("TRAIN_RANDOM_STATE", "42")))

# 出力先（未指定なら MODEL_GCS_URI から bucket を推定）
TRAIN_GCS_BUCKET = os.environ.get("TRAIN_GCS_BUCKET", "")
TRAIN_GCS_PREFIX = os.environ.get("TRAIN_GCS_PREFIX", "models")  # 例: models/<ver>/trade_ai_model.pkl

def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    # "gs://bucket/path/to.obj" -> ("bucket", "path/to.obj")
    u = ("" if uri is None else str(uri)).strip()
    if not u.startswith("gs://"):
        return "", ""
    parts = u.replace("gs://", "").split("/", 1)
    bucket = parts[0].strip()
    obj = parts[1].strip() if len(parts) > 1 else ""
    return bucket, obj

def _default_train_bucket() -> str:
    # TRAIN_GCS_BUCKET が無ければ MODEL_GCS_URI の bucket を使う
    if TRAIN_GCS_BUCKET:
        return str(TRAIN_GCS_BUCKET).strip()
    b, _ = _parse_gs_uri(MODEL_GCS_URI)
    return b

def _build_train_output_uri(version: str) -> str:
    bucket = _default_train_bucket()
    if not bucket:
        return ""
    v = ("" if version is None else str(version)).strip()
    if not v:
        v = datetime.now(JST).strftime("v%Y-%m-%d_%H%M%S")
    obj = f"{TRAIN_GCS_PREFIX}/{v}/trade_ai_model.pkl"
    return f"gs://{bucket}/{obj}"

def _sheet_rows_as_df(sheet_name: str, lookback_rows: int) -> pd.DataFrame:
    """
    指定シートの末尾 lookback_rows を DataFrame 化（ヘッダーはA1）
    """
    service = get_sheet_service()

    headers, _, okh = get_headers_and_len(sheet_name)
    if STRICT_HEADER_CHECK and not okh:
        raise RuntimeError(f"header_check_failed: sheet={sheet_name}")

    last_row = _get_row_count_cached(sheet_name)
    if last_row < 2:
        return pd.DataFrame(columns=headers)

    start_row = max(2, last_row - int(lookback_rows) + 1)
    rng = f"{sheet_name}!A{start_row}:{HEADER_COL_END}{last_row}"
    res = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=rng,
    ).execute()

    rows = res.get("values", []) or []
    if not rows:
        return pd.DataFrame(columns=headers)

    # 行の長さをヘッダーに合わせる
    padded = []
    ncol = len(headers)
    for r in rows:
        rr = list(r)
        if len(rr) < ncol:
            rr = rr + ([""] * (ncol - len(rr)))
        else:
            rr = rr[:ncol]
        padded.append(rr)

    df = pd.DataFrame(padded, columns=headers)
    return df

def _normalize_winlose(x: Any) -> str:
    s = ("" if x is None else str(x)).strip().lower()
    if s in {"win", "w", "1", "true"}:
        return "Win"
    if s in {"lose", "l", "0", "false"}:
        return "Lose"
    return ""

def _build_training_matrix_from_learn_log(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    learn_log から学習データを作る（推論側の9特徴量に揃える版）

    方針（統一 / NO IMPUTATION / NO GUESSING）:
    - 必須列（required_cols）に空白セル（"" / None / NaN）がある行は学習に使わない（埋めない）
    - Win/Lose が未確定の行は学習に使わない
    - BandWidth/BW_Change/Vol_Change/BTC_Ret/BTC_Vol は learn_log の実値のみ使う
      （0固定・推測生成は一切しない）
    """
    info: Dict[str, Any] = {
        "policy": "drop_rows_with_any_blank_required_feature_no_imputation",
        "required_cols": [
            "Win/Lose", "Side", "ScoreSigma", "VolSigma", "RSI",
            "BandWidth", "BW_Change", "Vol_Change", "BTC_Ret", "BTC_Vol",
        ],
        "missing_cols": [],
        "rows_total": 0,
        "rows_labeled": 0,
        "rows_skipped_blank_required": 0,
        "rows_used": 0,
        # NOTE: モデルの feature_names_in_ と一致させる（スペース入り）
        "feature_columns": [
            "Sigma", "BandWidth", "BW Change", "RSI", "Vol Change",
            "Rise Score", "Drop Score", "BTC Ret", "BTC Vol"
        ],
        "notes": [
            "NO IMPUTATION: rows with blank/non-numeric values in required features are dropped.",
            "Rise/Drop are derived from ScoreSigma + Side (directional encoding).",
        ],
    }

    if df is None or df.empty:
        return pd.DataFrame(columns=info["feature_columns"]), np.array([], dtype=int), info

    info["rows_total"] = int(len(df))

    needed = list(info["required_cols"])
    missing = [c for c in needed if c not in df.columns]
    info["missing_cols"] = list(missing)
    if missing:
        return pd.DataFrame(columns=info["feature_columns"]), np.array([], dtype=int), info

    df2 = df.copy()

    # Win/Lose を正規化してラベル行だけ残す
    df2["__winlose__"] = df2["Win/Lose"].apply(_normalize_winlose)
    df2 = df2[df2["__winlose__"].isin(["Win", "Lose"])].copy()
    info["rows_labeled"] = int(len(df2))
    if df2.empty:
        return pd.DataFrame(columns=info["feature_columns"]), np.array([], dtype=int), info

    # 必須列の空白行を除外（埋めない）
    required_cols = [
        "Side", "ScoreSigma", "VolSigma", "RSI",
        "BandWidth", "BW_Change", "Vol_Change", "BTC_Ret", "BTC_Vol",
    ]
    tmp = df2[required_cols].copy()

    blank = tmp.isna()
    for c in required_cols:
        blank[c] = blank[c] | tmp[c].astype(str).str.strip().eq("")

    blank_mask_any = blank.any(axis=1)
    info["rows_skipped_blank_required"] = int(blank_mask_any.sum())

    df2 = df2[~blank_mask_any].copy()
    if df2.empty:
        return pd.DataFrame(columns=info["feature_columns"]), np.array([], dtype=int), info

    y = (df2["__winlose__"] == "Win").astype(int).to_numpy()

    # 数値化（必須列は空白行を落としているので、ここで大量NaNにならない前提）
    score = pd.to_numeric(df2["ScoreSigma"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sigma = pd.to_numeric(df2["VolSigma"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rsi = pd.to_numeric(df2["RSI"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    bw = pd.to_numeric(df2["BandWidth"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    bw_chg = pd.to_numeric(df2["BW_Change"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    vol_chg = pd.to_numeric(df2["Vol_Change"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    btc_ret = pd.to_numeric(df2["BTC_Ret"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    btc_vol = pd.to_numeric(df2["BTC_Vol"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    side = df2["Side"].astype(str).fillna("").str.upper()
    is_long = side.str.contains("LONG") | side.str.contains("BUY")
    is_short = side.str.contains("SHORT") | side.str.contains("SELL")

    # 反対側は 0.0（欠損埋めではなく方向エンコードの定義）
    rise_score = np.where(is_long, score, 0.0)
    drop_score = np.where(is_short, score, 0.0)

    X = pd.DataFrame({
        "Sigma": sigma.astype(float),
        "BandWidth": bw.astype(float),
        "BW Change": bw_chg.astype(float),
        "RSI": rsi.astype(float),
        "Vol Change": vol_chg.astype(float),
        "Rise Score": pd.to_numeric(rise_score, errors="coerce").astype(float),
        "Drop Score": pd.to_numeric(drop_score, errors="coerce").astype(float),
        "BTC Ret": btc_ret.astype(float),
        "BTC Vol": btc_vol.astype(float),
    }, index=df2.index).replace([np.inf, -np.inf], np.nan)

    # NaN が残る行は学習に使わない（=欠損は埋めない）
    valid = ~X.isna().any(axis=1)
    X = X[valid].copy()
    y = y[valid.to_numpy()]

    info["rows_used"] = int(len(X))
    return X, y, info


# ==========================================
# GCS Upload (Unified Tuple Return)
# ==========================================
def _gcs_upload_from(src_path: str, uri: str) -> Tuple[bool, str]:
    """
    src_path のファイルを gs://bucket/path/to.obj にアップロードする。
    戻り値: (ok, message)
    """
    if (not uri) or (not uri.startswith("gs://")):
        return False, "uri is empty or not gs://"
    if (not src_path) or (not os.path.exists(src_path)):
        return False, f"src not found: {src_path}"

    try:
        parts = uri.replace("gs://", "").split("/", 1)
        bucket_name = (parts[0] if len(parts) >= 1 else "").strip()
        blob_name = (parts[1] if len(parts) >= 2 else "").strip()

        if not bucket_name:
            return False, f"invalid gs uri (bucket missing): {uri}"
        if not blob_name:
            return False, f"invalid gs uri (object path missing): {uri}"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(src_path)
        return True, "uploaded"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _sanitize_version_tag(tag: str) -> str:
    """
    version文字列をファイルパスやGCSパスに安全な形へ整形する。
    """
    t = ("" if tag is None else str(tag)).strip()
    if not t:
        return ""
    t = re.sub(r"[^0-9A-Za-z._-]+", "_", t).strip("_")
    return t[:64] if t else ""

def train_and_export_model(
    lookback_rows: int,
    out_version: str,
    hot_reload: bool,
    min_samples: int,
    upload: bool,
) -> Dict[str, Any]:
    """
    learn_log -> train -> (upload=1なら) gs://... に保存
    戻り値は /train のJSONレスポンス用
    """
    if not TRAIN_ENABLED:
        return {"ok": False, "error": "TRAIN_ENABLED=0 (train disabled)", "version": VERSION}

    if not SKLEARN_OK:
        return {"ok": False, "error": "scikit-learn not available (install scikit-learn).", "version": VERSION}

    df = _sheet_rows_as_df(LEARN_SHEET_NAME, lookback_rows=lookback_rows)
    X, y, info = _build_training_matrix_from_learn_log(df)

    n = int(len(y))
    need = int(min_samples) if int(min_samples) > 0 else int(TRAIN_MIN_SAMPLES)
    if n < need:
        return {
            "ok": False,
            "error": f"not enough labeled samples: {n} < min_samples={need}",
            "version": VERSION,
            "samples": n,
            "info": info,
        }

    # 1クラスしか無いと学習が必ず落ちるので事前に弾く
    uniq = np.unique(y)
    if int(len(uniq)) < 2:
        return {
            "ok": False,
            "error": f"need at least 2 classes in Win/Lose. got classes={uniq.tolist()}",
            "version": VERSION,
            "samples": n,
            "info": info,
        }

    # 分割（stratifyが落ちる場合は保険）
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TRAIN_TEST_SIZE, random_state=TRAIN_RANDOM_STATE, stratify=y
        )
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TRAIN_TEST_SIZE, random_state=TRAIN_RANDOM_STATE
        )

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model.fit(X_tr, y_tr)

    # 評価
    y_pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred)) if accuracy_score is not None else None

    auc = None
    try:
        proba = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba)) if roc_auc_score is not None else None
    except Exception:
        auc = None

    # version名を安全化
    ts_ver_raw = out_version if out_version else datetime.now(JST).strftime("v%Y%m%d_%H%M%S")
    ts_ver = _sanitize_version_tag(ts_ver_raw) or datetime.now(JST).strftime("v%Y%m%d_%H%M%S")

    # まずローカルに保存（upload=0でも hot_reload=1の時に使える）
    tmp_local = f"/tmp/trade_ai_model_{ts_ver}.pkl"
    try:
        joblib.dump(model, tmp_local)
    except Exception as e:
        return {"ok": False, "error": f"joblib.dump failed: {e}", "version": VERSION}

    out_uri = ""
    if upload:
        out_uri = _build_train_output_uri(ts_ver)
        if not out_uri:
            return {"ok": False, "error": "cannot determine GCS output uri (set TRAIN_GCS_BUCKET or MODEL_GCS_URI)", "version": VERSION}

        ok_up, up_msg = _gcs_upload_from(tmp_local, out_uri)
        if not ok_up:
            return {"ok": False, "error": f"GCS upload failed: {out_uri} ({up_msg})", "version": VERSION}

    # 任意：この実行中インスタンスだけ即時反映
    reloaded = False
    if hot_reload:
        try:
            global ai_model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME
            ai_model = model
            AI_MODEL_VERSION_RUNTIME = ts_ver
            AI_MODEL_SOURCE_RUNTIME = "trained" if upload else "trained(no-upload)"
            reloaded = True
        except Exception as e:
            print(f"[TRAIN] hot_reload failed: {e}")
            reloaded = False

    resp = {
        "ok": True,
        "version": VERSION,
        "trained_samples": n,
        "metrics": {"accuracy": acc, "auc": auc},
        "info": info,
        "new_model": {
            "model_version_suggest": ts_ver,
            "model_gcs_uri": out_uri,
            "uploaded": bool(upload),
            "hot_reloaded_in_this_instance": bool(reloaded),
        },
        "next_env_vars": ({"MODEL_VERSION": ts_ver, "MODEL_GCS_URI": out_uri} if upload else {}),
    }
    return resp


def _parse_kv_map(s: str) -> Dict[str, str]:
    """
    "BTC=gs://...;ETH=gs://..." -> {"BTC":"gs://...","ETH":"gs://..."}
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    parts = [p.strip() for p in str(s).split(";") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().upper()
        v = v.strip()
        if k and v:
            out[k] = v
    return out

def _gcs_download_to(uri: str, dst_path: str) -> bool:
    if (not uri) or (not uri.startswith("gs://")):
        return False

    try:
        parts = uri.replace("gs://", "").split("/", 1)
        bucket_name = (parts[0] if len(parts) >= 1 else "").strip()
        blob_name = (parts[1] if len(parts) >= 2 else "").strip()

        if not bucket_name or not blob_name:
            print(f"[AI] invalid gs uri (need gs://bucket/object): {uri}")
            return False

        print(f"[AI] Downloading model from {uri} -> {dst_path}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(dst_path)
        print("[AI] Download complete.")
        return True
    except Exception as e:
        print(f"[AI] GCS Download failed: {e}")
        return False


def _load_model_from_path(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[AI] Load Failed: {e}")
        return None

def reload_default_model() -> bool:
    """
    既定モデル(ai_model)を再ロードする。
    優先順位:
      1) MODEL_GCS_URI(gs://...) があれば毎回それを優先してダウンロード→ロード
      2) だめならローカル MODEL_LOCAL_PATH をロード
      3) だめなら ai_model=None として bypass（safe_predict_proba で 500 回避）
    """
    global ai_model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME, _model_cache

    # 銘柄別モデルキャッシュはリロード時にクリア
    _model_cache = {}

    # 1) GCS から（URIが変わったときに確実に反映されるよう、URIごとの tmp パスに保存）
    if MODEL_GCS_URI.startswith("gs://"):
        tmp_path = _safe_tmp_path_for_uri(MODEL_GCS_URI)
        ok_dl = _gcs_download_to(MODEL_GCS_URI, tmp_path)

        if ok_dl and os.path.exists(tmp_path):
            m = _load_model_from_path(tmp_path)
            if m is not None:
                ai_model = m
                AI_MODEL_VERSION_RUNTIME = MODEL_VERSION
                AI_MODEL_SOURCE_RUNTIME = "gcs"
                print(f"[AI] Model Loaded Successfully uri={MODEL_GCS_URI} path={tmp_path} ver={AI_MODEL_VERSION_RUNTIME}")
                return True

    # 2) ローカル fallback
    if os.path.exists(MODEL_LOCAL_PATH):
        m = _load_model_from_path(MODEL_LOCAL_PATH)
        if m is not None:
            ai_model = m
            AI_MODEL_VERSION_RUNTIME = MODEL_VERSION
            AI_MODEL_SOURCE_RUNTIME = "local"
            print(f"[AI] Model Loaded Successfully path={MODEL_LOCAL_PATH} ver={AI_MODEL_VERSION_RUNTIME}")
            return True

    # 3) 失敗 → AI ゲートは bypass（ただし FAIL_CLOSED_ON_AI_BYPASS によって挙動は変わる）
    print("[AI] model load failed -> AI gate is bypassed (ai_model=None).")
    ai_model = None
    AI_MODEL_VERSION_RUNTIME = MODEL_VERSION
    AI_MODEL_SOURCE_RUNTIME = "none"
    return False


def get_ai_model(force_reload: bool = False):
    """
    既定モデル(ai_model)を返す。
    - ai_model が None の場合は reload_default_model() で読み込みを試みる
    - force_reload=True の場合は必ず reload_default_model() を実行
    """
    global ai_model

    if force_reload:
        reload_default_model()
        return ai_model

    if ai_model is not None:
        return ai_model

    # ai_model が None のまま走り続けないよう、ここで必ずロードを試す
    reload_default_model()
    return ai_model

def _build_training_dataset_from_learn_log(lookback_rows: int) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    learn_log から学習用 X,y を作る。

    方針：
    - 学習データ（X,y）生成の「正」は _build_training_matrix_from_learn_log(df) に一本化する。
    - この関数は「Sheets からの取得」「lookback 範囲の制御」「DataFrame 化」だけを担当するラッパーにする。
    """
    meta: Dict[str, Any] = {"rows_scanned": 0, "rows_used": 0, "class_balance": {}}

    ok, msg = self_heal_prerequisites()
    if not ok:
        raise RuntimeError(f"self_heal_prerequisites failed: {msg}")

    headers = read_header_row(LEARN_SHEET_NAME)
    if not headers:
        raise RuntimeError("learn_log header is empty")

    # Win/Lose が無いと学習できないため早期に落とす（原因が分かりやすい）
    hm = _build_headers_map(headers)
    col_winlose = _resolve_col_idx(hm, "Win/Lose")
    if col_winlose == -1:
        raise RuntimeError(f"learn_log required column missing: Win/Lose. headers={headers[:40]}")

    last_row = _get_row_count_cached(LEARN_SHEET_NAME)
    if last_row < 2:
        raise RuntimeError("learn_log is empty")

    # lookback_rows は「末尾からの件数」
    lb = int(lookback_rows) if (lookback_rows is not None and int(lookback_rows) > 0) else 0
    start_row = 2 if lb <= 0 else max(2, last_row - lb + 1)

    service = get_sheet_service()
    res = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{LEARN_SHEET_NAME}!A{start_row}:{HEADER_COL_END}{last_row}",
    ).execute()
    rows = res.get("values", []) or []

    # DataFrame 化（列数が足りない行は右を空文字で埋める／多い行は切る）
    norm_rows: List[List[Any]] = []
    ncols = int(len(headers))
    for r in rows:
        rr = list(r) if r is not None else []
        if len(rr) < ncols:
            rr = rr + [""] * (ncols - len(rr))
        elif len(rr) > ncols:
            rr = rr[:ncols]
        norm_rows.append(rr)

    df = pd.DataFrame(norm_rows, columns=headers)

    meta["rows_scanned"] = int(len(df))
    meta["start_row"] = int(start_row)
    meta["last_row"] = int(last_row)
    meta["lookback_rows_requested"] = int(lb) if lb > 0 else 0

    # ここで「唯一の正」に委譲
    X_df, y_arr, m2 = _build_training_matrix_from_learn_log(df)

    # meta 統合（matrix側の情報を優先しつつ、ラッパー側も残す）
    m2 = dict(m2 or {})
    merged = dict(meta)
    merged.update(m2)

    # rows_used / class_balance が無ければ補完
    if "rows_used" not in merged:
        merged["rows_used"] = int(len(y_arr)) if y_arr is not None else 0

    if "class_balance" not in merged:
        if y_arr is None or len(y_arr) == 0:
            merged["class_balance"] = {}
        else:
            uniq, cnt = np.unique(np.asarray(y_arr, dtype=int), return_counts=True)
            merged["class_balance"] = {str(int(k)): int(v) for k, v in zip(uniq, cnt)}

    return X_df, np.asarray(y_arr, dtype=int), merged


def _train_model_from_learn_log(lookback_rows: int, min_samples: int) -> Tuple[Any, Dict[str, Any]]:
    """
    learn_log からモデルを学習して返す
    """
    X, y, info = _build_training_dataset_from_learn_log(lookback_rows=lookback_rows)
    if int(len(y)) < int(min_samples):
        raise RuntimeError(f"not enough samples: {len(y)} < {min_samples}")

    # sklearn は環境に入っている前提（既存モデルがjoblibでロードできているため）
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
    except Exception as e:
        raise RuntimeError(f"sklearn import failed: {type(e).__name__}: {e}")

    metrics: Dict[str, Any] = {"ok": True}
    strat = y if (len(np.unique(y)) >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    clf = LogisticRegression(
        max_iter=800,
        class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) >= 2 else None

    metrics["rows_used"] = int(info.get("rows_used", 0))
    metrics["rows_scanned"] = int(info.get("rows_scanned", 0))
    metrics["class_balance"] = info.get("class_balance", {})
    metrics["acc"] = float(accuracy_score(y_test, pred)) if len(y_test) > 0 else None
    if proba is not None:
        metrics["auc"] = float(roc_auc_score(y_test, proba))
    else:
        metrics["auc"] = None

    # feature_names を保持させる（DataFrame入力で feature_names_in_ が入る）
    return clf, metrics


def _safe_tmp_path_for_uri(uri: str) -> str:
    h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    d = "/tmp/models"
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return f"{d}/model_{h}.pkl"

def get_ai_model_for_symbol(symbol_code: str):
    """
    ENABLE_MULTI_MODEL=1 かつ MODEL_MAP に該当がある銘柄だけ別モデルを使う。
    それ以外は既定モデル(ai_model)を使う。
    """
    if not ENABLE_MULTI_MODEL:
        return ai_model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME

    sym = ("" if symbol_code is None else str(symbol_code)).strip().upper()
    m_map = _parse_kv_map(MODEL_MAP)
    v_map = _parse_kv_map(MODEL_VERSION_MAP)
    uri = m_map.get(sym, "")

    if (not uri) or (not uri.startswith("gs://")):
        return ai_model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME

    now = time.time()
    c = _model_cache.get(uri)
    if c and (now - float(c.get("ts", 0))) <= MODEL_CACHE_TTL_SEC:
        return c.get("model"), str(c.get("version", "")), str(c.get("source", "gcs"))

    tmp_path = _safe_tmp_path_for_uri(uri)
    ok_dl = _gcs_download_to(uri, tmp_path)
    if not ok_dl or (not os.path.exists(tmp_path)):
        return ai_model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME

    m = _load_model_from_path(tmp_path)
    if m is None:
        return ai_model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME

    ver = v_map.get(sym, "")
    _model_cache[uri] = {"model": m, "ts": now, "version": ver, "source": "gcs"}
    print(f"[AI] Multi-model loaded sym={sym} uri={uri} ver={ver}")
    return m, ver, "gcs"


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

def _pad_row_to_fields(row: List[Any], fields: List[str], fill: Any = "") -> List[Any]:
    """
    row を fields の長さに合わせる（列ズレ事故の防止）
    - 足りない: fill で右側を埋める
    - 多い: 右側を切る
    """
    n = len(fields) if fields else 0
    if n <= 0:
        return row
    if len(row) < n:
        return row + [fill] * (n - len(row))
    if len(row) > n:
        return row[:n]
    return row

# ==========================================
# ロジック本体 (/run)
# ==========================================
def logic_main(force: bool = False):
    global last_alert_records, last_candidate_records, ai_model

    start = time.time()
    now_jst = datetime.now(JST)
    print(f"[RUN] start {now_jst.isoformat()}  VERSION={VERSION} force={force}")

    # self heal
    ok, msg = self_heal_prerequisites()
    if not ok:
        send_discord_message(f"[WARN] self_heal_prerequisites failed: {msg}")
        return f"SelfHealFailed: {msg}"

    # --- AIモデルを確実に確保（Noneのまま走り続けない） ---
    try:
        # get_ai_model() が「必要ならロード」する想定（あなたの現行コードに存在している前提）
        m = get_ai_model()
        ai_model = m  # 念のためglobalも同期
        if ai_model is None:
            send_discord_message("[WARN] AI model is None (load failed). Run will continue with rule-based parts if allowed.")
    except Exception as e:
        send_discord_message(f"[WARN] get_ai_model failed: {type(e).__name__}: {e}")
        ai_model = None

    # 15分足の確定直後は取引所側の反映遅れがあるため、通常は「各15分の10分以降」に実行する
    # ...以下、BTCデータの取得へ続く

    # ただし /run?force=1 のときはこの待機をスキップする
    if (not force) and ((now_jst.minute % 15) < 10):
        print(f"[RUN] skip (waiting window) minute={now_jst.minute}")
        return "Waiting..."

    exchange = build_exchange()


    btc_mode = "Range"
    btc_1h_change = np.nan
    median_sigma = np.nan
    btc_ret = np.nan
    btc_vol = np.nan
    btc_ok = False


    try:
        btc_ohlcv = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe="15m", limit=60)
        if not btc_ohlcv or len(btc_ohlcv) < MIN_BARS:
            raise ValueError(f"BTC ohlcv bars不足: {0 if not btc_ohlcv else len(btc_ohlcv)}")

        btc_df = pd.DataFrame(btc_ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
        btc_df["Pct_Change"] = btc_df["Close"].pct_change(fill_method=None)
        btc_df["Dynamic_Sigma"] = btc_df["Pct_Change"].rolling(20).std()


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

    # BTCの地合いで片側を禁止するフィルタ（デフォルトON）
    # BTC_SIDE_FILTER=0 にすると Up/Down でも両建て（LONG/SHORT）を許可する
    BTC_SIDE_FILTER = (os.environ.get("BTC_SIDE_FILTER", "1").strip() == "1")

    if BTC_SIDE_FILTER:
        ALLOW_LONG = (btc_mode != "Down")
        ALLOW_SHORT = (btc_mode != "Up")
    else:
        ALLOW_LONG = True
        ALLOW_SHORT = True


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
            df["Dynamic_Sigma"] = df["Pct_Change"].rolling(20).std()
            
            df["MA20"] = df["Close"].rolling(20).mean()
            df["Upper2"] = df["MA20"] + (2 * df["Close"] * df["Dynamic_Sigma"])
            df["Lower2"] = df["MA20"] - (2 * df["Close"] * df["Dynamic_Sigma"])
            
            # MA20==0 は計算不能なので「0固定」せず NaN にする
            df["BandWidth"] = np.where(
                df["MA20"].abs() > 1e-12,
                (df["Upper2"] - df["Lower2"]) / df["MA20"],
                np.nan,
            )

            
            df["BW_Change"] = df["BandWidth"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            df["RSI"] = calculate_rsi(df["Close"]).replace([np.inf, -np.inf], np.nan)
            
            v = pd.to_numeric(df["Volume"], errors="coerce").replace(0, np.nan)
            df["Vol_Change"] = v.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


            df["Drop_Score"] = df["Pct_Change"].apply(lambda x: abs(x) if x < 0 else 0) / df["Dynamic_Sigma"]
            df["Rise_Score"] = df["Pct_Change"].apply(lambda x: abs(x) if x > 0 else 0) / df["Dynamic_Sigma"]

            df["Vol_MA20"] = pd.to_numeric(df["Volume"], errors="coerce").rolling(20).mean()

            row = df.iloc[-2]
            
            # 欠損・非finiteがある行は「補完せず」候補化しない（continue）
            required_cols = [
                "Dynamic_Sigma",
                "BandWidth",
                "BW_Change",
                "RSI",
                "Vol_Change",
                "Pct_Change",
            ]
            
            ok_row = True
            for c in required_cols:
                val = row.get(c, np.nan)
                if pd.isna(val):
                    ok_row = False
                    break
                try:
                    fv = float(val)
                    if not np.isfinite(fv):
                        ok_row = False
                        break
                except Exception:
                    ok_row = False
                    break
            
            # Dynamic_Sigma は正（>0）である必要
            if ok_row:
                try:
                    if float(row["Dynamic_Sigma"]) <= 0.0:
                        ok_row = False
                except Exception:
                    ok_row = False
            
            if not ok_row:
                continue



            
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


            # ===== 欠損・非finiteなら候補スキップ（補完しない）=====
            required_cols = [
                "Dynamic_Sigma",
                "BandWidth",
                "BW_Change",
                "RSI",
                "Vol_Change",
                "Pct_Change",
            ]

            ok_row = True
            for c in required_cols:
                val = row.get(c, np.nan)
                if pd.isna(val):
                    ok_row = False
                    break
                try:
                    fv = float(val)
                    if not np.isfinite(fv):
                        ok_row = False
                        break
                except Exception:
                    ok_row = False
                    break

            # Dynamic_Sigma は正（>0）である必要
            if ok_row:
                try:
                    if float(row["Dynamic_Sigma"]) <= 0.0:
                        ok_row = False
                except Exception:
                    ok_row = False

            if not ok_row:
                continue
            # =====================================================
            
            # ==========================================================
            # AIで「順張り/逆張り（LONG/SHORT）」を選ぶ（最小変更・泥沼回避）
            # - base_side: 現行ロジックが出した side
            # - flip_side: 反対 side
            # - 勝てそうな方（Win確率が高い方）を採用
            # - AIがbypassしたら従来通り（fail-open/closedに従う）
            # ==========================================================
            ai_score = None

            # 動的AI_TH（既存仕様）
            ai_th_used = AI_TH
            if DYNAMIC_AI_TH:
                ai_th_used = compute_dynamic_ai_th(AI_TH, btc_mode, median_sigma, btc_ok, BTC_CALM)

            # 銘柄別モデル（既存仕様）
            model_for_sym = ai_model
            sym_code = symbol.replace("/USDT", "")
            if ENABLE_MULTI_MODEL:
                model_for_sym, _ver, _src = get_ai_model_for_symbol(sym_code)

            # AIでside選択を有効化（envでOFF可）
            AI_SIDE_SELECT = (os.environ.get("AI_SIDE_SELECT", "1").strip() == "1")

            base_side = "LONG" if is_buy else "SHORT"
            flip_side = "SHORT" if base_side == "LONG" else "LONG"
            flip_allowed = (ALLOW_SHORT if flip_side == "SHORT" else ALLOW_LONG)

            # 学習側と整合：Sideに応じて Rise/Drop のどちらか一方だけに寄せる
            sig_score = float(max(float(row["Rise_Score"]), float(row["Drop_Score"])))

            def _make_feats(side: str) -> pd.DataFrame:
                rise = sig_score if side == "LONG" else 0.0
                drop = sig_score if side == "SHORT" else 0.0
                return pd.DataFrame([{
                    "Sigma": float(row["Dynamic_Sigma"]),
                    "BandWidth": float(row["BandWidth"]),
                    "BW_Change": float(row["BW_Change"]),
                    "RSI": float(row["RSI"]),
                    "Vol_Change": float(row["Vol_Change"]),
                    "Rise_Score": float(rise),
                    "Drop_Score": float(drop),
                    "BTC_Ret": float(btc_ret),
                    "BTC_Vol": float(btc_vol),
                }])


            def _score_side(side: str) -> Tuple[Optional[float], bool, Dict[str, Any]]:
                proba_x, bypass_x, dbg_x = safe_predict_proba(model_for_sym, _make_feats(side))
            
                # bypass なら予測しない（固定値判定もしない）
                if bypass_x:
                    return None, True, (dbg_x or {})
            
                # 念のため：None は絶対にパースしない（事故率低下）
                if proba_x is None:
                    d = (dbg_x or {})
                    if isinstance(d, dict):
                        d["error"] = "no_proba"
                    return None, True, d if isinstance(d, dict) else {"error": "no_proba"}
            
                try:
                    s = float(proba_x[0][1])
                except Exception:
                    return None, True, {"error": "proba_parse_failed"}
            
                if not np.isfinite(s):
                    return None, True, {"error": "non_finite_score"}
            
                return float(s), False, (dbg_x or {})


            # base 採点
            score_b, bypass_b, dbg_b = _score_side(base_side)

            # Crash/Trend 判定：荒れ相場（強トレンド/高ボラ）では flip（逆張り側）を禁止する
            crash_forbid_flip = False
            try:
                crash_forbid_flip = (float(sig_score) >= float(CRASH_SIGMA)) or (float(row["Dynamic_Sigma"]) >= float(CRASH_VOLSIGMA))
            except Exception:
                crash_forbid_flip = False

            # flip 採点（許可 & 有効 & 荒れ相場でない時だけ）
            score_f = None
            bypass_f = True
            dbg_f: Dict[str, Any] = {"skipped": True, "reason": "flip_disabled_or_not_allowed"}
            if AI_SIDE_SELECT and bool(flip_allowed) and (not crash_forbid_flip):
                score_f, bypass_f, dbg_f = _score_side(flip_side)
            else:
                if crash_forbid_flip:
                    dbg_f = {"skipped": True, "reason": "crash_forbid_flip"}
                else:
                    dbg_f = {"skipped": True, "reason": ("flip_not_allowed" if (not flip_allowed) else "AI_SIDE_SELECT=0")}

            # 採用判定：採点できた方があれば「高い方」。
            # ただし flip を採用するには、Baseより AI_SIDE_MARGIN 以上よいことを要求（ノイズ反転を減らす）
            chosen_side = base_side
            chosen_score = None

            if (not bypass_b) and (score_b is not None):
                chosen_side = base_side
                chosen_score = float(score_b)

            if (not bypass_f) and (score_f is not None):
                if chosen_score is None:
                    chosen_side = flip_side
                    chosen_score = float(score_f)
                else:
                    if float(score_f) > (float(chosen_score) + float(AI_SIDE_MARGIN)):
                        chosen_side = flip_side
                        chosen_score = float(score_f)

            flipped = (chosen_side != base_side)

            # dbg は item["ai_debug"] に入れる前提
            dbg = {
                "ai_side_select": bool(AI_SIDE_SELECT),
                "ai_side_margin": float(AI_SIDE_MARGIN),
                "crash_forbid_flip": bool(crash_forbid_flip),
                "crash_sig_score": float(sig_score),
                "crash_volsigma": float(row["Dynamic_Sigma"]),
                "crash_sig_th": float(CRASH_SIGMA),
                "crash_vol_th": float(CRASH_VOLSIGMA),
                "base_side": base_side,
                "flip_side": flip_side,
                "flip_allowed": bool(flip_allowed),
                "score_base": score_b,
                "score_flip": score_f,
                "bypassed_base": bool(bypass_b),
                "bypassed_flip": bool(bypass_f),
                "dbg_base": dbg_b,
                "dbg_flip": dbg_f,
                "chosen_side": chosen_side,
                "flipped": bool(flipped),
                "chosen_score": chosen_score,
                "ai_th_used": float(ai_th_used),
            }

            # side 確定（以降のTP/SL・Sheet・Discordがこのsideになる）
            if flipped:
                if chosen_side == "LONG":
                    is_buy = True
                    is_sell = False
                    signal_type = "LONG"
                else:
                    is_buy = False
                    is_sell = True
                    signal_type = "SHORT"


            # bypassed / ai_score を確定
            if (chosen_score is not None) and np.isfinite(float(chosen_score)):
                ai_score = float(chosen_score)
                bypassed = False
            else:
                ai_score = None
                bypassed = True

            # ai_pass 判定（方針：bypass時は「予測しない」＝通知しない）
            if bypassed:
                ai_pass = False
                print(f"[AI] bypassed -> SKIP (no alert) in logic_main sym={sym_code}: {dbg}")
            else:
                ai_pass = (float(ai_score) >= float(ai_th_used))

            # --- AI確率ログ（learn_log分析用）---
            # 方針:
            # - base/flip は _score_side() の結果（score_b / score_f）を使う
            # - used は最終採用の chosen_score（= ai_score と同義）
            # - 欠損やbypassは「補完せず」空欄 "" のまま残す
            ai_proba_base_val = ""
            ai_proba_flip_val = ""
            ai_proba_used_val = ""
            ai_margin_val = ""
            
            try:
                if (not bool(bypass_b)) and (score_b is not None):
                    sb = float(score_b)
                    if np.isfinite(sb):
                        ai_proba_base_val = sb
            except Exception:
                ai_proba_base_val = ""
            
            try:
                if (not bool(bypass_f)) and (score_f is not None):
                    sf = float(score_f)
                    if np.isfinite(sf):
                        ai_proba_flip_val = sf
            except Exception:
                ai_proba_flip_val = ""
            
            try:
                if (not bool(bypassed)) and (chosen_score is not None):
                    cs = float(chosen_score)
                    if np.isfinite(cs):
                        ai_proba_used_val = cs
                elif (ai_score is not None):
                    asv = float(ai_score)
                    if np.isfinite(asv):
                        ai_proba_used_val = asv
            except Exception:
                ai_proba_used_val = ""
            
            # margin = used - other（base/flipが両方あり、採用sideが判別できる時だけ）
            try:
                if (ai_proba_used_val != "") and (ai_proba_base_val != "") and (ai_proba_flip_val != ""):
                    other = ""
                    if str(chosen_side) == str(base_side):
                        other = ai_proba_flip_val
                    elif str(chosen_side) == str(flip_side):
                        other = ai_proba_base_val
            
                    if other != "":
                        ai_margin_val = float(ai_proba_used_val) - float(other)
            except Exception:
                ai_margin_val = ""            

            
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
                "ai_debug": dbg,
                "chg_pct": chg_pct_val,
                "vol_ratio": vol_ratio_val,

                # ★追加（learn_log分析用）
                "ai_proba_base": ai_proba_base_val,
                "ai_proba_flip": ai_proba_flip_val,
                "ai_proba_used": ai_proba_used_val,
                "ai_margin": ai_margin_val,
            
                # --- learn_log に書き戻したい値は item に持たせる（row参照を消す） ---
                "BandWidth": float(row["BandWidth"]),
                "BW_Change": float(row["BW_Change"]),
                "Vol_Change": float(row["Vol_Change"]),
                # market_ai_* は今このスニペット内で row に無いので、空で持たせる（将来拡張用）
                "market_ai_score": "",
                "market_ai_pass": "",
                "market_ai_debug": "",
            }
            
            pending_candidates.append(item)


            # ==========================================================
            # Guardrails (env): table/Discord 採用の前に弾く
            # ※learn_log 側の候補ログは残す（後で検証・学習に使える）
            # ==========================================================

            guard_ok = True
            sym_u = str(item.get("symbol", "")).strip().upper()

            # 1) 銘柄ブロック
            if sym_u in SYMBOL_BLOCKLIST:
                guard_ok = False
                print(f"[GR] skip alert (blocklist) sym={sym_u}")

            # 2) RSI ガード
            if guard_ok:
                try:
                    rsi_raw = item.get("rsi", None)
                    
                    # rsi が欠損/空/NaN/inf なら「推測で埋めずに」不採用
                    if rsi_raw is None or str(rsi_raw).strip() == "":
                        guard_ok = False
                        print(f"[GR] skip alert sym={sym_u} rsi=missing (no-signal)")
                    else:
                        try:
                            rsi_v = float(rsi_raw)
                            if not np.isfinite(rsi_v):
                                guard_ok = False
                                print(f"[GR] skip alert sym={sym_u} rsi=invalid (no-signal)")
                            else:
                                if bool(item.get("is_sell", False)) and (rsi_v < RSI_SHORT_MIN):
                                    guard_ok = False
                                    print(f"[GR] skip alert sym={sym_u} side=SHORT rsi={rsi_v:.2f} < {RSI_SHORT_MIN}")
                                if bool(item.get("is_buy", False)) and (rsi_v > RSI_LONG_MAX):
                                    guard_ok = False
                                    print(f"[GR] skip alert sym={sym_u} side=LONG rsi={rsi_v:.2f} > {RSI_LONG_MAX}")
                        except Exception:
                            guard_ok = False
                            print(f"[GR] skip alert sym={sym_u} rsi=parse_failed (no-signal)")

                except Exception:
                    pass

            # 2.6) VolSigma（sigma）チェック
            #      方針:
            #        - sigma が取れない / 変換できない / NaN / inf の候補は「アラート採用しない」
            #        - VOLSIGMA_BAN_RANGE が設定されている場合だけ、禁止レンジ内も「アラート採用しない」
            if guard_ok:
                try:
                    vs_raw = item.get("sigma", None)
                    if vs_raw is None or str(vs_raw).strip() == "":
                        guard_ok = False
                        print(f"[GR] skip alert sym={sym_u} vols=missing (no-signal)")
                    else:
                        vs = float(vs_raw)

                        # NaN / inf を invalid 扱い
                        if (vs != vs) or (vs == float("inf")) or (vs == float("-inf")):
                            guard_ok = False
                            print(f"[GR] skip alert sym={sym_u} vols=invalid (no-signal)")
                        else:
                            # 禁止レンジが有効な時だけ適用
                            if (VOLSIGMA_BAN_MIN is not None) and (VOLSIGMA_BAN_MAX is not None):
                                if float(VOLSIGMA_BAN_MIN) <= vs <= float(VOLSIGMA_BAN_MAX):
                                    guard_ok = False
                                    print(f"[GR] skip alert sym={sym_u} vols={vs:.6f} in [{VOLSIGMA_BAN_MIN},{VOLSIGMA_BAN_MAX}]")
                except Exception:
                    guard_ok = False
                    print(f"[GR] skip alert sym={sym_u} vols=invalid (no-signal)")

            # 3) VolRatio 上限（慎重銘柄は別閾値）
            if guard_ok:
                vr = item.get("vol_ratio", "")
                if vr != "":
                    try:
                        vr_f = float(vr)
                        vr_max = VOLRATIO_MAX_CAUTION if (sym_u in SYMBOL_CAUTIONLIST) else VOLRATIO_MAX
                        if vr_f > vr_max:
                            guard_ok = False
                            print(f"[GR] skip alert sym={sym_u} volratio={vr_f:.3f} > {vr_max}")
                    except Exception:
                        pass

            # 4) AI% レンジ（下限/上限。慎重銘柄は上限を別閾値）
            if guard_ok:
                if item.get("ai_score", None) is not None:
                    try:
                        ai_pct = float(item["ai_score"]) * 100.0
                        ai_max = AI_PCT_MAX_CAUTION if (sym_u in SYMBOL_CAUTIONLIST) else AI_PCT_MAX

                        if ai_pct < AI_PCT_MIN:
                            guard_ok = False
                            print(f"[GR] skip alert sym={sym_u} ai_pct={ai_pct:.2f} < {AI_PCT_MIN}")

                        if guard_ok and ai_pct > ai_max:
                            guard_ok = False
                            print(f"[GR] skip alert sym={sym_u} ai_pct={ai_pct:.2f} > {ai_max}")
                    except Exception:
                        pass

            # 5) 慎重銘柄：AI bypass（fail-open等）では採用しない
            #    ※bypassed 変数はこのスコープにある前提（現状コード通り）
            if guard_ok:
                if (sym_u in SYMBOL_CAUTIONLIST) and bypassed:
                    guard_ok = False
                    print(f"[GR] skip alert (caution + ai_bypassed) sym={sym_u}")

            # 重要：通知判定は「AI Pass」を最終ゲートにする（VolSigmaはランキング用で残す）
            if guard_ok and ai_pass and BTC_CALM:

                try:
                    _sc_raw = item.get("score", None)
                    if _sc_raw is not None and str(_sc_raw).strip() != "":
                        _sc = float(_sc_raw)
                        if np.isfinite(_sc) and (_sc < float(ALERT_SIGMA)):
                            print(f"[DBG] score below alert_sigma but still add. score={_sc} alert_sigma={ALERT_SIGMA}")
                except Exception:
                    pass


                pending_alerts.append(item)




        except Exception as e:
            print(f"[ERR] {symbol} fetch/compute: {e}")
            continue

    def calc_tp_sl(item):
        tp_mult = 3.5
        sl_mult = 1.8
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

    # learn_log に既存の ai_debug 列がある場合だけ、そこへラベルを書き込む
    learn_headers, _, _ = get_headers_and_len(LEARN_SHEET_NAME)
    
    # 基本は「実シートのヘッダー」に合わせる（列ズレ防止）
    # もし取得できなければ EXPECTED を使う
    learn_fields = list(learn_headers) if (learn_headers and len(learn_headers) > 0) else list(EXPECTED_HEADERS_LEARN)
    
    # ai_debug 列の存在確認（大文字小文字ゆれ対応）
    ai_debug_field = ""
    for h in (learn_headers or []):
        hs = str(h).strip()
        if hs.lower() == "ai_debug":
            ai_debug_field = hs  # 実際の表記（大文字小文字）を保持
            break
    
    # learn_fields 側に ai_debug が無ければ末尾に追加（重複は絶対に作らない）
    if ai_debug_field:
        if all(str(x).strip().lower() != "ai_debug" for x in (learn_fields or [])):
            learn_fields.append(ai_debug_field)
    
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

        # side選択（順張り/逆張り）の結果が分かるようにNoteへ埋め込む
        base_side = str(item.get("type", ""))  # ここは最終typeが入る（上で確定済み）
        dbg = item.get("ai_debug", None)


        flip_flag = ""
        chosen_side = ""
        score_b = ""
        score_f = ""

        # ★追加：空欄理由（flipが空の理由 / bypass理由）を短くログ化
        flip_reason = ""
        byp_b = ""
        byp_f = ""
        byp_all = ""

        if isinstance(dbg, dict):
            flip_flag = str(dbg.get("flipped", ""))
            chosen_side = str(dbg.get("chosen_side", ""))

            score_b = dbg.get("score_base", "")
            score_f = dbg.get("score_flip", "")

            # bypass状況（空欄の最大要因）
            try:
                byp_b = "1" if bool(dbg.get("bypassed_base", False)) else "0"
            except Exception:
                byp_b = ""

            try:
                byp_f = "1" if bool(dbg.get("bypassed_flip", False)) else "0"
            except Exception:
                byp_f = ""

            # flip側のスキップ理由（crash_forbid_flip / flip_not_allowed / AI_SIDE_SELECT=0 等）
            try:
                _df = dbg.get("dbg_flip", {})
                if isinstance(_df, dict):
                    flip_reason = str(_df.get("reason", "")) or ""
                    if (not flip_reason) and bool(_df.get("skipped", False)):
                        flip_reason = "skipped"
            except Exception:
                flip_reason = ""

        # 最終bypass は dbg ではなく item から推定（dbg にキーが無いので誤ログ防止）
        # ai_score が None なら「最終採用スコア無し」＝bypassed扱い
        try:
            byp_all = "1" if (item.get("ai_score", None) is None) else "0"
        except Exception:
            byp_all = ""

        # note_str に短い理由コードを追記（挙動は変えない）
        note_str = (
            f"AI:{ai_disp} Pass:{item['ai_pass']} "
            f"SideChosen:{chosen_side} Flip:{flip_flag} "
            f"BaseP:{score_b} FlipP:{score_f} "
            f"BByp:{byp_b} FByp:{byp_f} Bypassed:{byp_all} FReason:{flip_reason} "
            f"Calm:{BTC_CALM} SigmaMed:{median_sigma:.4f} BTC_OK:{btc_ok} "
            f"BTC:{btc_mode} 1h:{btc_1h_change:.2%}"
        )




        ai_debug_label = derive_ai_debug(
            btc_mode=btc_mode,
            signal_type=str(item.get("type", "")),
            side=("LONG" if item["is_buy"] else "SHORT"),
        )

        # learn_log に「必ず書き戻す」列（BandWidth〜BTC_Vol、market_ai_*）を準備
        def _f_or_blank(x):
            try:
                if x is None:
                    return ""
                v = float(x)
                if not np.isfinite(v):
                    return ""
                return v
            except Exception:
                return ""

        bw_val = _f_or_blank(item.get("BandWidth", ""))
        bw_chg_val = _f_or_blank(item.get("BW_Change", ""))
        vol_chg_val = _f_or_blank(item.get("Vol_Change", ""))
        btc_ret_val = _f_or_blank(btc_ret)
        btc_vol_val = _f_or_blank(btc_vol)
        
        m_ai_score = item.get("market_ai_score", "")
        m_ai_pass = item.get("market_ai_pass", "")
        m_ai_debug = item.get("market_ai_debug", "")


        row_out = [
            dt_cell, sym, ("LONG" if item["is_buy"] else "SHORT"),
            float(item["close"]), float(item["score"]), float(item["sigma"]), "CANDIDATE",
            float(tp), float(sl), float(tp_pct), float(sl_pct),
            DEFAULT_LEV, 0, 0, bool(item["ai_pass"]), bool(BTC_CALM),
            VERSION, item["type"], 0, 0,
            ("STORM" if not BTC_CALM else "CALM"), btc_mode, float(btc_1h_change),
            float(item["rsi"]), note_str,

            # --- ここは「候補段階」なので基本空欄（後でDONE時に埋まる） ---
            "", "", "", "", "", "", "",

            # --- optional-but-we-want-to-always-write ---
            bw_val, bw_chg_val, vol_chg_val, btc_ret_val, btc_vol_val,
            m_ai_score, m_ai_pass, m_ai_debug,
        ]


        # ai_debug 列が存在する場合は「末尾append」ではなく、その列位置に代入する（列ズレ防止）
        if ai_debug_field:
            ai_debug_idx = -1
            for i, h in enumerate(learn_fields):
                if str(h).strip().lower() == "ai_debug":
                    ai_debug_idx = i
                    break

            if ai_debug_idx >= 0:
                # 先に必要な長さまで伸ばしてから代入（index error回避）
                if len(row_out) <= ai_debug_idx:
                    row_out = row_out + [""] * (ai_debug_idx + 1 - len(row_out))
                row_out[ai_debug_idx] = ai_debug_label

        # ai_proba_* / ai_margin 列が存在する場合は、その列位置に代入する（列ズレ防止）
        ai_proba_base_out = _f_or_blank(item.get("ai_proba_base", ""))
        ai_proba_flip_out = _f_or_blank(item.get("ai_proba_flip", ""))
        ai_proba_used_out = _f_or_blank(item.get("ai_proba_used", ""))
        ai_margin_out = _f_or_blank(item.get("ai_margin", ""))
        
        for _col, _val in [
            ("ai_proba_base", ai_proba_base_out),
            ("ai_proba_flip", ai_proba_flip_out),
            ("ai_proba_used", ai_proba_used_out),
            ("ai_margin", ai_margin_out),
        ]:
            _idx = -1
            for j, h in enumerate(learn_fields):
                if str(h).strip().lower() == _col:
                    _idx = j
                    break
            if _idx >= 0:
                if len(row_out) <= _idx:
                    row_out = row_out + [""] * (_idx + 1 - len(row_out))
                row_out[_idx] = _val

        
        # ★列ズレ防止：必ずヘッダー長に合わせる
        row_out = _pad_row_to_fields(row_out, learn_fields, fill="")

        candidate_rows.append(row_out)

        


        learn_keys.add(k)

    if candidate_rows:
        append_rows_to_sheet(LEARN_SHEET_NAME, candidate_rows, learn_fields)
    
    # ===== DEBUG: observation only (no behavior change) =====
    try:
        _pc = pending_candidates or []
        _pa = pending_alerts or []
    
        _cnt = {
            "pending_candidates": int(len(_pc)),
            "pending_alerts": int(len(_pa)),
            "ai_missing": 0,          # ai_score が None
            "ai_pass_true": 0,        # ai_pass が True
            "ai_pass_false": 0,       # ai_pass が False/未設定
            "score_ge_alert_sigma": 0,
            "score_lt_alert_sigma": 0,
            "volratio_over_max": 0,
        }
    
        for _it in _pc:
            if not isinstance(_it, dict):
                continue
    
            _ai = _it.get("ai_score", None)
            if _ai is None:
                _cnt["ai_missing"] += 1
    
            if bool(_it.get("ai_pass", False)):
                _cnt["ai_pass_true"] += 1
            else:
                _cnt["ai_pass_false"] += 1
    
            # score と ALERT_SIGMA（存在する時だけ観測）
            try:
                _sc = float(_it.get("score", float("nan")))
                if np.isfinite(_sc):
                    if _sc >= float(ALERT_SIGMA):
                        _cnt["score_ge_alert_sigma"] += 1
                    else:
                        _cnt["score_lt_alert_sigma"] += 1
            except Exception:
                pass
    
            # vol_ratio 上限（存在する時だけ観測）
            try:
                _vr = _it.get("vol_ratio", None)
                if _vr is not None and np.isfinite(float(_vr)):
                    _sym_u = str(_it.get("symbol", "")).split("/")[0].split("-")[0].strip().upper()
                    _vr_max = float(VOLRATIO_MAX_CAUTION) if (_sym_u in SYMBOL_CAUTIONLIST) else float(VOLRATIO_MAX)
                    if float(_vr) > float(_vr_max):
                        _cnt["volratio_over_max"] += 1
            except Exception:
                pass
    
        print(
            f"[DBG] obs_summary "
            f"candidates={_cnt['pending_candidates']} alerts={_cnt['pending_alerts']} "
            f"BTC_CALM={BTC_CALM} btc_ok={btc_ok} btc_mode={btc_mode} "
            f"AI_TH={AI_TH} ALERT_SIGMA={ALERT_SIGMA} VOLRATIO_MAX={VOLRATIO_MAX} "
            f"cnt={_cnt}"
        )
    except Exception as _e:
        print(f"[DBG] obs_summary_failed: {_e}")
    # ===== /DEBUG =====
    
    # いったんスコア順に並べる（ここではまだ上位制限しない）
    filtered = sorted(pending_alerts, key=lambda x: x["score"], reverse=True)


    count = 0
    alert_rows: List[List[Any]] = []

    for item in filtered:
        if count >= 3:
            break

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

        # --- Expected value filter (SAFE DEFAULT: OFF) ---
        if ENABLE_E_FILTER:
            if item.get("ai_score", None) is None:
                print(f"[EV] filtered sym={sym} reason=no_ai_score_for_ev")
                continue
        
            p_win = float(item["ai_score"])
            exp_ret = (p_win * float(tp_pct)) - ((1.0 - p_win) * float(sl_pct))
        
            if exp_ret < float(E_TH):
                print(f"[EV] filtered sym={sym} exp_ret={exp_ret:.4f} < E_TH={E_TH}")
                continue


        cp = float(item["close"])
        lev = DEFAULT_LEV
        
        if item["is_buy"]:
            icon = "🚀"
            d_str = "買い(LONG)"
        else:
            icon = "☄️"
            d_str = "売り(SHORT)"
        
        ai_disp = "N/A" if item["ai_score"] is None else f"{float(item['ai_score']):.1%}"
        
        # --- Hyperliquid入力用の%（レバ反映）を計算 ---
        # sym が "STX/USDT" や "STX-USDT" のようでも判定できるようにベース銘柄に正規化
        sym_base = str(sym).split("/")[0].split("-")[0].strip().upper()
        
        # 5倍銘柄は 5、それ以外は DEFAULT_LEV（通常10）
        hl_lev = 5.0 if sym_base in MAX_LEV_5X_SYMBOLS else float(DEFAULT_LEV)
        
        # tp_pct / sl_pct は「0.58」のように“%表記の数値”前提（あなたの既存表示に合わせる）
        hl_tp_pct = float(tp_pct) * hl_lev
        hl_sl_pct = float(sl_pct) * hl_lev
        
        msg = (
            f"{icon} **{d_str}** {icon}\n"
            f"{VERSION}\n"
            f"💎 {sym} ({item['type']})\n"
            f"📈 Score:{item['score']:.2f}σ  AI:{ai_disp}\n"
            f"🟦 BTC:{btc_mode} 1h:{btc_1h_change:.2%}  Calm:{BTC_CALM}  BTC_OK:{btc_ok}\n"
            f"💰 {cp:.4f}\n"
            f"🎯 TP: {tp:.4f} ({tp_pct:.2f}%) HL:{hl_tp_pct:.1f}%\n"
            f"🛑 SL: {sl:.4f} ({sl_pct:.2f}%) HL:{hl_sl_pct:.1f}%"
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

        row_tbl = [
            dt_cell, sym, "LONG" if item["is_buy"] else "SHORT",
            float(cp), float(item["score"]), float(item["sigma"]), "AI_PASS",
            float(tp), float(sl), float(tp_pct), float(sl_pct),
            lev, float(tp_pct * lev), float(sl_pct * lev),
            bool(BTC_CALM), True, VERSION, note_compact,
            item["chg_pct"], item["vol_ratio"], "MARKET",
        ]
        
        # ★列ズレ防止：TABLE_FIELDS の長さに合わせる（tableが37列でも事故らない）
        row_tbl = _pad_row_to_fields(row_tbl, TABLE_FIELDS, fill="")
        
        alert_rows.append(row_tbl)


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

    # self heal
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

    res = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{sheet_name}!A{start_row}:{HEADER_COL_END}{last_row}",
    ).execute()

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
        service.spreadsheets().values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"valueInputOption": "USER_ENTERED", "data": updates},
        ).execute()
        _invalidate_sheet_caches(sheet_name)

    return judged

def judge_main():
    total = 0
    total += judge_sheet(MAIN_SHEET_NAME)
    total += judge_sheet(LEARN_SHEET_NAME)
    return f"Judged {total} rows (table + learn_log)"

# ==========================================
# /preflight（4点のうち「権限不足」を即判定して返す）
# ==========================================
def preflight_check() -> Tuple[bool, str]:
    global ai_model
    try:
        # metadata read
        _ = _get_sheets_meta()

        # self-heal
        ok, msg = self_heal_prerequisites()
        if not ok:
            return False, f"self_heal_ng: {msg}"

        # lazy model load (avoid heavy cold start)
        if ai_model is None:
            reload_default_model()

        # write test to lock cell (and revert)
        if RUN_MUTEX_ENABLED:
            cur = _mutex_read()
            _mutex_write(f"preflight|{int(time.time())}|{_INSTANCE_ID}")
            time.sleep(0.1)
            _mutex_write(cur)

        return True, f"ok model_loaded={ai_model is not None} source={AI_MODEL_SOURCE_RUNTIME}"
    except HttpError as e:
        # ここで 403 が出るなら共有権限不足
        return False, f"google_api_http_error: {str(e)}"
    except Exception as e:
        return False, f"preflight_error: {e}"


# ==========================================
# ルーティング
# ==========================================
@app.route("/routes", methods=["GET"])
def routes_list():
    # 今このインスタンスに登録されているルートを一覧で返す（原因切り分け用）
    rules = []
    for r in app.url_map.iter_rules():
        rules.append({
            "rule": str(r),
            "methods": sorted([m for m in r.methods if m not in {"HEAD", "OPTIONS"}]),
            "endpoint": str(r.endpoint),
        })
    rules = sorted(rules, key=lambda x: x["rule"])
    return jsonify({"ok": True, "version": VERSION, "routes": rules}), 200


@app.route("/", methods=["GET"])
def home():
    return f"{VERSION} is Active", 200



@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200


@app.route("/health", methods=["GET", "HEAD"])
def health():
    return "ok", 200


@app.route("/preflight", methods=["GET"])
def preflight():
    ok, msg = preflight_check()
    return (f"OK: {msg}", 200) if ok else (f"NG: {msg}", 500)

@app.route("/e_report", methods=["GET"])
def e_report():
    # 最小の簡易レポート（運用確認用）
    # /e_report を叩いたときに 404 にならず、Sheets疎通と行数が確認できる
    try:
        ok, msg = preflight_check()
        if not ok:
            return jsonify({"ok": False, "error": msg, "version": VERSION}), 500

        table_rows = _get_row_count_cached(MAIN_SHEET_NAME)
        learn_rows = _get_row_count_cached(LEARN_SHEET_NAME)

        return jsonify({
            "ok": True,
            "version": VERSION,
            "model_loaded": (ai_model is not None),
            "model_version": os.environ.get("MODEL_VERSION", ""),
            "sheet": {
                "spreadsheet_id": SPREADSHEET_ID,
                "table": {"name": MAIN_SHEET_NAME, "rows": table_rows},
                "learn_log": {"name": LEARN_SHEET_NAME, "rows": learn_rows},
            }
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "version": VERSION}), 500

@app.route("/ai_health", methods=["GET"])
def ai_health():
    """
    - 通常: 設定状態と、MODEL_MAPの登録状況、キャッシュ状況だけ返す（軽い）
    - probe=1: MODEL_MAPに登録された銘柄について get_ai_model_for_symbol() を実際に呼び、
               「ロードできるか」を返す（初回はGCSダウンロードが走るので少し重い）
    """
    loaded = (ai_model is not None)

    # どの銘柄がMODEL_MAPに登録されているか（設定の見える化）
    m_map = _parse_kv_map(MODEL_MAP)
    v_map = _parse_kv_map(MODEL_VERSION_MAP)

    # キャッシュ状況（どのURIがキャッシュされているか）
    cache_uris = list(_model_cache.keys()) if isinstance(_model_cache, dict) else []

    resp: Dict[str, Any] = {
        "ok": True,
        "default_model": {
            "model_loaded": loaded,
            "model_type": (str(type(ai_model)) if loaded else None),
            "model_version": str(AI_MODEL_VERSION_RUNTIME),
            "source": str(AI_MODEL_SOURCE_RUNTIME),
        },
        "flags": {
            "multi_model_enabled": bool(ENABLE_MULTI_MODEL),
            "dynamic_ai_th_enabled": bool(DYNAMIC_AI_TH),
            "e_filter_enabled": bool(ENABLE_E_FILTER),
        },
        "multi_model_config": {
            "model_map_keys": sorted(list(m_map.keys())),
            "model_map_uris": m_map,          # どこを指しているかをそのまま返す
            "model_version_map": v_map,       # 任意
            "cache_ttl_sec": int(MODEL_CACHE_TTL_SEC),
            "cached_uris": cache_uris,
        },
    }

    probe = str(request.args.get("probe", "0")).strip() == "1"
    if probe and bool(ENABLE_MULTI_MODEL) and len(m_map) > 0:
        # 実際にロードして結果を返す（候補が出なくても確認できる）
        results: Dict[str, Any] = {}
        for sym in sorted(list(m_map.keys())):
            try:
                m, ver, src = get_ai_model_for_symbol(sym)
                results[sym] = {
                    "loaded": (m is not None),
                    "model_type": (str(type(m)) if m is not None else None),
                    "version": str(ver),
                    "source": str(src),
                }
            except Exception as e:
                results[sym] = {
                    "loaded": False,
                    "error": f"{type(e).__name__}: {e}",
                }
        resp["multi_model_probe"] = results
    else:
        resp["multi_model_probe"] = {
            "ran": False,
            "hint": "Use /ai_health?probe=1 to actually try loading models (may download from GCS).",
        }

    return jsonify(resp), 200



@app.route("/ai_smoke", methods=["GET"])
def ai_smoke():
    """
    - safe_predict_proba が bypass / None / 形不正でも 500 を出さない
    - score は取れれば返す。取れなければ None のまま返す
    """
    # 固定値を入れない：欠損入力→safe_predict_proba が bypass できることを確認する
    feats = pd.DataFrame([{}])


    proba = None
    bypassed = True
    dbg = {"info": "init"}

    try:
        proba, bypassed, dbg = safe_predict_proba(ai_model, feats)
    except Exception as e:
        # safe_predict_proba 自体が例外でも 200 で返す（落とさない）
        dbg = {"error": f"{type(e).__name__}: {e}"}
        proba = None
        bypassed = True

    score = None
    try:
        if proba is not None and len(proba) > 0 and len(proba[0]) > 1:
            s = float(proba[0][1])
            if np.isfinite(s):
                score = s
            else:
                bypassed = True
                if isinstance(dbg, dict):
                    dbg["score_error"] = "non_finite"
    except Exception as e:
        bypassed = True
        if isinstance(dbg, dict):
            dbg["score_error"] = f"{type(e).__name__}: {e}"

    return jsonify({
        "ok": True,
        "model_loaded": (ai_model is not None),
        "score": score,
        "bypassed": bool(bypassed),
        "debug": dbg,
        "model_version": os.environ.get("MODEL_VERSION", ""),
    }), 200


@app.route("/reload_model", methods=["POST", "GET"])
def reload_model():
    ok = reload_default_model()
    return jsonify({
        "ok": True,
        "reloaded": bool(ok),
        "model_loaded": (ai_model is not None),
        "model_version": str(AI_MODEL_VERSION_RUNTIME),
        "source": str(AI_MODEL_SOURCE_RUNTIME),
    }), 200

@app.route("/train", methods=["GET", "POST"])
def train_process():
    """
    learn_log から学習してモデルを作る（Cloud Scheduler から GET で叩ける）。

    URL例:
      /train?lookback=2500&min_samples=60&hot_reload=1&upload=1

    パラメータ:
      - lookback: learn_log の末尾から読む行数
      - min_samples: Win/Lose が埋まっている行の最低数
      - hot_reload: 1 なら、このインスタンスだけ即時有効化（環境変数は変わらない）
      - upload: 1 なら GCSへアップロード
      - version: 任意のモデルバージョン名（未指定なら vYYYYMMDD_HHMMSS）
    """
    global ai_model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME

    # 同時実行ガード（run/judge/train）
    if not _run_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Busy (run/judge/train already in progress).", "version": VERSION}), 200

    mutex_token = ""
    try:
        ok, msg = preflight_check()
        if not ok:
            err = f"Preflight NG: {msg}"
            print(f"[WARN] /train {err}")
            send_discord_message(f"[WARN] /train {err}")
            # Scheduler を失敗扱いにしないため 200
            return jsonify({"ok": False, "error": err, "version": VERSION}), 200

        okm, token = acquire_run_mutex()
        if not okm:
            return jsonify({"ok": False, "error": "Busy (distributed mutex).", "version": VERSION}), 200
        mutex_token = token

        lookback = int(float(request.args.get("lookback", "2500")))
        min_samples = int(float(request.args.get("min_samples", str(TRAIN_MIN_SAMPLES))))
        hot_reload = str(request.args.get("hot_reload", "1")).strip() == "1"
        upload = str(request.args.get("upload", "1")).strip() == "1"

        ver = str(request.args.get("version", "")).strip()
        if not ver:
            ver = datetime.now(JST).strftime("v%Y%m%d_%H%M%S")

        t0 = time.time()

        # 学習→（必要なら）GCS保存→（必要なら）このインスタンスに即反映
        result = train_and_export_model(
            lookback_rows=lookback,
            out_version=ver,
            hot_reload=hot_reload,
            min_samples=min_samples,
            upload=upload,
        )

        elapsed = time.time() - t0

        result["elapsed_sec"] = float(elapsed)
        result["upload_requested"] = bool(upload)
        return jsonify(result), 200

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /train exception: {err}")
        send_discord_message(f"[ERR] /train crashed: {err}")
        # Scheduler を失敗扱いにしないため 200
        return jsonify({"ok": False, "error": err, "version": VERSION}), 200

    finally:
        if mutex_token:
            try:
                release_run_mutex(mutex_token)
            except Exception as e:
                msg = f"[WARN] /train release_run_mutex failed: {type(e).__name__}: {e}"
                print(msg)
                try:
                    send_discord_message(msg)
                except Exception as e2:
                    print(f"[WARN] /train send_discord_message failed: {type(e2).__name__}: {e2}")



        try:
            _run_lock.release()
        except RuntimeError:
            # 万一 release 済み or acquire 前なら無視
            pass


@app.route("/run", methods=["GET", "POST"])
def run_process():
    if not _run_lock.acquire(blocking=False):
        return "Busy (run/judge already in progress).", 200

    mutex_token = ""
    try:
        ok, msg = preflight_check()
        if not ok:
            err = f"Preflight NG: {msg}"
            print(f"[WARN] /run {err}")
            send_discord_message(f"[WARN] /run {err}")
            # Scheduler を失敗扱いにしないため 200
            return f"NG: {err}", 200

        okm, token = acquire_run_mutex()
        if not okm:
            return "Busy (distributed mutex).", 200
        mutex_token = token

        force = str(request.args.get("force", "0")).strip() == "1"

        try:
            res_run = str(logic_main(force=force))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[ERR] /run crashed: {err}")
            send_discord_message(f"[ERR] /run crashed: {err}")
            return f"ERR: {err}", 200

        if AUTO_JUDGE_AFTER_RUN:
            if not ENABLE_JUDGE:
                return res_run + " / Judge disabled", 200
            try:
                res_j = str(judge_main())
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[ERR] auto-judge crashed: {err}")
                send_discord_message(f"[ERR] auto-judge crashed: {err}")
                return res_run + f" / Judge ERR: {err}", 200
            return res_run + " / " + res_j, 200

        return res_run, 200

    except Exception as e:
        # preflight_check / acquire_run_mutex / release_run_mutex 等で例外が出ても 500 を避ける
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /run outer exception: {err}")
        send_discord_message(f"[ERR] /run outer crashed: {err}")
        return f"ERR: {err}", 200

    finally:
        if mutex_token:
            try:
                release_run_mutex(mutex_token)
            except Exception as e:
                msg = f"[WARN] /run release_run_mutex failed: {type(e).__name__}: {e}"
                print(msg)
                try:
                    send_discord_message(msg)
                except Exception as e2:
                    print(f"[WARN] /run send_discord_message failed: {type(e2).__name__}: {e2}")


        try:
            _run_lock.release()
        except RuntimeError:
            pass



@app.route("/judge", methods=["GET", "POST"])
def judge_process():
    if not ENABLE_JUDGE:
        return "Judge is disabled (set ENABLE_JUDGE=1 to enable).", 403

    if not _run_lock.acquire(blocking=False):
        return "Busy (run/judge already in progress).", 200

    mutex_token = ""
    try:
        ok, msg = preflight_check()
        if not ok:
            err = f"Preflight NG: {msg}"
            print(f"[WARN] /judge {err}")
            send_discord_message(f"[WARN] /judge {err}")
            # Scheduler を失敗扱いにしないため 200
            return f"NG: {err}", 200

        okm, token = acquire_run_mutex()
        if not okm:
            return "Busy (distributed mutex).", 200
        mutex_token = token

        try:
            return str(judge_main()), 200
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[ERR] /judge crashed: {err}")
            send_discord_message(f"[ERR] /judge crashed: {err}")
            return f"ERR: {err}", 200

    except Exception as e:
        # preflight_check / acquire_run_mutex / release_run_mutex 等で例外が出ても 500 を避ける
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /judge outer exception: {err}")
        send_discord_message(f"[ERR] /judge outer crashed: {err}")
        return f"ERR: {err}", 200

    finally:
        if mutex_token:
            try:
                release_run_mutex(mutex_token)
            except Exception as e:
                msg = f"[WARN] /judge release_run_mutex failed: {type(e).__name__}: {e}"
                print(msg)
                try:
                    send_discord_message(msg)
                except Exception as e2:
                    print(f"[WARN] /judge send_discord_message failed: {type(e2).__name__}: {e2}")


        try:
            _run_lock.release()
        except RuntimeError:
            pass





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



















