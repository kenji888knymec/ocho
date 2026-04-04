import os
import time
import threading
import hashlib
import subprocess
import re
import json
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
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    SKLEARN_OK = True
except Exception as _e:
    LogisticRegression = None
    RandomForestClassifier = None
    HistGradientBoostingClassifier = None
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
daily_discord_webhook_url = os.environ.get("DAILY_DISCORD_WEBHOOK_URL", "").strip()
V2_NOTIFY_IGNORE_RUNTIME_DEDUP = str(
    os.environ.get("V2_NOTIFY_IGNORE_RUNTIME_DEDUP", "0")
).strip().lower() in ("1", "true", "yes", "on")

# Hyperliquid: 最大5倍銘柄（この銘柄だけ HL 表示を x5 にする）
# ※環境変数 MAX_LEV_5X_SYMBOLS が未設定（空）の場合は、このデフォルトセットを使う
HL_MAX5_SYMBOLS = {"STX", "XLM", "FET", "HBAR", "POL"}

SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1XwWkzijIwRlafg2zDgPHQ4tgjYModapFI3T_wbYS9_8")
MAIN_SHEET_NAME = os.environ.get("MAIN_SHEET_NAME", "table")
LEARN_SHEET_NAME = os.environ.get("LEARN_SHEET_NAME", "learn_log")

VERSION = "Ver7.7 ModelReloadFix (Code v3.5.5)"

SIGNAL_ENGINE = os.environ.get("SIGNAL_ENGINE", "v1").strip().lower()

ENABLE_LEGACY_REGIME_NOTIFY = str(
    os.environ.get("ENABLE_LEGACY_REGIME_NOTIFY", "0")
).strip().lower() in ("1", "true", "yes", "on")

V1_DISABLE = str(
    os.environ.get("V1_DISABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_LONG_FR_FAIL_OPEN = str(
    os.environ.get("V2_LONG_FR_FAIL_OPEN", "0")
).strip().lower() in ("1", "true", "yes", "on")

V2_FET_HTF_NEUTRAL_FAIL_OPEN = str(
    os.environ.get("V2_FET_HTF_NEUTRAL_FAIL_OPEN", "0")
).strip().lower() in ("1", "true", "yes", "on")

# --- Thresholds (env configurable) ---
CAND_SIGMA = float(os.environ.get("CAND_SIGMA", "1.2"))
ALERT_SIGMA = float(os.environ.get("ALERT_SIGMA", "2.0"))
AI_TH = float(os.environ.get("AI_TH", "0.55"))

# --- 方向別AI閾値（Phase1） ---

# LONG_AI_TH / SHORT_AI_TH / SHORT_AI_TH_UP 未設定時は AI_TH をそのまま使う
# LONG          : より厳しくするために使う
# SHORT         : 通常SHORTの独立閾値
# SHORT_AI_TH_UP: BTC_Mode=Up の SHORT だけに使う追加閾値

LONG_AI_TH = float(os.environ.get("LONG_AI_TH", str(AI_TH)))
SHORT_AI_TH = float(os.environ.get("SHORT_AI_TH", str(AI_TH)))
SHORT_AI_TH_UP = float(os.environ.get("SHORT_AI_TH_UP", str(SHORT_AI_TH)))


# --- LONG BTC_Upバイパス（Phase2） ---

# LONG_BYPASS_ON_BTC_UP=1 のときだけ有効（デフォルト=0で完全OFF）

LONG_BYPASS_ON_BTC_UP = (os.environ.get("LONG_BYPASS_ON_BTC_UP", "0").strip() == "1")


# --- SHORT BTC下落深度ガード ---

# SHORTを出すにはBTC_1h_changeがこの値以下であることを要求

# 例: -0.003 → BTCが1hで-0.3%以上下がった時だけSHORT許可

# 未設定(-inf)ならチェック無効（現行動作と同一）


# 1 のときだけ「Win確率」を反転して扱う（score_used = 1 - score_raw）
AI_PROBA_INVERT = (os.environ.get("AI_PROBA_INVERT", "0").strip() == "1")

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

# --- LONG BTC_Upバイパス（Phase2）: _env_float使用分（定義後に配置） ---
LONG_BYPASS_RSI_MAX = _env_float("LONG_BYPASS_RSI_MAX", 50.0)
LONG_BYPASS_SCORE_MIN = _env_float("LONG_BYPASS_SCORE_MIN", 1.8)

# --- SHORT BTC下落深度ガード: _env_float使用分（定義後に配置） ---
SHORT_BTC_1H_MIN = _env_float("SHORT_BTC_1H_MIN", float("-inf"))

# --- SHORT × BTC_Mode=Down × BandWidth ブロック ---
SHORT_DOWN_BW_BLOCK = _env_float("SHORT_DOWN_BW_BLOCK", float("inf"))

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

# R1: Score上限（例: 3.0 を設定するとScore>3.0をブロック）
SCORE_MAX = _env_float("SCORE_MAX", float("inf"))
# R3追加: SHORTのRSI上限（例: 70 を設定するとSHORT×RSI>70をブロック）
RSI_SHORT_MAX = _env_float("RSI_SHORT_MAX", float("inf"))


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
    f"SCORE_MAX={SCORE_MAX} RSI_SHORT_MAX={RSI_SHORT_MAX} "
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

# --- Side別AIモデル設定 ---
LONG_AI_ENABLE = (os.environ.get("LONG_AI_ENABLE", "1").strip() == "1")
SHORT_AI_ENABLE = (os.environ.get("SHORT_AI_ENABLE", "1").strip() == "1")

LONG_MODEL_GCS_URI = os.environ.get("LONG_MODEL_GCS_URI", "").strip()
SHORT_MODEL_GCS_URI = os.environ.get("SHORT_MODEL_GCS_URI", "").strip()

LONG_MODEL_VERSION = os.environ.get("LONG_MODEL_VERSION", "").strip()
SHORT_MODEL_VERSION = os.environ.get("SHORT_MODEL_VERSION", "").strip()

LONG_AI_PROBA_INVERT = (os.environ.get("LONG_AI_PROBA_INVERT", "1" if AI_PROBA_INVERT else "0").strip() == "1")
SHORT_AI_PROBA_INVERT = (os.environ.get("SHORT_AI_PROBA_INVERT", "1" if AI_PROBA_INVERT else "0").strip() == "1")

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

print(

    "[CFG] "

    f"LONG_AI_TH={LONG_AI_TH} "
    f"SHORT_AI_TH={SHORT_AI_TH} "
    f"SHORT_AI_TH_UP={SHORT_AI_TH_UP} "

    f"LONG_BYPASS_ON_BTC_UP={LONG_BYPASS_ON_BTC_UP} "

    f"LONG_BYPASS_RSI_MAX={LONG_BYPASS_RSI_MAX} "

    f"LONG_BYPASS_SCORE_MIN={LONG_BYPASS_SCORE_MIN} "

    f"SHORT_BTC_1H_MIN={SHORT_BTC_1H_MIN}"

)

# --- V2 AI 推論設定 ---
V2_LONG_AI_MIN = float(os.environ.get("V2_LONG_AI_MIN", str(LONG_AI_TH)))
V2_SHORT_AI_MIN = float(os.environ.get("V2_SHORT_AI_MIN", str(SHORT_AI_TH)))
V2_AI_FAIL_OPEN = (os.environ.get("V2_AI_FAIL_OPEN", "0").strip() == "1")
V2_AI_NOTE_DBG_MAXLEN = int(float(os.environ.get("V2_AI_NOTE_DBG_MAXLEN", "220")))


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
    # --- AI evidence columns (already exist in your sheet) ---
    "ai_debug", "ai_proba_base", "ai_proba_flip", "ai_proba_used", "ai_margin",
    "tag", "proba_raw", "proba_used", "invert_applied", "is_flip",
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
ai_model_long = None
ai_model_short = None

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


def send_daily_discord_message(text: str):
    """
    日次レポート専用Webhookに送る。
    DAILY_DISCORD_WEBHOOK_URL が未設定なら通常Webhookへフォールバックする。
    """
    webhook = daily_discord_webhook_url or discord_webhook_url
    if (not webhook) or ("ここに" in webhook):
        print("[DBG] daily discord webhook url empty or placeholder")
        return

    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)] or [text]
    for chunk in chunks:
        try:
            r = http.post(webhook, json={"content": chunk}, timeout=10)
            print(f"[DBG] daily discord status={r.status_code}")
            if r.status_code >= 300:
                print(f"[DBG] daily discord body={r.text[:200]}")
        except Exception as e:
            print(f"[ERR] daily discord webhook post: {e}")

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
    if V1_DISABLE and sheet_name in (LEARN_SHEET_NAME, MAIN_SHEET_NAME):
        print(f"[V1-DISABLED] skip append_rows_to_sheet sheet={sheet_name}")
        return

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


def _v2_notify_key(sig: Dict[str, Any]) -> str:
    """
    同じ候補を /run のたびに重複通知しないためのキー。
    V2シグナル本体の key（dt / tp / sl）も読む。
    """
    raw_dt = sig.get("time", "") or sig.get("Datetime_JST", "") or sig.get("dt", "") or ""
    dt_str = normalize_dt_str(str(raw_dt).strip()) if str(raw_dt).strip() else ""

    return "|".join([
        dt_str,
        str(sig.get("symbol", "") or ""),
        str(sig.get("direction", "") or ""),
        str(sig.get("entry_price", "") or sig.get("EntryPrice", "") or ""),
        str(sig.get("tp_price", "") or sig.get("TP_Price", "") or sig.get("tp", "") or sig.get("TP", "") or ""),
        str(sig.get("sl_price", "") or sig.get("SL_Price", "") or sig.get("sl", "") or sig.get("SL", "") or ""),
    ])


def _safe_num_str(v, digits: int = 4) -> str:
    try:
        if v is None or v == "":
            return ""
        x = float(v)
        if not np.isfinite(x):
            return ""
        return f"{x:.{digits}f}"
    except Exception:
        return ""


def _safe_pct_str(v, digits: int = 2) -> str:
    try:
        if v is None or v == "":
            return ""
        x = float(v)
        if not np.isfinite(x):
            return ""
        return f"{x:.{digits}f}"
    except Exception:
        return ""


def _get_v2_symbol_recent_stats(symbol: str, direction: str, lookback_n: int = 20) -> Dict[str, Any]:
    """
    v2_shadow_ai から同一銘柄・同方向の直近成績を取得する。
    通知本文に「直近20件勝率」を出すための補助。
    """
    try:
        df = get_v2_shadow_ai_data()
        if df is None or df.empty:
            return {"n": 0, "wins": 0, "wr": np.nan}

        work = df.copy()

        if "Symbol" not in work.columns or "Direction" not in work.columns:
            return {"n": 0, "wins": 0, "wr": np.nan}

        if "Datetime_JST" in work.columns:
            work["Datetime_JST"] = pd.to_datetime(
                work["Datetime_JST"].astype(str).str.strip(),
                errors="coerce",
            )
            work = work[work["Datetime_JST"].notna()].copy()
            work = work.sort_values("Datetime_JST", ascending=False)

        work["Symbol"] = work["Symbol"].astype(str).str.strip().str.upper()
        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()

        if "EvalStatus" in work.columns:
            work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
            work = work[work["EvalStatus"] == "DONE"].copy()

        sub = work[
            (work["Symbol"] == str(symbol).strip().upper()) &
            (work["Direction"] == str(direction).strip().upper())
        ].copy()

        if sub.empty:
            return {"n": 0, "wins": 0, "wr": np.nan}

        sub = sub.head(int(lookback_n))

        if "WinLose" not in sub.columns:
            return {"n": int(len(sub)), "wins": 0, "wr": np.nan}

        wl = sub["WinLose"].astype(str).str.strip().str.lower()
        wins = int(wl.isin(["win", "w", "1", "true"]).sum())
        n = int(len(sub))
        wr = float(wins / n) if n > 0 else np.nan

        return {"n": n, "wins": wins, "wr": wr}

    except Exception:
        return {"n": 0, "wins": 0, "wr": np.nan}


def _build_v2_discord_message(sig: Dict[str, Any]) -> str:
    """
    V2 の実運用向けDiscord通知。
    実行に必要な情報を先頭に出し、
    判断補助として AI 判定% と直近成績だけを追加する。
    V2シグナル本体の key（dt / tp / sl）も読む。
    """
    symbol = str(sig.get("symbol", "") or "").strip().upper()
    direction = str(sig.get("direction", "") or "").strip().upper()

    dt_raw = sig.get("time", "") or sig.get("Datetime_JST", "") or sig.get("dt", "") or ""
    if hasattr(dt_raw, "strftime"):
        dt_str = dt_raw.strftime("%Y-%m-%d %H:%M:%S")
    else:
        dt_str = str(dt_raw).strip()
        dt_str = normalize_dt_str(dt_str) if dt_str else ""

    entry = _safe_num_str(
        sig.get("entry_price", sig.get("EntryPrice", sig.get("entry", "")))
    )
    tp_price = _safe_num_str(
        sig.get("tp_price", sig.get("TP_Price", sig.get("tp", sig.get("TP", ""))))
    )
    sl_price = _safe_num_str(
        sig.get("sl_price", sig.get("SL_Price", sig.get("sl", sig.get("SL", ""))))
    )

    tp_pct = _safe_pct_str(sig.get("tp_pct", sig.get("TP_Pct", "")))
    sl_pct = _safe_pct_str(sig.get("sl_pct", sig.get("SL_Pct", "")))

    lev = sig.get("leverage", sig.get("Lev", sig.get("Leverage", DEFAULT_LEV)))
    try:
        lev_i = int(float(lev))
    except Exception:
        lev_i = int(DEFAULT_LEV)

    display_lev = 5 if symbol in MAX_LEV_5X_SYMBOLS else lev_i

    tp_lev_pct = _safe_pct_str(sig.get("tp_lev_pct", sig.get("TP_Lev%", "")))
    sl_lev_pct = _safe_pct_str(sig.get("sl_lev_pct", sig.get("SL_Lev%", "")))

    if tp_lev_pct == "" and tp_pct != "":
        try:
            tp_lev_pct = f"{float(tp_pct) * float(display_lev):.2f}"
        except Exception:
            tp_lev_pct = ""
    if sl_lev_pct == "" and sl_pct != "":
        try:
            sl_lev_pct = f"{float(sl_pct) * float(display_lev):.2f}"
        except Exception:
            sl_lev_pct = ""

    ai_prob = _safe_pct_str(sig.get("ai_prob_win", sig.get("AI_Prob_Win", "")), digits=1)
    recent = _get_v2_symbol_recent_stats(symbol, direction, lookback_n=20)

    if direction == "LONG":
        side_emoji = "🟢"
        tp_emoji = "🎯"
        sl_emoji = "🛑"
    else:
        side_emoji = "🔴"
        tp_emoji = "🎯"
        sl_emoji = "🛑"

    recent_line = ""
    if int(recent.get("n", 0)) >= 10 and np.isfinite(recent.get("wr", np.nan)):
        recent_line = (
            f"📊 直近{int(recent['n'])}件勝率: "
            f"{float(recent['wr']) * 100:.0f}% "
            f"（{int(recent['wins'])}/{int(recent['n'])}）"
        )
    elif int(recent.get("n", 0)) > 0:
        recent_line = f"📊 直近{int(recent['n'])}件: 参考不足"

    lines = [
        f"{side_emoji}【{symbol} {direction}】",
        f"💰 Entry: {entry}" if entry else "",
        f"{tp_emoji} TP: {tp_price}" if tp_price else "",
        f"{sl_emoji} SL: {sl_price}" if sl_price else "",
        f"📌 TP入力: {tp_pct}%（x{display_lev}で {tp_lev_pct}%）" if tp_pct else "",
        f"📌 SL入力: {sl_pct}%（x{display_lev}で {sl_lev_pct}%）" if sl_pct else "",
        f"🤖 AI判定: {ai_prob}%" if ai_prob else "",
        recent_line,
        f"🕒 {dt_str}" if dt_str else "",
    ]

    return "\n".join([x for x in lines if x])


def send_v2_live_discord_alerts(notify_candidates: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    V2 の notify_pass=1 候補を Discord に送る。
    戻り値:
      (sent_n, runtime_dedup_skipped_n)
    """
    global last_alert_records

    sent = 0
    runtime_dedup_skipped = 0
    now_ts = int(time.time())

    try:
        expire_before = now_ts - 86400
        last_alert_records = {
            k: v for k, v in last_alert_records.items()
            if int(v) >= expire_before
        }
    except Exception:
        pass

    for sig in notify_candidates:
        key = _v2_notify_key(sig)

        if (not V2_NOTIFY_IGNORE_RUNTIME_DEDUP) and key in last_alert_records:
            runtime_dedup_skipped += 1
            print(
                f"[V2-NOTIFY-SKIP] runtime_dedup "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')} "
                f"key={key}"
            )
            continue

        msg = _build_v2_discord_message(sig)
        if not msg:
            print(
                f"[V2-NOTIFY-SKIP] empty_message "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')}"
            )
            continue

        try:
            send_discord_message(msg)
            last_alert_records[key] = now_ts
            sent += 1
            print(
                f"[V2-NOTIFY-SENT] "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')} "
                f"key={key}"
            )
        except Exception as e:
            print(f"[WARN] send_v2_live_discord_alerts failed: {type(e).__name__}: {e}")

    return sent, runtime_dedup_skipped


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
                     since: Optional[int] = None, retries: Optional[int] = None) -> Optional[List[List[Any]]]:
    """ccxt fetch_ohlcv を安全に呼ぶ。/judge の NameError 回避のためグローバル定義。"""
    last_err = None
    sym = _resolve_okx_symbol(exchange, symbol)

    if retries is None:
        retries = int(globals().get("FETCH_RETRY", 2))
    sleep_sec = float(globals().get("FETCH_RETRY_SLEEP_SEC", 1.0))

    for k in range(retries + 1):
        try:
            if since is None:
                return exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            return exchange.fetch_ohlcv(sym, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            last_err = e
            emsg = str(e)

            if "does not have market symbol" in emsg:
                print(f"[WARN] fetch_ohlcv_safe failed: {symbol} (resolved={sym}) err={emsg}")
                raise RuntimeError(emsg)

            if k < retries:
                time.sleep(sleep_sec * (k + 1))
                continue
            break

    print(f"[WARN] fetch_ohlcv_safe failed: {symbol} (resolved={sym}) err={last_err}")
    return None


def _log_judge_15m_bar(bar_open_ts_ms: int, label: str = "") -> None:
    """
    /judge が参照している 15m足が「形成中か」を1行で出す。
    bar_open_ts_ms: 足の open 時刻（ms）
    """
    try:
        jst = timezone(timedelta(hours=9))
        now_jst = datetime.now(timezone.utc).astimezone(jst)

        bar_open_jst = datetime.fromtimestamp(bar_open_ts_ms / 1000.0, tz=timezone.utc).astimezone(jst)
        bar_close_jst = bar_open_jst + timedelta(minutes=15)

        forming = now_jst < bar_close_jst

        print(
            f"[JUDGE15M]{'[' + label + ']' if label else ''} "
            f"now_jst={now_jst.isoformat(timespec='seconds')} "
            f"bar_open_jst={bar_open_jst.isoformat(timespec='seconds')} "
            f"bar_close_jst={bar_close_jst.isoformat(timespec='seconds')} "
            f"forming={forming}"
        )
    except Exception as e:
        print(f"[WARN] _log_judge_15m_bar failed: {type(e).__name__}: {e}")

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
                debug["nan_columns"] = list(nan_cols)
                debug["nan_n_cols"] = int(len(nan_cols))
                debug["nan_cnt"] = nan_cnt
                debug["nan_filled"] = 0
                debug["action"] = "nan_input_bypass"
                debug["error"] = "input contains NaN; skip predict_proba"

                try:
                    print(
                        f"[AI-NAN-BYPASS] "
                        f"nan_cols={nan_cols} "
                        f"nan_cnt={nan_cnt} "
                        f"input_n_features={debug.get('input_n_features', -1)} "
                        f"expected_n_features={debug.get('expected_n_features', -1)}"
                    )
                except Exception:
                    pass

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
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "").strip()

AI_MODEL_VERSION_RUNTIME = MODEL_VERSION
AI_MODEL_SOURCE_RUNTIME = "none"

AI_MODEL_VERSION_RUNTIME_LONG = LONG_MODEL_VERSION
AI_MODEL_SOURCE_RUNTIME_LONG = "none"

AI_MODEL_VERSION_RUNTIME_SHORT = SHORT_MODEL_VERSION
AI_MODEL_SOURCE_RUNTIME_SHORT = "none"

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

# --- V2 side別学習設定 ---
TRAIN_V2_LOOKBACK_ROWS = int(float(os.environ.get("TRAIN_V2_LOOKBACK_ROWS", "20000")))
TRAIN_V2_LONG_MIN_SAMPLES = int(float(os.environ.get("TRAIN_V2_LONG_MIN_SAMPLES", "80")))
TRAIN_V2_SHORT_MIN_SAMPLES = int(float(os.environ.get("TRAIN_V2_SHORT_MIN_SAMPLES", "80")))

TRAIN_V2_LONG_GCS_PREFIX = os.environ.get("TRAIN_V2_LONG_GCS_PREFIX", "side_models/long")
TRAIN_V2_SHORT_GCS_PREFIX = os.environ.get("TRAIN_V2_SHORT_GCS_PREFIX", "side_models/short")

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

def _build_train_output_uri_for_side(side: str, version: str) -> str:
    bucket = _default_train_bucket()
    if not bucket:
        return ""

    s = str(side or "").strip().upper()
    v = ("" if version is None else str(version)).strip()
    if not v:
        ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        v = f"v2_{s.lower()}_{ts}"

    if s == "LONG":
        prefix = str(TRAIN_V2_LONG_GCS_PREFIX).strip().strip("/")
        obj = f"{prefix}/{v}/long_model.pkl"
        return f"gs://{bucket}/{obj}"

    if s == "SHORT":
        prefix = str(TRAIN_V2_SHORT_GCS_PREFIX).strip().strip("/")
        obj = f"{prefix}/{v}/short_model.pkl"
        return f"gs://{bucket}/{obj}"

    return ""

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
        # 修正後
        "feature_columns": [
            "Sigma", "BandWidth", "BW Change", "RSI", "Vol Change",
            "BTC Ret", "BTC Vol",
            "Score", "Is Long",
            "Long x BTC Ret", "Long x RSI"
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

    # 修正後
    score_raw = pd.to_numeric(df2["ScoreSigma"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    is_long_enc = is_long.astype(float)
    
    # 修正後（交互作用項追加）
    X = pd.DataFrame({
        "Sigma": sigma.astype(float),
        "BandWidth": bw.astype(float),
        "BW Change": bw_chg.astype(float),
        "RSI": rsi.astype(float),
        "Vol Change": vol_chg.astype(float),
        "BTC Ret": btc_ret.astype(float),
        "BTC Vol": btc_vol.astype(float),
        "Score": score_raw.astype(float),
        "Is Long": is_long_enc.astype(float),
        "Long x BTC Ret": (is_long_enc * btc_ret).astype(float),
        "Long x RSI": (is_long_enc * rsi).astype(float),
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
        
                try:
                    fn = getattr(ai_model, "feature_names_in_", None)
                    if fn is not None:
                        print(f"[AI] model_feature_names_in_={list(fn)}")
                except Exception:
                    pass
        
                return True


    # 2) ローカル fallback
    if os.path.exists(MODEL_LOCAL_PATH):
        m = _load_model_from_path(MODEL_LOCAL_PATH)
        if m is not None:
            ai_model = m
            AI_MODEL_VERSION_RUNTIME = MODEL_VERSION
            AI_MODEL_SOURCE_RUNTIME = "local"
            print(f"[AI] Model Loaded Successfully path={MODEL_LOCAL_PATH} ver={AI_MODEL_VERSION_RUNTIME}")
    
            try:
                fn = getattr(ai_model, "feature_names_in_", None)
                if fn is not None:
                    print(f"[AI] model_feature_names_in_={list(fn)}")
            except Exception:
                pass
    
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
    
    try:
        fn = getattr(m, "feature_names_in_", None)
        if fn is not None:
            print(f"[AI] model_feature_names_in_({sym})={list(fn)}")
    except Exception:
        pass
    
    return m, ver, "gcs"


def _get_side_model_uri(side: str) -> str:
    s = str(side or "").strip().upper()
    if s == "LONG":
        return LONG_MODEL_GCS_URI
    if s == "SHORT":
        return SHORT_MODEL_GCS_URI
    return ""

def _get_side_model_version(side: str) -> str:
    s = str(side or "").strip().upper()
    if s == "LONG":
        return LONG_MODEL_VERSION
    if s == "SHORT":
        return SHORT_MODEL_VERSION
    return ""

def _get_side_model_invert(side: str) -> bool:
    s = str(side or "").strip().upper()
    if s == "LONG":
        return bool(LONG_AI_PROBA_INVERT)
    if s == "SHORT":
        return bool(SHORT_AI_PROBA_INVERT)
    return bool(AI_PROBA_INVERT)

def _is_side_ai_enabled(side: str) -> bool:
    s = str(side or "").strip().upper()
    if s == "LONG":
        return bool(LONG_AI_ENABLE)
    if s == "SHORT":
        return bool(SHORT_AI_ENABLE)
    return True

def get_ai_model_for_side(side: str, sym_code: str):
    """
    優先順位:
      1) side専用モデル
      2) 既存の銘柄別モデル
      3) 既存の単一モデル
    """
    global ai_model_long, ai_model_short
    global AI_MODEL_VERSION_RUNTIME_LONG, AI_MODEL_SOURCE_RUNTIME_LONG
    global AI_MODEL_VERSION_RUNTIME_SHORT, AI_MODEL_SOURCE_RUNTIME_SHORT

    s = str(side or "").strip().upper()

    if not _is_side_ai_enabled(s):
        return None, "", "disabled"

    side_uri = _get_side_model_uri(s)
    side_ver = _get_side_model_version(s)

    if side_uri.startswith("gs://"):
        cached = _model_cache.get(side_uri)
        now_ts = time.time()

        if cached and (now_ts - float(cached.get("ts", 0.0)) <= float(MODEL_CACHE_TTL_SEC)):
            model = cached.get("model")
            ver = str(cached.get("version", side_ver))
            src = str(cached.get("source", side_uri))
            if s == "LONG":
                ai_model_long = model
                AI_MODEL_VERSION_RUNTIME_LONG = ver
                AI_MODEL_SOURCE_RUNTIME_LONG = src
            elif s == "SHORT":
                ai_model_short = model
                AI_MODEL_VERSION_RUNTIME_SHORT = ver
                AI_MODEL_SOURCE_RUNTIME_SHORT = src
            return model, ver, src

        tmp_path = _safe_tmp_path_for_uri(side_uri)
        ok_dl = _gcs_download_to(side_uri, tmp_path)
        if ok_dl and os.path.exists(tmp_path):
            model = _load_model_from_path(tmp_path)
            if model is not None:
                src = side_uri
                ver = side_ver or ""
                _model_cache[side_uri] = {
                    "model": model,
                    "ts": now_ts,
                    "version": ver,
                    "source": src,
                }
                if s == "LONG":
                    ai_model_long = model
                    AI_MODEL_VERSION_RUNTIME_LONG = ver
                    AI_MODEL_SOURCE_RUNTIME_LONG = src
                elif s == "SHORT":
                    ai_model_short = model
                    AI_MODEL_VERSION_RUNTIME_SHORT = ver
                    AI_MODEL_SOURCE_RUNTIME_SHORT = src
                print(f"[AI] Side-model loaded side={s} uri={side_uri} ver={ver}")
                try:
                    fn = getattr(model, "feature_names_in_", None)
                    if fn is not None:
                        print(f"[AI] model_feature_names_in_({s})={list(fn)}")
                except Exception:
                    pass
                return model, ver, src

    if ENABLE_MULTI_MODEL:
        try:
            model_for_sym, ver_sym, src_sym = get_ai_model_for_symbol(sym_code)
            if model_for_sym is not None:
                return model_for_sym, ver_sym, src_sym
        except Exception as e:
            print(f"[AI] get_ai_model_for_side symbol fallback failed side={s} sym={sym_code} err={type(e).__name__}: {e}")

    try:
        model = get_ai_model()
        return model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME
    except Exception as e:
        print(f"[AI] get_ai_model_for_side fallback failed side={s} err={type(e).__name__}: {e}")
        return None, "", "none"



# ==========================================
# ★起動時に「必要シート/ヘッダー」を自己修復
# ==========================================
def self_heal_prerequisites() -> Tuple[bool, str]:
    try:
        ok_lock = ensure_sheet_exists(RUN_MUTEX_SHEET, min_rows=50, min_cols=5)

        if V1_DISABLE:
            ok_all = bool(ok_lock)
            return ok_all, f"lock={ok_lock} learn=skipped table=skipped v1_disable={V1_DISABLE}"

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


# --- Pillar1: 多時間足 ---
V2_HTF_TIMEFRAME   = os.environ.get("V2_HTF_TIMEFRAME", "1h")
V2_HTF_LIMIT       = int(float(os.environ.get("V2_HTF_LIMIT", "60")))
V2_EMA_FAST        = int(float(os.environ.get("V2_EMA_FAST", "8")))
V2_EMA_SLOW        = int(float(os.environ.get("V2_EMA_SLOW", "21")))
V2_PILLAR1_MIN     = _env_float("V2_PILLAR1_MIN", 0.5)  # Pillar1 最低スコア（必須ゲート）

# --- Pillar2: ファンディングレート ---
V2_FR_LONG_TH      = _env_float("V2_FR_LONG_TH", -0.0001)
V2_FR_SHORT_TH     = _env_float("V2_FR_SHORT_TH", 0.0003)
V2_FR_WEIGHT       = _env_float("V2_FR_WEIGHT", 0.8)

# --- Pillar3: 出来高 ---
V2_VOL_CONFIRM     = _env_float("V2_VOL_CONFIRM", 1.2)

# --- レジーム転換検出 ---
V2_REGIME_CHECK_ENABLE     = str(os.environ.get("V2_REGIME_CHECK_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_REGIME_LTF_EMA_FAST     = int(float(os.environ.get("V2_REGIME_LTF_EMA_FAST", "5")))
V2_REGIME_LTF_EMA_SLOW     = int(float(os.environ.get("V2_REGIME_LTF_EMA_SLOW", "13")))

# --- BTC矛盾チェック閾値 ---
V2_BTC_CONFLICT_STRONG_TH  = _env_float("V2_BTC_CONFLICT_STRONG_TH", 0.8)

# --- TP倍率 ---
V2_TP_MULT_FIXED           = _env_float("V2_TP_MULT_FIXED", 2.0)
V2_TP_SCORE_DEPENDENT      = str(os.environ.get("V2_TP_SCORE_DEPENDENT", "0")).strip().lower() in ("1", "true", "yes", "on")

# --- 合計スコア ---
V2_MIN_SCORE                    = _env_float("V2_MIN_SCORE", 1.3)
V2_JUDGE_MAX_BARS               = int(float(os.environ.get("V2_JUDGE_MAX_BARS", "96")))
V2_JUDGE_LOOKBACK_ROWS          = int(float(os.environ.get("V2_JUDGE_LOOKBACK_ROWS", "2500")))
V2_JUDGE_CLOSE_WINDOW_SEC       = int(float(os.environ.get("V2_JUDGE_CLOSE_WINDOW_SEC", "240")))
V2_JUDGE_MAX_ROWS               = int(float(os.environ.get("V2_JUDGE_MAX_ROWS", "300")))
V2_JUDGE_BACKFILL_ENABLE        = str(os.environ.get("V2_JUDGE_BACKFILL_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_JUDGE_BACKFILL_MAX_ROWS      = int(float(os.environ.get("V2_JUDGE_BACKFILL_MAX_ROWS", "150")))
V2_JUDGE_BACKFILL_LOOKBACK_ROWS = int(float(os.environ.get("V2_JUDGE_BACKFILL_LOOKBACK_ROWS", "20000")))
V2_JUDGE_PRIORITY_BANDS         = [
    s.strip().upper()
    for s in str(os.environ.get("V2_JUDGE_PRIORITY_BANDS", "RULE_QS,RULE_QS_CAUTION,RULE_QS_NO_RANK")).split(",")
    if s.strip()
]
V2_DEBUG_REJECTS                = str(os.environ.get("V2_DEBUG_REJECTS", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_SYMBOL_BLOCKLIST        = [s.strip().upper() for s in str(os.environ.get("V2_LONG_SYMBOL_BLOCKLIST", "")).split(",") if s.strip()]
V2_SHORT_SYMBOL_BLOCKLIST       = [s.strip().upper() for s in str(os.environ.get("V2_SHORT_SYMBOL_BLOCKLIST", "")).split(",") if s.strip()]

# --- V2 SHORT Selection Pipeline ---
V2_SHORT_DEF_ENABLE             = str(os.environ.get("V2_SHORT_DEF_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_DEF_P1_MIN             = _env_float("V2_SHORT_DEF_P1_MIN", 1.5)
V2_SHORT_DEF_SCORE_MIN          = _env_float("V2_SHORT_DEF_SCORE_MIN", 2.5)
V2_SHORT_DEF_BTCSTR_MIN         = _env_float("V2_SHORT_DEF_BTCSTR_MIN", 0.3)
V2_SHORT_DEF_BTCSTR_MAX         = _env_float("V2_SHORT_DEF_BTCSTR_MAX", float("inf"))
V2_SHORT_DEF_RSI_MIN            = _env_float("V2_SHORT_DEF_RSI_MIN", 40.0)
V2_SHORT_DEF_RSI_MAX            = _env_float("V2_SHORT_DEF_RSI_MAX", 60.0)
V2_SHORT_DEF_RSI_MIN_DOWN       = _env_float("V2_SHORT_DEF_RSI_MIN_DOWN", 35.0)
V2_SHORT_DEF_RSI_MAX_DOWN       = _env_float("V2_SHORT_DEF_RSI_MAX_DOWN", 60.0)
V2_SHORT_DEF_REQUIRE_VOLCONF    = str(os.environ.get("V2_SHORT_DEF_REQUIRE_VOLCONF", "0")).strip().lower() in ("1", "true", "yes", "on")

V2_SHORT_RANK_ENABLE            = str(os.environ.get("V2_SHORT_RANK_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_RANK_TOP_N             = int(float(os.environ.get("V2_SHORT_RANK_TOP_N", "2")))
V2_SHORT_QS_MIN                 = _env_float("V2_SHORT_QS_MIN", float("-inf"))

V2_SHORT_RSI_SWEET_MIN          = _env_float("V2_SHORT_RSI_SWEET_MIN", 40.0)
V2_SHORT_RSI_SWEET_MAX          = _env_float("V2_SHORT_RSI_SWEET_MAX", 55.0)

# --- SHORTの帯別制御 ---
V2_SHORT_RSI_WEAK_MIN      = _env_float("V2_SHORT_RSI_WEAK_MIN", 35.0)
V2_SHORT_RSI_WEAK_MAX      = _env_float("V2_SHORT_RSI_WEAK_MAX", 40.0)

V2_SHORT_RSI_DEEP_MAX      = _env_float("V2_SHORT_RSI_DEEP_MAX", 35.0)
V2_SHORT_RSI_REBOUND_MIN   = _env_float("V2_SHORT_RSI_REBOUND_MIN", 45.0)
V2_SHORT_RSI_REBOUND_MAX   = _env_float("V2_SHORT_RSI_REBOUND_MAX", 50.0)

V2_SHORT_QS_W_P2_NEG       = _env_float("V2_SHORT_QS_W_P2_NEG", -0.20)
V2_SHORT_QS_W_P1_GE2       = _env_float("V2_SHORT_QS_W_P1_GE2", -0.15)
V2_SHORT_QS_W_P3_FLAT      = _env_float("V2_SHORT_QS_W_P3_FLAT", 0.15)

V2_SHORT_ALLOWED_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_SHORT_ALLOWED_HOURS", "10,11,12,15,16,17,18,19,20")).split(",")
    if str(x).strip()
}
V2_SHORT_STRONG_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_SHORT_STRONG_HOURS", "10,11,12,15,16,17,18,19,20")).split(",")
    if str(x).strip()
}
V2_SHORT_BUCKET_BTC_MODES = {
    s.strip().upper()
    for s in str(os.environ.get("V2_SHORT_BUCKET_BTC_MODES", "DOWN,RANGE")).split(",")
    if s.strip()
}
V2_SHORT_BUCKET_P1_MIN       = _env_float("V2_SHORT_BUCKET_P1_MIN", 1.3)
V2_SHORT_BUCKET_P1_MAX       = _env_float("V2_SHORT_BUCKET_P1_MAX", 3.0)
V2_SHORT_BUCKET_RSI_MIN      = _env_float("V2_SHORT_BUCKET_RSI_MIN", 30.0)
V2_SHORT_BUCKET_RSI_MAX      = _env_float("V2_SHORT_BUCKET_RSI_MAX", 50.0)
V2_SHORT_BUCKET_P2_MIN       = _env_float("V2_SHORT_BUCKET_P2_MIN", -0.30)
V2_SHORT_STRONG_P2_MIN       = _env_float("V2_SHORT_STRONG_P2_MIN", 0.0)
V2_SHORT_STRONG_VOLRATIO_MIN = _env_float("V2_SHORT_STRONG_VOLRATIO_MIN", 1.2)
V2_SHORT_GOOD_HOUR_AI_TH     = _env_float("V2_SHORT_GOOD_HOUR_AI_TH", 0.15)
V2_SHORT_STRONG_AI_TH        = _env_float("V2_SHORT_STRONG_AI_TH", 0.10)

# --- V2 SHORT Training Filters ---
V2_SHORT_TRAIN_REQUIRE_VOLCONF     = str(os.environ.get("V2_SHORT_TRAIN_REQUIRE_VOLCONF", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_TRAIN_EXCLUDE_BLOCKLIST   = str(os.environ.get("V2_SHORT_TRAIN_EXCLUDE_BLOCKLIST", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_TRAIN_MODE                = str(os.environ.get("V2_SHORT_TRAIN_MODE", "ANY")).strip().upper()

# --- V2 LONG Training / Guard ---
V2_LONG_DEF_REQUIRE_VOLCONF        = str(os.environ.get("V2_LONG_DEF_REQUIRE_VOLCONF", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_TRAIN_REQUIRE_VOLCONF      = str(os.environ.get("V2_LONG_TRAIN_REQUIRE_VOLCONF", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_TRAIN_EXCLUDE_BLOCKLIST    = str(os.environ.get("V2_LONG_TRAIN_EXCLUDE_BLOCKLIST", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_TRAIN_MODE                 = str(os.environ.get("V2_LONG_TRAIN_MODE", "UP")).strip().upper()

V2_LONG_ALLOWED_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_LONG_ALLOWED_HOURS", "11,12,13,14,19,20")).split(",")
    if str(x).strip()
}
V2_LONG_BUCKET_P1_MIN  = _env_float("V2_LONG_BUCKET_P1_MIN", 2.0)
V2_LONG_BUCKET_P1_MAX  = _env_float("V2_LONG_BUCKET_P1_MAX", 2.5)
V2_LONG_BUCKET_RSI_MIN = _env_float("V2_LONG_BUCKET_RSI_MIN", 40.0)
V2_LONG_BUCKET_RSI_MAX = _env_float("V2_LONG_BUCKET_RSI_MAX", 50.0)
V2_LONG_BUCKET_P2_MIN  = _env_float("V2_LONG_BUCKET_P2_MIN", 0.0)

V2_LONG_ALT_ENABLE = str(os.environ.get("V2_LONG_ALT_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_ALT_ALLOWED_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_LONG_ALT_ALLOWED_HOURS", "11,12,13,14")).split(",")
    if str(x).strip()
}
V2_LONG_ALT_BUCKET_P1_MIN  = _env_float("V2_LONG_ALT_BUCKET_P1_MIN", 1.3)
V2_LONG_ALT_BUCKET_P1_MAX  = _env_float("V2_LONG_ALT_BUCKET_P1_MAX", 3.0)
V2_LONG_ALT_BUCKET_RSI_MIN = _env_float("V2_LONG_ALT_BUCKET_RSI_MIN", 40.0)
V2_LONG_ALT_BUCKET_RSI_MAX = _env_float("V2_LONG_ALT_BUCKET_RSI_MAX", 55.0)
V2_LONG_ALT_BUCKET_P2_MIN  = _env_float("V2_LONG_ALT_BUCKET_P2_MIN", 0.0)
V2_LONG_ALT_AI_TH          = _env_float("V2_LONG_ALT_AI_TH", 0.15)

V2_SHORT_PAUSE_ALLOW_STRONG_BUCKET = str(
    os.environ.get("V2_SHORT_PAUSE_ALLOW_STRONG_BUCKET", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_LONG_STRONG_ENABLE = str(os.environ.get("V2_LONG_STRONG_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_STRONG_ALLOWED_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_LONG_STRONG_ALLOWED_HOURS", "11,12,13,14")).split(",")
    if str(x).strip()
}
V2_LONG_STRONG_BUCKET_BTC_MODES = {
    s.strip().upper()
    for s in str(os.environ.get("V2_LONG_STRONG_BUCKET_BTC_MODES", "DOWN,UP")).split(",")
    if s.strip()
}
V2_LONG_STRONG_BUCKET_P1_MIN  = _env_float("V2_LONG_STRONG_BUCKET_P1_MIN", 2.0)
V2_LONG_STRONG_BUCKET_P1_MAX  = _env_float("V2_LONG_STRONG_BUCKET_P1_MAX", 2.5)
V2_LONG_STRONG_BUCKET_RSI_MIN = _env_float("V2_LONG_STRONG_BUCKET_RSI_MIN", 40.0)
V2_LONG_STRONG_BUCKET_RSI_MAX = _env_float("V2_LONG_STRONG_BUCKET_RSI_MAX", 60.0)
V2_LONG_STRONG_BUCKET_P2_MIN  = _env_float("V2_LONG_STRONG_BUCKET_P2_MIN", 0.0)
V2_LONG_STRONG_AI_TH          = _env_float("V2_LONG_STRONG_AI_TH", 0.15)

V2_SHORT_PAUSE_ENABLE = str(os.environ.get("V2_SHORT_PAUSE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_PAUSE_LOOKBACK_HOURS = int(float(os.environ.get("V2_SHORT_PAUSE_LOOKBACK_HOURS", "24")))
V2_SHORT_PAUSE_MIN_DONE = int(float(os.environ.get("V2_SHORT_PAUSE_MIN_DONE", "8")))
V2_SHORT_PAUSE_STOP_WINRATE = _env_float("V2_SHORT_PAUSE_STOP_WINRATE", 0.40)
V2_SHORT_PAUSE_STOP_AVGPNL = _env_float("V2_SHORT_PAUSE_STOP_AVGPNL", 0.0)
V2_SHORT_PAUSE_CACHE_TTL_SEC = int(float(os.environ.get("V2_SHORT_PAUSE_CACHE_TTL_SEC", "300")))

# --- V2 Trainer ---
V2_CLASSIFIER_TYPE                 = str(os.environ.get("V2_CLASSIFIER_TYPE", "HGB")).strip().upper()
# --- V2 SHORT Brake / Recovery ---
V2_SHORT_BRAKE_ENABLE          = str(os.environ.get("V2_SHORT_BRAKE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_BRAKE_LOOKBACK_SLOTS  = int(float(os.environ.get("V2_SHORT_BRAKE_LOOKBACK_SLOTS", "2")))
V2_SHORT_BRAKE_MIN_SLOTS       = int(float(os.environ.get("V2_SHORT_BRAKE_MIN_SLOTS", "2")))
V2_SHORT_BRAKE_MIN_DONE        = int(float(os.environ.get("V2_SHORT_BRAKE_MIN_DONE", "20")))
V2_SHORT_BRAKE_STOP_WINRATE    = _env_float("V2_SHORT_BRAKE_STOP_WINRATE", 25.0)
V2_SHORT_BRAKE_STOP_AVGPNL     = _env_float("V2_SHORT_BRAKE_STOP_AVGPNL", -0.20)
V2_SHORT_BRAKE_CAUTION_WINRATE = _env_float("V2_SHORT_BRAKE_CAUTION_WINRATE", 40.0)
V2_SHORT_BRAKE_CAUTION_AVGPNL  = _env_float("V2_SHORT_BRAKE_CAUTION_AVGPNL", 0.00)
V2_SHORT_BRAKE_RECOVER_WINRATE = _env_float("V2_SHORT_BRAKE_RECOVER_WINRATE", 55.0)
V2_SHORT_BRAKE_RECOVER_AVGPNL  = _env_float("V2_SHORT_BRAKE_RECOVER_AVGPNL", 0.05)
V2_SHORT_BRAKE_PROBE_TOP_N     = int(float(os.environ.get("V2_SHORT_BRAKE_PROBE_TOP_N", "1")))

# --- V2 LONG Brake / Recovery ---
V2_LONG_BRAKE_ENABLE          = str(os.environ.get("V2_LONG_BRAKE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_BRAKE_LOOKBACK_SLOTS  = int(float(os.environ.get("V2_LONG_BRAKE_LOOKBACK_SLOTS", "2")))
V2_LONG_BRAKE_MIN_SLOTS       = int(float(os.environ.get("V2_LONG_BRAKE_MIN_SLOTS", "2")))
V2_LONG_BRAKE_MIN_DONE        = int(float(os.environ.get("V2_LONG_BRAKE_MIN_DONE", "6")))
V2_LONG_BRAKE_STOP_WINRATE    = _env_float("V2_LONG_BRAKE_STOP_WINRATE", 35.0)
V2_LONG_BRAKE_STOP_AVGPNL     = _env_float("V2_LONG_BRAKE_STOP_AVGPNL", -0.10)
V2_LONG_BRAKE_CAUTION_WINRATE = _env_float("V2_LONG_BRAKE_CAUTION_WINRATE", 50.0)
V2_LONG_BRAKE_CAUTION_AVGPNL  = _env_float("V2_LONG_BRAKE_CAUTION_AVGPNL", 0.00)
V2_LONG_BRAKE_RECOVER_WINRATE = _env_float("V2_LONG_BRAKE_RECOVER_WINRATE", 50.0)
V2_LONG_BRAKE_RECOVER_AVGPNL  = _env_float("V2_LONG_BRAKE_RECOVER_AVGPNL", 0.00)
V2_LONG_BRAKE_PROBE_TOP_N     = int(float(os.environ.get("V2_LONG_BRAKE_PROBE_TOP_N", "1")))

# --- V2 LONG Selection Pipeline ---
V2_LONG_DEF_ENABLE         = str(os.environ.get("V2_LONG_DEF_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_DEF_P1_MIN         = _env_float("V2_LONG_DEF_P1_MIN", 2.0)
V2_LONG_DEF_RSI_MIN        = _env_float("V2_LONG_DEF_RSI_MIN", 40.0)
V2_LONG_DEF_SCORE_MIN      = _env_float("V2_LONG_DEF_SCORE_MIN", 1.5)
V2_LONG_DEF_RSI_MAX        = _env_float("V2_LONG_DEF_RSI_MAX", 60.0)
V2_LONG_BLOCK_RANGE        = str(os.environ.get("V2_LONG_BLOCK_RANGE", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_REGIME_CONFLICT_HARD_SPREAD = _env_float("V2_LONG_REGIME_CONFLICT_HARD_SPREAD", -0.0010)
V2_LONG_P1_RESCUE_ENABLE   = str(os.environ.get("V2_LONG_P1_RESCUE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_P1_RESCUE_MIN      = _env_float("V2_LONG_P1_RESCUE_MIN", 1.8)
V2_LONG_P1_RESCUE_RSI_MAX  = _env_float("V2_LONG_P1_RESCUE_RSI_MAX", 60.0)
V2_LONG_RANK_ENABLE        = str(os.environ.get("V2_LONG_RANK_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_RANK_TOP_N         = int(float(os.environ.get("V2_LONG_RANK_TOP_N", "2")))
V2_LONG_QS_MIN             = _env_float("V2_LONG_QS_MIN", float("-inf"))
V2_LONG_RSI_SWEET_MIN      = _env_float("V2_LONG_RSI_SWEET_MIN", 40.0)
V2_LONG_RSI_SWEET_MAX      = _env_float("V2_LONG_RSI_SWEET_MAX", 45.0)
V2_LONG_BTCSTR_SOFT_MAX    = _env_float("V2_LONG_BTCSTR_SOFT_MAX", 0.5)
V2_LONG_BTCSTR_HARD_MAX    = _env_float("V2_LONG_BTCSTR_HARD_MAX", 0.7)
V2_LONG_QS_W_P1            = _env_float("V2_LONG_QS_W_P1", 1.0)
V2_LONG_QS_W_RSI           = _env_float("V2_LONG_QS_W_RSI", 0.6)
V2_LONG_QS_W_P3            = _env_float("V2_LONG_QS_W_P3", 0.5)
V2_LONG_QS_W_BTC           = _env_float("V2_LONG_QS_W_BTC", 0.2)
V2_LONG_QS_W_OTHER         = _env_float("V2_LONG_QS_W_OTHER", 0.2)
V2_LONG_QS_W_BAD_SYMBOL    = _env_float("V2_LONG_QS_W_BAD_SYMBOL", -0.8)
# LONG mode bonus は一旦無効化
# 現在のデータでは Up を罰し、Range/Down を優遇する既定値が逆向きなので、まずはニュートラルに戻す
V2_LONG_QS_W_MODE          = _env_float("V2_LONG_QS_W_MODE", 0.0)
V2_LONG_MODE_UP_PENALTY    = _env_float("V2_LONG_MODE_UP_PENALTY", 0.0)
V2_LONG_MODE_RANGE_BONUS   = _env_float("V2_LONG_MODE_RANGE_BONUS", 0.0)
V2_LONG_MODE_DOWN_BONUS    = _env_float("V2_LONG_MODE_DOWN_BONUS", 0.0)

# --- LONG raw rescue ---
V2_LONG_RAW_RESCUE_ENABLE          = str(os.environ.get("V2_LONG_RAW_RESCUE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_RAW_RESCUE_ONLY_BTC_LONG   = str(os.environ.get("V2_LONG_RAW_RESCUE_ONLY_BTC_LONG", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_RAW_RESCUE_LTF_MIN         = _env_float("V2_LONG_RAW_RESCUE_LTF_MIN", 0.0)
V2_LONG_RAW_RESCUE_RSI_MAX         = _env_float("V2_LONG_RAW_RESCUE_RSI_MAX", 65.0)
V2_LONG_RAW_RESCUE_SHORT_HTF_MAX   = _env_float("V2_LONG_RAW_RESCUE_SHORT_HTF_MAX", 2.0)
V2_LONG_RAW_RESCUE_ALLOW_NEUTRAL   = str(os.environ.get("V2_LONG_RAW_RESCUE_ALLOW_NEUTRAL", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_RAW_RESCUE_ALLOW_SHORT     = str(os.environ.get("V2_LONG_RAW_RESCUE_ALLOW_SHORT", "1")).strip().lower() in ("1", "true", "yes", "on")

# --- LONG RANGE rescue strict filter ---
V2_LONG_RANGE_RESCUE_RSI_MAX       = _env_float("V2_LONG_RANGE_RESCUE_RSI_MAX", 43.0)
V2_LONG_RANGE_RESCUE_P2_MIN        = _env_float("V2_LONG_RANGE_RESCUE_P2_MIN", 0.05)
V2_LONG_RANGE_RESCUE_P3_MIN        = _env_float("V2_LONG_RANGE_RESCUE_P3_MIN", 0.00)
V2_LONG_RANGE_RESCUE_BLOCK_SYMBOLS = [s.strip().upper() for s in str(os.environ.get("V2_LONG_RANGE_RESCUE_BLOCK_SYMBOLS", "APT,SUI,STX,BONK")).split(",") if s.strip()]
# 通常の RANGE rescue では落ちるが、実績上まだ触る価値がある一部だけを細く救う
V2_LONG_RANGE_RESCUE_THIN_ENABLE        = str(os.environ.get("V2_LONG_RANGE_RESCUE_THIN_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_RANGE_RESCUE_THIN_RSI_MAX       = _env_float("V2_LONG_RANGE_RESCUE_THIN_RSI_MAX", 55.0)
V2_LONG_RANGE_RESCUE_THIN_P2_MIN        = _env_float("V2_LONG_RANGE_RESCUE_THIN_P2_MIN", 0.00)
V2_LONG_RANGE_RESCUE_THIN_P3_MIN        = _env_float("V2_LONG_RANGE_RESCUE_THIN_P3_MIN", 0.0001)
V2_LONG_RANGE_RESCUE_THIN_ALLOW_SYMBOLS = [s.strip().upper() for s in str(os.environ.get("V2_LONG_RANGE_RESCUE_THIN_ALLOW_SYMBOLS", "POL,HBAR,SOL")).split(",") if s.strip()]

# --- LONG notify qualification gate ---
V2_LONG_NOTIFY_GATE_ENABLE         = str(os.environ.get("V2_LONG_NOTIFY_GATE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_NOTIFY_GATE_BLOCK_SYMBOLS  = [s.strip().upper() for s in str(os.environ.get("V2_LONG_NOTIFY_GATE_BLOCK_SYMBOLS", "APT,SUI,STX,BONK")).split(",") if s.strip()]

V2_LONG_NOTIFY_GATE_RANGE_ENABLE   = str(os.environ.get("V2_LONG_NOTIFY_GATE_RANGE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_NOTIFY_GATE_RANGE_RSI_MAX  = _env_float("V2_LONG_NOTIFY_GATE_RANGE_RSI_MAX", 43.0)
V2_LONG_NOTIFY_GATE_RANGE_P2_MIN   = _env_float("V2_LONG_NOTIFY_GATE_RANGE_P2_MIN", 0.05)
V2_LONG_NOTIFY_GATE_RANGE_P3_MIN   = _env_float("V2_LONG_NOTIFY_GATE_RANGE_P3_MIN", 0.00)

V2_LONG_NOTIFY_GATE_UP_ENABLE      = str(os.environ.get("V2_LONG_NOTIFY_GATE_UP_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_NOTIFY_GATE_UP_RSI_MAX     = _env_float("V2_LONG_NOTIFY_GATE_UP_RSI_MAX", 48.0)
V2_LONG_NOTIFY_GATE_UP_P2_MIN      = _env_float("V2_LONG_NOTIFY_GATE_UP_P2_MIN", 0.15)
V2_LONG_NOTIFY_GATE_UP_P3_MIN      = _env_float("V2_LONG_NOTIFY_GATE_UP_P3_MIN", 0.00)

# --- SHORT raw rescue ---
V2_SHORT_RAW_RESCUE_ENABLE         = str(os.environ.get("V2_SHORT_RAW_RESCUE_ENABLE", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_RAW_RESCUE_ONLY_BTC_SHORT = str(os.environ.get("V2_SHORT_RAW_RESCUE_ONLY_BTC_SHORT", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_RAW_RESCUE_LTF_MIN        = _env_float("V2_SHORT_RAW_RESCUE_LTF_MIN", 0.0)
V2_SHORT_RAW_RESCUE_RSI_MIN        = _env_float("V2_SHORT_RAW_RESCUE_RSI_MIN", 35.0)
V2_SHORT_RAW_RESCUE_LONG_HTF_MAX   = _env_float("V2_SHORT_RAW_RESCUE_LONG_HTF_MAX", 2.0)
V2_SHORT_RAW_RESCUE_ALLOW_NEUTRAL  = str(os.environ.get("V2_SHORT_RAW_RESCUE_ALLOW_NEUTRAL", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_RAW_RESCUE_ALLOW_LONG     = str(os.environ.get("V2_SHORT_RAW_RESCUE_ALLOW_LONG", "1")).strip().lower() in ("1", "true", "yes", "on")

# --- LONG guard soften in UP ---
V2_LONG_DEF_REQUIRE_VOLCONF_UP     = str(os.environ.get("V2_LONG_DEF_REQUIRE_VOLCONF_UP", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_BAD_SYMBOLS        = [s.strip().upper() for s in str(os.environ.get("V2_LONG_BAD_SYMBOLS", "SUI,UNI,AAVE,STX,XLM")).split(",") if s.strip()]
V2_LONG_REJECT_MODE_DOWN   = str(os.environ.get("V2_LONG_REJECT_MODE_DOWN", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_DEF_RSI_HARD_MAX   = _env_float("V2_LONG_DEF_RSI_HARD_MAX", 55.0)
V2_LONG_DEF_P1_HARD_MAX    = _env_float("V2_LONG_DEF_P1_HARD_MAX", 2.5)

# --- LONG AI bad-regime veto / caution ---
V2_LONG_AI_BADREGIME_ENABLE           = str(os.environ.get("V2_LONG_AI_BADREGIME_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_AI_BADREGIME_HOURS            = os.environ.get("V2_LONG_AI_BADREGIME_HOURS", "23,1,2,4")
V2_LONG_AI_BADREGIME_P2_MAX           = _env_float("V2_LONG_AI_BADREGIME_P2_MAX", 0.0)
V2_LONG_AI_BADREGIME_P3_MAX           = _env_float("V2_LONG_AI_BADREGIME_P3_MAX", 0.0)
V2_LONG_AI_BADREGIME_STOP_HITS        = int(float(os.environ.get("V2_LONG_AI_BADREGIME_STOP_HITS", "4")))
V2_LONG_AI_BADREGIME_CAUTION_HITS     = int(float(os.environ.get("V2_LONG_AI_BADREGIME_CAUTION_HITS", "3")))
V2_LONG_AI_BADREGIME_CAUTION_PENALTY  = _env_float("V2_LONG_AI_BADREGIME_CAUTION_PENALTY", 0.08)

# --- LONG rescue notify / rank tuning ---
V2_LONG_RESCUE_NOTIFY_BLOCK_HOURS     = os.environ.get("V2_LONG_RESCUE_NOTIFY_BLOCK_HOURS", "0,1,2,3,4,5")
V2_LONG_RESCUE_NOTIFY_DOWN_TOP_N      = int(float(os.environ.get("V2_LONG_RESCUE_NOTIFY_DOWN_TOP_N", "3")))
V2_LONG_RANK_P2_WEAK_PENALTY          = _env_float("V2_LONG_RANK_P2_WEAK_PENALTY", 0.08)
V2_LONG_RANK_P3_WEAK_PENALTY          = _env_float("V2_LONG_RANK_P3_WEAK_PENALTY", 0.08)
V2_LONG_RANK_BAD_HOUR_PENALTY         = _env_float("V2_LONG_RANK_BAD_HOUR_PENALTY", 0.06)

# --- LONG rank blend (AI + rule composite) ---
V2_LONG_RANK_BLEND_ENABLE          = str(os.environ.get("V2_LONG_RANK_BLEND_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_RANK_W_AI                  = _env_float("V2_LONG_RANK_W_AI", 1.00)
V2_LONG_RANK_W_QS                  = _env_float("V2_LONG_RANK_W_QS", 0.35)
V2_LONG_RANK_W_P2                  = _env_float("V2_LONG_RANK_W_P2", 0.30)
V2_LONG_RANK_W_P3                  = _env_float("V2_LONG_RANK_W_P3", 0.20)
V2_LONG_RANK_W_RSI                 = _env_float("V2_LONG_RANK_W_RSI", 0.10)
V2_LONG_RANK_RAW_RESCUE_PENALTY    = _env_float("V2_LONG_RANK_RAW_RESCUE_PENALTY", 0.12)
V2_LONG_RANK_MODE_UP_PENALTY       = _env_float("V2_LONG_RANK_MODE_UP_PENALTY", 0.10)
V2_LONG_RANK_MODE_RANGE_PENALTY    = _env_float("V2_LONG_RANK_MODE_RANGE_PENALTY", 0.05)

# --- monitor / guard / retrain cadence ---
V2_MONITOR_1H_ENABLE                  = str(os.environ.get("V2_MONITOR_1H_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_MONITOR_1H_LOOKBACK_HOURS          = int(float(os.environ.get("V2_MONITOR_1H_LOOKBACK_HOURS", "1")))

V2_GUARDRAIL_ENABLE                   = str(os.environ.get("V2_GUARDRAIL_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_GUARDRAIL_LOOKBACK_HOURS           = int(float(os.environ.get("V2_GUARDRAIL_LOOKBACK_HOURS", "6")))
V2_GUARDRAIL_MIN_DONE                 = int(float(os.environ.get("V2_GUARDRAIL_MIN_DONE", "12")))
V2_GUARDRAIL_CAUTION_WR_GAP           = _env_float("V2_GUARDRAIL_CAUTION_WR_GAP", -0.03)
V2_GUARDRAIL_STOP_WR_GAP              = _env_float("V2_GUARDRAIL_STOP_WR_GAP", -0.05)
V2_GUARDRAIL_CAUTION_PNL_GAP          = _env_float("V2_GUARDRAIL_CAUTION_PNL_GAP", -0.05)
V2_GUARDRAIL_STOP_PNL_GAP             = _env_float("V2_GUARDRAIL_STOP_PNL_GAP", -0.10)
V2_GUARDRAIL_CAUTION_TOP_N            = int(float(os.environ.get("V2_GUARDRAIL_CAUTION_TOP_N", "1")))

V2_RETRAIN_REVIEW_ENABLE              = str(os.environ.get("V2_RETRAIN_REVIEW_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_RETRAIN_LOOKBACK_HOURS             = int(float(os.environ.get("V2_RETRAIN_LOOKBACK_HOURS", "24")))
V2_RETRAIN_MIN_DONE                   = int(float(os.environ.get("V2_RETRAIN_MIN_DONE", "200")))
V2_RETRAIN_TRIGGER_WR_GAP             = _env_float("V2_RETRAIN_TRIGGER_WR_GAP", -0.05)
V2_RETRAIN_TRIGGER_PNL_GAP            = _env_float("V2_RETRAIN_TRIGGER_PNL_GAP", -0.10)

# --- V2 Shadow 出力先シート ---
V2_SHADOW_SHEET            = os.environ.get("V2_SHADOW_SHEET", "v2_shadow_ai")

# --- V2 regime-aware notify policy ---
V2_REGIME_POLICY_ENABLE         = str(os.environ.get("V2_REGIME_POLICY_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_REGIME_SHORT_MODE            = str(os.environ.get("V2_REGIME_SHORT_MODE", "DOWN")).strip().upper()
V2_REGIME_LONG_MODE             = str(os.environ.get("V2_REGIME_LONG_MODE", "UP")).strip().upper()
V2_REGIME_LOOKBACK_HOURS        = int(float(os.environ.get("V2_REGIME_LOOKBACK_HOURS", "72")))
V2_REGIME_MIN_DONE              = int(float(os.environ.get("V2_REGIME_MIN_DONE", "20")))
V2_REGIME_MIN_UPLIFT_WINRATE    = _env_float("V2_REGIME_MIN_UPLIFT_WINRATE", 0.03)
V2_REGIME_MIN_UPLIFT_PNL        = _env_float("V2_REGIME_MIN_UPLIFT_PNL", 0.0)

V2_SHORT_REGIME_CONFLICT_BLOCK_ENABLE = str(
    os.environ.get("V2_SHORT_REGIME_CONFLICT_BLOCK_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

# --- V2 Shadow ヘッダー（旧列を流用しない） ---
V2_HEADERS = [
    "Datetime_JST", "Symbol", "Direction", "EntryPrice",
    # スコア
    "TotalScore", "P1_TrendScore", "P2_FundingScore", "P3_VolumeScore",
    # Pillar1 詳細
    "SymHTF_Dir", "SymHTF_Strength", "BTC_HTF_Dir", "BTC_HTF_Strength",
    "LTF_Aligned", "LTF_Reasons",
    # Pillar2 詳細
    "FundingRate", "FR_Available",
    # Pillar3 詳細
    "VolRatio", "VolConfirmed",
    # TP/SL（ATRベース）
    "TP_Price", "SL_Price", "TP_Pct", "SL_Pct", "ATR",
    # 分析用
    "RSI", "Hour_JST", "BTC_Mode_Compat",
    # AI用（将来の V2-AI / LONG-AI / SHORT-AI 用）
    "AI_Prob_Win", "AI_Pass", "AI_Band", "AI_Model_Version", "AI_Model_Type", "AI_Note",
    # 勝敗（judge が後から埋める）
    "EvalStatus", "ExitTime", "ExitPrice", "ExitReason", "PnL_Pct", "WinLose", "HoldMin",
    # メタ
    "Version", "Note",
]

print(f"[V2-CFG] HTF={V2_HTF_TIMEFRAME} "
      f"EMA={V2_EMA_FAST}/{V2_EMA_SLOW} P1_MIN={V2_PILLAR1_MIN} "
      f"MIN_SCORE={V2_MIN_SCORE} SHEET={V2_SHADOW_SHEET}")


# ==========================================
# 補助関数
# ==========================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ==========================================
# Pillar 1: 多時間足トレンド整合
# ==========================================

def assess_htf_trend(htf_df: pd.DataFrame) -> Dict[str, Any]:
    """1H足のEMAトレンド方向と強度。"""
    if len(htf_df) < V2_EMA_SLOW + 5:
        return {
            "direction": "NEUTRAL",
            "strength": 0.0,
            "ema_spread": 0.0,
            "ema_fast": np.nan,
            "ema_slow": np.nan,
            "close": np.nan,
            "slope": np.nan,
            "strength_raw": 0.0,
        }

    htf_df = htf_df.copy()
    htf_df["EMA_F"] = _ema(htf_df["Close"], V2_EMA_FAST)
    htf_df["EMA_S"] = _ema(htf_df["Close"], V2_EMA_SLOW)

    ema_f = float(htf_df["EMA_F"].iloc[-1])
    ema_s = float(htf_df["EMA_S"].iloc[-1])
    close = float(htf_df["Close"].iloc[-1])

    spread = (ema_f - ema_s) / ema_s if ema_s != 0 else 0.0
    strength_raw = abs(spread) / 0.005
    strength = min(strength_raw, 1.0)

    ema_gap_ratio = abs(ema_f - ema_s) / max(abs(close), 1e-9)
    close_vs_ema_s_ratio = abs(close - ema_s) / max(abs(close), 1e-9)

    # 差が小さい時は無理に方向を付けず NEUTRAL に寄せる
    if ema_gap_ratio < 0.0015 or close_vs_ema_s_ratio < 0.0015:
        direction = "NEUTRAL"
    elif ema_f > ema_s and close > ema_s:
        direction = "LONG"
    elif ema_f < ema_s and close < ema_s:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    slope = np.nan
    # EMAの傾き（減速検出）
    if len(htf_df) >= 4:
        slope = float(htf_df["EMA_F"].iloc[-1] - htf_df["EMA_F"].iloc[-3])
        if direction == "LONG" and slope < 0:
            strength *= 0.5
        elif direction == "SHORT" and slope > 0:
            strength *= 0.5

    return {
        "direction": direction,
        "strength": float(strength),
        "ema_spread": float(spread),
        "ema_fast": float(ema_f),
        "ema_slow": float(ema_s),
        "close": float(close),
        "slope": float(slope) if np.isfinite(slope) else np.nan,
        "strength_raw": float(strength_raw),
    }


def detect_btc_regime_conflict(exchange, htf_direction: str) -> Dict[str, Any]:
    """
    BTC 1H方向と 15m短期EMA の矛盾を方向別に返す。
    short_conflict: 1H=SHORT なのに 15m EMA5 > EMA13
    long_conflict:  1H=LONG なのに 15m EMA5 < EMA13
    変更点:
    - SHORT は従来どおり
    - LONG は「少しでも逆行」で即hard blockせず、
      spread が V2_LONG_REGIME_CONFLICT_HARD_SPREAD 以下のときだけ hard conflict にする
    - hard に満たない逆行は long_conflict_soft=True で返し、ログで追えるようにする
    """
    result = {
        "short_conflict": False,
        "long_conflict": False,
        "long_conflict_soft": False,
        "reason": "no_conflict",
        "ltf_ema_fast": np.nan,
        "ltf_ema_slow": np.nan,
        "spread": np.nan,
    }

    if not V2_REGIME_CHECK_ENABLE:
        result["reason"] = "disabled"
        return result

    try:
        btc_ltf = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe="15m", limit=30)
        if not btc_ltf or len(btc_ltf) < V2_REGIME_LTF_EMA_SLOW + 3:
            result["reason"] = "btc_ltf_insufficient"
            return result

        btc_ltf_df = pd.DataFrame(btc_ltf, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
        ema_fast = _ema(btc_ltf_df["Close"], V2_REGIME_LTF_EMA_FAST)
        ema_slow = _ema(btc_ltf_df["Close"], V2_REGIME_LTF_EMA_SLOW)

        ltf_ema_f = float(ema_fast.iloc[-2])
        ltf_ema_s = float(ema_slow.iloc[-2])

        result["ltf_ema_fast"] = ltf_ema_f
        result["ltf_ema_slow"] = ltf_ema_s

        if not (np.isfinite(ltf_ema_f) and np.isfinite(ltf_ema_s)):
            result["reason"] = "ema_not_finite"
            return result

        if htf_direction == "SHORT" and ltf_ema_f > ltf_ema_s:
            spread = (ltf_ema_f - ltf_ema_s) / ltf_ema_s if ltf_ema_s != 0 else 0.0
            result["spread"] = float(spread)
            result["short_conflict"] = True
            result["reason"] = f"short_conflict spread={spread:.6f}"
            return result

        if htf_direction == "LONG" and ltf_ema_f < ltf_ema_s:
            spread = (ltf_ema_f - ltf_ema_s) / ltf_ema_s if ltf_ema_s != 0 else 0.0
            result["spread"] = float(spread)
            if spread <= V2_LONG_REGIME_CONFLICT_HARD_SPREAD:
                result["long_conflict"] = True
                result["reason"] = (
                    f"long_conflict_hard spread={spread:.6f} "
                    f"hard={V2_LONG_REGIME_CONFLICT_HARD_SPREAD:.6f}"
                )
                return result
            result["long_conflict_soft"] = True
            result["reason"] = (
                f"long_conflict_soft spread={spread:.6f} "
                f"hard={V2_LONG_REGIME_CONFLICT_HARD_SPREAD:.6f}"
            )
            return result

        return result

    except Exception as e:
        print(f"[V2-REGIME] detect failed: {e}")
        result["reason"] = f"error:{e}"
        return result


def assess_ltf_long(ltf_df: pd.DataFrame) -> Dict[str, Any]:
    """
    LONG専用のLTF評価。
    押し目買い + 押し目からの回復を重視する。
    高RSIの追いかけ加点はしない。
    """
    if len(ltf_df) < 20:
        return {"score": 0.0, "reasons": []}

    df = ltf_df.copy()
    df["RSI"] = _rsi(df["Close"])
    df["EMA20"] = _ema(df["Close"], 20)

    cur = df.iloc[-2]   # 確定足
    prev = df.iloc[-3]
    rsi = _safe_float(cur["RSI"], np.nan)
    close = float(cur["Close"])
    ema20 = float(cur["EMA20"])
    prev_rsi = _safe_float(prev["RSI"], np.nan)

    score = 0.0
    reasons = []

    # 押し目: 価格がEMA20付近（-1%～+0.5%）
    dist = (close - ema20) / ema20 if ema20 != 0 else 0
    if -0.01 <= dist <= 0.005:
        score += 0.5
        reasons.append(f"pullback({dist:.4f})")

    # RSI 40-60帯で上昇中 → 押し目からの回復
    if np.isfinite(rsi) and np.isfinite(prev_rsi) and 40 <= rsi <= 60 and rsi > prev_rsi:
        score += 0.5
        reasons.append(f"rsi_recovering({rsi:.1f})")

    # 陽線確認
    if close > float(cur["Open"]):
        score += 0.2
        reasons.append("bullish_bar")

    return {"score": float(score), "reasons": reasons, "rsi": rsi}


def assess_ltf_short(ltf_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ★修正2: SHORT専用のLTF評価。
    戻り売り。LONG とは非対称。

    ★修正5: RSI<35 は加点しない。
    実績データ:
      SHORT × RSI<30  = WR 20.0%, PnL -0.535%
      SHORT × RSI 30-40 = WR 40.6%（最良）
      SHORT × RSI 40-50 = WR 44.0%（最良）
    つまりSHORT は RSI 30-55 帯がベスト。極端に低いRSIは危険。
    """
    if len(ltf_df) < 20:
        return {"score": 0.0, "reasons": []}

    df = ltf_df.copy()
    df["RSI"] = _rsi(df["Close"])
    df["EMA20"] = _ema(df["Close"], 20)

    cur = df.iloc[-2]
    prev = df.iloc[-3]
    rsi = _safe_float(cur["RSI"], np.nan)
    close = float(cur["Close"])
    ema20 = float(cur["EMA20"])
    prev_rsi = _safe_float(prev["RSI"], np.nan)

    score = 0.0
    reasons = []

    # 戻り: 価格がEMA20付近（-0.5%～+1%）
    dist = (close - ema20) / ema20 if ema20 != 0 else 0
    if -0.005 <= dist <= 0.01:
        score += 0.5
        reasons.append(f"retracement({dist:.4f})")

    # ★修正5: SHORT最適帯 RSI 30-55 で下降中
    if np.isfinite(rsi) and np.isfinite(prev_rsi) and 30 <= rsi <= 55 and rsi < prev_rsi:
        score += 0.7
        reasons.append(f"rsi_optimal_short({rsi:.1f})")
    elif np.isfinite(rsi) and np.isfinite(prev_rsi) and 55 < rsi <= 70 and rsi < prev_rsi:
        # RSI 55-70 はまだ許容（戻りが深い）
        score += 0.3
        reasons.append(f"rsi_declining({rsi:.1f})")

    # ★修正5: RSI<30 は減点（追いかけ危険）
    if np.isfinite(rsi) and rsi < 30:
        score -= 0.5
        reasons.append(f"rsi_oversold_penalty({rsi:.1f})")

    # ★修正5: RSI>70 も減点（SHORTには逆風）
    # 実績: SHORT × RSI 70+ = WR 13.4%
    if np.isfinite(rsi) and rsi > 70:
        score -= 0.8
        reasons.append(f"rsi_high_penalty({rsi:.1f})")

    # 陰線確認
    if close < float(cur["Open"]):
        score += 0.2
        reasons.append("bearish_bar")

    return {"score": float(score), "reasons": reasons, "rsi": rsi}


# ==========================================
# Pillar 2: ファンディングレート
# ==========================================

def _build_fr_symbol_candidates(exchange, symbol: str) -> List[str]:
    """
    Funding Rate 取得用の候補シンボルを構築する。
    方針:
      1) 生の symbol
      2) _resolve_okx_symbol() の解決結果
      3) exchange.markets 上で base/quote が一致する swap / linear 候補
      4) markets の id / symbol から拾える候補
      5) FET/USDT のように markets 走査で拾えない場合の明示 fallback
    を重複なく並べる。
    """
    out: List[str] = []
    seen: Set[str] = set()

    def _add(v: Any) -> None:
        s = ("" if v is None else str(v)).strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    raw = ("" if symbol is None else str(symbol)).strip()
    if not raw:
        return out

    _add(raw)

    try:
        resolved = _resolve_okx_symbol(exchange, raw)
        _add(resolved)
    except Exception:
        resolved = raw

    try:
        exchange.load_markets()
    except Exception:
        pass

    base = ""
    quote = "USDT"

    src = resolved if resolved else raw
    if "/" in src:
        left, right = src.split("/", 1)
        base = left.strip().upper()
        quote = right.split(":", 1)[0].strip().upper() or "USDT"
    else:
        base = src.strip().upper()

    mk = getattr(exchange, "markets", None) or {}
    if isinstance(mk, dict) and mk:
        for cand_symbol, info in mk.items():
            try:
                info = info or {}

                cand_symbol = str(cand_symbol).strip()
                market_id = str(info.get("id", "") or "").strip()
                market_symbol = str(info.get("symbol", "") or "").strip()

                b = str(info.get("base", "") or "").strip().upper()
                q = str(info.get("quote", "") or "").strip().upper()

                active = info.get("active", True)
                is_swap = bool(info.get("swap", False))
                is_linear = bool(info.get("linear", False))
                typ = str(info.get("type", "") or "").strip().lower()

                if active is False:
                    continue
                if b != base or q != quote:
                    continue
                if not (is_swap or is_linear or typ == "swap"):
                    continue

                _add(cand_symbol)
                _add(market_symbol)
                _add(market_id)

                if market_id:
                    _add(f"{b}/{q}:{q}")
                    _add(f"{b}/{q}")
            except Exception:
                continue

    # 従来候補
    if ":" not in raw and "/USDT" in raw:
        _add(raw + ":USDT")

    # FET/USDT 専用 fallback
    if base == "FET" and quote == "USDT":
        _add("FET/USDT")
        _add("FET/USDT:USDT")
        _add("FET-USDT-SWAP")
        _add("FET-USDT")
        _add("FETUSDT")
        _add("FET-USDT-PERP")

    return out


def fetch_funding_rate_safe(exchange, symbol: str) -> Optional[float]:
    """
    ccxt で Funding Rate を取得する。
    取得できなければ None を返す。
    重要:
      - 候補は _build_fr_symbol_candidates() で広く作る
      - fetch_funding_rate で失敗した候補は fetch_ticker でも fundingRate を試す
    """
    candidates = _build_fr_symbol_candidates(exchange, symbol)
    if not candidates:
        candidates = [symbol]

    last_err = ""

    for sym in candidates:
        # 1) まず fetch_funding_rate
        try:
            result = exchange.fetch_funding_rate(sym)
            if result and "fundingRate" in result:
                fr = result["fundingRate"]
                if fr is not None:
                    v = float(fr)
                    if np.isfinite(v):
                        if V2_DEBUG_REJECTS:
                            print(f"[V2-FR] success sym={symbol} used={sym} fr={v}")
                        return v
                    last_err = f"non_finite fundingRate={fr}"
                else:
                    last_err = "fundingRate is None"
            else:
                last_err = "no fundingRate key in fetch_funding_rate result"
        except Exception as e:
            last_err = f"fetch_funding_rate {type(e).__name__}:{e}"

        # 2) ダメなら fetch_ticker も試す
        try:
            ticker = exchange.fetch_ticker(sym)
            if ticker:
                fr = ticker.get("fundingRate", None)
                if fr is not None:
                    v = float(fr)
                    if np.isfinite(v):
                        if V2_DEBUG_REJECTS:
                            print(f"[V2-FR] success sym={symbol} used={sym} fr={v} via=ticker")
                        return v
                    last_err = f"non_finite ticker fundingRate={fr}"
                else:
                    info = ticker.get("info", {}) or {}
                    for key in ("fundingRate", "lastFundingRate", "nextFundingRate"):
                        raw_v = info.get(key, None)
                        if raw_v is None or str(raw_v).strip() == "":
                            continue
                        try:
                            v = float(raw_v)
                            if np.isfinite(v):
                                if V2_DEBUG_REJECTS:
                                    print(f"[V2-FR] success sym={symbol} used={sym} fr={v} via=ticker.info.{key}")
                                return v
                        except Exception:
                            continue
                    last_err = "ticker has no usable fundingRate"
            else:
                last_err = "fetch_ticker returned empty"
        except Exception as e:
            last_err = f"fetch_ticker {type(e).__name__}:{e}"

    if V2_DEBUG_REJECTS:
        print(f"[V2-FR] unavailable sym={symbol} candidates={candidates} last_err={last_err}")
    return None


def assess_funding(fr: Optional[float], direction: str) -> Dict[str, Any]:
    """
    ★修正2: LONG/SHORT で閾値の意味が違う。
    ★修正1: FR が取れなければスコア0（全一致不要）。
    """
    if fr is None:
        return {"score": 0.0, "available": False, "reason": "no_data"}

    score = 0.0
    reason = ""

    if direction == "LONG":
        if fr < V2_FR_LONG_TH:
            # 負のFR → ショートが溜まっている → LONG有利
            score = min(abs(fr - V2_FR_LONG_TH) / 0.0005, 1.0) * V2_FR_WEIGHT
            reason = f"negative_fr({fr:.6f})"
        elif fr > V2_FR_SHORT_TH:
            # 高い正のFR → LONG過密 → 逆風
            score = -0.3 * V2_FR_WEIGHT
            reason = f"crowded_long({fr:.6f})"
    elif direction == "SHORT":
        if fr > V2_FR_SHORT_TH:
            # 正のFR → ロングが溜まっている → SHORT有利
            score = min(abs(fr - V2_FR_SHORT_TH) / 0.0005, 1.0) * V2_FR_WEIGHT
            reason = f"positive_fr({fr:.6f})"
        elif fr < V2_FR_LONG_TH:
            # 負のFR → SHORT過密 → 逆風
            score = -0.3 * V2_FR_WEIGHT
            reason = f"crowded_short({fr:.6f})"

    return {"score": float(score), "available": True, "funding_rate": fr, "reason": reason}


# ==========================================
# Pillar 3: 出来高確認
# ==========================================

def assess_volume(ltf_df: pd.DataFrame, direction: str) -> Dict[str, Any]:
    if len(ltf_df) < 20:
        return {"score": 0.0, "confirmed": False, "vol_ratio": None}

    vol = pd.to_numeric(ltf_df["Volume"], errors="coerce")
    vol_ma = vol.rolling(20).mean()
    cur_vol = float(vol.iloc[-2])
    cur_ma = float(vol_ma.iloc[-2])

    if cur_ma <= 0 or pd.isna(cur_ma):
        return {"score": 0.0, "confirmed": False, "vol_ratio": None}

    vr = cur_vol / cur_ma
    chg = float(ltf_df["Close"].iloc[-2] - ltf_df["Close"].iloc[-3])

    score = 0.0
    confirmed = False

    if direction == "LONG" and chg > 0 and vr >= V2_VOL_CONFIRM:
        score = min(vr / 2.0, 1.0)
        confirmed = True
    elif direction == "SHORT" and chg < 0 and vr >= V2_VOL_CONFIRM:
        score = min(vr / 2.0, 1.0)
        confirmed = True
    elif vr < 0.8:
        score = -0.2  # 方向は合っているが出来高が伴わない

    return {"score": float(score), "confirmed": confirmed, "vol_ratio": float(vr)}


# ==========================================
# 動的TP/SL（ATRベース）
# ==========================================

def calc_atr_tp_sl(ltf_df: pd.DataFrame, direction: str, entry: float, total_score: float) -> Dict:
    atr_val = _safe_float(_atr(ltf_df).iloc[-2], abs(entry * 0.005))
    if atr_val <= 0:
        atr_val = abs(entry * 0.005)

    sl_mult = 1.5
    if V2_TP_SCORE_DEPENDENT:
        tp_mult = 2.0 + min(total_score / 5.0, 1.5)
    else:
        tp_mult = V2_TP_MULT_FIXED

    if direction == "LONG":
        sl = entry - atr_val * sl_mult
        tp = entry + atr_val * tp_mult
    else:
        sl = entry + atr_val * sl_mult
        tp = entry - atr_val * tp_mult

    return {
        "tp": float(tp), "sl": float(sl),
        "tp_pct": abs(tp - entry) / entry * 100,
        "sl_pct": abs(sl - entry) / entry * 100,
        "atr": float(atr_val),
    }


# ==========================================
# 統合シグナル生成
# ==========================================

def v2_generate_signal(
    exchange,
    symbol: str,
    btc_htf: Dict[str, Any],
    now_jst: datetime,
    regime_info: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    1銘柄を評価。合意なければ None。

    ★修正1: Pillar1 必須、Pillar2/3 は加点減点
    ★修正2: LONG/SHORT で assess_ltf を分ける
    ★追加: どこで不採用になったかをログで見える化
    """
    raw_symbol = str(symbol).strip()
    base_symbol = raw_symbol.split(":", 1)[0]
    sym = base_symbol.split("/", 1)[0].strip().upper()
    hour = None

    def _v2_reject(reason: str, extra: str = "") -> None:
        if V2_DEBUG_REJECTS:
            msg = f"[V2-REJECT] sym={sym} raw={raw_symbol} reason={reason}"
            if extra:
                msg += f" {extra}"
            print(msg)

    # ---- データ取得 ----
    ltf = fetch_ohlcv_safe(exchange, symbol, timeframe="15m", limit=60)
    if not ltf or len(ltf) < 30:
        _v2_reject("ltf_insufficient", f"len={0 if not ltf else len(ltf)} need=30")
        return None
    ltf_df = pd.DataFrame(ltf, columns=["Time", "Open", "High", "Low", "Close", "Volume"])

    try:
        closed_bar_time_ms = int(ltf_df["Time"].iloc[-2])
        closed_bar_dt = datetime.fromtimestamp(closed_bar_time_ms / 1000, JST)
        hour = closed_bar_dt.hour
    except Exception as e:
        _v2_reject("closed_bar_time_invalid", f"e={e}")
        return None

    htf = fetch_ohlcv_safe(exchange, symbol, timeframe=V2_HTF_TIMEFRAME, limit=V2_HTF_LIMIT)
    if not htf or len(htf) < V2_EMA_SLOW + 5:
        _v2_reject("htf_insufficient", f"len={0 if not htf else len(htf)} need={V2_EMA_SLOW + 5}")
        return None
    htf_df = pd.DataFrame(htf, columns=["Time", "Open", "High", "Low", "Close", "Volume"])

    # ---- Pillar1: 多時間足トレンド ----
    sym_htf = assess_htf_trend(htf_df)
    direction = str(sym_htf.get("direction", "") or "").strip().upper()
    sym_htf_strength = float(sym_htf.get("strength", 0.0) or 0.0)

    # BTC整合チェック用を先に取る
    btc_dir = str(btc_htf.get("direction", "NEUTRAL") or "").strip().upper()
    btc_str = float(btc_htf.get("strength", 0.0) or 0.0)

    # レジーム転換ガード: 方向別。SHORT/LONG の両方を見る。
    short_regime_conflict = bool((regime_info or {}).get("short_conflict", False))
    long_regime_conflict = bool((regime_info or {}).get("long_conflict", False))

    # 先に両方向LTF評価を作る
    ltf_eval_long = assess_ltf_long(ltf_df)
    ltf_eval_short = assess_ltf_short(ltf_df)

    long_rescue = False
    long_rescue_reason = ""
    short_rescue = False
    short_rescue_reason = ""

    ema_fast = _safe_float_or_nan(sym_htf.get("ema_fast"))
    ema_slow = _safe_float_or_nan(sym_htf.get("ema_slow"))
    htf_close = _safe_float_or_nan(sym_htf.get("close"))

    ema_close_long_ok = (np.isfinite(htf_close) and np.isfinite(ema_slow) and htf_close >= ema_slow)
    ema_close_short_ok = (np.isfinite(htf_close) and np.isfinite(ema_slow) and htf_close <= ema_slow)
    ema_gap_small_ok = (
        np.isfinite(ema_fast)
        and np.isfinite(ema_slow)
        and np.isfinite(htf_close)
        and abs(ema_fast - ema_slow) / max(abs(htf_close), 1e-9) < 0.002
    )

    # LONG raw rescue:
    # ここは一時的に無効化する。
    # 目的は、SHORT raw rescue が本当に発火するかを確認すること。
    # direction は assess_htf_trend() の結果のまま維持する。
    long_rescue = False
    long_rescue_reason = ""

    # SHORT raw rescue:
    # HTFがNEUTRALまたは弱いLONGでも、BTCがSHORT寄りでLTF_SHORTが十分強ければSHORT候補を救う
    if V2_SHORT_RAW_RESCUE_ENABLE:
        btc_short_ok = (btc_dir == "SHORT")
        allow_by_btc = (not V2_SHORT_RAW_RESCUE_ONLY_BTC_SHORT) or btc_short_ok

        ltf_short_score = float(ltf_eval_short.get("score", 0.0) or 0.0)
        ltf_short_rsi = _safe_float_or_nan(ltf_eval_short.get("rsi"))

        weak_long_ok = (
            V2_SHORT_RAW_RESCUE_ALLOW_LONG
            and direction == "LONG"
            and sym_htf_strength <= V2_SHORT_RAW_RESCUE_LONG_HTF_MAX
        )
        neutral_ok = V2_SHORT_RAW_RESCUE_ALLOW_NEUTRAL and (direction == "NEUTRAL")

        rescue_target_ok = neutral_ok or weak_long_ok or ema_close_short_ok or ema_gap_small_ok

        if (
            allow_by_btc
            and rescue_target_ok
            and (not short_regime_conflict)
            and ltf_short_score >= V2_SHORT_RAW_RESCUE_LTF_MIN
            and ((not np.isfinite(ltf_short_rsi)) or ltf_short_rsi >= V2_SHORT_RAW_RESCUE_RSI_MIN)
        ):
            direction = "SHORT"
            short_rescue = True
            short_rescue_reason = (
                f"short_raw_rescue from={sym_htf.get('direction')} "
                f"sym_htf_strength={sym_htf_strength:.4f} "
                f"btc_dir={btc_dir} btc_str={btc_str:.4f} "
                f"ltf_short_score={ltf_short_score:.4f} "
                f"ltf_short_rsi={ltf_short_rsi} "
                f"ema_close_short_ok={ema_close_short_ok} "
                f"ema_gap_small_ok={ema_gap_small_ok}"
            )

    if direction == "NEUTRAL":
        if V2_FET_HTF_NEUTRAL_FAIL_OPEN and sym == "FET":
            print(
                f"[V2-FET-HTF-BYPASS] "
                f"sym={sym} "
                f"reason=sym_htf_neutral "
                f"sym_htf_dir={sym_htf.get('direction')} "
                f"sym_htf_strength={sym_htf_strength}"
            )
            direction = "LONG"
        else:
            _v2_reject(
                "sym_htf_neutral",
                f"sym_htf_dir={sym_htf.get('direction')} sym_htf_strength={sym_htf_strength}"
            )
            return None

    # side別シンボル blocklist
    long_block = set(V2_LONG_SYMBOL_BLOCKLIST or [])
    short_block = set(V2_SHORT_SYMBOL_BLOCKLIST or [])

    if direction == "LONG" and sym in long_block:
        _v2_reject(
            "long_symbol_block",
            f"sym={sym} blocklist={sorted(long_block)}"
        )
        return None

    if direction == "SHORT" and sym in short_block:
        _v2_reject(
            "short_symbol_block",
            f"sym={sym} blocklist={sorted(short_block)}"
        )
        return None

    # BTC整合チェック
    # LONG raw rescue で方向を LONG にした場合は、ここで即死させず後段に回す
    if btc_dir != "NEUTRAL" and btc_dir != direction and btc_str >= V2_BTC_CONFLICT_STRONG_TH:
        if not (direction == "LONG" and long_rescue):
            _v2_reject(
                "btc_conflict_block",
                f"direction={direction} btc_dir={btc_dir} btc_str={btc_str} min={V2_BTC_CONFLICT_STRONG_TH}"
            )
            return None

    if V2_SHORT_REGIME_CONFLICT_BLOCK_ENABLE and short_regime_conflict and direction == "SHORT":
        _v2_reject(
            "regime_conflict_block",
            f"direction={direction} btc_dir={btc_dir} short_regime_conflict=True"
        )
        return None

    if long_regime_conflict and direction == "LONG":
        _v2_reject(
            "regime_conflict_block",
            f"direction={direction} btc_dir={btc_dir} long_regime_conflict=True"
        )
        return None

    # 方向別LTF評価
    if direction == "LONG":
        ltf_eval = ltf_eval_long
    else:
        ltf_eval = ltf_eval_short

    p1_score = sym_htf_strength + ltf_eval["score"]
    if btc_dir == direction:
        p1_score += btc_str * 0.5

    if long_rescue:
        print(f"[V2-LONG-RAW-RESCUE] sym={sym} {long_rescue_reason}")

    if short_rescue:
        print(f"[V2-SHORT-RAW-RESCUE] sym={sym} {short_rescue_reason}")

    # ★修正1: Pillar1 必須ゲート
    if p1_score < V2_PILLAR1_MIN:
        _v2_reject(
            "pillar1_below_min",
            f"direction={direction} p1={p1_score:.4f} min={V2_PILLAR1_MIN} "
            f"sym_htf_strength={sym_htf['strength']:.4f} ltf_score={ltf_eval['score']:.4f} "
            f"btc_dir={btc_dir} btc_str={btc_str:.4f}"
        )
        return None

    # ---- Pillar2: ファンディングレート ----
    fr = fetch_funding_rate_safe(exchange, symbol)
    p2 = assess_funding(fr, direction)
    p2_score = p2["score"]

    # ---- Pillar3: 出来高 ----
    p3 = assess_volume(ltf_df, direction)
    p3_score = p3["score"]

    # ---- 合計 ----
    total = p1_score + p2_score + p3_score
    if total < V2_MIN_SCORE:
        _v2_reject(
            "total_below_min",
            f"direction={direction} total={total:.4f} min={V2_MIN_SCORE} "
            f"p1={p1_score:.4f} p2={p2_score:.4f} p3={p3_score:.4f}"
        )
        return None

    # ---- TP/SL ----
    entry = float(ltf_df["Close"].iloc[-2])
    tpsl = calc_atr_tp_sl(ltf_df, direction, entry, total)

    # ★修正3: btc_mode 互換変換
    btc_mode_compat = {"LONG": "Up", "SHORT": "Down", "NEUTRAL": "Range"}.get(btc_dir, "Range")

    score_for_ai = float(total)

    ltf_ai = ltf_df.copy()

    close_s = pd.to_numeric(ltf_ai["Close"], errors="coerce")
    vol_s = pd.to_numeric(ltf_ai["Volume"], errors="coerce")

    ltf_ai["Pct_Change"] = close_s.pct_change(fill_method=None)
    ltf_ai["Dynamic_Sigma"] = ltf_ai["Pct_Change"].rolling(20).std()

    ma20 = close_s.rolling(20).mean()
    std20 = close_s.rolling(20).std()
    bb_upper = ma20 + (2.0 * std20)
    bb_lower = ma20 - (2.0 * std20)

    ltf_ai["BandWidth"] = np.where(
        ma20.abs() > 1e-12,
        (bb_upper - bb_lower) / ma20.abs(),
        np.nan,
    )

    ltf_ai["BW_Change"] = (
        pd.to_numeric(ltf_ai["BandWidth"], errors="coerce")
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
    )

    ltf_ai["Vol_Change"] = (
        vol_s.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
    )

    sigma_for_ai = _safe_float_or_nan(ltf_ai["Dynamic_Sigma"].iloc[-2])
    bandwidth_for_ai = _safe_float_or_nan(ltf_ai["BandWidth"].iloc[-2])
    bw_change_for_ai = _safe_float_or_nan(ltf_ai["BW_Change"].iloc[-2])
    vol_change_for_ai = _safe_float_or_nan(ltf_ai["Vol_Change"].iloc[-2])

    btc_ret_for_ai = _safe_float_or_nan(np.nan)
    btc_vol_for_ai = _safe_float_or_nan(np.nan)

    return {
        "symbol": sym,
        "direction": direction,
        "entry_price": entry,

        "total_score": float(total),
        "score": score_for_ai,

        "p1_score": float(p1_score),
        "p2_score": float(p2_score),
        "p3_score": float(p3_score),

        "sym_htf_dir": sym_htf["direction"],
        "sym_htf_strength": sym_htf["strength"],
        "btc_htf_dir": btc_dir,
        "btc_htf_strength": btc_str,

        "ltf_aligned": ltf_eval["score"] >= 0.5,
        "ltf_reasons": ltf_eval.get("reasons", []),

        "funding_rate": fr,
        "fr_available": p2["available"],
        "vol_ratio": p3.get("vol_ratio"),
        "vol_confirmed": p3["confirmed"],

        "tp": tpsl["tp"],
        "sl": tpsl["sl"],
        "tp_pct": tpsl["tp_pct"],
        "sl_pct": tpsl["sl_pct"],
        "atr": tpsl["atr"],
        "rsi": ltf_eval.get("rsi", np.nan),
        "hour": hour,
        "btc_mode_compat": btc_mode_compat,

        "sigma": sigma_for_ai,
        "BandWidth": bandwidth_for_ai,
        "BW_Change": bw_change_for_ai,
        "Vol_Change": vol_change_for_ai,
        "btc_ret": btc_ret_for_ai,
        "btc_vol": btc_vol_for_ai,

        "time_ms": closed_bar_time_ms,
        "dt": closed_bar_dt,
        "long_raw_rescue": bool(long_rescue),
        "short_raw_rescue": bool(short_rescue),
    }


# ==========================================
# V2 Selection Pipeline (SHORT専用)
# ==========================================
# Stage A: ディフェンシブフィルター（最低品質ライン）
# Stage B: QualityScore 計算（将来AIに差し替える唯一の場所）
# Stage C: 15分スロットごとに上位N件のみ残す
# ==========================================

def _safe_float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _fmt_sig_num_or_blank(sig: Dict[str, Any], key: str, fmt: str = ".4f") -> str:
    """sig から key の値を取り出し、finite なら指定フォーマットで文字列化、非 finite なら空文字を返す。"""
    v = _safe_float_or_nan(sig.get(key))
    if np.isfinite(v):
        return format(v, fmt)
    return ""


def _is_fr_available_flag(sig: Dict[str, Any]) -> bool:
    v = sig.get("fr_available", None)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _sig_hour_jst(sig: Dict[str, Any]) -> Optional[int]:
    for key in ("hour", "hour_jst", "Hour_JST"):
        try:
            v = sig.get(key, None)
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv):
                return int(fv)
        except Exception:
            continue
    return None


def _is_short_win_bucket(sig: Dict[str, Any]) -> bool:
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    hour = _sig_hour_jst(sig)

    if not _is_fr_available_flag(sig):
        return False
    if hour is None:
        return False
    if hour not in V2_SHORT_ALLOWED_HOURS:
        return False
    if btc_mode_compat not in V2_SHORT_BUCKET_BTC_MODES:
        return False
    if not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(rsi):
        return False
    if not (float(V2_SHORT_BUCKET_P1_MIN) <= float(p1) < float(V2_SHORT_BUCKET_P1_MAX)):
        return False
    if not (float(V2_SHORT_BUCKET_RSI_MIN) <= float(rsi) < float(V2_SHORT_BUCKET_RSI_MAX)):
        return False
    if float(p2) < float(V2_SHORT_BUCKET_P2_MIN):
        return False
    return True


def _is_short_strong_hour_bucket(sig: Dict[str, Any]) -> bool:
    if not _is_short_win_bucket(sig):
        return False

    hour = _sig_hour_jst(sig)
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    volratio = _safe_float_or_nan(sig.get("vol_ratio"))

    if hour is None:
        return False
    if hour not in V2_SHORT_STRONG_HOURS:
        return False
    if not np.isfinite(p2) or not np.isfinite(volratio):
        return False
    if float(p2) < float(V2_SHORT_STRONG_P2_MIN):
        return False
    if float(volratio) < float(V2_SHORT_STRONG_VOLRATIO_MIN):
        return False

    return True


def _is_long_down_win_bucket(sig: Dict[str, Any]) -> bool:
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    hour = _sig_hour_jst(sig)

    if not _is_fr_available_flag(sig):
        return False
    if hour is None:
        return False
    if hour not in V2_LONG_ALLOWED_HOURS:
        return False
    if btc_mode_compat != "DOWN":
        return False
    if not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(rsi):
        return False
    if not (float(V2_LONG_BUCKET_P1_MIN) <= float(p1) < float(V2_LONG_BUCKET_P1_MAX)):
        return False
    if not (float(V2_LONG_BUCKET_RSI_MIN) <= float(rsi) < float(V2_LONG_BUCKET_RSI_MAX)):
        return False
    if float(p2) < float(V2_LONG_BUCKET_P2_MIN):
        return False
    return True


def _is_long_alt_win_bucket(sig: Dict[str, Any]) -> bool:
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    hour = _sig_hour_jst(sig)

    if not V2_LONG_ALT_ENABLE:
        return False

    if not _is_fr_available_flag(sig):
        return False

    if hour is None:
        return False

    if hour not in V2_LONG_ALT_ALLOWED_HOURS:
        return False

    if not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(rsi):
        return False

    if not (float(V2_LONG_ALT_BUCKET_P1_MIN) <= float(p1) < float(V2_LONG_ALT_BUCKET_P1_MAX)):
        return False

    if not (float(V2_LONG_ALT_BUCKET_RSI_MIN) <= float(rsi) < float(V2_LONG_ALT_BUCKET_RSI_MAX)):
        return False

    if float(p2) < float(V2_LONG_ALT_BUCKET_P2_MIN):
        return False

    return True


def _check_long_strong_bucket(sig: Dict[str, Any]) -> Tuple[bool, str]:
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    hour = _sig_hour_jst(sig)
    fr_ok = _is_fr_available_flag(sig)

    if not V2_LONG_STRONG_ENABLE:
        return False, "long_strong_disabled"

    if not fr_ok:
        if V2_LONG_FR_FAIL_OPEN:
            try:
                sig["fr_fail_open_applied"] = True
            except Exception:
                pass
            print(
                f"[V2-LONG-STRONG-FR-BYPASS] "
                f"sym={str(sig.get('symbol', ''))} "
                f"direction={str(sig.get('direction', 'LONG'))} "
                f"reason=long_strong_fr_unavailable "
                f"p2_as_zero=1"
            )
        else:
            return False, "long_strong_fr_unavailable"

    if hour is None:
        return False, "long_strong_hour_missing"

    if hour not in V2_LONG_STRONG_ALLOWED_HOURS:
        return False, (
            f"long_strong_hour_out_of_range "
            f"hour={hour} "
            f"allowed={sorted(list(V2_LONG_STRONG_ALLOWED_HOURS))}"
        )

    if btc_mode_compat not in V2_LONG_STRONG_BUCKET_BTC_MODES:
        return False, (
            f"long_strong_btc_mode_miss "
            f"btc_mode={btc_mode_compat} "
            f"allowed={sorted(list(V2_LONG_STRONG_BUCKET_BTC_MODES))}"
        )

    if not np.isfinite(p1):
        return False, "long_strong_missing_p1"

    if not np.isfinite(p2):
        return False, "long_strong_missing_p2"

    if not np.isfinite(rsi):
        return False, "long_strong_missing_rsi"

    if not (float(V2_LONG_STRONG_BUCKET_P1_MIN) <= float(p1) < float(V2_LONG_STRONG_BUCKET_P1_MAX)):
        return False, (
            f"long_strong_p1_out_of_range "
            f"p1={float(p1):.4f} "
            f"min={float(V2_LONG_STRONG_BUCKET_P1_MIN):.4f} "
            f"max={float(V2_LONG_STRONG_BUCKET_P1_MAX):.4f}"
        )

    if not (float(V2_LONG_STRONG_BUCKET_RSI_MIN) <= float(rsi) < float(V2_LONG_STRONG_BUCKET_RSI_MAX)):
        return False, (
            f"long_strong_rsi_out_of_range "
            f"rsi={float(rsi):.4f} "
            f"min={float(V2_LONG_STRONG_BUCKET_RSI_MIN):.4f} "
            f"max={float(V2_LONG_STRONG_BUCKET_RSI_MAX):.4f}"
        )

    if float(p2) < float(V2_LONG_STRONG_BUCKET_P2_MIN):
        return False, (
            f"long_strong_p2_below_min "
            f"p2={float(p2):.4f} "
            f"min={float(V2_LONG_STRONG_BUCKET_P2_MIN):.4f}"
        )

    return True, "long_strong_bucket_pass"


def _is_long_strong_bucket(sig: Dict[str, Any]) -> bool:
    ok, _ = _check_long_strong_bucket(sig)
    return ok


def defensive_filter(sig: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Stage A:
    SHORT専用の最低品質ライン。
    LONGはここでは触らない。
    欠損は埋めずに reject する。

    今回の方針:
      - SHORT は「勝てる時間帯 + 共通条件」だけを通す
      - FR_Available=False は切る
      - 時間帯 / BTC_Mode / P1 / RSI / P2 の共通条件でまず絞る
      - 既存の score / btcstr / volconf / weak pocket の hard reject は SHORTでは使わない
    """
    direction = str(sig.get("direction", "")).strip().upper()
    if direction != "SHORT":
        return True, "not_short"

    if not V2_SHORT_DEF_ENABLE:
        return True, "short_def_disabled"

    short_recent = _get_recent_short_strict_health()
    if bool(short_recent.get("paused")):
        if V2_SHORT_PAUSE_ALLOW_STRONG_BUCKET and _is_short_strong_hour_bucket(sig):
            print(
                f"[V2-PAUSE-BYPASS] strong_short "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')} "
                f"reason={short_recent.get('reason_text', '')}"
            )
        else:
            return False, f"short_recent_pause {short_recent.get('reason_text', '')}"

    p1 = _safe_float_or_nan(sig.get("p1_score"))
    total = _safe_float_or_nan(sig.get("total_score"))
    btcstr = _safe_float_or_nan(sig.get("btc_htf_strength"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    hour = _sig_hour_jst(sig)

    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    fr_ok = _is_fr_available_flag(sig)

    if not np.isfinite(p1):
        return False, "missing_p1_score"

    if not np.isfinite(total):
        return False, "missing_total_score"

    if not np.isfinite(btcstr):
        return False, "missing_btc_htf_strength"

    if not np.isfinite(rsi):
        return False, "missing_rsi"

    if not np.isfinite(p2):
        return False, "missing_p2_score"

    if not fr_ok:
        return False, "fr_unavailable"

    if not _is_short_win_bucket(sig):
        return False, (
            f"short_bucket_miss "
            f"p1={p1:.4f} "
            f"p2={p2:.4f} "
            f"rsi={rsi:.4f} "
            f"hour={hour} "
            f"btc_mode={btc_mode_compat} "
            f"fr_ok={fr_ok}"
        )

    return True, "pass"


def _calc_rsi_zone_bonus(rsi: float) -> float:
    try:
        r = float(rsi)
    except Exception:
        return 0.0

    if not np.isfinite(r):
        return 0.0

    if r < V2_SHORT_RSI_DEEP_MAX:
        return 1.0

    if V2_SHORT_RSI_REBOUND_MIN <= r <= V2_SHORT_RSI_REBOUND_MAX:
        return 1.0

    if V2_SHORT_RSI_WEAK_MIN <= r < V2_SHORT_RSI_WEAK_MAX:
        return -1.0

    if V2_SHORT_DEF_RSI_MIN <= r <= V2_SHORT_DEF_RSI_MAX:
        return 0.25

    return 0.0


def calc_quality_score(sig: Dict[str, Any]) -> float:
    """
    Stage B:
    SHORT用の仮QualityScore。

    今回の方針:
      - 弱い帯の 35〜40 を嫌う
      - 強い帯の <35 と 45〜50 を残す
      - P2<0 を減点する
      - P1>=2.0 を減点する
      - P3が0〜0.5の帯を少し優遇する
      - total と p1 の二重効きを抑える
    """
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    total = _safe_float_or_nan(sig.get("total_score"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    p3 = _safe_float_or_nan(sig.get("p3_score"))
    btcstr = _safe_float_or_nan(sig.get("btc_htf_strength"))
    rsi = _safe_float_or_nan(sig.get("rsi"))

    volconf_raw = sig.get("vol_confirmed", None)
    volconf = 1.0 if str(volconf_raw).strip().lower() in ("1", "true", "yes", "on") else 0.0

    if not np.isfinite(p1):
        p1 = 0.0
    if not np.isfinite(total):
        total = 0.0
    if not np.isfinite(p2):
        p2 = 0.0
    if not np.isfinite(p3):
        p3 = 0.0
    if not np.isfinite(btcstr):
        btcstr = 0.0

    rsi_zone_bonus = _calc_rsi_zone_bonus(rsi)

    other_score = total - p1
    other_score = max(-1.5, min(other_score, 1.5))

    btc_over_penalty = -0.30 if (np.isfinite(btcstr) and btcstr > 0.90) else 0.0
    p2_neg_penalty = V2_SHORT_QS_W_P2_NEG if p2 < 0.0 else 0.0
    p1_high_penalty = V2_SHORT_QS_W_P1_GE2 if p1 >= 2.0 else 0.0
    p3_flat_bonus = V2_SHORT_QS_W_P3_FLAT if (0.0 <= p3 <= 0.5) else 0.0

    qs = (
        (p1 * 0.20) +
        (other_score * 0.30) +
        (p3 * 0.20) +
        (btcstr * 0.05) +
        (rsi_zone_bonus * 0.15) +
        (volconf * 0.20) +
        (p3_flat_bonus) +
        (p2_neg_penalty) +
        (p1_high_penalty) +
        (btc_over_penalty)
    )
    return float(qs)


def rank_and_select(short_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage C:
    SHORT候補を ai_prob_win 降順に並べる。
    - SHORT AI が入っていれば ai_prob_win は AI確率
    - bypass/fallback のときは RULE fallback の値
    - SHORT brake は従来通り適用
    """
    if not short_signals:
        return []

    for sig in short_signals:
        sig["rule_qs"] = calc_quality_score(sig)

    ranked = sorted(
        short_signals,
        key=lambda x: (
            float(x.get("ai_prob_win", 0.0) or 0.0),
            float(x.get("rule_qs", 0.0) or 0.0),
        ),
        reverse=True,
    )

    base_top_n = max(1, int(V2_SHORT_RANK_TOP_N))
    probe_top_n = max(1, int(V2_SHORT_BRAKE_PROBE_TOP_N))
    brake = get_v2_short_brake_state(datetime.now(JST))
    brake_mode = str(brake.get("mode", "NORMAL") or "NORMAL").upper()
    effective_top_n = int(brake.get("effective_top_n", base_top_n) or base_top_n)

    if not V2_SHORT_RANK_ENABLE:
        for rank_idx, sig in enumerate(ranked, start=1):
            slot_total = len(ranked)
            sig["ai_pass"] = "1"
            sig["ai_band"] = _v2_rank_selected_band(sig)
            current_note = str(sig.get("ai_note", "") or "")
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n=ALL;selected=true;rank_disabled=true;brake_mode={brake_mode}"
            )
        return ranked

    slot_total = len(ranked)

    for rank_idx, sig in enumerate(ranked, start=1):
        current_note = str(sig.get("ai_note", "") or "")

        if brake_mode == "STOP":
            sig["ai_pass"] = "0"
            if rank_idx <= probe_top_n:
                sig["ai_band"] = "BRAKE_STOP_OBSERVE"
                sig["ai_note"] = (
                    f"{current_note};rank={rank_idx};slot_total={slot_total};"
                    f"top_n={probe_top_n};selected=false;brake_mode=STOP;observe_only=true"
                )
            else:
                sig["ai_band"] = "BRAKE_STOP_DROP"
                sig["ai_note"] = (
                    f"{current_note};rank={rank_idx};slot_total={slot_total};"
                    f"top_n={probe_top_n};selected=false;brake_mode=STOP;observe_only=false"
                )
            continue

        if rank_idx <= effective_top_n:
            sig["ai_pass"] = "1"
            sig["ai_band"] = _v2_rank_selected_band(sig)
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n={effective_top_n};selected=true;brake_mode={brake_mode}"
            )
        else:
            sig["ai_pass"] = "0"
            sig["ai_band"] = _v2_rank_reject_band(sig)
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n={effective_top_n};selected=false;reason=rank_exceeded;brake_mode={brake_mode}"
            )

    return ranked


def _allow_long_p1_rescue(sig: Dict[str, Any]) -> bool:
    """
    LONG の P1 1.8〜2.0 帯を条件付きで救う。
    何でも救うのではなく、UP・低RSI・研究タグが良いものだけ。
    """
    if not V2_LONG_P1_RESCUE_ENABLE:
        return False
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    if not np.isfinite(p1) or not np.isfinite(rsi):
        return False
    if p1 < V2_LONG_P1_RESCUE_MIN or p1 >= V2_LONG_DEF_P1_MIN:
        return False
    if rsi >= V2_LONG_P1_RESCUE_RSI_MAX:
        return False
    if btc_mode_compat != "UP":
        return False
    tags = _get_long_research_tags(sig)
    if "LONG_CORE_P18_RSI60_UP" not in tags and "LONG_CORE_P20_RSI60_UP" not in tags:
        return False
    if "LONG_DANGER_RSI70" in tags or "LONG_DANGER_RANGE" in tags:
        return False
    return True


def _allow_long_p1_hard_max_rescue(sig: Dict[str, Any], p1: float, hard_max: float) -> Tuple[bool, str]:
    """
    LONG の p1_above_hard_max を限定救済する。
    全解除ではなく、Up相場で実績が良かった帯だけを通す。
    """
    try:
        side_u = str(sig.get("side", sig.get("direction", ""))).strip().upper()
        if side_u != "LONG":
            return False, ""

        btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
        rsi = _safe_float_or_nan(sig.get("rsi"))
        p3 = _safe_float_or_nan(sig.get("p3_score"))

        if btc_mode_compat != "UP":
            return False, ""

        if not np.isfinite(rsi) or not np.isfinite(p3):
            return False, ""

        if p1 < hard_max:
            return False, ""

        if not (55.0 <= rsi < 60.0):
            return False, ""

        if not (p3 < 0.0):
            return False, ""

        reason = (
            "rescued_p1_above_hard_max "
            f"p1={float(p1):.4f} "
            f"hard_max={float(hard_max):.4f} "
            f"rsi={float(rsi):.4f} "
            f"p3={float(p3):.4f} "
            f"btc_mode={btc_mode_compat}"
        )
        return True, reason

    except Exception:
        return False, ""


def _allow_long_rsi_out_of_range_rescue(
    sig: Dict[str, Any],
    rsi: float,
    rsi_min_eff: float,
    rsi_max_eff: float,
) -> Tuple[bool, str]:
    """
    LONG の rsi_out_of_range を限定救済する。
    Down 相場の強い反発帯だけを通す。
    """
    try:
        side_u = str(sig.get("side", sig.get("direction", ""))).strip().upper()
        if side_u != "LONG":
            return False, ""

        btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
        p1 = _safe_float_or_nan(sig.get("p1_score"))
        p2 = _safe_float_or_nan(sig.get("p2_score"))
        p3 = _safe_float_or_nan(sig.get("p3_score"))

        if btc_mode_compat != "DOWN":
            return False, ""

        if not np.isfinite(rsi) or not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(p3):
            return False, ""

        if not (float(rsi) < 35.0):
            return False, ""

        if not (1.5 <= float(p1) < 1.8):
            return False, ""

        if not (float(p2) <= 0.25):
            return False, ""

        p3v = float(p3)
        if not (abs(p3v - (-0.2)) <= 1e-9 or abs(p3v - 0.0) <= 1e-9):
            return False, ""

        reason = (
            "rescued_rsi_out_of_range "
            f"rsi={float(rsi):.4f} "
            f"range={float(rsi_min_eff):.4f}-{float(rsi_max_eff):.4f} "
            f"p1={float(p1):.4f} "
            f"p2={float(p2):.4f} "
            f"p3={float(p3):.4f} "
            f"btc_mode={btc_mode_compat}"
        )
        return True, reason

    except Exception:
        return False, ""


def _allow_long_p1_below_min_rescue(
    sig: Dict[str, Any],
    p1: float,
    p1_min_eff: float,
) -> Tuple[bool, str]:
    """
    LONG の p1_below_min を限定救済する。
    Down 相場のかなり狭い勝ち帯だけを通す。
    """
    try:
        side_u = str(sig.get("side", sig.get("direction", ""))).strip().upper()
        if side_u != "LONG":
            return False, ""

        btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
        rsi = _safe_float_or_nan(sig.get("rsi"))
        p2 = _safe_float_or_nan(sig.get("p2_score"))
        p3 = _safe_float_or_nan(sig.get("p3_score"))

        if btc_mode_compat != "DOWN":
            return False, ""

        if not np.isfinite(rsi) or not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(p3):
            return False, ""

        if not (35.0 <= float(rsi) < 40.0):
            return False, ""

        if abs(float(p2)) > 1e-9:
            return False, ""

        if not (float(p3) > 0.0):
            return False, ""

        reason = (
            "rescued_p1_below_min "
            f"p1={float(p1):.4f} "
            f"min={float(p1_min_eff):.4f} "
            f"rsi={float(rsi):.4f} "
            f"p2={float(p2):.4f} "
            f"p3={float(p3):.4f} "
            f"btc_mode={btc_mode_compat}"
        )
        return True, reason

    except Exception:
        return False, ""


def defensive_filter_long(sig: Dict[str, Any]) -> Tuple[bool, str]:
    """
    LONG専用の最低品質ライン。
    SHORTはここでは触らない。
    追加方針:
      - LONGは当面厳しめ継続
      - VolConfirmed を必須化できるようにする
    """
    direction = str(sig.get("direction", "")).strip().upper()
    if direction != "LONG":
        return True, "not_long"

    if not V2_LONG_DEF_ENABLE:
        return True, "long_def_disabled"

    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    long_raw_rescue = str(sig.get("long_raw_rescue", "")).strip().lower() in ("1", "true", "yes", "on")

    # まず LONG_STRONG を最優先で通す
    strong_ok, strong_reason = _check_long_strong_bucket(sig)
    if strong_ok:
        return True, strong_reason

    # 次に、必要なら LONG_ALT を通す
    if _is_long_alt_win_bucket(sig):
        return True, "long_alt_bucket_pass"

    if V2_LONG_BLOCK_RANGE and btc_mode_compat == "RANGE":
        if not long_raw_rescue:
            return False, "range_mode_block"

    if V2_LONG_REJECT_MODE_DOWN and btc_mode_compat == "DOWN":
        if not long_raw_rescue and (not _is_long_down_win_bucket(sig)):
            return False, "down_mode_block"

    p1 = _safe_float_or_nan(sig.get("p1_score"))
    total = _safe_float_or_nan(sig.get("total_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    hour = _sig_hour_jst(sig)

    volconf_raw = sig.get("vol_confirmed", None)
    volconf = str(volconf_raw).strip().lower() in ("1", "true", "yes", "on")
    fr_ok = _is_fr_available_flag(sig)

    if not fr_ok:
        if V2_LONG_FR_FAIL_OPEN:
            sig["fr_fail_open_applied"] = True
            print(
                f"[V2-LONG-FR-BYPASS] "
                f"sym={str(sig.get('symbol', ''))} "
                f"direction={str(sig.get('direction', ''))} "
                f"reason=fr_unavailable "
                f"p2_as_zero=1"
            )
        else:
            return False, "fr_unavailable"

    primary_long_ok = _is_long_down_win_bucket(sig)
    alt_long_ok = _is_long_alt_win_bucket(sig)

    if alt_long_ok:
        return True, "long_alt_bucket_pass"

    if not primary_long_ok:
        if btc_mode_compat != "DOWN":
            return False, f"long_mode_not_whitelisted btc_mode={btc_mode_compat} strong_reason={strong_reason}"

        return False, (
            f"long_bucket_miss "
            f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')} "
            f"p2={_fmt_sig_num_or_blank(sig, 'p2_score')} "
            f"rsi={_fmt_sig_num_or_blank(sig, 'rsi')} "
            f"hour={hour if hour is not None else ''} "
            f"btc_mode={btc_mode_compat} "
            f"fr_ok={fr_ok}"
        )

    if not np.isfinite(p1):
        return False, "missing_p1_score"

    if not np.isfinite(total):
        return False, "missing_total_score"

    if not np.isfinite(rsi):
        return False, "missing_rsi"

    if V2_LONG_DEF_REQUIRE_VOLCONF and (not volconf):
        if not long_raw_rescue:
            if not (btc_mode_compat == "UP" and (not V2_LONG_DEF_REQUIRE_VOLCONF_UP)):
                return False, "vol_not_confirmed"

    p1_min_eff = float(V2_LONG_DEF_P1_MIN)
    if long_raw_rescue:
        p1_min_eff = min(p1_min_eff, 1.5)

    if p1 < p1_min_eff:
        if _allow_long_p1_rescue(sig):
            pass
        else:
            allow_rescue, rescue_reason = _allow_long_p1_below_min_rescue(
                sig, p1, p1_min_eff
            )
            if allow_rescue:
                return True, rescue_reason
            return False, f"p1_below_min p1={p1:.4f} min={p1_min_eff:.4f}"

    if p1 >= V2_LONG_DEF_P1_HARD_MAX:
        allow_rescue, rescue_reason = _allow_long_p1_hard_max_rescue(
            sig, p1, float(V2_LONG_DEF_P1_HARD_MAX)
        )
        if allow_rescue:
            return True, rescue_reason
        return False, (
            f"p1_above_hard_max p1={p1:.4f} "
            f"hard_max={V2_LONG_DEF_P1_HARD_MAX:.4f}"
        )

    score_min_eff = float(V2_LONG_DEF_SCORE_MIN)
    rsi_min_eff = float(V2_LONG_DEF_RSI_MIN)
    rsi_max_eff = float(V2_LONG_DEF_RSI_MAX)

    if long_raw_rescue:
        score_min_eff = min(score_min_eff, 1.3)
        rsi_min_eff = min(rsi_min_eff, 35.0)

    p2 = _safe_float_or_nan(sig.get("p2_score"))
    p3 = _safe_float_or_nan(sig.get("p3_score"))
    sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()
    range_block_syms = {s.strip().upper() for s in V2_LONG_RANGE_RESCUE_BLOCK_SYMBOLS if str(s).strip()}

    if long_raw_rescue and btc_mode_compat == "RANGE":
        thin_allow_syms = {s.strip().upper() for s in V2_LONG_RANGE_RESCUE_THIN_ALLOW_SYMBOLS if str(s).strip()}
        thin_rescue_hit = (
            V2_LONG_RANGE_RESCUE_THIN_ENABLE
            and sym in thin_allow_syms
            and np.isfinite(rsi) and np.isfinite(p2) and np.isfinite(p3)
            and rsi <= float(V2_LONG_RANGE_RESCUE_THIN_RSI_MAX)
            and p2 >= float(V2_LONG_RANGE_RESCUE_THIN_P2_MIN)
            and p3 >= float(V2_LONG_RANGE_RESCUE_THIN_P3_MIN)
        )
        if not thin_rescue_hit:
            if sym in range_block_syms:
                return False, f"range_rescue_symbol_block sym={sym}"

            if (not np.isfinite(rsi)) or rsi > float(V2_LONG_RANGE_RESCUE_RSI_MAX):
                return False, (
                    f"range_rescue_rsi_too_high rsi={rsi:.4f} "
                    f"max={float(V2_LONG_RANGE_RESCUE_RSI_MAX):.1f}"
                )

            if (not np.isfinite(p2)) or p2 < float(V2_LONG_RANGE_RESCUE_P2_MIN):
                return False, (
                    f"range_rescue_p2_below_min p2={p2:.4f} "
                    f"min={float(V2_LONG_RANGE_RESCUE_P2_MIN):.4f}"
                )

            if (not np.isfinite(p3)) or p3 < float(V2_LONG_RANGE_RESCUE_P3_MIN):
                return False, (
                    f"range_rescue_p3_below_min p3={p3:.4f} "
                    f"min={float(V2_LONG_RANGE_RESCUE_P3_MIN):.4f}"
                )

    if total < score_min_eff:
        return False, f"score_below_min total={total:.4f} min={score_min_eff:.4f}"

    if rsi < rsi_min_eff or rsi > rsi_max_eff:
        allow_rescue, rescue_reason = _allow_long_rsi_out_of_range_rescue(
            sig, rsi, rsi_min_eff, rsi_max_eff
        )
        if allow_rescue:
            return True, rescue_reason
        return False, (
            f"rsi_out_of_range rsi={rsi:.4f} "
            f"min={rsi_min_eff:.1f} max={rsi_max_eff:.1f}"
        )

    if rsi > V2_LONG_DEF_RSI_HARD_MAX:
        return False, (
            f"rsi_above_hard_max rsi={rsi:.4f} "
            f"hard_max={V2_LONG_DEF_RSI_HARD_MAX:.1f}"
        )

    return True, "pass"


def _calc_long_rsi_bonus(rsi: float) -> float:
    if not np.isfinite(rsi):
        return 0.0

    if V2_LONG_RSI_SWEET_MIN <= rsi <= V2_LONG_RSI_SWEET_MAX:
        return 1.0

    if V2_LONG_DEF_RSI_MIN <= rsi <= V2_LONG_DEF_RSI_MAX:
        return 0.4

    return 0.0


def _calc_long_p3_bonus(p3_score: float) -> float:
    if not np.isfinite(p3_score):
        return 0.0

    p3 = float(p3_score)

    if -0.25 <= p3 <= -0.15:
        return 1.0
    if -0.05 <= p3 <= 0.05:
        return 0.3
    if 0.95 <= p3 <= 1.05:
        return -0.3

    return 0.0


def _calc_long_btc_bonus(btcstr: float) -> float:
    if not np.isfinite(btcstr):
        return 0.0

    if btcstr < V2_LONG_BTCSTR_SOFT_MAX:
        return 1.0
    if btcstr < V2_LONG_BTCSTR_HARD_MAX:
        return 0.2

    return -0.5


def _calc_long_mode_bonus(btc_mode_compat: str) -> float:
    mode = str(btc_mode_compat or "").strip().upper()
    if mode == "UP":
        return V2_LONG_MODE_UP_PENALTY
    if mode == "RANGE":
        return V2_LONG_MODE_RANGE_BONUS
    if mode == "DOWN":
        return V2_LONG_MODE_DOWN_BONUS
    return 0.0


def _calc_long_bad_symbol_flag(symbol: str) -> float:
    sym = str(symbol).split("/", 1)[0].strip().upper()
    return 1.0 if sym in V2_LONG_BAD_SYMBOLS else 0.0


def calc_long_quality_score(sig: Dict[str, Any]) -> float:
    """
    LONG用の仮QualityScore。
    将来LONG-AIに差し替える場所。
    LONGは gate で P1 と RSI を見ているので、
    ここでは補助情報（P3 / BTC強度 / BTCモード / other_score / bad_symbol）を中心に順位づけする。
    """
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    total = _safe_float_or_nan(sig.get("total_score"))
    p3_score = _safe_float_or_nan(sig.get("p3_score"))
    btcstr = _safe_float_or_nan(sig.get("btc_htf_strength"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "")

    if not np.isfinite(p1) or not np.isfinite(total):
        return float("nan")

    rsi_bonus = _calc_long_rsi_bonus(rsi)
    p3_bonus = _calc_long_p3_bonus(p3_score)
    btc_bonus = _calc_long_btc_bonus(btcstr)
    mode_bonus = _calc_long_mode_bonus(btc_mode_compat)

    other_score = total - p1
    other_score = max(-1.5, min(other_score, 1.5))

    bad_symbol_flag = _calc_long_bad_symbol_flag(sig.get("symbol", ""))

    qs = (
        (p1 * V2_LONG_QS_W_P1) +
        (rsi_bonus * V2_LONG_QS_W_RSI) +
        (p3_bonus * V2_LONG_QS_W_P3) +
        (btc_bonus * V2_LONG_QS_W_BTC) +
        (mode_bonus * V2_LONG_QS_W_MODE) +
        (other_score * V2_LONG_QS_W_OTHER) +
        (bad_symbol_flag * V2_LONG_QS_W_BAD_SYMBOL)
    )
    return float(qs)


def _get_long_research_tags(sig: Dict[str, Any]) -> List[str]:
    """
    LONG研究用タグ。
    本番判定には使わず、v2_shadow_ai の AI_Note に残して後から集計しやすくする。
    """
    tags: List[str] = []
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()

    if btc_mode_compat == "DOWN":
        tags.append("mode_down")

    if np.isfinite(rsi) and rsi > V2_LONG_DEF_RSI_HARD_MAX:
        tags.append("rsi_gt_hard_max")

    if np.isfinite(p1) and p1 >= V2_LONG_DEF_P1_HARD_MAX:
        tags.append("p1_ge_hard_max")

    if np.isfinite(p1) and np.isfinite(rsi):
        if p1 >= 2.0 and rsi < 60 and btc_mode_compat == "UP":
            tags.append("LONG_CORE_P20_RSI60_UP")
        elif p1 >= 1.8 and rsi < 60 and btc_mode_compat == "UP":
            tags.append("LONG_CORE_P18_RSI60_UP")
        if rsi >= 70:
            tags.append("LONG_DANGER_RSI70")
        elif rsi >= 60:
            tags.append("LONG_WARN_RSI60")
    if btc_mode_compat == "RANGE":
        tags.append("LONG_DANGER_RANGE")
    return tags


def _long_notify_gate(sig: Dict[str, Any]) -> Tuple[bool, str]:
    if not V2_LONG_NOTIFY_GATE_ENABLE:
        return True, "notify_gate_disabled"

    mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()
    rsi = _safe_float_or_nan(sig.get("rsi"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    p3 = _safe_float_or_nan(sig.get("p3_score"))

    block_syms = {s.strip().upper() for s in V2_LONG_NOTIFY_GATE_BLOCK_SYMBOLS if str(s).strip()}
    if sym in block_syms:
        return False, f"notify_gate_symbol_block sym={sym}"

    if mode == "RANGE" and V2_LONG_NOTIFY_GATE_RANGE_ENABLE:
        thin_allow_syms_gate = {s.strip().upper() for s in V2_LONG_RANGE_RESCUE_THIN_ALLOW_SYMBOLS if str(s).strip()}
        thin_gate_hit = (
            V2_LONG_RANGE_RESCUE_THIN_ENABLE
            and sym in thin_allow_syms_gate
            and np.isfinite(rsi) and np.isfinite(p2) and np.isfinite(p3)
            and rsi <= float(V2_LONG_RANGE_RESCUE_THIN_RSI_MAX)
            and p2 >= float(V2_LONG_RANGE_RESCUE_THIN_P2_MIN)
            and p3 >= float(V2_LONG_RANGE_RESCUE_THIN_P3_MIN)
        )
        if not thin_gate_hit:
            if (not np.isfinite(rsi)) or rsi > float(V2_LONG_NOTIFY_GATE_RANGE_RSI_MAX):
                return False, (
                    f"notify_gate_range_rsi_high rsi={rsi:.4f} "
                    f"max={float(V2_LONG_NOTIFY_GATE_RANGE_RSI_MAX):.1f}"
                )

            if (not np.isfinite(p2)) or p2 < float(V2_LONG_NOTIFY_GATE_RANGE_P2_MIN):
                return False, (
                    f"notify_gate_range_p2_low p2={p2:.4f} "
                    f"min={float(V2_LONG_NOTIFY_GATE_RANGE_P2_MIN):.4f}"
                )

            if (not np.isfinite(p3)) or p3 < float(V2_LONG_NOTIFY_GATE_RANGE_P3_MIN):
                return False, (
                    f"notify_gate_range_p3_low p3={p3:.4f} "
                    f"min={float(V2_LONG_NOTIFY_GATE_RANGE_P3_MIN):.4f}"
                )

    if mode == "UP" and V2_LONG_NOTIFY_GATE_UP_ENABLE:
        if (not np.isfinite(rsi)) or rsi > float(V2_LONG_NOTIFY_GATE_UP_RSI_MAX):
            return False, (
                f"notify_gate_up_rsi_high rsi={rsi:.4f} "
                f"max={float(V2_LONG_NOTIFY_GATE_UP_RSI_MAX):.1f}"
            )

        if (not np.isfinite(p2)) or p2 < float(V2_LONG_NOTIFY_GATE_UP_P2_MIN):
            return False, (
                f"notify_gate_up_p2_low p2={p2:.4f} "
                f"min={float(V2_LONG_NOTIFY_GATE_UP_P2_MIN):.4f}"
            )

        if (not np.isfinite(p3)) or p3 < float(V2_LONG_NOTIFY_GATE_UP_P3_MIN):
            return False, (
                f"notify_gate_up_p3_low p3={p3:.4f} "
                f"min={float(V2_LONG_NOTIFY_GATE_UP_P3_MIN):.4f}"
            )

    return True, "pass"


def _v2_bool01(x: Any) -> float:
    s = ("" if x is None else str(x)).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return 1.0
    if s in {"0", "false", "f", "no", "n", "off"}:
        return 0.0
    return np.nan


def _v2_mode_onehot(mode_raw: Any) -> Tuple[float, float, float]:
    mode = ("" if mode_raw is None else str(mode_raw)).strip().upper()
    if mode == "UP":
        return 1.0, 0.0, 0.0
    if mode == "RANGE":
        return 0.0, 1.0, 0.0
    if mode == "DOWN":
        return 0.0, 0.0, 1.0
    return np.nan, np.nan, np.nan


def _v2_sig_symbol_code(sig: Dict[str, Any]) -> str:
    raw = str(sig.get("symbol", "") or "").strip().upper()
    base = raw.split(":", 1)[0]
    return base.split("/", 1)[0].strip().upper()


def _v2_sig_hour(sig: Dict[str, Any]) -> float:
    h = _safe_float_or_nan(sig.get("hour"))
    if np.isfinite(h):
        return float(h)

    dt = sig.get("dt")
    try:
        if dt is not None and hasattr(dt, "hour"):
            return float(dt.hour)
    except Exception:
        pass

    return np.nan


def _v2_compact_dbg(dbg: Any) -> str:
    d = dbg if isinstance(dbg, dict) else {"dbg": str(dbg)}
    parts: List[str] = []

    err = str(d.get("error", "") or "").strip()
    if err:
        parts.append(f"error={err}")

    detail = str(d.get("detail", "") or "").strip()
    if detail:
        parts.append(f"detail={detail}")

    miss = d.get("missing_columns")
    if not isinstance(miss, (list, tuple)) or not miss:
        miss = d.get("missing_cols")
    if isinstance(miss, (list, tuple)) and miss:
        miss_txt = ",".join(str(x) for x in list(miss)[:8])
        if len(miss) > 8:
            miss_txt += ",..."
        parts.append(f"missing={miss_txt}")

    nan_cols = d.get("nan_columns")
    if not isinstance(nan_cols, (list, tuple)) or not nan_cols:
        nan_cols = d.get("nan_cols")
    if isinstance(nan_cols, (list, tuple)) and nan_cols:
        nan_txt = ",".join(str(x) for x in list(nan_cols)[:8])
        if len(nan_cols) > 8:
            nan_txt += ",..."
        parts.append(f"nan={nan_txt}")

    nan_cnt = d.get("nan_cnt")
    try:
        if nan_cnt is not None:
            parts.append(f"nan_cnt={int(nan_cnt)}")
    except Exception:
        pass

    if not parts:
        raw = str(d.get("dbg", "") or "").strip()
        parts.append(raw if raw else "bypass")

    txt = ";".join(parts)
    maxlen = max(60, int(V2_AI_NOTE_DBG_MAXLEN))
    return txt[:maxlen]


def _build_v2_ai_feature_frame(sig: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    本番の V2 LONG/SHORT AI 推論用特徴量を、現行 serving model が期待する旧9列で作る。

    期待列:
      Sigma, BandWidth, BW_Change, RSI, Vol_Change,
      Rise_Score, Drop_Score, BTC_Ret, BTC_Vol

    重要:
      - ここでは固定値埋めをしない
      - 欠損があれば None を返し、上位で bypass させる
      - side に応じて Rise_Score / Drop_Score の片側だけにスコアを載せる
    """
    side_u = str(sig.get("side", "") or "").strip().upper()
    is_long = 1.0 if side_u == "LONG" else 0.0
    is_short = 1.0 - is_long

    score_val = _safe_float_or_nan(sig.get("score"))
    sigma_val = _safe_float_or_nan(sig.get("sigma"))
    bw_val = _safe_float_or_nan(sig.get("BandWidth"))
    bw_chg_val = _safe_float_or_nan(sig.get("BW_Change"))
    rsi_val = _safe_float_or_nan(sig.get("rsi"))
    vol_chg_val = _safe_float_or_nan(sig.get("Vol_Change"))
    btc_ret_val = _safe_float_or_nan(sig.get("btc_ret"))
    btc_vol_val = _safe_float_or_nan(sig.get("btc_vol"))

    feats = pd.DataFrame([{
        "Sigma": sigma_val,
        "BandWidth": bw_val,
        "BW_Change": bw_chg_val,
        "RSI": rsi_val,
        "Vol_Change": vol_chg_val,
        "Rise_Score": score_val * is_long,
        "Drop_Score": score_val * is_short,
        "BTC_Ret": btc_ret_val,
        "BTC_Vol": btc_vol_val,
    }]).replace([np.inf, -np.inf], np.nan)

    mask = feats.isna()
    if mask.any().any():
        bad_cols = [c for c in feats.columns if mask[c].any()]
        try:
            print(
                f"[V2-AI-FEAT-SKIP] sym={sig.get('symbol', '')} "
                f"side={side_u} bad_cols={bad_cols} "
                f"score={sig.get('score', '')} "
                f"sigma={sig.get('sigma', '')} "
                f"bw={sig.get('BandWidth', '')} "
                f"bw_chg={sig.get('BW_Change', '')} "
                f"rsi={sig.get('rsi', '')} "
                f"vol_chg={sig.get('Vol_Change', '')} "
                f"btc_ret={sig.get('btc_ret', '')} "
                f"btc_vol={sig.get('btc_vol', '')}"
            )
        except Exception:
            pass
        return None

    return feats


def _v2_rank_selected_band(sig: Dict[str, Any]) -> str:
    mt = str(sig.get("ai_model_type", "") or "").strip().upper()
    regime = str(sig.get("_long_bad_regime_mode", "") or "").strip().upper()
    if mt == "LONG_AI":
        return "LONG_AI_CAUTION_RANK" if regime == "CAUTION" else "LONG_AI_RANK"
    if mt == "SHORT_AI":
        return "SHORT_AI_RANK"
    if mt == "LONG_AI_FALLBACK":
        return "LONG_AI_CAUTION_FALLBACK_RANK" if regime == "CAUTION" else "LONG_AI_FALLBACK_RANK"
    if mt == "SHORT_AI_FALLBACK":
        return "SHORT_AI_FALLBACK_RANK"
    return "RULE_QS"


def _v2_rank_reject_band(sig: Dict[str, Any]) -> str:
    mt = str(sig.get("ai_model_type", "") or "").strip().upper()
    if mt == "LONG_AI":
        return "LONG_AI_RANK_REJECT"
    if mt == "SHORT_AI":
        return "SHORT_AI_RANK_REJECT"
    if mt == "LONG_AI_FALLBACK":
        return "LONG_AI_FALLBACK_RANK_REJECT"
    if mt == "SHORT_AI_FALLBACK":
        return "SHORT_AI_FALLBACK_RANK_REJECT"
    return "RANK_REJECT"


def _allow_long_ai_below_min_rescue(sig: Dict[str, Any], ai_prob: float, ai_th: float) -> Tuple[bool, str]:
    """
    LONG の ai_below_min を限定救済する。
    AIを無効化するのではなく、実績で強かった組み合わせだけを通す。
    """

    try:
        side_u = str(sig.get("side", sig.get("direction", ""))).strip().upper()
        if side_u != "LONG":
            return False, ""

        btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
        rsi = _safe_float_or_nan(sig.get("rsi"))
        p1 = _safe_float_or_nan(sig.get("p1_score"))

        if btc_mode_compat != "DOWN":
            return False, ""

        if not np.isfinite(rsi) or not np.isfinite(p1):
            return False, ""

        if not (40.0 <= rsi < 45.0):
            return False, ""

        if not (1.5 <= p1 < 2.0):
            return False, ""

        reason = (
            "rescued_ai_below_min:"
            f"down_mode;"
            f"rsi={rsi:.4f};"
            f"p1={p1:.4f};"
            f"ai_prob={float(ai_prob):.4f};"
            f"ai_th={float(ai_th):.4f}"
        )
        return True, reason

    except Exception:
        return False, ""


def _allow_long_ai_bypass_rescue(sig: Dict[str, Any], dbg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    LONG の ai_bypass を限定救済する。
    全解除ではなく、最新 v2_shadow_ai 実績で強かった Down 反発帯だけを通す。
    """
    try:
        side_u = str(sig.get("side", sig.get("direction", ""))).strip().upper()
        if side_u != "LONG":
            return False, ""

        btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
        rsi = _safe_float_or_nan(sig.get("rsi"))
        p1 = _safe_float_or_nan(sig.get("p1_score"))
        p2 = _safe_float_or_nan(sig.get("p2_score"))
        p3 = _safe_float_or_nan(sig.get("p3_score"))

        if btc_mode_compat != "DOWN":
            return False, ""

        if not np.isfinite(rsi) or not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(p3):
            return False, ""

        if not (40.0 <= rsi < 44.5):
            return False, ""

        if not (1.5 <= p1 < 1.8):
            return False, ""

        if abs(float(p2)) > 1e-9:
            return False, ""

        if abs(float(p3) - (-0.2)) > 1e-9:
            return False, ""

        reason = (
            "rescued_ai_bypass "
            f"p1={float(p1):.4f} "
            f"p2={float(p2):.4f} "
            f"p3={float(p3):.4f} "
            f"rsi={float(rsi):.4f} "
            f"btc_mode={btc_mode_compat} "
            f"dbg={_v2_compact_dbg(dbg)}"
        )
        return True, reason

    except Exception:
        return False, ""


def _predict_v2_ai_score(sig: Dict[str, Any], side: str, fallback_score: float) -> Dict[str, Any]:
    """
    V2 sig から事前特徴量だけを使って side別AI推論。
    欠損は埋めない。safe_predict_proba() に渡して bypass させる。
    """
    side_u = str(side or "").strip().upper()
    sym_code = _v2_sig_symbol_code(sig)

    sig_for_feats = dict(sig)
    sig_for_feats["side"] = side_u
    feats = _build_v2_ai_feature_frame(sig_for_feats)

    threshold = V2_LONG_AI_MIN if side_u == "LONG" else V2_SHORT_AI_MIN

    model_for_side, model_ver_side, model_src_side = get_ai_model_for_side(side_u, sym_code)
    invert_for_side = _get_side_model_invert(side_u)

    base = {
        "side": side_u,
        "score": np.nan,
        "ok": False,
        "used_fallback": False,
        "model_version": model_ver_side or "",
        "model_source": model_src_side or "",
        "model_type": f"{side_u}_AI",
        "note": "",
    }

    if feats is None:
        if V2_AI_FAIL_OPEN and np.isfinite(fallback_score):
            base["score"] = float(fallback_score)
            base["ok"] = True
            base["used_fallback"] = True
            base["model_type"] = f"{side_u}_AI_FALLBACK"
            base["note"] = "v2_ai_feature_nan_rule_fallback"
            return base

        base["note"] = "v2_ai_feature_nan_precheck_skip"
        return base

    if model_for_side is None:
        if V2_AI_FAIL_OPEN and np.isfinite(fallback_score):
            base["score"] = float(fallback_score)
            base["ok"] = True
            base["used_fallback"] = True
            base["model_type"] = f"{side_u}_AI_FALLBACK"
            base["note"] = "ai_model_none_rule_fallback"
            return base
        base["note"] = "ai_model_none"
        return base

    proba_x, bypass_x, dbg_x = safe_predict_proba(model_for_side, feats)
    dbg = (dbg_x or {})
    if not isinstance(dbg, dict):
        dbg = {"dbg": str(dbg)}

    if bypass_x or proba_x is None:
        allow_rescue, rescue_reason = _allow_long_ai_bypass_rescue(sig, dbg)

        if allow_rescue and np.isfinite(fallback_score):
            base["score"] = float(fallback_score)
            base["ok"] = True
            base["used_fallback"] = False
            base["model_type"] = f"{side_u}_AI_RESCUE"
            base["note"] = rescue_reason
            return base

        if V2_AI_FAIL_OPEN and np.isfinite(fallback_score):
            base["score"] = float(fallback_score)
            base["ok"] = True
            base["used_fallback"] = True
            base["model_type"] = f"{side_u}_AI_FALLBACK"
            base["note"] = f"ai_bypass_rule_fallback:{_v2_compact_dbg(dbg)}"
            return base

        base["note"] = f"ai_bypass:{_v2_compact_dbg(dbg)}"
        return base

    try:
        p = np.asarray(proba_x, dtype=float)

        win_idx = 1
        cls_list = None
        try:
            cls = getattr(model_for_side, "classes_", None)
            cls_list = list(cls) if cls is not None else []
            if cls_list and (1 in cls_list):
                win_idx = cls_list.index(1)
        except Exception:
            win_idx = 1

        s_raw = float(p[0][win_idx])
        if not np.isfinite(s_raw):
            raise ValueError("non_finite_proba")

        s_used = (1.0 - s_raw) if invert_for_side else s_raw
        if not np.isfinite(s_used):
            raise ValueError("non_finite_score_used")

        if float(s_used) < float(threshold):
            allow_rescue, rescue_reason = _allow_long_ai_below_min_rescue(sig, float(s_used), float(threshold))

            if allow_rescue:
                base["score"] = float(s_used)
                base["ok"] = True
                base["note"] = (
                    f"rescued:{rescue_reason};"
                    f"ai={float(s_used):.6f};min={float(threshold):.6f};"
                    f"raw={float(s_raw):.6f};invert={int(bool(invert_for_side))}"
                )
                return base

            base["score"] = float(s_used)
            base["ok"] = False
            base["note"] = (
                f"ai_below_min ai={float(s_used):.6f};min={float(threshold):.6f};"
                f"raw={float(s_raw):.6f};invert={int(bool(invert_for_side))}"
            )
            return base

        base["score"] = float(s_used)
        base["ok"] = True
        base["note"] = (
            f"ai={float(s_used):.6f};min={float(threshold):.6f};"
            f"raw={float(s_raw):.6f};invert={int(bool(invert_for_side))};"
            f"win_idx={int(win_idx)}"
        )
        return base

    except Exception as e:
        if V2_AI_FAIL_OPEN and np.isfinite(fallback_score):
            base["score"] = float(fallback_score)
            base["ok"] = True
            base["used_fallback"] = True
            base["model_type"] = f"{side_u}_AI_FALLBACK"
            base["note"] = f"ai_parse_failed_rule_fallback err={type(e).__name__}:{e}"
            return base

        base["note"] = f"ai_parse_failed err={type(e).__name__}:{e}"
        return base


def _parse_hour_csv_to_set(csv_str: str) -> set:
    """'23,1,2,4' → {23, 1, 2, 4}"""
    result = set()
    for tok in str(csv_str or "").split(","):
        tok = tok.strip()
        if tok:
            try:
                result.add(int(float(tok)))
            except (ValueError, TypeError):
                pass
    return result


def _get_long_sig_hour(sig: Dict[str, Any]):
    """sigからJST時刻(int)を取得。取得できない場合はNone。"""
    h = _v2_sig_hour(sig)
    if np.isfinite(h):
        return int(h)
    return None


def _assess_long_ai_bad_regime(sig: Dict[str, Any]) -> Dict[str, Any]:
    """
    4条件をカウントして bad-regime を判定。
      - is_up_mode  : btc_mode_compat == "UP"
      - is_bad_hour : hour in V2_LONG_AI_BADREGIME_HOURS set
      - p2_weak     : p2_score <= V2_LONG_AI_BADREGIME_P2_MAX
      - p3_weak     : p3_score <= V2_LONG_AI_BADREGIME_P3_MAX
    hits >= STOP_HITS    → mode="STOP"
    hits >= CAUTION_HITS → mode="CAUTION"
    それ以外             → mode="NORMAL"
    戻り値: {"mode": str, "hit_count": int, "tags": list, "note": str}
    """
    if not V2_LONG_AI_BADREGIME_ENABLE:
        return {"mode": "NORMAL", "hit_count": 0, "tags": [], "note": "disabled"}

    hits = 0
    tags = []

    btc_mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    if btc_mode == "UP":
        hits += 1
        tags.append("up_mode")

    hour = _get_long_sig_hour(sig)
    if hour is not None:
        bad_hours = _parse_hour_csv_to_set(V2_LONG_AI_BADREGIME_HOURS)
        if hour in bad_hours:
            hits += 1
            tags.append(f"bad_hour:{hour}")

    p2 = _safe_float_or_nan(sig.get("p2_score"))
    if np.isfinite(p2) and p2 <= V2_LONG_AI_BADREGIME_P2_MAX:
        hits += 1
        tags.append(f"p2_weak:{p2:.4f}")

    p3 = _safe_float_or_nan(sig.get("p3_score"))
    if np.isfinite(p3) and p3 <= V2_LONG_AI_BADREGIME_P3_MAX:
        hits += 1
        tags.append(f"p3_weak:{p3:.4f}")

    if hits >= V2_LONG_AI_BADREGIME_STOP_HITS:
        mode = "STOP"
    elif hits >= V2_LONG_AI_BADREGIME_CAUTION_HITS:
        mode = "CAUTION"
    else:
        mode = "NORMAL"

    note = f"hits={hits};stop_th={V2_LONG_AI_BADREGIME_STOP_HITS};caution_th={V2_LONG_AI_BADREGIME_CAUTION_HITS}"
    return {"mode": mode, "hit_count": hits, "tags": tags, "note": note}


def _long_rank_rsi_bonus(sig: Dict[str, Any]) -> float:
    rsi = _safe_float_or_nan(sig.get("rsi"))
    if not np.isfinite(rsi):
        return 0.0

    if rsi <= 38.0:
        return 0.20
    if rsi <= 43.0:
        return 0.10
    if rsi <= 48.0:
        return 0.00
    return -0.10


def _long_rank_mode_penalty(sig: Dict[str, Any]) -> float:
    mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    if mode == "UP":
        return float(V2_LONG_RANK_MODE_UP_PENALTY)
    if mode == "RANGE":
        return float(V2_LONG_RANK_MODE_RANGE_PENALTY)
    return 0.0


def _long_rank_raw_rescue_penalty(sig: Dict[str, Any]) -> float:
    raw = sig.get("long_raw_rescue", None)
    s = ("" if raw is None else str(raw)).strip().lower()
    return float(V2_LONG_RANK_RAW_RESCUE_PENALTY) if s in {"1", "true", "yes", "on"} else 0.0


def _long_rank_blend_score(sig: Dict[str, Any]) -> float:
    score = 0.0

    base_ai = _safe_float_or_nan(sig.get("ai_prob_win"))
    qs = _safe_float_or_nan(sig.get("qs"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    p3 = _safe_float_or_nan(sig.get("p3_score"))

    if np.isfinite(base_ai):
        score += float(V2_LONG_RANK_W_AI) * float(base_ai)
    if np.isfinite(qs):
        score += float(V2_LONG_RANK_W_QS) * float(qs)
    if np.isfinite(p2):
        score += float(V2_LONG_RANK_W_P2) * float(p2)
    if np.isfinite(p3):
        score += float(V2_LONG_RANK_W_P3) * float(p3)

    rsi_bonus = _long_rank_rsi_bonus(sig)
    score += float(V2_LONG_RANK_W_RSI) * float(rsi_bonus)

    score -= _long_rank_mode_penalty(sig)
    score -= _long_rank_raw_rescue_penalty(sig)

    return float(score)


def rank_and_select_long(long_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LONG候補を ai_prob_win 降順に並べる。
    追加仕様:
    - v2_shadow_ai の直近 LONG / DONE 実績からブレーキ状態を判定する
    - NORMAL  : 通常の TOP_N
    - CAUTION : TOP_N を probe 値まで絞る
    - STOP    : 通知は止めるが、上位 probe 件だけ観測用に残す
    """
    if not long_signals:
        return []

    def _long_rank_score(sig: Dict[str, Any]) -> float:
        base_ai = float(sig.get("ai_prob_win", 0.0) or 0.0)

        bad = _assess_long_ai_bad_regime(sig)
        tags = list(bad.get("tags", []) or [])

        penalty = 0.0
        for t in tags:
            if str(t).startswith("p2_weak:"):
                penalty += float(V2_LONG_RANK_P2_WEAK_PENALTY)
            elif str(t).startswith("p3_weak:"):
                penalty += float(V2_LONG_RANK_P3_WEAK_PENALTY)
            elif str(t).startswith("bad_hour:"):
                penalty += float(V2_LONG_RANK_BAD_HOUR_PENALTY)

        if not V2_LONG_RANK_BLEND_ENABLE:
            return float(base_ai - penalty)

        blend_score = _long_rank_blend_score(sig)
        return float(blend_score - penalty)

    ranked = sorted(
        long_signals,
        key=lambda x: _long_rank_score(x),
        reverse=True,
    )

    base_top_n = max(1, int(V2_LONG_RANK_TOP_N))
    probe_top_n = max(1, int(V2_LONG_BRAKE_PROBE_TOP_N))
    brake = get_v2_long_brake_state(datetime.now(JST))
    brake_mode = str(brake.get("mode", "NORMAL") or "NORMAL").upper()

    if brake_mode == "STOP":
        effective_top_n = 0
        observe_top_n = probe_top_n
    elif brake_mode == "CAUTION":
        effective_top_n = max(1, int(brake.get("effective_top_n", probe_top_n) or probe_top_n))
        observe_top_n = effective_top_n
    else:
        effective_top_n = base_top_n
        observe_top_n = effective_top_n

    wr = brake.get("wr", float("nan"))
    avg_pnl = brake.get("avg_pnl", float("nan"))
    wr_str = "nan" if pd.isna(wr) else f"{float(wr):.1f}"
    pnl_str = "nan" if pd.isna(avg_pnl) else f"{float(avg_pnl):+.4f}"
    slot_list = "|".join(brake.get("slots", []) or [])

    print(
        f"[V2-LONG-BRAKE] mode={brake_mode} "
        f"base_top_n={base_top_n} effective_top_n={effective_top_n} observe_top_n={observe_top_n} "
        f"n={int(brake.get('n', 0) or 0)} slots={int(brake.get('slot_count', 0) or 0)} "
        f"wr={wr_str} avg_pnl={pnl_str} reason={brake.get('reason', '')} "
        f"slot_list={slot_list}"
    )

    if (not V2_LONG_RANK_ENABLE) and brake_mode == "NORMAL":
        for rank_idx, sig in enumerate(ranked, start=1):
            sig["_long_rank_pos"] = int(rank_idx)
            slot_total = len(ranked)
            current_note = str(sig.get("ai_note", "") or "")
            sig["ai_band"] = "RULE_QS_NO_RANK"
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n=ALL;selected=true;rank_disabled=true;"
                f"brake_mode={brake_mode};brake_wr={wr_str};brake_pnl={pnl_str};"
                f"brake_n={int(brake.get('n', 0) or 0)};brake_slots={slot_list};"
                f"brake_reason={brake.get('reason', '')}"
            )
        return ranked

    slot_total = len(ranked)

    for rank_idx, sig in enumerate(ranked, start=1):
        sig["_long_rank_pos"] = int(rank_idx)
        current_note = str(sig.get("ai_note", "") or "")
        rank_score = _long_rank_score(sig)

        if brake_mode == "STOP":
            sig["ai_pass"] = "0"
            if rank_idx <= observe_top_n:
                sig["ai_band"] = "LONG_BRAKE_PROBE"
                reject_reason = "long_brake_probe"
            else:
                sig["ai_band"] = "LONG_BRAKE_STOP"
                reject_reason = "long_brake_stop"
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n={effective_top_n};probe_top_n={observe_top_n};selected=false;reason={reject_reason};"
                f"brake_mode={brake_mode};brake_wr={wr_str};brake_pnl={pnl_str};"
                f"brake_n={int(brake.get('n', 0) or 0)};brake_slots={slot_list};"
                f"brake_reason={brake.get('reason', '')}"
            )
            continue

        if rank_idx <= effective_top_n:
            sig["ai_pass"] = "1"
            current_band = str(sig.get("ai_band", "") or "").strip().upper()
            if current_band in {"", "RULE_QS", "AI_RANK", "AI_FALLBACK_RANK"}:
                sig["ai_band"] = current_band if current_band else "RULE_QS"
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n={effective_top_n};selected=true;rank_score={rank_score:.6f};"
                f"brake_mode={brake_mode};brake_wr={wr_str};brake_pnl={pnl_str};"
                f"brake_n={int(brake.get('n', 0) or 0)};brake_slots={slot_list};"
                f"brake_reason={brake.get('reason', '')}"
            )
        else:
            sig["ai_pass"] = "0"
            if brake_mode == "CAUTION":
                sig["ai_band"] = "LONG_BRAKE_CAUTION"
                reject_reason = "long_brake_caution"
            else:
                sig["ai_band"] = "RANK_REJECT"
                reject_reason = "rank_exceeded"
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n={effective_top_n};selected=false;rank_score={rank_score:.6f};reason={reject_reason};"
                f"brake_mode={brake_mode};brake_wr={wr_str};brake_pnl={pnl_str};"
                f"brake_n={int(brake.get('n', 0) or 0)};brake_slots={slot_list};"
                f"brake_reason={brake.get('reason', '')}"
            )

    return ranked


def v2_selection_pipeline(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage A -> B -> C
    SHORT:
        defensive_filter -> calc_quality_score -> V2 side-AI score -> rank_and_select
    LONG:
        defensive_filter_long -> calc_long_quality_score -> V2 side-AI score -> rank_and_select_long

    REJECT も rank落ちも記録用に残す。
    """
    if not signals:
        return []

    output: List[Dict[str, Any]] = []
    short_passed: List[Dict[str, Any]] = []
    long_passed: List[Dict[str, Any]] = []

    for sig in signals:
        direction = str(sig.get("direction", "")).strip().upper()

        if direction == "SHORT":
            ok, reason = defensive_filter(sig)

            if not ok:
                sig["ai_prob_win"] = ""
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = "QS_v1"
                sig["ai_model_type"] = "SHORT_RULE_RANK"
                sig["ai_note"] = f"REJECTED:{reason}"
                output.append(sig)
                print(f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=SHORT reason={reason}")
                continue

            qs = calc_quality_score(sig)

            if qs < V2_SHORT_QS_MIN:
                sig["ai_prob_win"] = round(qs, 6)
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = "QS_v1"
                sig["ai_model_type"] = "SHORT_RULE_RANK"
                sig["ai_note"] = f"REJECTED:qs_below_min qs={qs:.4f} min={V2_SHORT_QS_MIN:.4f}"
                output.append(sig)
                print(
                    f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=SHORT "
                    f"reason=qs_below_min qs={qs:.4f} min={V2_SHORT_QS_MIN:.4f}"
                )
                continue

            ai_res = _predict_v2_ai_score(sig, "SHORT", qs)

            if not bool(ai_res.get("ok")):
                ai_score = ai_res.get("score", np.nan)
                sig["ai_prob_win"] = round(float(ai_score), 6) if np.isfinite(ai_score) else ""
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = str(ai_res.get("model_version", "") or "")
                sig["ai_model_type"] = str(ai_res.get("model_type", "SHORT_AI") or "SHORT_AI")
                sig["ai_note"] = (
                    f"REJECTED:{ai_res.get('note', 'ai_reject')};"
                    f"qs={qs:.4f};"
                    f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')};"
                    f"total={_fmt_sig_num_or_blank(sig, 'total_score')};"
                    f"btcstr={_fmt_sig_num_or_blank(sig, 'btc_htf_strength')};"
                    f"rsi={sig.get('rsi', '')}"
                )
                output.append(sig)
                print(f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=SHORT reason={ai_res.get('note', 'ai_reject')}")
                continue

            ai_score = float(ai_res["score"])
            sig["ai_prob_win"] = round(ai_score, 6)
            sig["ai_pass"] = "1"
            sig["ai_band"] = "SHORT_AI_SCORE" if not ai_res.get("used_fallback") else "SHORT_AI_FALLBACK_SCORE"
            sig["ai_model_version"] = str(ai_res.get("model_version", "") or "")
            sig["ai_model_type"] = str(ai_res.get("model_type", "SHORT_AI") or "SHORT_AI")
            sig["ai_note"] = (
                f"{ai_res.get('note', '')};"
                f"qs={qs:.4f};"
                f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')};"
                f"total={_fmt_sig_num_or_blank(sig, 'total_score')};"
                f"btcstr={_fmt_sig_num_or_blank(sig, 'btc_htf_strength')};"
                f"rsi={sig.get('rsi', '')}"
            )

            short_passed.append(sig)
            continue

        if direction == "LONG":
            research_tags = _get_long_research_tags(sig)
            research_note = "|".join(research_tags)
            rescued_p1 = _allow_long_p1_rescue(sig)
            ok, reason = defensive_filter_long(sig)

            if not ok:
                sig["ai_prob_win"] = ""
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = "LONG_QS_v2"
                sig["ai_model_type"] = "LONG_RULE_RANK"
                sig["ai_note"] = f"REJECTED:{reason};rescued_p1={rescued_p1};research_tags={research_note}"
                output.append(sig)
                print(f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=LONG reason={reason}")
                continue

            qs = calc_long_quality_score(sig)
            sig["qs"] = round(float(qs), 6) if np.isfinite(qs) else ""

            if not np.isfinite(qs):
                sig["ai_prob_win"] = ""
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = "LONG_QS_v2"
                sig["ai_model_type"] = "LONG_RULE_RANK"
                sig["ai_note"] = f"REJECTED:missing_qs_feature;rescued_p1={rescued_p1};research_tags={research_note}"
                output.append(sig)
                print(f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=LONG reason=missing_qs_feature")
                continue

            if qs < V2_LONG_QS_MIN:
                sig["ai_prob_win"] = round(qs, 6)
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = "LONG_QS_v2"
                sig["ai_model_type"] = "LONG_RULE_RANK"
                sig["ai_note"] = (
                    f"REJECTED:qs_below_min qs={qs:.4f} min={V2_LONG_QS_MIN:.4f};"
                    f"rescued_p1={rescued_p1};"
                    f"research_tags={research_note}"
                )
                output.append(sig)
                print(
                    f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=LONG "
                    f"reason=qs_below_min qs={qs:.4f} min={V2_LONG_QS_MIN:.4f}"
                )
                continue

            bad_regime = _assess_long_ai_bad_regime(sig)
            sig["_long_bad_regime_mode"] = str(bad_regime.get("mode", "NORMAL") or "NORMAL").upper()
            sig["_long_bad_regime_hits"] = int(bad_regime.get("hit_count", 0) or 0)
            sig["_long_bad_regime_tags"] = list(bad_regime.get("tags", []) or [])

            bad_tags_text = "|".join(str(x) for x in sig["_long_bad_regime_tags"]) if sig["_long_bad_regime_tags"] else "none"
            bad_regime_note = (
                f"bad_mode={sig['_long_bad_regime_mode']};"
                f"bad_hits={sig['_long_bad_regime_hits']};"
                f"bad_tags={bad_tags_text};"
                f"bad_note={bad_regime.get('note', '')}"
            )

            print(
                f"[V2-LONG-BADREGIME] sym={sig.get('symbol')} "
                f"mode={sig['_long_bad_regime_mode']} "
                f"hits={sig['_long_bad_regime_hits']} "
                f"tags={bad_tags_text} "
                f"btc_mode={sig.get('btc_mode_compat', '')} "
                f"hour={sig.get('hour', '')} "
                f"p2={_fmt_sig_num_or_blank(sig, 'p2_score')} "
                f"p3={_fmt_sig_num_or_blank(sig, 'p3_score')}"
            )

            if sig["_long_bad_regime_mode"] == "STOP":
                sig["ai_prob_win"] = ""
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = "LONG_QS_v2"
                sig["ai_model_type"] = "LONG_RULE_RANK"
                sig["ai_note"] = (
                    f"REJECTED:bad_regime_stop;"
                    f"{bad_regime_note};"
                    f"qs={qs:.4f};"
                    f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')};"
                    f"p2={_fmt_sig_num_or_blank(sig, 'p2_score')};"
                    f"p3={_fmt_sig_num_or_blank(sig, 'p3_score')};"
                    f"btcstr={_fmt_sig_num_or_blank(sig, 'btc_htf_strength')};"
                    f"btc_mode={sig.get('btc_mode_compat', '')};"
                    f"hour={sig.get('hour', '')};"
                    f"rsi={sig.get('rsi', '')};"
                    f"rescued_p1={rescued_p1};"
                    f"research_tags={research_note}"
                )
                output.append(sig)
                print(
                    f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=LONG "
                    f"reason=bad_regime_stop;{bad_regime_note}"
                )
                continue

            ai_res = _predict_v2_ai_score(sig, "LONG", qs)

            if sig["_long_bad_regime_mode"] == "CAUTION":
                caution_penalty = float(V2_LONG_AI_BADREGIME_CAUTION_PENALTY)
                caution_note = (
                    f"{bad_regime_note};"
                    f"bad_penalty={caution_penalty:.4f}"
                )

                if np.isfinite(float(ai_res.get("score", np.nan))) and bool(ai_res.get("ok")):
                    raw_score_before_penalty = float(ai_res["score"])
                    penalized_score = max(0.0, raw_score_before_penalty - caution_penalty)
                    ai_res["score"] = float(penalized_score)
                    ai_res["note"] = (
                        f"{ai_res.get('note', '')};"
                        f"{caution_note};"
                        f"raw_before_penalty={raw_score_before_penalty:.6f}"
                    )

                    if penalized_score < float(V2_LONG_AI_MIN):
                        ai_res["ok"] = False
                        ai_res["note"] = f"{ai_res.get('note', '')};bad_regime_caution_below_min"
                else:
                    ai_res["note"] = f"{ai_res.get('note', '')};{caution_note};penalty_not_applied"
            else:
                ai_res["note"] = f"{ai_res.get('note', '')};{bad_regime_note}"

            if not bool(ai_res.get("ok")):
                ai_score = ai_res.get("score", np.nan)
                sig["ai_prob_win"] = round(float(ai_score), 6) if np.isfinite(ai_score) else ""
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = str(ai_res.get("model_version", "") or "")
                sig["ai_model_type"] = str(ai_res.get("model_type", "LONG_AI") or "LONG_AI")
                sig["ai_note"] = (
                    f"REJECTED:{ai_res.get('note', 'ai_reject')};"
                    f"qs={qs:.4f};"
                    f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')};"
                    f"p2={_fmt_sig_num_or_blank(sig, 'p2_score')};"
                    f"p3={_fmt_sig_num_or_blank(sig, 'p3_score')};"
                    f"btcstr={_fmt_sig_num_or_blank(sig, 'btc_htf_strength')};"
                    f"btc_mode={sig.get('btc_mode_compat', '')};"
                    f"hour={sig.get('hour', '')};"
                    f"rsi={sig.get('rsi', '')};"
                    f"rescued_p1={rescued_p1};"
                    f"research_tags={research_note}"
                )
                output.append(sig)
                print(
                    f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=LONG "
                    f"reason={ai_res.get('note', 'ai_reject')}"
                )
                continue

            ai_score = float(ai_res["score"])
            sig["ai_prob_win"] = round(ai_score, 6)
            sig["ai_pass"] = "1"
            sig["ai_band"] = "LONG_AI_SCORE" if not ai_res.get("used_fallback") else "LONG_AI_FALLBACK_SCORE"
            sig["ai_model_version"] = str(ai_res.get("model_version", "") or "")
            sig["ai_model_type"] = str(ai_res.get("model_type", "LONG_AI") or "LONG_AI")
            sig["ai_note"] = (
                f"{ai_res.get('note', '')};"
                f"qs={qs:.4f};"
                f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')};"
                f"p2={_fmt_sig_num_or_blank(sig, 'p2_score')};"
                f"p3={_fmt_sig_num_or_blank(sig, 'p3_score')};"
                f"btcstr={_fmt_sig_num_or_blank(sig, 'btc_htf_strength')};"
                f"btc_mode={sig.get('btc_mode_compat', '')};"
                f"hour={sig.get('hour', '')};"
                f"rsi={sig.get('rsi', '')};"
                f"rescued_p1={rescued_p1};"
                f"research_tags={research_note}"
            )

            gate_ok, gate_reason = _long_notify_gate(sig)
            if not gate_ok:
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_note"] = (
                    f"REJECTED:{gate_reason};"
                    f"{ai_res.get('note', '')};"
                    f"qs={qs:.4f};"
                    f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')};"
                    f"p2={_fmt_sig_num_or_blank(sig, 'p2_score')};"
                    f"p3={_fmt_sig_num_or_blank(sig, 'p3_score')};"
                    f"btcstr={_fmt_sig_num_or_blank(sig, 'btc_htf_strength')};"
                    f"btc_mode={sig.get('btc_mode_compat', '')};"
                    f"hour={sig.get('hour', '')};"
                    f"rsi={sig.get('rsi', '')};"
                    f"rescued_p1={rescued_p1};"
                    f"research_tags={research_note}"
                )
                output.append(sig)
                print(
                    f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=LONG "
                    f"reason={gate_reason}"
                )
                continue

            print(
                f"[V2-LONG-AI] sym={sig.get('symbol')} "
                f"ai_prob_win={sig.get('ai_prob_win', '')} "
                f"band={sig.get('ai_band', '')} "
                f"{bad_regime_note}"
            )

            long_passed.append(sig)
            continue

        sig["ai_prob_win"] = ""
        sig["ai_pass"] = ""
        sig["ai_band"] = "COMMON"
        sig["ai_model_version"] = ""
        sig["ai_model_type"] = "COMMON"
        sig["ai_note"] = ""
        output.append(sig)

    ranked_shorts = rank_and_select(short_passed)
    ranked_longs = rank_and_select_long(long_passed)

    output.extend(ranked_shorts)
    output.extend(ranked_longs)

    snapshot = get_v2_regime_health_snapshot(datetime.now(JST))
    output = v2_apply_regime_notify_policy(output, snapshot)

    try:
        print(
            "[V2-REGIME-SNAPSHOT] "
            f"SHORT mode={snapshot['SHORT']['target_mode']} "
            f"enabled={snapshot['SHORT']['enabled']} "
            f"reason={snapshot['SHORT']['reason']} "
            f"selected_n={snapshot['SHORT']['selected']['n']} "
            f"control_n={snapshot['SHORT']['control']['n']} "
            f"wr_uplift={snapshot['SHORT']['uplift']['win_rate']} "
            f"pnl_uplift={snapshot['SHORT']['uplift']['avg_pnl']} | "
            f"LONG mode={snapshot['LONG']['target_mode']} "
            f"enabled={snapshot['LONG']['enabled']} "
            f"reason={snapshot['LONG']['reason']} "
            f"selected_n={snapshot['LONG']['selected']['n']} "
            f"control_n={snapshot['LONG']['control']['n']} "
            f"wr_uplift={snapshot['LONG']['uplift']['win_rate']} "
            f"pnl_uplift={snapshot['LONG']['uplift']['avg_pnl']}"
        )
    except Exception:
        pass

    return output


# ==========================================
# V2 Shadow シートへの書き込み
# ==========================================

def v2_build_shadow_row(sig: Dict) -> List[Any]:
    """
    V2専用ヘッダーの行を構築。
    将来の V2-AI / LONG-AI / SHORT-AI を載せられるように AI列も保持する。
    """
    return [
        "'" + sig["dt"].strftime("%Y-%m-%d %H:%M:%S"),
        sig["symbol"],
        sig["direction"],
        float(sig["entry_price"]),
        # スコア
        float(sig["total_score"]),
        float(sig["p1_score"]),
        float(sig["p2_score"]),
        float(sig["p3_score"]),
        # Pillar1 詳細
        sig["sym_htf_dir"],
        float(sig["sym_htf_strength"]),
        sig["btc_htf_dir"],
        float(sig["btc_htf_strength"]),
        sig["ltf_aligned"],
        json.dumps(sig["ltf_reasons"], ensure_ascii=False) if sig["ltf_reasons"] else "",
        # Pillar2 詳細
        sig["funding_rate"] if sig["funding_rate"] is not None else "",
        sig["fr_available"],
        # Pillar3 詳細
        sig["vol_ratio"] if sig["vol_ratio"] is not None else "",
        sig["vol_confirmed"],
        # TP/SL
        float(sig["tp"]),
        float(sig["sl"]),
        float(sig["tp_pct"]),
        float(sig["sl_pct"]),
        float(sig["atr"]),
        # 分析用
        float(sig["rsi"]) if np.isfinite(sig["rsi"]) else "",
        int(sig["hour"]),
        sig["btc_mode_compat"],
        # AI用（未実装時は空欄で保存）
        sig.get("ai_prob_win", ""),
        sig.get("ai_pass", ""),
        sig.get("ai_band", ""),
        sig.get("ai_model_version", ""),
        sig.get("ai_model_type", ""),
        sig.get("ai_note", ""),
        # 勝敗（後で judge が埋める）
        "", "", "", "", "", "", "",
        # メタ
        sig.get("v2_version", "V2-Shadow"),
        sig.get("note", ""),
    ]


def v2_ensure_shadow_sheet(svc, spreadsheet_id: str):
    """v2_shadow シートが無ければ作成し、ヘッダーを書く。"""
    ensure_sheet_exists(V2_SHADOW_SHEET, min_rows=5000, min_cols=len(V2_HEADERS) + 5)
    current = read_header_row(V2_SHADOW_SHEET)
    if current != V2_HEADERS:
        write_header_row(V2_SHADOW_SHEET, V2_HEADERS)
        print(f"[V2] shadow sheet headers written: {V2_SHADOW_SHEET}")


def v2_write_shadow_rows(rows: List[List[Any]]):
    """v2_shadow シートに行を追記（Datetime_JST + Symbol + Direction で重複排除）。"""
    if not rows:
        return

    try:
        # 既存シートの直近キーを拾う（Datetime_JST, Symbol, Direction）
        existing = set()
        try:
            service = get_sheet_service()
            last_row = _get_row_count_cached(V2_SHADOW_SHEET)
            if last_row >= 2:
                start = max(2, last_row - int(DEDUP_LOOKBACK_ROWS) + 1)
                res = service.spreadsheets().values().get(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f"{V2_SHADOW_SHEET}!A{start}:C{last_row}",
                ).execute()
                vals = res.get("values", []) or []
                for r in vals:
                    dt_raw = str(r[0]).strip() if len(r) > 0 else ""
                    sym = str(r[1]).strip() if len(r) > 1 else ""
                    direction = str(r[2]).strip().upper() if len(r) > 2 else ""
                    if not dt_raw or not sym or not direction:
                        continue
                    dt_key = normalize_dt_str(dt_raw)
                    existing.add(f"{sym}|{dt_key}|{direction}")
        except Exception as e:
            print(f"[WARN] v2 shadow dedup pre-scan failed: {e}")
            existing = set()

        # 今回書く分も重複排除（同じバッチ内の重複も除外）
        filtered: List[List[Any]] = []
        for row in rows:
            dt_raw = str(row[0]).strip() if len(row) > 0 else ""
            sym = str(row[1]).strip() if len(row) > 1 else ""
            direction = str(row[2]).strip().upper() if len(row) > 2 else ""
            if not dt_raw or not sym or not direction:
                continue
            dt_key = normalize_dt_str(dt_raw)
            k = f"{sym}|{dt_key}|{direction}"
            if k in existing:
                continue
            existing.add(k)
            filtered.append(row)

        if not filtered:
            print(f"[V2] wrote 0 rows to {V2_SHADOW_SHEET} (all deduped)")
            return

        append_rows_to_sheet(V2_SHADOW_SHEET, filtered, V2_HEADERS)
        print(f"[V2] wrote {len(filtered)} rows to {V2_SHADOW_SHEET} (deduped)")

    except Exception as e:
        print(f"[V2-ERR] shadow write failed: {e}")


# ==========================================
# V2 Shadow 分析レポート
# ==========================================

def get_v2_shadow_ai_data() -> pd.DataFrame:
    """v2_shadow_ai シートを全件取得して DataFrame で返す（列名ベース）。"""
    service = get_sheet_service()

    hdr_res = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A1:{HEADER_COL_END}1",
    ).execute()
    raw_hdr = (hdr_res.get("values", [[]]) or [[]])[0]
    headers = _normalize_headers(raw_hdr)

    if not headers:
        return pd.DataFrame()

    data_res = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A2:{HEADER_COL_END}",
    ).execute()
    rows = data_res.get("values", []) or []

    if not rows:
        return pd.DataFrame(columns=headers)

    n_cols = len(headers)
    fixed = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
    return pd.DataFrame(fixed, columns=headers)


def _v2_monitor_prepare_done(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "Datetime_JST" not in work.columns:
        return pd.DataFrame()

    work["Datetime_JST"] = pd.to_datetime(work["Datetime_JST"].astype(str).str.strip(), errors="coerce")
    work = work[work["Datetime_JST"].notna()].copy()

    if "EvalStatus" in work.columns:
        work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
        work = work[work["EvalStatus"] == "DONE"].copy()

    if "Direction" in work.columns:
        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
    else:
        work["Direction"] = ""

    if "WinLose" in work.columns:
        wl = work["WinLose"].astype(str).str.strip().str.lower()
        work["_win"] = wl.isin(["win", "w", "1", "true"])
        work["_lose"] = wl.isin(["lose", "l", "0", "false"])
    else:
        work["_win"] = False
        work["_lose"] = False

    if "PnL_Pct" in work.columns:
        work["_pnl"] = pd.to_numeric(
            work["PnL_Pct"].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
    else:
        work["_pnl"] = np.nan

    if "AI_Band" in work.columns:
        work["AI_Band"] = work["AI_Band"].astype(str).str.strip().str.upper()
    else:
        work["AI_Band"] = ""

    note_col = None
    for c in ["AI_Note", "ai_note"]:
        if c in work.columns:
            note_col = c
            break

    if note_col:
        note = work[note_col].astype(str)
        work["_notify_1"] = note.str.contains("notify_pass=1", na=False)
    else:
        work["_notify_1"] = False

    return work


def _v2_monitor_stats(sub: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(sub))
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg_pnl": np.nan}

    wins = int(sub["_win"].sum()) if "_win" in sub.columns else 0
    wr = float(wins / n) if n > 0 else np.nan
    pnl = pd.to_numeric(sub["_pnl"], errors="coerce")
    avg_pnl = float(pnl.dropna().mean()) if len(pnl.dropna()) > 0 else np.nan
    return {"n": n, "wr": wr, "avg_pnl": avg_pnl}


def _get_v2_window_report(hours: int, now_jst: Optional[datetime] = None) -> Dict[str, Any]:
    now_jst = now_jst or datetime.now(JST)

    out = {
        "hours": int(hours),
        "notify": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "rank_reject": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "rejected": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "wr_gap_vs_rank_reject": np.nan,
        "pnl_gap_vs_rank_reject": np.nan,
    }

    df = get_v2_shadow_ai_data()
    done = _v2_monitor_prepare_done(df)
    if done.empty:
        return out

    since = pd.Timestamp(now_jst.replace(tzinfo=None)) - pd.Timedelta(hours=int(hours))
    done = done[done["Datetime_JST"] >= since].copy()
    done = done[done["Direction"] == "LONG"].copy()

    if done.empty:
        return out

    notify_sub = done[done["_notify_1"]].copy()
    rank_reject_sub = done[done["AI_Band"].eq("RANK_REJECT")].copy()
    rejected_sub = done[done["AI_Band"].eq("REJECTED")].copy()

    notify_stats = _v2_monitor_stats(notify_sub)
    rank_reject_stats = _v2_monitor_stats(rank_reject_sub)
    rejected_stats = _v2_monitor_stats(rejected_sub)

    out["notify"] = notify_stats
    out["rank_reject"] = rank_reject_stats
    out["rejected"] = rejected_stats

    if np.isfinite(notify_stats["wr"]) and np.isfinite(rank_reject_stats["wr"]):
        out["wr_gap_vs_rank_reject"] = float(notify_stats["wr"] - rank_reject_stats["wr"])
    if np.isfinite(notify_stats["avg_pnl"]) and np.isfinite(rank_reject_stats["avg_pnl"]):
        out["pnl_gap_vs_rank_reject"] = float(notify_stats["avg_pnl"] - rank_reject_stats["avg_pnl"])

    return out


def get_v2_guardrail_state(now_jst: Optional[datetime] = None) -> Dict[str, Any]:
    now_jst = now_jst or datetime.now(JST)

    state = {
        "enabled": bool(V2_GUARDRAIL_ENABLE),
        "mode": "NORMAL",
        "reason": "guardrail_disabled" if not V2_GUARDRAIL_ENABLE else "insufficient_data",
        "lookback_hours": int(V2_GUARDRAIL_LOOKBACK_HOURS),
        "notify": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "rank_reject": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "rejected": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "wr_gap_vs_rank_reject": np.nan,
        "pnl_gap_vs_rank_reject": np.nan,
        "effective_down_top_n": int(V2_LONG_RESCUE_NOTIFY_DOWN_TOP_N),
    }

    if not V2_GUARDRAIL_ENABLE:
        return state

    rep = _get_v2_window_report(int(V2_GUARDRAIL_LOOKBACK_HOURS), now_jst)
    state.update(rep)

    if (
        rep["notify"]["n"] < int(V2_GUARDRAIL_MIN_DONE)
        or rep["rank_reject"]["n"] < int(V2_GUARDRAIL_MIN_DONE)
        or rep["rejected"]["n"] < int(V2_GUARDRAIL_MIN_DONE)
    ):
        state["reason"] = "insufficient_data"
        return state

    wr_gap = rep["wr_gap_vs_rank_reject"]
    pnl_gap = rep["pnl_gap_vs_rank_reject"]

    wr_gap_vs_rejected = np.nan
    pnl_gap_vs_rejected = np.nan

    if np.isfinite(rep["notify"]["wr"]) and np.isfinite(rep["rejected"]["wr"]):
        wr_gap_vs_rejected = float(rep["notify"]["wr"] - rep["rejected"]["wr"])

    if np.isfinite(rep["notify"]["avg_pnl"]) and np.isfinite(rep["rejected"]["avg_pnl"]):
        pnl_gap_vs_rejected = float(rep["notify"]["avg_pnl"] - rep["rejected"]["avg_pnl"])

    stop_hit = (
        (np.isfinite(wr_gap) and wr_gap <= float(V2_GUARDRAIL_STOP_WR_GAP))
        or (np.isfinite(pnl_gap) and pnl_gap <= float(V2_GUARDRAIL_STOP_PNL_GAP))
        or (np.isfinite(wr_gap_vs_rejected) and wr_gap_vs_rejected <= float(V2_GUARDRAIL_STOP_WR_GAP))
        or (np.isfinite(pnl_gap_vs_rejected) and pnl_gap_vs_rejected <= float(V2_GUARDRAIL_STOP_PNL_GAP))
    )

    if stop_hit:
        state["mode"] = "STOP"
        state["reason"] = (
            f"stop wr_gap={wr_gap} pnl_gap={pnl_gap} "
            f"wr_gap_vs_rejected={wr_gap_vs_rejected} "
            f"pnl_gap_vs_rejected={pnl_gap_vs_rejected}"
        )
        state["effective_down_top_n"] = 0
        return state

    caution_hit = (
        (np.isfinite(wr_gap) and wr_gap <= float(V2_GUARDRAIL_CAUTION_WR_GAP))
        or (np.isfinite(pnl_gap) and pnl_gap <= float(V2_GUARDRAIL_CAUTION_PNL_GAP))
        or (np.isfinite(wr_gap_vs_rejected) and wr_gap_vs_rejected <= float(V2_GUARDRAIL_CAUTION_WR_GAP))
        or (np.isfinite(pnl_gap_vs_rejected) and pnl_gap_vs_rejected <= float(V2_GUARDRAIL_CAUTION_PNL_GAP))
    )

    if caution_hit:
        state["mode"] = "CAUTION"
        state["reason"] = (
            f"caution wr_gap={wr_gap} pnl_gap={pnl_gap} "
            f"wr_gap_vs_rejected={wr_gap_vs_rejected} "
            f"pnl_gap_vs_rejected={pnl_gap_vs_rejected}"
        )
        state["effective_down_top_n"] = min(
            int(V2_LONG_RESCUE_NOTIFY_DOWN_TOP_N),
            int(V2_GUARDRAIL_CAUTION_TOP_N),
        )
        return state

    state["mode"] = "NORMAL"
    state["reason"] = (
        f"normal wr_gap={wr_gap} pnl_gap={pnl_gap} "
        f"wr_gap_vs_rejected={wr_gap_vs_rejected} "
        f"pnl_gap_vs_rejected={pnl_gap_vs_rejected}"
    )
    return state


def get_v2_retrain_review_state(now_jst: Optional[datetime] = None) -> Dict[str, Any]:
    now_jst = now_jst or datetime.now(JST)

    state = {
        "enabled": bool(V2_RETRAIN_REVIEW_ENABLE),
        "should_retrain": False,
        "reason": "retrain_review_disabled" if not V2_RETRAIN_REVIEW_ENABLE else "insufficient_data",
        "lookback_hours": int(V2_RETRAIN_LOOKBACK_HOURS),
        "notify": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "rank_reject": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "rejected": {"n": 0, "wr": np.nan, "avg_pnl": np.nan},
        "wr_gap_vs_rank_reject": np.nan,
        "pnl_gap_vs_rank_reject": np.nan,
    }

    if not V2_RETRAIN_REVIEW_ENABLE:
        return state

    rep = _get_v2_window_report(int(V2_RETRAIN_LOOKBACK_HOURS), now_jst)
    state.update(rep)

    if rep["notify"]["n"] < int(V2_RETRAIN_MIN_DONE):
        state["reason"] = "insufficient_data"
        return state

    wr_gap = rep["wr_gap_vs_rank_reject"]
    pnl_gap = rep["pnl_gap_vs_rank_reject"]

    if (
        (np.isfinite(wr_gap) and wr_gap <= float(V2_RETRAIN_TRIGGER_WR_GAP)) or
        (np.isfinite(pnl_gap) and pnl_gap <= float(V2_RETRAIN_TRIGGER_PNL_GAP))
    ):
        state["should_retrain"] = True
        state["reason"] = f"trigger wr_gap={wr_gap} pnl_gap={pnl_gap}"
        return state

    state["reason"] = f"no_trigger wr_gap={wr_gap} pnl_gap={pnl_gap}"
    return state


def _v2_shadow_last_per_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    if "Datetime_JST" not in work.columns or "Symbol" not in work.columns or "Direction" not in work.columns:
        return work

    work["__dt__"] = pd.to_datetime(work["Datetime_JST"], errors="coerce")
    work["__sym__"] = work["Symbol"].astype(str).str.strip().str.upper()
    work["__side__"] = work["Direction"].astype(str).str.strip().str.upper()

    work = work.sort_values("__dt__", kind="stable")
    work = work.drop_duplicates(subset=["__dt__", "__sym__", "__side__"], keep="last")
    return work


def _v2_health_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"n": 0, "win_rate": np.nan, "avg_pnl": np.nan}

    work = df.copy()
    winlose = work["WinLose"].apply(_normalize_v2_winlose) if "WinLose" in work.columns else pd.Series([], dtype=str)
    pnl = pd.to_numeric(work["PnL_Pct"], errors="coerce") if "PnL_Pct" in work.columns else pd.Series([], dtype=float)

    n = int(len(work))
    win_n = int((winlose == "Win").sum()) if len(winlose) else 0
    wr = (win_n / n) if n > 0 else np.nan
    avg_pnl = float(pnl.dropna().mean()) if len(pnl.dropna()) > 0 else np.nan

    return {
        "n": n,
        "win_rate": wr,
        "avg_pnl": avg_pnl,
    }


def _v2_selected_bands_for_side(side: str) -> set:
    side_u = str(side or "").strip().upper()
    if side_u == "SHORT":
        return {
            "SHORT_AI_RANK",
            "SHORT_AI_FALLBACK_RANK",
            "RULE_QS",
        }
    if side_u == "LONG":
        return {
            "LONG_AI_RANK",
            "LONG_AI_CAUTION_RANK",
            "LONG_AI_FALLBACK_RANK",
            "LONG_AI_CAUTION_FALLBACK_RANK",
            "RULE_QS",
        }
    return set()


def _v2_control_bands_for_side(side: str) -> set:
    side_u = str(side or "").strip().upper()
    if side_u == "SHORT":
        return {
            "SHORT_AI_RANK_REJECT",
            "SHORT_AI_FALLBACK_RANK_REJECT",
            "REJECTED",
            "RANK_REJECT",
        }
    if side_u == "LONG":
        return {
            "LONG_AI_RANK_REJECT",
            "LONG_AI_FALLBACK_RANK_REJECT",
            "REJECTED",
            "RANK_REJECT",
        }
    return set()


def get_v2_regime_health_snapshot(now_jst: Optional[datetime] = None) -> Dict[str, Any]:
    """
    v2_shadow_ai から regime内 uplift を計算して、
    LONG/SHORT の通知可否を決めるためのスナップショットを返す。
    """
    now_jst = now_jst or datetime.now(JST)

    out: Dict[str, Any] = {
        "SHORT": {
            "target_mode": str(V2_REGIME_SHORT_MODE),
            "enabled": False,
            "reason": "init",
            "selected": {"n": 0, "win_rate": np.nan, "avg_pnl": np.nan},
            "control": {"n": 0, "win_rate": np.nan, "avg_pnl": np.nan},
            "uplift": {"win_rate": np.nan, "avg_pnl": np.nan},
        },
        "LONG": {
            "target_mode": str(V2_REGIME_LONG_MODE),
            "enabled": False,
            "reason": "init",
            "selected": {"n": 0, "win_rate": np.nan, "avg_pnl": np.nan},
            "control": {"n": 0, "win_rate": np.nan, "avg_pnl": np.nan},
            "uplift": {"win_rate": np.nan, "avg_pnl": np.nan},
        },
    }

    try:
        df = get_v2_shadow_ai_data()
        if df is None or df.empty:
            out["SHORT"]["reason"] = "shadow_empty"
            out["LONG"]["reason"] = "shadow_empty"
            return out

        work = df.copy()
        if "Datetime_JST" not in work.columns:
            out["SHORT"]["reason"] = "missing_datetime"
            out["LONG"]["reason"] = "missing_datetime"
            return out

        work["Datetime_JST"] = pd.to_datetime(work["Datetime_JST"], errors="coerce")
        work = work.dropna(subset=["Datetime_JST"]).copy()

        if V2_REGIME_LOOKBACK_HOURS > 0:
            since = now_jst - timedelta(hours=int(V2_REGIME_LOOKBACK_HOURS))
            since = pd.Timestamp(since)
            if since.tzinfo is not None:
                since = since.tz_localize(None)
            work = work[work["Datetime_JST"] >= since].copy()

        if work.empty:
            out["SHORT"]["reason"] = "no_rows_in_window"
            out["LONG"]["reason"] = "no_rows_in_window"
            return out

        work = _v2_shadow_last_per_key(work)

        if "EvalStatus" in work.columns:
            work = work[work["EvalStatus"].astype(str).str.strip().str.upper() == "DONE"].copy()

        if "WinLose" in work.columns:
            work["WinLose"] = work["WinLose"].apply(_normalize_v2_winlose)
            work = work[work["WinLose"].isin(["Win", "Lose"])].copy()

        if work.empty:
            out["SHORT"]["reason"] = "no_done_rows"
            out["LONG"]["reason"] = "no_done_rows"
            return out

        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
        work["BTC_Mode_Compat"] = work["BTC_Mode_Compat"].astype(str).str.strip().str.upper()
        work["AI_Band"] = work["AI_Band"].astype(str).str.strip().str.upper()

        for side_u, target_mode in [("SHORT", str(V2_REGIME_SHORT_MODE)), ("LONG", str(V2_REGIME_LONG_MODE))]:
            sub = work[
                (work["Direction"] == side_u) &
                (work["BTC_Mode_Compat"] == target_mode)
            ].copy()

            if sub.empty:
                out[side_u]["reason"] = "no_rows_in_target_mode"
                continue

            selected_bands = _v2_selected_bands_for_side(side_u)
            control_bands = _v2_control_bands_for_side(side_u)

            selected = sub[sub["AI_Band"].isin(selected_bands)].copy()
            control = sub[sub["AI_Band"].isin(control_bands)].copy()

            s_stats = _v2_health_stats(selected)
            c_stats = _v2_health_stats(control)

            wr_uplift = np.nan
            pnl_uplift = np.nan
            if np.isfinite(s_stats["win_rate"]) and np.isfinite(c_stats["win_rate"]):
                wr_uplift = float(s_stats["win_rate"] - c_stats["win_rate"])
            if np.isfinite(s_stats["avg_pnl"]) and np.isfinite(c_stats["avg_pnl"]):
                pnl_uplift = float(s_stats["avg_pnl"] - c_stats["avg_pnl"])

            out[side_u]["selected"] = s_stats
            out[side_u]["control"] = c_stats
            out[side_u]["uplift"] = {
                "win_rate": wr_uplift,
                "avg_pnl": pnl_uplift,
            }

            if s_stats["n"] < int(V2_REGIME_MIN_DONE):
                out[side_u]["enabled"] = False
                out[side_u]["reason"] = f"selected_n_lt_min_done:{s_stats['n']}<{V2_REGIME_MIN_DONE}"
                continue

            if c_stats["n"] < int(V2_REGIME_MIN_DONE):
                out[side_u]["enabled"] = False
                out[side_u]["reason"] = f"control_n_lt_min_done:{c_stats['n']}<{V2_REGIME_MIN_DONE}"
                continue

            if not np.isfinite(wr_uplift) or wr_uplift < float(V2_REGIME_MIN_UPLIFT_WINRATE):
                out[side_u]["enabled"] = False
                out[side_u]["reason"] = (
                    f"wr_uplift_lt_min:{wr_uplift}<{V2_REGIME_MIN_UPLIFT_WINRATE}"
                )
                continue

            if not np.isfinite(pnl_uplift) or pnl_uplift < float(V2_REGIME_MIN_UPLIFT_PNL):
                out[side_u]["enabled"] = False
                out[side_u]["reason"] = (
                    f"pnl_uplift_lt_min:{pnl_uplift}<{V2_REGIME_MIN_UPLIFT_PNL}"
                )
                continue

            out[side_u]["enabled"] = True
            out[side_u]["reason"] = "enabled_by_regime_uplift"

        return out

    except Exception as e:
        out["SHORT"]["reason"] = f"snapshot_error:{type(e).__name__}:{e}"
        out["LONG"]["reason"] = f"snapshot_error:{type(e).__name__}:{e}"
        return out


_v2_short_pause_cache: Dict[str, Any] = {
    "ts": 0.0,
    "state": {
        "paused": False,
        "reason": "init",
        "reason_text": "",
        "n": 0,
        "win_rate": np.nan,
        "avg_pnl": np.nan,
    },
}


def _get_recent_short_strict_health(now_jst: Optional[datetime] = None) -> Dict[str, Any]:
    now_jst = now_jst or datetime.now(JST)
    now_ts = time.time()

    cached = _v2_short_pause_cache.get("state", {})
    if (now_ts - float(_v2_short_pause_cache.get("ts", 0.0))) <= float(V2_SHORT_PAUSE_CACHE_TTL_SEC):
        return dict(cached)

    out: Dict[str, Any] = {
        "paused": False,
        "reason": "init",
        "reason_text": "",
        "n": 0,
        "win_rate": np.nan,
        "avg_pnl": np.nan,
    }

    if not V2_SHORT_PAUSE_ENABLE:
        out["reason"] = "disabled"
        out["reason_text"] = "short_pause_disabled"
        _v2_short_pause_cache["ts"] = now_ts
        _v2_short_pause_cache["state"] = dict(out)
        return out

    try:
        df = get_v2_shadow_ai_data()
        if df is None or df.empty:
            out["reason"] = "shadow_empty"
            out["reason_text"] = "shadow_empty"
            _v2_short_pause_cache["ts"] = now_ts
            _v2_short_pause_cache["state"] = dict(out)
            return out

        work = df.copy()

        if "Datetime_JST" not in work.columns:
            out["reason"] = "missing_datetime"
            out["reason_text"] = "missing_datetime"
            _v2_short_pause_cache["ts"] = now_ts
            _v2_short_pause_cache["state"] = dict(out)
            return out

        work["Datetime_JST"] = pd.to_datetime(work["Datetime_JST"], errors="coerce")
        work = work.dropna(subset=["Datetime_JST"]).copy()

        since = now_jst - timedelta(hours=int(V2_SHORT_PAUSE_LOOKBACK_HOURS))
        since_ts = pd.Timestamp(since)
        if since_ts.tzinfo is not None:
            since_ts = since_ts.tz_localize(None)

        work = work[work["Datetime_JST"] >= since_ts].copy()

        if "EvalStatus" in work.columns:
            work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
            work = work[work["EvalStatus"] == "DONE"].copy()

        if {"Datetime_JST", "Symbol", "Direction"}.issubset(work.columns):
            work = work.sort_values("Datetime_JST").drop_duplicates(
                subset=["Datetime_JST", "Symbol", "Direction"],
                keep="last",
            )

        if "Direction" in work.columns:
            work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
            work = work[work["Direction"] == "SHORT"].copy()

        if "FR_Available" in work.columns:
            work["FR_Available"] = work["FR_Available"].apply(_normalize_bool01)
            work = work[work["FR_Available"] == 1.0].copy()
        else:
            out["reason"] = "missing_fr_available"
            out["reason_text"] = "missing_fr_available"
            _v2_short_pause_cache["ts"] = now_ts
            _v2_short_pause_cache["state"] = dict(out)
            return out

        for col in ("Hour_JST", "P1_TrendScore", "P2_FundingScore", "RSI", "PnL_Pct"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        if "BTC_Mode_Compat" in work.columns:
            work["BTC_Mode_Compat"] = work["BTC_Mode_Compat"].astype(str).str.strip().str.upper()
        else:
            out["reason"] = "missing_btc_mode"
            out["reason_text"] = "missing_btc_mode"
            _v2_short_pause_cache["ts"] = now_ts
            _v2_short_pause_cache["state"] = dict(out)
            return out

        work = work[
            work["Hour_JST"].isin(V2_SHORT_ALLOWED_HOURS)
            & work["BTC_Mode_Compat"].isin(V2_SHORT_BUCKET_BTC_MODES)
            & (work["P1_TrendScore"] >= float(V2_SHORT_BUCKET_P1_MIN))
            & (work["P1_TrendScore"] < float(V2_SHORT_BUCKET_P1_MAX))
            & (work["RSI"] >= float(V2_SHORT_BUCKET_RSI_MIN))
            & (work["RSI"] < float(V2_SHORT_BUCKET_RSI_MAX))
            & (work["P2_FundingScore"] >= float(V2_SHORT_BUCKET_P2_MIN))
        ].copy()

        out["n"] = int(len(work))
        if out["n"] <= 0:
            out["reason"] = "no_recent_strict_short"
            out["reason_text"] = "no_recent_strict_short"
            _v2_short_pause_cache["ts"] = now_ts
            _v2_short_pause_cache["state"] = dict(out)
            return out

        if "WinLose" not in work.columns:
            out["reason"] = "missing_winlose"
            out["reason_text"] = "missing_winlose"
            _v2_short_pause_cache["ts"] = now_ts
            _v2_short_pause_cache["state"] = dict(out)
            return out

        work["__win__"] = work["WinLose"].astype(str).str.strip().str.upper().eq("WIN")
        wr = float(work["__win__"].mean()) if len(work) > 0 else np.nan
        avg_pnl = float(work["PnL_Pct"].mean()) if len(work) > 0 else np.nan

        out["win_rate"] = wr
        out["avg_pnl"] = avg_pnl

        paused = (
            int(out["n"]) >= int(V2_SHORT_PAUSE_MIN_DONE)
            and (
                (np.isfinite(wr) and float(wr) < float(V2_SHORT_PAUSE_STOP_WINRATE))
                or (np.isfinite(avg_pnl) and float(avg_pnl) <= float(V2_SHORT_PAUSE_STOP_AVGPNL))
            )
        )

        out["paused"] = bool(paused)
        out["reason"] = "paused" if paused else "ok"
        out["reason_text"] = (
            f"n={int(out['n'])} "
            f"wr={'' if not np.isfinite(wr) else format(float(wr), '.4f')} "
            f"avg_pnl={'' if not np.isfinite(avg_pnl) else format(float(avg_pnl), '.4f')} "
            f"lookback_h={int(V2_SHORT_PAUSE_LOOKBACK_HOURS)}"
        )

    except Exception as e:
        out["paused"] = False
        out["reason"] = "error"
        out["reason_text"] = f"short_pause_check_error:{type(e).__name__}:{e}"

    _v2_short_pause_cache["ts"] = now_ts
    _v2_short_pause_cache["state"] = dict(out)
    return out


def v2_apply_regime_notify_policy(signals: List[Dict[str, Any]], snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    ai_pass は shadow上の「選抜結果」として残し、
    _notify_pass は実通知可否として別で持つ。
    これで「通知停止」と「shadow記録停止」を分離する。
    """
    out: List[Dict[str, Any]] = []
    guard_state = get_v2_guardrail_state(datetime.now(JST))

    for sig in signals:
        side = str(sig.get("direction", "")).strip().upper()
        mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
        ai_pass = str(sig.get("ai_pass", "")).strip()

        notify_pass = "0"
        notify_reason = "init"

        if ai_pass != "1":
            notify_pass = "0"
            notify_reason = "ai_pass_0"
        elif not V2_REGIME_POLICY_ENABLE:
            notify_pass = "1"
            notify_reason = "regime_policy_disabled"
        elif side not in {"SHORT", "LONG"}:
            notify_pass = "0"
            notify_reason = "unsupported_side"
        else:
            side_policy = snapshot.get(side, {}) or {}
            target_mode = str(side_policy.get("target_mode", "") or "").strip().upper()
            enabled = bool(side_policy.get("enabled", False))
            reason = str(side_policy.get("reason", "") or "")

            long_raw_rescue = str(sig.get("long_raw_rescue", "")).strip().lower() in ("1", "true", "yes", "on")
            short_raw_rescue = str(sig.get("short_raw_rescue", "")).strip().lower() in ("1", "true", "yes", "on")
            long_rank_pos = int(float(sig.get("_long_rank_pos", 9999) or 9999))

            if mode != target_mode:
                if side == "LONG" and long_raw_rescue and str(mode).upper() in {"DOWN", "RANGE"}:
                    hour = _get_long_sig_hour(sig)
                    blocked_hours = _parse_hour_csv_to_set(V2_LONG_RESCUE_NOTIFY_BLOCK_HOURS)

                    if guard_state.get("mode") == "STOP":
                        notify_pass = "0"
                        notify_reason = f"guardrail_stop:{guard_state.get('reason', '')}"
                    elif hour is not None and hour in blocked_hours:
                        notify_pass = "0"
                        notify_reason = f"long_rescue_blocked_bad_hour:{hour}"
                    else:
                        eff_top_n = int(guard_state.get("effective_down_top_n", V2_LONG_RESCUE_NOTIFY_DOWN_TOP_N))
                        if eff_top_n <= 0:
                            notify_pass = "0"
                            notify_reason = f"guardrail_top_n_zero:{guard_state.get('reason', '')}"
                        elif long_rank_pos > eff_top_n:
                            notify_pass = "0"
                            notify_reason = (
                                f"long_rescue_rank_cut:rank={long_rank_pos}>{eff_top_n};"
                                f"guard={guard_state.get('mode', 'NORMAL')}"
                            )
                        else:
                            notify_pass = "1"
                            notify_reason = (
                                f"notify_enabled_by_long_rescue:{mode}!={target_mode};"
                                f"rank={long_rank_pos};"
                                f"guard={guard_state.get('mode', 'NORMAL')}"
                            )
                elif side == "SHORT" and short_raw_rescue and str(mode).upper() in {"UP", "RANGE"}:
                    notify_pass = "1"
                    notify_reason = f"notify_enabled_by_short_rescue:{mode}!={target_mode}"
                else:
                    notify_pass = "0"
                    notify_reason = f"mode_mismatch:{mode}!={target_mode}"
            elif not enabled:
                notify_pass = "0"
                notify_reason = f"regime_off:{reason}"
            else:
                notify_pass = "1"
                notify_reason = "notify_enabled"

        sig["_notify_pass"] = notify_pass
        sig["_notify_reason"] = notify_reason

        cur_note = str(sig.get("ai_note", "") or "")
        if cur_note:
            sig["ai_note"] = f"{cur_note};notify_pass={notify_pass};notify_reason={notify_reason}"
        else:
            sig["ai_note"] = f"notify_pass={notify_pass};notify_reason={notify_reason}"

        out.append(sig)

    return out


def _normalize_bool01(x: Any):
    s = ("" if x is None else str(x)).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return 1.0
    if s in {"0", "false", "f", "no", "n", "off"}:
        return 0.0
    return np.nan


def _normalize_v2_winlose(x: Any) -> str:
    s = ("" if x is None else str(x)).strip().lower()
    if s in {"win", "w", "1", "true"}:
        return "Win"
    if s in {"lose", "l", "0", "false"}:
        return "Lose"
    return ""


def _build_training_matrix_from_v2_shadow_ai(df: pd.DataFrame, side: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    v2_shadow_ai から、LONG/SHORT 別の学習用 X, y を作る。
    方針:
      - DONE + Win/Lose確定行だけ使う
      - Datetime_JST + Symbol + Direction で重複除去（keep='last'）
      - 欠損は埋めない
      - AI列 / 結果列は特徴量に使わない
    """
    side_u = str(side or "").strip().upper()

    info: Dict[str, Any] = {
        "policy": "v2_shadow_ai_done_only_drop_blank_no_imputation_no_leakage",
        "side": side_u,
        "required_cols": [
            "Datetime_JST", "Symbol", "Direction", "EvalStatus", "WinLose",
            "TotalScore", "P1_TrendScore", "P2_FundingScore", "P3_VolumeScore",
            "SymHTF_Strength", "BTC_HTF_Strength",
            "VolRatio", "ATR", "RSI", "Hour_JST",
            "LTF_Aligned", "FR_Available", "VolConfirmed", "BTC_Mode_Compat",
        ],
        "missing_cols": [],
        "rows_total": 0,
        "rows_after_side": 0,
        "rows_after_done": 0,
        "rows_labeled": 0,
        "rows_skipped_blank_required": 0,
        "rows_used": 0,
        "feature_columns": [
            "TotalScore",
            "P1_TrendScore",
            "P2_FundingScore",
            "P3_VolumeScore",
            "SymHTF_Strength",
            "BTC_HTF_Strength",
            "VolRatio",
            "ATR",
            "RSI",
            "Hour_JST",
            "LTF_Aligned",
            "FR_Available",
            "VolConfirmed",
            "BTC_Mode_UP",
            "BTC_Mode_RANGE",
            "BTC_Mode_DOWN",
        ],
        "notes": [
            "NO IMPUTATION: rows with blank/non-numeric values in required features are dropped.",
            "NO LEAKAGE: AI_* and post-exit columns are not used as features.",
            "WinLose is target only.",
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

    work = df.copy()
    work["_row_order"] = np.arange(len(work))

    work = work.sort_values("_row_order").drop_duplicates(
        subset=[
            "Datetime_JST",
            "Symbol",
            "Direction",
            "TotalScore",
            "P1_TrendScore",
            "P2_FundingScore",
            "P3_VolumeScore",
        ],
        keep="last",
    )

    work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
    work = work[work["Direction"] == side_u].copy()
    info["rows_after_side"] = int(len(work))
    if work.empty:
        return pd.DataFrame(columns=info["feature_columns"]), np.array([], dtype=int), info

    work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
    work = work[work["EvalStatus"] == "DONE"].copy()
    info["rows_after_done"] = int(len(work))
    if work.empty:
        return pd.DataFrame(columns=info["feature_columns"]), np.array([], dtype=int), info

    work["__winlose__"] = work["WinLose"].apply(_normalize_v2_winlose)
    work = work[work["__winlose__"].isin(["Win", "Lose"])].copy()
    info["rows_labeled"] = int(len(work))
    if work.empty:
        return pd.DataFrame(columns=info["feature_columns"]), np.array([], dtype=int), info

    # 数値化（事前特徴量のみ）
    work["TotalScore"] = pd.to_numeric(work["TotalScore"], errors="coerce")
    work["P1_TrendScore"] = pd.to_numeric(work["P1_TrendScore"], errors="coerce")
    work["P2_FundingScore"] = pd.to_numeric(work["P2_FundingScore"], errors="coerce")
    work["P3_VolumeScore"] = pd.to_numeric(work["P3_VolumeScore"], errors="coerce")
    work["SymHTF_Strength"] = pd.to_numeric(work["SymHTF_Strength"], errors="coerce")
    work["BTC_HTF_Strength"] = pd.to_numeric(work["BTC_HTF_Strength"], errors="coerce")
    work["VolRatio"] = pd.to_numeric(work["VolRatio"], errors="coerce")
    work["ATR"] = pd.to_numeric(work["ATR"], errors="coerce")
    work["RSI"] = pd.to_numeric(work["RSI"], errors="coerce")
    work["Hour_JST"] = pd.to_numeric(work["Hour_JST"], errors="coerce")

    work["LTF_Aligned"] = work["LTF_Aligned"].apply(_normalize_bool01)
    work["FR_Available"] = work["FR_Available"].apply(_normalize_bool01)
    work["VolConfirmed"] = work["VolConfirmed"].apply(_normalize_bool01)

    mode = work["BTC_Mode_Compat"].astype(str).str.strip().str.upper()
    work["BTC_Mode_UP"] = np.where(mode == "UP", 1.0, np.where(mode.isin(["RANGE", "DOWN"]), 0.0, np.nan))
    work["BTC_Mode_RANGE"] = np.where(mode == "RANGE", 1.0, np.where(mode.isin(["UP", "DOWN"]), 0.0, np.nan))
    work["BTC_Mode_DOWN"] = np.where(mode == "DOWN", 1.0, np.where(mode.isin(["UP", "RANGE"]), 0.0, np.nan))

    X = work[info["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    y = (work["__winlose__"] == "Win").astype(int).to_numpy()

    blank_mask_any = X.isna().any(axis=1)
    info["rows_skipped_blank_required"] = int(blank_mask_any.sum())

    X = X[~blank_mask_any].copy()
    y = y[~blank_mask_any.to_numpy()]

    info["rows_used"] = int(len(X))
    return X, y, info


def _build_training_dataset_from_v2_shadow_ai(side: str, lookback_rows: int) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    v2_shadow_ai から LONG/SHORT 別の学習用 X, y を作るラッパー。
    追加方針:
      - SHORT: VolConfirmed=True 優先、SHORT blocklist除外、mode選別
      - LONG : VolConfirmed=True 優先、LONG blocklist除外、UP中心
    """
    side_u = str(side or "").strip().upper()
    meta: Dict[str, Any] = {
        "rows_scanned": 0,
        "rows_used": 0,
        "class_balance": {},
        "side": side_u,
        "train_filters": {},
    }

    df = get_v2_shadow_ai_data()
    if df is None or df.empty:
        raise RuntimeError("v2_shadow_ai is empty")

    if lookback_rows is not None and int(lookback_rows) > 0:
        df = df.tail(int(lookback_rows)).copy()

    meta["rows_scanned"] = int(len(df))

    work = df.copy()

    if "Direction" in work.columns:
        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()

    if "Symbol" in work.columns:
        work["__sym__"] = work["Symbol"].astype(str).str.split("/", n=1).str[0].str.strip().str.upper()
    else:
        work["__sym__"] = ""

    if "BTC_Mode_Compat" in work.columns:
        work["__mode__"] = work["BTC_Mode_Compat"].astype(str).str.strip().str.upper()
    else:
        work["__mode__"] = ""

    if "VolConfirmed" in work.columns:
        work["__volconf__"] = work["VolConfirmed"].apply(_normalize_bool01)
    else:
        work["__volconf__"] = np.nan

    if side_u == "SHORT":
        if V2_SHORT_TRAIN_EXCLUDE_BLOCKLIST:
            short_block = set(V2_SHORT_SYMBOL_BLOCKLIST or [])
            if short_block:
                work = work[~work["__sym__"].isin(short_block)].copy()

        if V2_SHORT_TRAIN_REQUIRE_VOLCONF:
            work = work[work["__volconf__"] == 1.0].copy()

        if V2_SHORT_TRAIN_MODE == "DOWN":
            work = work[work["__mode__"] == "DOWN"].copy()
        elif V2_SHORT_TRAIN_MODE == "UP":
            work = work[work["__mode__"] == "UP"].copy()
        elif V2_SHORT_TRAIN_MODE == "RANGE":
            work = work[work["__mode__"] == "RANGE"].copy()

        meta["train_filters"] = {
            "require_volconf": bool(V2_SHORT_TRAIN_REQUIRE_VOLCONF),
            "exclude_blocklist": bool(V2_SHORT_TRAIN_EXCLUDE_BLOCKLIST),
            "mode": str(V2_SHORT_TRAIN_MODE),
            "blocklist": list(V2_SHORT_SYMBOL_BLOCKLIST or []),
        }

    elif side_u == "LONG":
        if V2_LONG_TRAIN_EXCLUDE_BLOCKLIST:
            long_block = set(V2_LONG_SYMBOL_BLOCKLIST or [])
            if long_block:
                work = work[~work["__sym__"].isin(long_block)].copy()

        if V2_LONG_TRAIN_REQUIRE_VOLCONF:
            work = work[work["__volconf__"] == 1.0].copy()

        if V2_LONG_TRAIN_MODE == "DOWN":
            work = work[work["__mode__"] == "DOWN"].copy()
        elif V2_LONG_TRAIN_MODE == "RANGE":
            work = work[work["__mode__"] == "RANGE"].copy()
        elif V2_LONG_TRAIN_MODE == "UP":
            work = work[work["__mode__"] == "UP"].copy()

        meta["train_filters"] = {
            "require_volconf": bool(V2_LONG_TRAIN_REQUIRE_VOLCONF),
            "exclude_blocklist": bool(V2_LONG_TRAIN_EXCLUDE_BLOCKLIST),
            "mode": str(V2_LONG_TRAIN_MODE),
            "blocklist": list(V2_LONG_SYMBOL_BLOCKLIST or []),
        }

    X, y, info = _build_training_matrix_from_v2_shadow_ai(work, side_u)

    meta["rows_used"] = int(len(X))
    if len(y) > 0:
        uniq, cnt = np.unique(y, return_counts=True)
        meta["class_balance"] = {str(int(k)): int(v) for k, v in zip(uniq, cnt)}

    meta["build_info"] = info
    return X, y, meta


def get_v2_done_records(df: pd.DataFrame) -> pd.DataFrame:
    """EvalStatus == 'DONE' の行だけ返す。"""
    if "EvalStatus" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    status = df["EvalStatus"].astype(str).str.strip().str.upper()
    return df[status == "DONE"].copy()


def _series_to_jst_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            return dt.dt.tz_convert(JST).dt.tz_localize(None)
    except Exception:
        pass
    return dt


def _prepare_v2_short_done_for_brake(df: pd.DataFrame) -> pd.DataFrame:
    """
    v2_shadow_ai 全件から、SHORT / DONE のうち
    SHORTブレーキ判定に使う観測対象だけを取り出して
    Datetime_JST × Symbol × Direction の最後の行に正規化する。
    観測対象:
      - SHORT_AI_RANK
      - SHORT_AI_FALLBACK_RANK
      - SHORT_BRAKE_PROBE
      - RULE_QS
      - RULE_QS_NO_RANK
    除外:
      - SHORT_AI_RANK_REJECT
      - REJECTED
      - SHORT_BRAKE_STOP
      - SHORT_BRAKE_CAUTION
    """
    if df is None or df.empty:
        return pd.DataFrame()
    need_cols = {
        "Datetime_JST", "Symbol", "Direction", "EvalStatus",
        "PnL_Pct", "WinLose", "AI_Band", "AI_Model_Version"
    }
    if not need_cols.issubset(set(df.columns)):
        return pd.DataFrame()
    work = df.copy()
    work["_row_order"] = np.arange(len(work))
    work["Datetime_JST"] = _series_to_jst_naive(work["Datetime_JST"])
    if "ExitTime" in work.columns:
        work["ExitTime"] = _series_to_jst_naive(work["ExitTime"])
    work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
    work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
    work["AI_Band"] = work["AI_Band"].astype(str).str.strip().str.upper()
    work["AI_Model_Version"] = work["AI_Model_Version"].astype(str).str.strip()
    observe_bands = {
        "SHORT_AI_RANK",
        "SHORT_AI_FALLBACK_RANK",
        "SHORT_BRAKE_PROBE",
        "RULE_QS",
        "RULE_QS_NO_RANK",
    }
    work = work[
        (work["Direction"] == "SHORT") &
        (work["EvalStatus"] == "DONE") &
        (work["AI_Model_Version"] != "") &
        (work["AI_Band"].isin(observe_bands))
    ].copy()
    if work.empty:
        return work
    work = work.sort_values("_row_order").drop_duplicates(
        subset=["Datetime_JST", "Symbol", "Direction"],
        keep="last",
    )
    work["_pnl"] = pd.to_numeric(
        work["PnL_Pct"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )
    wl = work["WinLose"].astype(str).str.strip().str.lower()
    work["_win"] = wl.isin(["win", "w", "1", "true"])
    work["_lose"] = wl.isin(["lose", "l", "0", "false"])
    work = work[work["Datetime_JST"].notna()].copy()
    return work


def get_v2_short_brake_state(now_jst: Optional[datetime] = None) -> Dict[str, Any]:
    """
    v2_shadow_ai の最近の SHORT / DONE 実績から、SHORT のブレーキ状態を返す。
    mode:
      - NORMAL  : 通常運転
      - CAUTION : TOP_N を絞る（probe 運転）
      - STOP    : 今回の SHORT 選抜は停止し、記録だけ残す
    """
    base_top_n = max(1, int(V2_SHORT_RANK_TOP_N))
    probe_top_n = max(1, int(V2_SHORT_BRAKE_PROBE_TOP_N))
    default_state = {
        "enabled": bool(V2_SHORT_BRAKE_ENABLE),
        "mode": "NORMAL",
        "base_top_n": base_top_n,
        "effective_top_n": base_top_n,
        "n": 0,
        "slot_count": 0,
        "wr": float("nan"),
        "avg_pnl": float("nan"),
        "reason": "brake_disabled" if not V2_SHORT_BRAKE_ENABLE else "insufficient_done_data",
        "slots": [],
    }
    if not V2_SHORT_BRAKE_ENABLE:
        return default_state
    if now_jst is None:
        now_jst = datetime.now(JST)
    try:
        current_slot = pd.Timestamp(now_jst.astimezone(JST).replace(tzinfo=None)).floor("15min")
        df = get_v2_shadow_ai_data()
        done = _prepare_v2_short_done_for_brake(df)
        if done.empty:
            return default_state
        done = done[done["Datetime_JST"] < current_slot].copy()
        if done.empty:
            return default_state
        unique_slots = sorted(done["Datetime_JST"].dropna().unique())
        lookback_slots = max(1, int(V2_SHORT_BRAKE_LOOKBACK_SLOTS))
        slots = unique_slots[-lookback_slots:]
        sample = done[done["Datetime_JST"].isin(slots)].copy()
        slot_count = len(slots)
        n = len(sample)
        if slot_count < max(1, int(V2_SHORT_BRAKE_MIN_SLOTS)) or n < max(1, int(V2_SHORT_BRAKE_MIN_DONE)):
            state = dict(default_state)
            state["n"] = n
            state["slot_count"] = slot_count
            state["reason"] = "insufficient_done_data"
            state["slots"] = [pd.Timestamp(s).strftime("%Y-%m-%d %H:%M") for s in slots]
            return state
        wins = int(sample["_win"].sum())
        losses = int(sample["_lose"].sum())
        wr = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else float("nan")
        avg_pnl = float(sample["_pnl"].mean()) if n > 0 else float("nan")
        stop_wr = float(V2_SHORT_BRAKE_STOP_WINRATE)
        stop_pnl = float(V2_SHORT_BRAKE_STOP_AVGPNL)
        caution_wr = float(V2_SHORT_BRAKE_CAUTION_WINRATE)
        caution_pnl = float(V2_SHORT_BRAKE_CAUTION_AVGPNL)
        recover_wr = float(V2_SHORT_BRAKE_RECOVER_WINRATE)
        recover_pnl = float(V2_SHORT_BRAKE_RECOVER_AVGPNL)
        if (not pd.isna(wr)) and (not pd.isna(avg_pnl)) and wr < stop_wr and avg_pnl < stop_pnl:
            mode = "STOP"
            effective_top_n = 0
            reason = "recent_done_under_stop_threshold"
        elif (not pd.isna(wr)) and (not pd.isna(avg_pnl)) and wr >= recover_wr and avg_pnl >= recover_pnl:
            mode = "NORMAL"
            effective_top_n = base_top_n
            reason = "recent_done_recovered"
        elif (not pd.isna(wr)) and (not pd.isna(avg_pnl)) and wr < caution_wr and avg_pnl < caution_pnl:
            mode = "CAUTION"
            effective_top_n = min(base_top_n, probe_top_n)
            reason = "recent_done_under_caution_threshold"
        else:
            mode = "CAUTION"
            effective_top_n = min(base_top_n, probe_top_n)
            reason = "recent_done_not_recovered_yet"
        return {
            "enabled": True,
            "mode": mode,
            "base_top_n": base_top_n,
            "effective_top_n": int(effective_top_n),
            "n": n,
            "slot_count": slot_count,
            "wr": float(wr),
            "avg_pnl": float(avg_pnl),
            "reason": reason,
            "slots": [pd.Timestamp(s).strftime("%Y-%m-%d %H:%M") for s in slots],
        }
    except Exception as e:
        print(f"[V2-SHORT-BRAKE-WARN] evaluation failed: {type(e).__name__}: {e}")
        return {
            "enabled": True,
            "mode": "CAUTION",
            "base_top_n": base_top_n,
            "effective_top_n": probe_top_n,
            "n": 0,
            "slot_count": 0,
            "wr": float("nan"),
            "avg_pnl": float("nan"),
            "reason": f"brake_eval_error:{type(e).__name__}",
            "slots": [],
        }


def _prepare_v2_long_done_for_brake(df: pd.DataFrame) -> pd.DataFrame:
    """
    v2_shadow_ai 全件から、LONG / DONE のうち
    LONGブレーキ判定に使う観測対象だけを取り出して
    Datetime_JST × Symbol × Direction の最後の行に正規化する。
    観測対象:
      - LONG_AI_RANK
      - LONG_AI_FALLBACK_RANK
      - LONG_BRAKE_PROBE
      - RULE_QS
      - RULE_QS_NO_RANK
    除外:
      - LONG_AI_RANK_REJECT
      - REJECTED
      - LONG_BRAKE_STOP
      - LONG_BRAKE_CAUTION
    """
    if df is None or df.empty:
        return pd.DataFrame()
    need_cols = {
        "Datetime_JST", "Symbol", "Direction", "EvalStatus",
        "PnL_Pct", "WinLose", "AI_Band", "AI_Model_Version"
    }
    if not need_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    work = df.copy()
    work["_row_order"] = np.arange(len(work))
    work["Datetime_JST"] = _series_to_jst_naive(work["Datetime_JST"])
    if "ExitTime" in work.columns:
        work["ExitTime"] = _series_to_jst_naive(work["ExitTime"])

    work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
    work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
    work["AI_Band"] = work["AI_Band"].astype(str).str.strip().str.upper()
    work["AI_Model_Version"] = work["AI_Model_Version"].astype(str).str.strip()

    observe_bands = {
        "LONG_AI_RANK",
        "LONG_AI_FALLBACK_RANK",
        "LONG_BRAKE_PROBE",
        "RULE_QS",
        "RULE_QS_NO_RANK",
    }

    work = work[
        (work["Direction"] == "LONG") &
        (work["EvalStatus"] == "DONE") &
        (work["AI_Model_Version"] != "") &
        (work["AI_Band"].isin(observe_bands))
    ].copy()

    if work.empty:
        return work

    work = work.sort_values("_row_order").drop_duplicates(
        subset=["Datetime_JST", "Symbol", "Direction"],
        keep="last",
    )

    work["_pnl"] = pd.to_numeric(
        work["PnL_Pct"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )
    wl = work["WinLose"].astype(str).str.strip().str.lower()
    work["_win"] = wl.isin(["win", "w", "1", "true"])
    work["_lose"] = wl.isin(["lose", "l", "0", "false"])
    work = work[work["Datetime_JST"].notna()].copy()
    return work


def get_v2_long_brake_state(now_jst: Optional[datetime] = None) -> Dict[str, Any]:
    """
    v2_shadow_ai の最近の LONG / DONE 実績から、LONG のブレーキ状態を返す。
    mode:
      - NORMAL  : 通常運転
      - CAUTION : TOP_N を絞る（probe 運転）
      - STOP    : 今回の LONG 選抜は停止し、記録だけ残す
    """
    base_top_n = max(1, int(V2_LONG_RANK_TOP_N))
    probe_top_n = max(1, int(V2_LONG_BRAKE_PROBE_TOP_N))

    default_state = {
        "enabled": bool(V2_LONG_BRAKE_ENABLE),
        "mode": "NORMAL",
        "base_top_n": base_top_n,
        "effective_top_n": base_top_n,
        "n": 0,
        "slot_count": 0,
        "wr": float("nan"),
        "avg_pnl": float("nan"),
        "reason": "brake_disabled" if not V2_LONG_BRAKE_ENABLE else "insufficient_done_data",
        "slots": [],
    }

    if not V2_LONG_BRAKE_ENABLE:
        return default_state

    if now_jst is None:
        now_jst = datetime.now(JST)

    try:
        current_slot = pd.Timestamp(now_jst.astimezone(JST).replace(tzinfo=None)).floor("15min")
        df = get_v2_shadow_ai_data()
        done = _prepare_v2_long_done_for_brake(df)

        if done.empty:
            return default_state

        done = done[done["Datetime_JST"] < current_slot].copy()
        if done.empty:
            return default_state

        unique_slots = sorted(done["Datetime_JST"].dropna().unique())
        lookback_slots = max(1, int(V2_LONG_BRAKE_LOOKBACK_SLOTS))
        slots = unique_slots[-lookback_slots:]
        sample = done[done["Datetime_JST"].isin(slots)].copy()

        slot_count = len(slots)
        n = len(sample)

        if slot_count < max(1, int(V2_LONG_BRAKE_MIN_SLOTS)) or n < max(1, int(V2_LONG_BRAKE_MIN_DONE)):
            state = dict(default_state)
            state["n"] = n
            state["slot_count"] = slot_count
            state["reason"] = "insufficient_done_data"
            state["slots"] = [pd.Timestamp(s).strftime("%Y-%m-%d %H:%M") for s in slots]
            return state

        wins = int(sample["_win"].sum())
        losses = int(sample["_lose"].sum())
        wr = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else float("nan")
        avg_pnl = float(sample["_pnl"].mean()) if n > 0 else float("nan")

        stop_wr = float(V2_LONG_BRAKE_STOP_WINRATE)
        stop_pnl = float(V2_LONG_BRAKE_STOP_AVGPNL)
        caution_wr = float(V2_LONG_BRAKE_CAUTION_WINRATE)
        caution_pnl = float(V2_LONG_BRAKE_CAUTION_AVGPNL)
        recover_wr = float(V2_LONG_BRAKE_RECOVER_WINRATE)
        recover_pnl = float(V2_LONG_BRAKE_RECOVER_AVGPNL)

        mode = "NORMAL"
        effective_top_n = base_top_n
        reason = "normal"

        if (not pd.isna(wr)) and (not pd.isna(avg_pnl)):
            if (wr <= stop_wr) or (avg_pnl <= stop_pnl):
                mode = "STOP"
                effective_top_n = 0
                reason = "stop_threshold_hit"
            elif (wr <= caution_wr) or (avg_pnl <= caution_pnl):
                mode = "CAUTION"
                effective_top_n = probe_top_n
                reason = "caution_threshold_hit"
            elif (wr >= recover_wr) and (avg_pnl >= recover_pnl):
                mode = "NORMAL"
                effective_top_n = base_top_n
                reason = "recovered"

        return {
            "enabled": True,
            "mode": mode,
            "base_top_n": base_top_n,
            "effective_top_n": int(effective_top_n),
            "n": int(n),
            "slot_count": int(slot_count),
            "wr": float(wr) if not pd.isna(wr) else float("nan"),
            "avg_pnl": float(avg_pnl) if not pd.isna(avg_pnl) else float("nan"),
            "reason": reason,
            "slots": [pd.Timestamp(s).strftime("%Y-%m-%d %H:%M") for s in slots],
        }

    except Exception as e:
        state = dict(default_state)
        state["reason"] = f"exception:{type(e).__name__}"
        return state


def _build_short_selection_report(df_done: pd.DataFrame, now_jst: str) -> str:
    """
    v2_shadow_ai の DONE データから SHORT 選抜効果を集計してレポート文字列を返す。
    対象: Direction=SHORT AND EvalStatus=DONE AND AI_Model_Version が空でない行。
    """
    MIN_SAMPLES = 10  # これ未満は「判定保留」

    lines = [f"=== V2 SHORT 選抜効果 ({now_jst}) ==="]

    # AI_Model_Version が存在し空でない行のみ（パイプライン稼働後）
    if "AI_Model_Version" not in df_done.columns or "Direction" not in df_done.columns:
        lines.append("（AI_Model_Version または Direction 列なし — スキップ）")
        return "\n".join(lines)

    mv = df_done["AI_Model_Version"].astype(str).str.strip()
    dir_col = df_done["Direction"].astype(str).str.strip().str.upper()
    df_short_pipeline = df_done[(dir_col == "SHORT") & (mv != "") & (mv != "NAN") & (mv != "NONE")].copy()

    n_target = len(df_short_pipeline)
    if n_target == 0:
        lines.append("（選抜パイプライン稼働後のSHORT DONEデータなし）")
        return "\n".join(lines)

    # _pnl / _win / _lose が親で付与済みのはずだが念のため
    if "_pnl" not in df_short_pipeline.columns:
        if "PnL_Pct" in df_short_pipeline.columns:
            df_short_pipeline["_pnl"] = pd.to_numeric(
                df_short_pipeline["PnL_Pct"].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce",
            )
        else:
            df_short_pipeline["_pnl"] = float("nan")

    if "_win" not in df_short_pipeline.columns:
        if "WinLose" in df_short_pipeline.columns:
            wl = df_short_pipeline["WinLose"].astype(str).str.strip().str.lower()
            df_short_pipeline["_win"] = wl.isin(["win", "w", "1", "true"])
            df_short_pipeline["_lose"] = wl.isin(["lose", "l", "0", "false"])
        else:
            df_short_pipeline["_win"] = False
            df_short_pipeline["_lose"] = False

    def _fmt(sub: pd.DataFrame, label: str) -> str:
        n = len(sub)
        if n == 0:
            return f"  {label}: 0件"
        w = int(sub["_win"].sum())
        lo = int(sub["_lose"].sum())
        wr = w / (w + lo) * 100 if (w + lo) > 0 else float("nan")
        avg_pnl = sub["_pnl"].mean()
        wr_str = f"{wr:.1f}%" if not pd.isna(wr) else "N/A"
        pnl_str = f"{avg_pnl:+.4f}%" if not pd.isna(avg_pnl) else "N/A"
        return f"  {label}: n={n}, WR={wr_str}, PnL={pnl_str}"

    # ---- 1. A/B/C 群比較 ----
    ai_pass_col = df_short_pipeline["AI_Pass"].astype(str).str.strip() if "AI_Pass" in df_short_pipeline.columns else pd.Series([""] * n_target, index=df_short_pipeline.index)
    df_pass1 = df_short_pipeline[ai_pass_col == "1"]
    df_pass0 = df_short_pipeline[ai_pass_col == "0"]

    lines.append(f"SHORT全体:    {_fmt(df_short_pipeline, 'A群').strip()}")
    lines.append(f"選抜通過(1):  {_fmt(df_pass1, 'B群').strip()}")
    lines.append(f"選抜落ち(0):  {_fmt(df_pass0, 'C群').strip()}")

    # 判定
    n_b = len(df_pass1)
    n_c = len(df_pass0)
    w_b = int(df_pass1["_win"].sum())
    lo_b = int(df_pass1["_lose"].sum())
    w_c = int(df_pass0["_win"].sum())
    lo_c = int(df_pass0["_lose"].sum())
    wr_b = w_b / (w_b + lo_b) * 100 if (w_b + lo_b) > 0 else None
    wr_c = w_c / (w_c + lo_c) * 100 if (w_c + lo_c) > 0 else None

    if n_b < MIN_SAMPLES:
        verdict = f"判定保留（B群 n={n_b} < {MIN_SAMPLES}件）"
    elif wr_b is None or wr_c is None:
        verdict = "判定保留（WinLose データ不足）"
    elif wr_b > wr_c + 2:
        verdict = "効いている（B群 > C群）"
    elif wr_b < wr_c - 2:
        verdict = "逆効果（B群 < C群 — 要調整）"
    else:
        verdict = "差なし（±2%以内）"
    lines.append(f"判定: {verdict}")

    # ---- 2. AI_Band 別 ----
    lines.append("")
    lines.append("--- AI_Band別 ---")
    if "AI_Band" in df_short_pipeline.columns:
        band_col = df_short_pipeline["AI_Band"].astype(str).str.strip()
        for band in ["RULE_QS", "RULE_QS_NO_RANK", "RANK_REJECT", "REJECTED"]:
            sub = df_short_pipeline[band_col == band]
            if len(sub) > 0:
                lines.append(_fmt(sub, band))
    else:
        lines.append("  AI_Band列なし")

    # ---- 3. QualityScore (AI_Prob_Win) 帯別 ----
    lines.append("")
    lines.append("--- QualityScore帯別 ---")
    if "AI_Prob_Win" in df_short_pipeline.columns:
        qs = pd.to_numeric(df_short_pipeline["AI_Prob_Win"], errors="coerce")
        df_qs = df_short_pipeline.copy()
        df_qs["_qs"] = qs
        df_qs_valid = df_qs[df_qs["_qs"].notna()]
        if len(df_qs_valid) == 0:
            lines.append("  数値データなし")
        else:
            bands_qs = [
                ("QS <3.0",    df_qs_valid[df_qs_valid["_qs"] < 3.0]),
                ("QS 3.0-3.5", df_qs_valid[(df_qs_valid["_qs"] >= 3.0) & (df_qs_valid["_qs"] < 3.5)]),
                ("QS 3.5-4.0", df_qs_valid[(df_qs_valid["_qs"] >= 3.5) & (df_qs_valid["_qs"] < 4.0)]),
                ("QS 4.0+",    df_qs_valid[df_qs_valid["_qs"] >= 4.0]),
            ]
            for label, sub in bands_qs:
                if len(sub) > 0:
                    lines.append(_fmt(sub, label))
    else:
        lines.append("  AI_Prob_Win列なし")

    # ---- 4. 日次推移 ----
    lines.append("")
    lines.append("--- 日次推移 ---")
    if "Datetime_JST" in df_short_pipeline.columns:
        dt_s = pd.to_datetime(
            df_short_pipeline["Datetime_JST"].astype(str).str.strip(),
            errors="coerce",
        )
        df_daily = df_short_pipeline.copy()
        df_daily["_date"] = dt_s.dt.date
        dates_valid = df_daily[df_daily["_date"].notna()]["_date"].unique()
        dates_sorted = sorted(dates_valid)
        if len(dates_sorted) == 0:
            lines.append("  日付データなし")
        else:
            def _wr_str(sub: pd.DataFrame) -> str:
                w = int(sub["_win"].sum()); lo = int(sub["_lose"].sum())
                return f"{w/(w+lo)*100:.1f}%" if (w + lo) > 0 else "N/A"
            for d in dates_sorted:
                day_df = df_daily[df_daily["_date"] == d]
                if "AI_Pass" in day_df.columns:
                    ap = day_df["AI_Pass"].astype(str).str.strip()
                    pass1_day = day_df[ap == "1"]
                    pass0_day = day_df[ap == "0"]
                else:
                    pass1_day = day_df.iloc[0:0]
                    pass0_day = day_df.iloc[0:0]
                lines.append(
                    f"  {d}: Pass n={len(pass1_day)} WR={_wr_str(pass1_day)} | "
                    f"Reject n={len(pass0_day)} WR={_wr_str(pass0_day)}"
                )
    else:
        lines.append("  Datetime_JST列なし")

    return "\n".join(lines)


def _summarize_simple(sub: pd.DataFrame, label: str, emoji: str = "📌") -> str:
    n = len(sub)
    if n == 0:
        return f"{emoji} {label}: 0件"

    w = int(sub["_win"].sum()) if "_win" in sub.columns else 0
    lo = int(sub["_lose"].sum()) if "_lose" in sub.columns else 0
    wr = (w / (w + lo) * 100) if (w + lo) > 0 else float("nan")
    avg_pnl = sub["_pnl"].mean() if "_pnl" in sub.columns else float("nan")

    wr_str = f"{wr:.1f}%" if not pd.isna(wr) else "N/A"
    pnl_str = f"{avg_pnl:+.2f}%" if not pd.isna(avg_pnl) else "N/A"

    return f"{emoji} {label}: {n}件 / 勝率 {wr_str} / 平均 {pnl_str}"


def _build_symbol_ranking_lines(
    df: pd.DataFrame,
    title: str,
    top_n: int = 5,
    sort_bad: bool = False,
    min_n: int = 3,
) -> list:
    """
    銘柄別ランキングを daily 用に出す。
    sort_bad=False: 良い順
    sort_bad=True : 悪い順
    """
    lines = [f"【{title}】"]

    if df is None or df.empty or "Symbol" not in df.columns:
        lines.append("・データなし")
        return lines

    work = df.copy()
    work["Symbol"] = work["Symbol"].astype(str).str.strip().str.upper()

    rows = []
    for sym, sub in work.groupby("Symbol"):
        n = len(sub)
        if n < int(min_n):
            continue

        w = int(sub["_win"].sum()) if "_win" in sub.columns else 0
        lo = int(sub["_lose"].sum()) if "_lose" in sub.columns else 0
        wr = (w / (w + lo) * 100.0) if (w + lo) > 0 else float("nan")
        avg_pnl = sub["_pnl"].mean() if "_pnl" in sub.columns else float("nan")

        rows.append({
            "symbol": sym,
            "n": n,
            "wins": w,
            "wr": wr,
            "avg_pnl": avg_pnl,
        })

    if not rows:
        lines.append("・対象なし（3件以上の銘柄なし）")
        return lines

    rank_df = pd.DataFrame(rows)

    if sort_bad:
        rank_df = rank_df.sort_values(
            ["avg_pnl", "wr", "n"],
            ascending=[True, True, False],
        ).head(int(top_n))
    else:
        rank_df = rank_df.sort_values(
            ["avg_pnl", "wr", "n"],
            ascending=[False, False, False],
        ).head(int(top_n))

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

    for i, row in enumerate(rank_df.itertuples(index=False), start=1):
        medal = medals[i - 1] if i - 1 < len(medals) else f"{i}."
        lines.append(
            f"{medal} {row.symbol}: {row.n}件 / 勝率 {row.wr:.1f}% / 平均 {row.avg_pnl:+.2f}%"
        )

    return lines


def analyze_v2_performance() -> str:
    """
    daily report 用。
    daily 専用Webhookへ送り、
    内容は薄すぎず・長すぎずにする。
    - 全体
    - 採用
    - 惜しくも落選
    - 除外
    - 良い銘柄ランキング
    - 悪い銘柄ランキング
    - 惜しくも落選ランキング
    その後に AI サマリーも送る。
    """
    try:
        df_all = get_v2_shadow_ai_data()
    except Exception as e:
        msg = f"[V2-REPORT] シート取得失敗: {e}"
        print(msg)
        send_daily_discord_message(msg)
        return msg

    total_rows = len(df_all)
    df_done = get_v2_done_records(df_all).copy()
    n_done = len(df_done)

    if "EvalStatus" in df_all.columns:
        status_all = df_all["EvalStatus"].astype(str).str.strip().str.upper()
        n_open = int((status_all == "OPEN").sum())
    else:
        n_open = 0

    now_jst = datetime.now(JST)
    now_str = now_jst.strftime("%Y-%m-%d %H:%M JST")

    if n_done == 0:
        msg = "\n".join([
            "📘【V2 デイリーレポート】",
            f"🕒 {now_str}",
            f"📦 総件数 {total_rows} / OPEN {n_open} / DONE {n_done}",
            "・完了データなし",
        ])
        send_daily_discord_message(msg)
        return msg

    if "PnL_Pct" in df_done.columns:
        df_done["_pnl"] = pd.to_numeric(
            df_done["PnL_Pct"].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
    else:
        df_done["_pnl"] = float("nan")

    if "WinLose" in df_done.columns:
        wl = df_done["WinLose"].astype(str).str.strip().str.lower()
        df_done["_win"] = wl.eq("win")
        df_done["_lose"] = wl.eq("lose")
    else:
        df_done["_win"] = False
        df_done["_lose"] = False

    if "Datetime_JST" in df_done.columns:
        df_done["_dt"] = pd.to_datetime(df_done["Datetime_JST"], errors="coerce")
    else:
        df_done["_dt"] = pd.NaT

    since = pd.Timestamp(now_jst.replace(tzinfo=None)) - pd.Timedelta(hours=24)
    df_24h = df_done[df_done["_dt"] >= since].copy()

    if df_24h.empty:
        msg = "\n".join([
            "📘【V2 デイリーレポート】",
            f"🕒 {now_str}",
            f"📦 総件数 {total_rows} / OPEN {n_open} / DONE {n_done}",
            "・直近24時間の完了データなし",
        ])
        send_daily_discord_message(msg)
        return msg

    note_col = None
    for c in ["AI_Note", "ai_note"]:
        if c in df_24h.columns:
            note_col = c
            break

    if note_col:
        note = df_24h[note_col].astype(str)
        df_24h["_notify_1"] = note.str.contains("notify_pass=1", na=False)
    else:
        df_24h["_notify_1"] = False

    if "AI_Band" in df_24h.columns:
        df_24h["AI_Band"] = df_24h["AI_Band"].astype(str).str.strip().str.upper()
    else:
        df_24h["AI_Band"] = ""

    notify_df = df_24h[df_24h["_notify_1"]].copy()
    rank_reject_df = df_24h[df_24h["AI_Band"] == "RANK_REJECT"].copy()
    rejected_df = df_24h[df_24h["AI_Band"] == "REJECTED"].copy()

    long_n = 0
    short_n = 0
    if "Direction" in df_24h.columns:
        dir_col = df_24h["Direction"].astype(str).str.strip().str.upper()
        long_n = int((dir_col == "LONG").sum())
        short_n = int((dir_col == "SHORT").sum())

    lines = [
        "📘【V2 デイリーレポート】",
        f"🕒 {now_str}",
        f"📦 直近24h DONE {len(df_24h)}件 / OPEN {n_open}件",
        f"🟢 LONG {long_n}件 / 🔴 SHORT {short_n}件",
        "",
        _summarize_simple(notify_df, "採用", "✅"),
        _summarize_simple(rank_reject_df, "惜しくも落選", "🟡"),
        _summarize_simple(rejected_df, "除外", "🔴"),
        "",
    ]

    lines += _build_symbol_ranking_lines(
        notify_df,
        "良い銘柄ランキング",
        top_n=5,
        sort_bad=False,
        min_n=3,
    )
    lines += [""]
    lines += _build_symbol_ranking_lines(
        notify_df,
        "悪い銘柄ランキング",
        top_n=5,
        sort_bad=True,
        min_n=3,
    )
    lines += [""]
    lines += _build_symbol_ranking_lines(
        rank_reject_df,
        "惜しくも落選ランキング",
        top_n=5,
        sort_bad=False,
        min_n=3,
    )

    msg = "\n".join([x for x in lines if x is not None])

    send_daily_discord_message(msg)
    print(
        f"[V2-REPORT] daily report sent. "
        f"done24h={len(df_24h)} notify={len(notify_df)} "
        f"rank_reject={len(rank_reject_df)} rejected={len(rejected_df)}"
    )

    try:
        call_claude_for_analysis_daily(msg)
    except Exception as e:
        print(f"[V2-REPORT] Claude daily summary failed: {e}")

    return msg


# ==========================================
# Claude AI 分析（/v2_report から呼び出す）
# ==========================================

def call_claude_for_analysis(
    short_before: str, short_after: str,
    long_before: str, long_after: str,
    change_summary_short: str, change_summary_long: str,
    cutoff_short_str: str, cutoff_long_str: str,
) -> str:
    """
    SHORT/LONG それぞれの変更前後レポートを Claude API に送り、比較分析を生成して Discord に送信する。
    ANTHROPIC_API_KEY が未設定の場合はスキップする。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[V2-AI] ANTHROPIC_API_KEY 未設定 — AI分析をスキップ")
        return ""

    prompt = f"""あなたは暗号通貨botのV2ルール改善アドバイザーです。
以下の変更前・変更後の集計データを分析し、日本語で回答してください。

【ロードマップ現在地】
第4段階：V2ルールの成績レビュー・3分類

【厳守事項】
- V1コードには絶対に触らない
- AI学習・LONG/SHORTモデル実装はまだしない
- 時間帯フィルターは観察するが実装しない
- 最小修正（1〜2個）を超える変更はしない
- v2_live化はまだしない

【背景】
SHORT カットオフ（{cutoff_short_str} JST）に実施したコード修正：
{change_summary_short}

LONG カットオフ（{cutoff_long_str} JST）に実施したコード修正：
{change_summary_long}

【重要な分析ルール】
- SHORTとLONGを混ぜて分析しない
- 変更前データと変更後データを絶対に混ぜて分析しない
- 変更前後を必ず比較して差分を明示する
- コード修正による効果を具体的に評価する
- 変更後のデータが少ない場合はその旨を明記し、判断を急がない

【SHORT変更前集計データ（{cutoff_short_str}より前）】
{short_before}

【SHORT変更後集計データ（{cutoff_short_str}以降）】
{short_after}

【LONG変更前集計データ（{cutoff_long_str}より前）】
{long_before}

【LONG変更後集計データ（{cutoff_long_str}以降）】
{long_after}

【出力形式（この形式を必ず守ること）】
## SHORT 全体評価
（SHORT変更前・変更後それぞれの現状を2〜3行で要約。混ぜない）

## SHORT 変更前後の比較・差分
- 勝率の変化：
- 平均PnLの変化：
- コード修正の効果：

## LONG 全体評価
（LONG変更前・変更後それぞれの現状を2〜3行で要約。混ぜない）

## LONG 変更前後の比較・差分
- 勝率の変化：
- 平均PnLの変化：
- フィルター追加の効果：

## 切る候補（SHORT/LONG共通）
- 対象：
- 理由：
- 具体的な除外条件（コード修正イメージ）：

## 残す候補
- 対象：
- 理由：

## まだ観察
- 対象：
- 理由：
- 次の再判断タイミング：

## 改善した点
（変更後で明確に良くなった点）

## まだ課題の点
（変更後でも残る問題）

## 最小修正案（1〜2個のみ）
1. （具体的な変更内容）
2. （あれば）
"""

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = response.content[0].text if response.content else "(応答なし)"
        header = "=== AI分析レポート ⑥（Claude） ==="
        full_msg = f"{header}\n{ai_text}"
        send_discord_message(full_msg)
        print("[V2-AI] AI分析送信完了")
        return full_msg
    except Exception as e:
        err_msg = f"[V2-AI] AI分析エラー: {type(e).__name__}: {e}"
        print(err_msg)
        send_discord_message(err_msg)
        return err_msg


def call_claude_for_analysis_daily(daily_report_text: str) -> str:
    """
    daily report を読んで、改善だけでなく
    今日の評価コメントを短く送る。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[V2-AI] ANTHROPIC_API_KEY 未設定 — daily AIサマリーをスキップ")
        return ""

    prompt = f"""あなたは暗号通貨botのV2運用レビュー担当です。
以下の daily report を読み、Discord向けに短い日本語コメントを作ってください。

【目的】
改善点だけを並べるのではなく、
その日の結果をまず評価し、必要なら次の見方を少しだけ足してください。

【出力ルール】
- 4〜8行くらい
- 専門用語を減らす
- 難しい言葉を避ける
- 先頭に絵文字をつける
- 「良かった点」「注意点」「次に見るポイント」を短く入れる
- Discord向けに読みやすくする

【daily report】
{daily_report_text}
"""

    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 900,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=90,
        )
        if r.status_code >= 300:
            print(f"[V2-AI] daily Claude HTTP {r.status_code}: {r.text[:300]}")
            return ""

        data = r.json()
        text = ""
        for c in data.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "text":
                text += str(c.get("text", ""))

        text = text.strip()
        if not text:
            return ""

        msg = "🧠【AIサマリー】\n" + text
        send_daily_discord_message(msg)
        return msg

    except Exception as e:
        print(f"[V2-AI] daily Claude summary error: {e}")
        return ""


# ==========================================
# Shadow Mode メインループ
# ==========================================

def v2_shadow_run(exchange, now_jst: datetime, force: bool = False) -> str:
    """
    V1 の logic_main() の後に呼ばれる。
    現在は V2 シグナルを評価して v2_shadow シートに記録する。
    将来は SIGNAL_ENGINE に応じて V2本番通知へ昇格できるようにする。

    今回追加:
    - SHORTだけ Defensive Filter を適用
    - SHORTだけ QualityScore でランキング
    - SHORTは上位N件だけ残す（残りは ai_pass=0 で記録）
    - LONGは今まで通り通す
    - REJECTされたSHORTも ai_pass=0 でシートに記録（検証用）
    """
    engine_mode = str(SIGNAL_ENGINE).strip().lower()

    # 15分足確定待ち
    if (not force) and ((now_jst.minute % 15) < 10):
        return "V2: waiting"

    # シート準備
    try:
        v2_ensure_shadow_sheet(get_sheet_service(), SPREADSHEET_ID)
    except Exception as e:
        print(f"[V2-ERR] sheet setup: {e}")
        return f"V2: sheet error {e}"

    # BTC 上位足トレンド
    btc_htf_ohlcv = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe=V2_HTF_TIMEFRAME, limit=V2_HTF_LIMIT)
    if not btc_htf_ohlcv or len(btc_htf_ohlcv) < V2_EMA_SLOW + 5:
        return "V2: BTC HTF insufficient"

    btc_htf_df = pd.DataFrame(btc_htf_ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    btc_htf = assess_htf_trend(btc_htf_df)

    # BTC 15m Pct_Change / Vol (AI 特徴量用)
    btc_ret = np.nan
    btc_vol = np.nan
    try:
        btc_15m_ohlcv = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe="15m", limit=30)
        if btc_15m_ohlcv and len(btc_15m_ohlcv) >= 3:
            btc_15m_df = pd.DataFrame(btc_15m_ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
            btc_15m_df["Pct_Change"] = pd.to_numeric(btc_15m_df["Close"], errors="coerce").pct_change(fill_method=None)
            btc_ret = float(btc_15m_df["Pct_Change"].iloc[-2])
            btc_vol = abs(btc_ret)
    except Exception as e:
        print(f"[V2-WARN] BTC 15m fetch for AI feats failed: {e}")

    # レジーム転換検出
    regime = detect_btc_regime_conflict(exchange, btc_htf["direction"])
    short_conflict = bool(regime.get("short_conflict", False))
    long_conflict = bool(regime.get("long_conflict", False))
    long_conflict_soft = bool(regime.get("long_conflict_soft", False))
    regime_spread = regime.get("spread", np.nan)
    spread_str = "nan"
    try:
        if np.isfinite(float(regime_spread)):
            spread_str = f"{float(regime_spread):.6f}"
    except Exception:
        pass
    if short_conflict or long_conflict or long_conflict_soft:
        print(
            f"[V2-REGIME] CONFLICT DETECTED: reason={regime['reason']} "
            f"short_conflict={short_conflict} "
            f"long_conflict={long_conflict} "
            f"long_conflict_soft={long_conflict_soft} "
            f"spread={spread_str}"
        )
    else:
        print(f"[V2-REGIME] no conflict: {regime['reason']}")
    print(
        f"[V2] BTC HTF: {btc_htf['direction']} "
        f"str={btc_htf['strength']:.3f} "
        f"str_raw={float(btc_htf.get('strength_raw', np.nan)):.3f} "
        f"ema_spread={float(btc_htf.get('ema_spread', np.nan)):.6f} "
        f"ema_fast={float(btc_htf.get('ema_fast', np.nan)):.2f} "
        f"ema_slow={float(btc_htf.get('ema_slow', np.nan)):.2f} "
        f"close={float(btc_htf.get('close', np.nan)):.2f} "
        f"slope={float(btc_htf.get('slope', np.nan)):.6f} "
        f"mode={engine_mode} "
        f"short_conflict={short_conflict} "
        f"long_conflict={long_conflict} "
        f"long_conflict_soft={long_conflict_soft} "
        f"regime_spread={spread_str}"
    )

    symbols = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "AAVE/USDT", "NEAR/USDT", "ATOM/USDT", "AVAX/USDT",
        "APT/USDT", "FET/USDT", "LTC/USDT", "LINK/USDT",
        "DOT/USDT", "DOGE/USDT", "UNI/USDT", "XRP/USDT",
        "ARB/USDT", "INJ/USDT", "SUI/USDT", "SEI/USDT",
        "ADA/USDT", "XLM/USDT", "HBAR/USDT", "SHIB/USDT",
        "BONK/USDT", "TRX/USDT", "STX/USDT", "POL/USDT",
    ]

    raw_signals: List[Dict[str, Any]] = []

    for symbol in symbols:
        try:
            sig = v2_generate_signal(exchange, symbol, btc_htf, now_jst, regime_info=regime)
            if sig is not None:
                sig["v2_version"] = "V2-Shadow"
                sig["note"] = ""
                sig["btc_ret"] = _safe_float_or_nan(btc_ret)
                sig["btc_vol"] = _safe_float_or_nan(btc_vol)

                raw_signals.append(sig)

                print(
                    f"[V2] RAW-SIG: {sig['symbol']} {sig['direction']} "
                    f"total={sig['total_score']:.2f} "
                    f"P1={sig['p1_score']:.2f} P2={sig['p2_score']:.2f} P3={sig['p3_score']:.2f} "
                    f"RSI={sig.get('rsi', '')} BTCstr={sig.get('btc_htf_strength', '')}"
                )
        except Exception as e:
            print(f"[V2-ERR] {symbol}: {e}")

    # Selection Pipeline: フィルター → スコア → ランキング
    # REJECTされたSHORTも ai_pass="0" で含まれる
    final_signals = v2_selection_pipeline(raw_signals)

    n_short_pass = sum(
        1 for x in final_signals
        if str(x.get("direction", "")).upper() == "SHORT" and str(x.get("ai_pass", "")) == "1"
    )
    n_short_reject = sum(
        1 for x in final_signals
        if str(x.get("direction", "")).upper() == "SHORT" and str(x.get("ai_pass", "")) == "0"
    )
    n_long_pass = sum(
        1 for x in final_signals
        if str(x.get("direction", "")).upper() == "LONG" and str(x.get("ai_pass", "")) == "1"
    )
    n_long_reject = sum(
        1 for x in final_signals
        if str(x.get("direction", "")).upper() == "LONG" and str(x.get("ai_pass", "")) == "0"
    )

    n_short_notify = sum(
        1 for x in final_signals
        if str(x.get("direction", "")).upper() == "SHORT" and str(x.get("_notify_pass", "0")) == "1"
    )
    n_long_notify = sum(
        1 for x in final_signals
        if str(x.get("direction", "")).upper() == "LONG" and str(x.get("_notify_pass", "0")) == "1"
    )

    print(
        f"[V2] RAW={len(raw_signals)} FINAL={len(final_signals)} "
        f"SHORT_PASS={n_short_pass} SHORT_REJECT={n_short_reject} SHORT_NOTIFY={n_short_notify} "
        f"LONG_PASS={n_long_pass} LONG_REJECT={n_long_reject} LONG_NOTIFY={n_long_notify}"
    )

    for sig in final_signals:
        direction = str(sig.get("direction", "")).upper()
        ai_pass = str(sig.get("ai_pass", ""))
        notify_pass = str(sig.get("_notify_pass", "0"))
        notify_reason = str(sig.get("_notify_reason", "") or "")

        if direction == "SHORT" and ai_pass == "1":
            print(
                f"[V2] SHORT-SELECTED: {sig['symbol']} {sig['direction']} "
                f"total={sig['total_score']:.2f} "
                f"P1={sig['p1_score']:.2f} "
                f"QS={sig.get('ai_prob_win', '')} "
                f"TYPE={sig.get('ai_model_type', '')} "
                f"NOTIFY={notify_pass} "
                f"NOTIFY_REASON={notify_reason}"
            )
        elif direction == "SHORT" and ai_pass == "0":
            print(
                f"[V2] SHORT-REJECTED: {sig['symbol']} {sig['direction']} "
                f"BAND={sig.get('ai_band', '')} NOTE={sig.get('ai_note', '')}"
            )
        elif direction == "LONG" and ai_pass == "1":
            print(
                f"[V2] LONG-SELECTED: {sig['symbol']} {sig['direction']} "
                f"total={sig['total_score']:.2f} "
                f"P1={sig['p1_score']:.2f} "
                f"QS={sig.get('ai_prob_win', '')} "
                f"TYPE={sig.get('ai_model_type', '')} "
                f"NOTIFY={notify_pass} "
                f"NOTIFY_REASON={notify_reason}"
            )
        elif direction == "LONG" and ai_pass == "0":
            print(
                f"[V2] LONG-REJECTED: {sig['symbol']} {sig['direction']} "
                f"BAND={sig.get('ai_band', '')} NOTE={sig.get('ai_note', '')}"
            )

    # シート書き込み（選抜通過もREJECTも全部書く）
    rows = [v2_build_shadow_row(sig) for sig in final_signals]
    v2_write_shadow_rows(rows)

    if engine_mode == "v2_live":
        notify_candidates = [
            x for x in final_signals
            if str(x.get("_notify_pass", "0")) == "1"
        ]

        sent_n, runtime_dedup_skipped_n = send_v2_live_discord_alerts(notify_candidates)

        print(
            f"[V2] engine_mode=v2_live "
            f"notify_candidates={len(notify_candidates)} "
            f"discord_sent={sent_n} "
            f"discord_runtime_dedup_skipped={runtime_dedup_skipped_n} "
            f"notify_ignore_runtime_dedup={int(V2_NOTIFY_IGNORE_RUNTIME_DEDUP)}"
        )

    return f"V2-shadow: final={len(final_signals)} (short_pass={n_short_pass} short_reject={n_short_reject} long_pass={n_long_pass} long_reject={n_long_reject}) raw={len(raw_signals)}"




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
        # 修正後（元に戻す）
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


            # High/Low を使って「同一足で上も下も動いた」を拾えるようにする
            if "High" not in df.columns:
                df["High"] = df["Close"]
            if "Low" not in df.columns:
                df["Low"] = df["Close"]
            
            prev_close = df["Close"].shift(1)
            
            drop_move = ((prev_close - df["Low"]) / prev_close).clip(lower=0)
            rise_move = ((df["High"] - prev_close) / prev_close).clip(lower=0)
            
            df["Drop_Score"] = drop_move / df["Dynamic_Sigma"]
            df["Rise_Score"] = rise_move / df["Dynamic_Sigma"]

            
            df["Vol_MA20"] = pd.to_numeric(df["Volume"], errors="coerce").rolling(20).mean()

            row = df.iloc[-2]
            
            # 欠損・非finiteがある行は「補完せず」候補化しない（continue）
            # Rise_Score / Drop_Score もここで必ずチェックして、
            # AI入力まで NaN を流さない
            required_cols = [
                "Dynamic_Sigma",
                "BandWidth",
                "BW_Change",
                "RSI",
                "Vol_Change",
                "Pct_Change",
                "Rise_Score",
                "Drop_Score",
            ]

            ok_row = True
            bad_col = ""

            for c in required_cols:
                val = row.get(c, np.nan)
                if pd.isna(val):
                    ok_row = False
                    bad_col = c
                    break
                try:
                    fv = float(val)
                    if not np.isfinite(fv):
                        ok_row = False
                        bad_col = c
                        break
                except Exception:
                    ok_row = False
                    bad_col = c
                    break

            # Dynamic_Sigma は正（>0）である必要
            if ok_row:
                try:
                    sigma_val = float(row["Dynamic_Sigma"])
                    if sigma_val <= 0.0:
                        ok_row = False
                        bad_col = "Dynamic_Sigma<=0"
                except Exception:
                    ok_row = False
                    bad_col = "Dynamic_Sigma"

            if not ok_row:
                try:
                    print(
                        f"[CAND-SKIP] sym={symbol} bad_col={bad_col} "
                        f"sigma={row.get('Dynamic_Sigma', np.nan)} "
                        f"rise={row.get('Rise_Score', np.nan)} "
                        f"drop={row.get('Drop_Score', np.nan)}"
                    )
                except Exception:
                    pass
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

            _short_ok = (row["Drop_Score"] >= CAND_SIGMA) and ALLOW_SHORT
            _long_ok = (row["Rise_Score"] >= CAND_SIGMA) and ALLOW_LONG and (row["Close"] > df.iloc[-6]["Close"])

            if _short_ok and _long_ok:
                # 両方成立 → Scoreが高い方を採用
                if row["Drop_Score"] >= row["Rise_Score"]:
                    is_sell = True
                    signal_type = "SHORT"
                else:
                    is_buy = True
                    signal_type = "LONG"
            elif _short_ok:
                is_sell = True
                signal_type = "SHORT"
            elif _long_ok:
                is_buy = True
                signal_type = "LONG"

            if not (is_buy or is_sell):
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
            bypassed = True
            
            # 動的AI_TH（既存仕様）
            ai_th_used = AI_TH
            if DYNAMIC_AI_TH:
                ai_th_used = compute_dynamic_ai_th(AI_TH, btc_mode, median_sigma, btc_ok, BTC_CALM)
            
            # side別AIモデル
            sym_code = symbol.replace("/USDT", "")

            # AIでside選択を有効化（envでOFF可）
            AI_SIDE_SELECT = (os.environ.get("AI_SIDE_SELECT", "1").strip() == "1")

            # --- NameError防止：AIブロック突入時点で必ず初期値を持たせる（挙動は変えない）---
            dbg = {}
            ai_debug_str = ""

            proba_raw_val = ""
            proba_used_val = ""
            invert_applied_val = ""

            ai_proba_base_val = ""
            ai_proba_flip_val = ""
            ai_proba_used_val = ""
            ai_margin_val = ""

            base_side = "LONG" if is_buy else "SHORT"
            flip_side = "SHORT" if base_side == "LONG" else "LONG"
            flip_allowed = (ALLOW_SHORT if flip_side == "SHORT" else ALLOW_LONG)

            # 学習側と整合：Sideに応じて Rise/Drop のどちらか一方だけに寄せる
            sig_score = float(max(float(row["Rise_Score"]), float(row["Drop_Score"])))

            def _make_feats(side: str) -> Optional[pd.DataFrame]:
                _is_long = 1.0 if side == "LONG" else 0.0
                _is_short = 1.0 - _is_long

                feats = pd.DataFrame([{
                    "Sigma": float(row["Dynamic_Sigma"]),
                    "BandWidth": float(row["BandWidth"]),
                    "BW_Change": float(row["BW_Change"]),
                    "RSI": float(row["RSI"]),
                    "Vol_Change": float(row["Vol_Change"]),
                    "Rise_Score": float(sig_score) * _is_long,
                    "Drop_Score": float(sig_score) * _is_short,
                    "BTC_Ret": float(btc_ret),
                    "BTC_Vol": float(btc_vol),
                }])

                feats = feats.replace([np.inf, -np.inf], np.nan)

                mask = feats.isna()
                if mask.any().any():
                    bad_cols = [c for c in feats.columns if mask[c].any()]
                    try:
                        print(
                            f"[AI-FEAT-SKIP] sym={symbol} side={side} "
                            f"bad_cols={bad_cols} "
                            f"btc_ret={btc_ret} btc_vol={btc_vol} "
                            f"volchg={row.get('Vol_Change', np.nan)} "
                            f"sigma={row.get('Dynamic_Sigma', np.nan)} "
                            f"score={sig_score}"
                        )
                    except Exception:
                        pass
                    return None

                return feats

            def _score_side(side: str) -> Tuple[Optional[float], Optional[float], bool, Dict[str, Any]]:
                """
                戻り:
                  - score_used: 判定に使うスコア（side別 invert を反映）
                  - score_raw : model.predict_proba の生値（反転前）
                  - bypass    : Trueなら予測できていない
                  - dbg       : safe_predict_proba のデバッグ（追跡用情報を必ず追記）
                """
                model_for_side, model_ver_side, model_src_side = get_ai_model_for_side(side, sym_code)
                invert_for_side = _get_side_model_invert(side)

                d_base: Dict[str, Any] = {
                    "side": str(side),
                    "model_version": model_ver_side,
                    "model_source": model_src_side,
                    "model_type": f"{str(side).strip().upper()}_AI",
                }

                feats_side = _make_feats(side)
                if feats_side is None:
                    d = dict(d_base)
                    d["action"] = "precheck_feature_skip"
                    d["error"] = "features contain NaN before safe_predict_proba"
                    d["proba_raw"] = None
                    d["proba_used"] = None
                    return None, None, True, d

                proba_x, bypass_x, dbg_x = safe_predict_proba(model_for_side, feats_side)

                d = (dbg_x or {})
                if not isinstance(d, dict):
                    d = {"dbg": str(d)}
                d.update(d_base)

                # bypass なら予測しない（固定値判定もしない）
                if bypass_x:
                    d["ai_proba_invert_enabled"] = bool(invert_for_side)
                    d["ai_proba_invert_applied"] = False
                    d["proba_raw"] = None
                    d["proba_used"] = None
                    return None, None, True, d

                # 念のため：None は絶対にパースしない
                if proba_x is None:
                    d["error"] = "no_proba"
                    d["ai_proba_invert_enabled"] = bool(invert_for_side)
                    d["ai_proba_invert_applied"] = False
                    d["proba_raw"] = None
                    d["proba_used"] = None
                    return None, None, True, d

                try:
                    p = np.asarray(proba_x, dtype=float)

                    # 1列目が「Win」だと決め打ちせず、classes_ を見て win_idx を確定
                    win_idx = 1
                    cls_list = None
                    try:
                        cls = getattr(model_for_side, "classes_", None)
                        cls_list = list(cls) if cls is not None else []
                        if cls_list and (1 in cls_list):
                            win_idx = cls_list.index(1)
                    except Exception:
                        win_idx = 1

                    s_raw = float(p[0][win_idx])
                    if not np.isfinite(s_raw):
                        d["error"] = "non_finite_proba"
                        d["ai_proba_invert_enabled"] = bool(invert_for_side)
                        d["ai_proba_invert_applied"] = False
                        d["proba_raw"] = None
                        d["proba_used"] = None
                        return None, None, True, d

                    s_used = (1.0 - s_raw) if invert_for_side else s_raw

                    d["classes_"] = [int(x) for x in cls_list] if cls_list is not None else None
                    d["win_index"] = int(win_idx)
                    d["ai_proba_invert_enabled"] = bool(invert_for_side)
                    d["ai_proba_invert_applied"] = bool(invert_for_side)
                    d["proba_raw"] = float(s_raw)
                    d["proba_used"] = float(s_used)
                    d["score_raw"] = float(s_raw)
                    d["score_used"] = float(s_used)
                    d["invert_applied"] = bool(invert_for_side)

                    return float(s_used), float(s_raw), False, d

                except Exception as e:
                    d["error"] = "proba_parse_failed"
                    d["detail"] = str(e)
                    d["ai_proba_invert_enabled"] = bool(invert_for_side)
                    d["ai_proba_invert_applied"] = False
                    d["proba_raw"] = None
                    d["proba_used"] = None
                    return None, None, True, d



            # AI_SIDE_SELECT=0 のときは、旧 logic_main 側の AI 判定を完全にスキップする。
            # V2 selection pipeline 側の AI だけで判定させたいので、
            # base/flip のどちらも _score_side() を呼ばない。
            if not AI_SIDE_SELECT:
                score_b = None
                score_b_raw = None
                bypass_b = True
                dbg_b: Dict[str, Any] = {"skipped": True, "reason": "AI_SIDE_SELECT=0(base_disabled)"}

                crash_forbid_flip = False
                try:
                    crash_forbid_flip = (float(sig_score) >= float(CRASH_SIGMA)) or (float(row["Dynamic_Sigma"]) >= float(CRASH_VOLSIGMA))
                except Exception:
                    crash_forbid_flip = False

                score_f = None
                score_f_raw = None
                bypass_f = True
                if crash_forbid_flip:
                    dbg_f: Dict[str, Any] = {"skipped": True, "reason": "crash_forbid_flip"}
                else:
                    dbg_f: Dict[str, Any] = {"skipped": True, "reason": "AI_SIDE_SELECT=0(flip_disabled)"}

                chosen_side = base_side
                chosen_score = None
                flipped = False

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
                    "old_logic_ai_skipped": True,
                }

                ai_score = None
                bypassed = False
                ai_pass = True

            else:
                # base 採点（used=判定用, raw=反転前）
                score_b, score_b_raw, bypass_b, dbg_b = _score_side(base_side)

                # Crash/Trend 判定：荒れ相場（強トレンド/高ボラ）では flip（逆張り側）を禁止する
                crash_forbid_flip = False
                try:
                    crash_forbid_flip = (float(sig_score) >= float(CRASH_SIGMA)) or (float(row["Dynamic_Sigma"]) >= float(CRASH_VOLSIGMA))
                except Exception:
                    crash_forbid_flip = False

                # flip 採点（許可 & 有効 & 荒れ相場でない時だけ）
                score_f = None
                score_f_raw = None
                bypass_f = True
                dbg_f: Dict[str, Any] = {"skipped": True, "reason": "flip_disabled_or_not_allowed"}
                if AI_SIDE_SELECT and bool(flip_allowed) and (not crash_forbid_flip):
                    score_f, score_f_raw, bypass_f, dbg_f = _score_side(flip_side)
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

                    # ★Phase1: 方向別閾値
                    # LONG  : ai_th_used と LONG_AI_TH の高い方を使う
                    #         → LONG を厳しくできる
                    # SHORT : 通常は ai_th_used と SHORT_AI_TH の低い方を使う
                    #         ただし BTC_Mode=Up の SHORT は SHORT_AI_TH_UP も考慮して厳しくする

                    if chosen_side == "LONG":

                        ai_th_effective = max(float(ai_th_used), float(LONG_AI_TH))

                        src_sig = locals().get("item", {})
                        if not isinstance(src_sig, dict):
                            src_sig = {}

                        long_ai_sig = {
                            "symbol": src_sig.get("symbol", row.get("Symbol", row.get("symbol", sym_code))),
                            "p1_score": src_sig.get("p1_score", row.get("P1_TrendScore", row.get("p1_score", np.nan))),
                            "p2_score": src_sig.get("p2_score", row.get("P2_FundingScore", row.get("p2_score", np.nan))),
                            "rsi": src_sig.get("rsi", row.get("RSI", row.get("rsi", np.nan))),
                            "btc_mode_compat": src_sig.get("btc_mode_compat", row.get("BTC_Mode_Compat", row.get("btc_mode_compat", btc_mode))),
                            "hour_jst": src_sig.get("hour", row.get("Hour_JST", row.get("hour", row.get("hour_jst", np.nan)))),
                            "fr_available": src_sig.get("fr_available", row.get("FR_Available", row.get("fr_available", ""))),
                        }

                        strong_ok, strong_reason = _check_long_strong_bucket(long_ai_sig)
                        alt_ok = _is_long_alt_win_bucket(long_ai_sig)

                        if strong_ok:
                            ai_th_effective = min(float(ai_th_effective), float(V2_LONG_STRONG_AI_TH))
                        elif alt_ok:
                            ai_th_effective = min(float(ai_th_effective), float(V2_LONG_ALT_AI_TH))

                        dbg["long_ai_mode"] = "soft"
                        dbg["long_ai_strong_ok"] = bool(strong_ok)
                        dbg["long_ai_strong_reason"] = str(strong_reason)
                        dbg["long_ai_alt_ok"] = bool(alt_ok)
                        dbg["long_ai_score"] = float(ai_score)
                        dbg["long_ai_th_effective"] = float(ai_th_effective)
                        dbg["long_ai_sig"] = {
                            "symbol": str(long_ai_sig.get("symbol", "")),
                            "p1_score": _safe_float_or_nan(long_ai_sig.get("p1_score")),
                            "p2_score": _safe_float_or_nan(long_ai_sig.get("p2_score")),
                            "rsi": _safe_float_or_nan(long_ai_sig.get("rsi")),
                            "btc_mode_compat": str(long_ai_sig.get("btc_mode_compat", "")),
                            "hour_jst": long_ai_sig.get("hour_jst"),
                            "fr_available": long_ai_sig.get("fr_available"),
                        }

                        print(
                            "[V2-LONG-AI-CHECK] "
                            f"sym={str(long_ai_sig.get('symbol', ''))} "
                            f"strong_ok={strong_ok} "
                            f"strong_reason={strong_reason} "
                            f"alt_ok={alt_ok} "
                            f"ai_score={float(ai_score):.6f} "
                            f"ai_th_used={float(ai_th_used):.6f} "
                            f"long_ai_th={float(LONG_AI_TH):.6f} "
                            f"strong_ai_th={float(V2_LONG_STRONG_AI_TH):.6f} "
                            f"alt_ai_th={float(V2_LONG_ALT_AI_TH):.6f} "
                            f"ai_th_effective={float(ai_th_effective):.6f} "
                            f"p1={_safe_float_or_nan(long_ai_sig.get('p1_score'))} "
                            f"p2={_safe_float_or_nan(long_ai_sig.get('p2_score'))} "
                            f"rsi={_safe_float_or_nan(long_ai_sig.get('rsi'))} "
                            f"btc_mode={str(long_ai_sig.get('btc_mode_compat', ''))} "
                            f"hour={long_ai_sig.get('hour_jst')} "
                            f"fr_available={long_ai_sig.get('fr_available')}"
                        )

                        ai_pass = (float(ai_score) >= float(ai_th_effective))

                    else:

                        short_ai_sig = {
                            "p1_score": row.get("P1_TrendScore", row.get("p1_score", np.nan)),
                            "p2_score": row.get("P2_FundingScore", row.get("p2_score", np.nan)),
                            "rsi": row.get("RSI", row.get("rsi", np.nan)),
                            "btc_mode_compat": row.get("BTC_Mode_Compat", row.get("btc_mode_compat", btc_mode)),
                            "hour_jst": row.get("Hour_JST", row.get("hour", row.get("hour_jst", np.nan))),
                            "fr_available": row.get("FR_Available", row.get("fr_available", "")),
                            "vol_ratio": row.get("VolRatio", row.get("vol_ratio", np.nan)),
                        }

                        short_core_ok = _is_short_win_bucket(short_ai_sig)
                        short_push_ok = _is_short_strong_hour_bucket(short_ai_sig)

                        # 強いSHORTは AI hard gate を bypass
                        ai_th_effective = 0.0
                        if short_push_ok and V2_SHORT_PAUSE_ALLOW_STRONG_BUCKET:
                            ai_pass = True
                        else:
                            ai_pass = bool(short_core_ok)

                        dbg["short_ai_mode"] = "soft"
                        dbg["short_ai_core_ok"] = bool(short_core_ok)
                        dbg["short_ai_push_ok"] = bool(short_push_ok)
                        dbg["short_ai_score"] = float(ai_score)
                        dbg["short_ai_th_effective"] = 0.0
                        dbg["short_ai_below_old_threshold"] = bool(
                            float(ai_score) < float(min(float(ai_th_used), float(SHORT_AI_TH)))
                        )

                        if short_push_ok:
                            signal_type = "SHORT_PUSH"



            if (not ai_pass) and (not bypassed) and LONG_BYPASS_ON_BTC_UP:

                try:

                    if (chosen_side == "LONG"

                            and str(btc_mode) == "Up"

                            and bool(BTC_CALM)

                            and float(item.get("rsi", 999)) < float(LONG_BYPASS_RSI_MAX)

                            and float(item.get("score", 0)) >= float(LONG_BYPASS_SCORE_MIN)):

                        ai_pass = True

                        print(f"[AI] LONG bypass activated sym={sym_code} btc_mode=Up calm=True"

                              f" rsi={item.get('rsi','')} score={item.get('score','')}"

                              f" ai_score={ai_score} ai_th={ai_th_effective}")

                except Exception as _bp_err:

                    print(f"[AI] LONG bypass check failed sym={sym_code}: {_bp_err}")

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


            # --- NameError防止：どの分岐でも必ず定義しておく ---
            if "dbg" not in locals() or not isinstance(dbg, dict):
                dbg = {}

            if "ai_proba_base_val" not in locals():
                ai_proba_base_val = ""
            if "ai_proba_flip_val" not in locals():
                ai_proba_flip_val = ""
            if "ai_proba_used_val" not in locals():
                ai_proba_used_val = ""
            if "ai_margin_val" not in locals():
                ai_margin_val = ""

            # --- dbg から「証拠」用の値を抜く（chosen_side の dbg を採用）---
            chosen_dbg = None
            try:
                if str(chosen_side) == str(base_side):
                    chosen_dbg = dbg.get("dbg_base", None)
                elif str(chosen_side) == str(flip_side):
                    chosen_dbg = dbg.get("dbg_flip", None)
            except Exception:
                chosen_dbg = None
            
            if isinstance(chosen_dbg, dict):
                proba_raw_val = chosen_dbg.get("proba_raw", "")
                proba_used_val = chosen_dbg.get("proba_used", "")
                invert_applied_val = chosen_dbg.get("ai_proba_invert_applied", "")
            else:
                proba_raw_val = ""
                proba_used_val = ""
                invert_applied_val = ""
            
            # --- DIRECT/REVERSE は ai_debug(AO) ではなく tag に逃がす ---
            invert_mode_tag = "REVERSE" if bool(AI_PROBA_INVERT) else "DIRECT"
            
            # --- Sheetsへ書く ai_debug(AO) は「必ず」JSON文字列にする ---
            # ※ これで AO が DIRECT/REVERSE になる事故を潰します（ここで上書き確定）
            try:
                ai_debug_str = json.dumps(dbg, ensure_ascii=False, separators=(",", ":"), default=str)
            except Exception:
                ai_debug_str = str(dbg)
            
            item = {
                "symbol": symbol.replace("/USDT", ""),
                "time": int(row["Time"]),
                "is_buy": bool(is_buy),
                "is_sell": bool(is_sell),
                "close": float(row["Close"]),
                "score": float(max(row["Drop_Score"], row["Rise_Score"])),
                "sigma": float(row["Dynamic_Sigma"]),   # 実質 VolSigma として扱う
                "rsi": float(row["RSI"]),
                "type": signal_type,
                "dt": datetime.fromtimestamp(int(row["Time"]) / 1000, JST),
                "ai_score": ai_score,
                "ai_pass": bool(ai_pass),

                # learn_log(AO) ：JSON文字列
                "ai_debug": ai_debug_str,

                # DIRECT/REVERSE は tag 列へ（learn_log側で追跡できる）
                "tag": invert_mode_tag,

                # AI_PROBA_INVERT の「証拠」も item に入れる（必要なら後で列追加できる）
                "proba_raw": proba_raw_val,
                "proba_used": proba_used_val,
                "invert_applied": invert_applied_val,
                "is_flip": bool(flipped),

                "chg_pct": chg_pct_val,
                "vol_ratio": vol_ratio_val,

                # learn_log(AP..AS) 用
                "ai_proba_base": ai_proba_base_val,
                "ai_proba_flip": ai_proba_flip_val,
                "ai_proba_used": ai_proba_used_val,
                "ai_margin": ai_margin_val,

                "BandWidth": float(row["BandWidth"]),
                "BW_Change": float(row["BW_Change"]),
                "Vol_Change": float(row["Vol_Change"]),

                # --- 追加：通知レジーム判定・通知フィルター用（Phase1） ---
                "btc_mode": str(btc_mode),
                "btc_1h_change": float(btc_1h_change),
                "btc_calm": bool(BTC_CALM),
                "market_tag": ("STORM" if not BTC_CALM else "CALM"),

                # NOTE:
                # ここでは _to_finite_float_or_none(...) を使わない（この位置では未定義の可能性があるため）
                # 後段の notify filter 側で _nf(...) により安全変換する
                "btc_ret": btc_ret,
                "btc_vol": btc_vol,

                # --- market AI（既存） ---
                "market_ai_score": "",
                "market_ai_pass": "",
                "market_ai_debug": "",

                # --- 追加：通知判定結果の一時保持用（Phase1/Phase2で使用） ---
                "_regime_mode": "",
                "_notify_tier": "",
                "_notify_reason": "",
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

            # 1.5) R1: Score上限ブロック
            if guard_ok:
                try:
                    sc_raw = item.get("score", None)
                    if sc_raw is not None:
                        sc_v = float(sc_raw)
                        if np.isfinite(sc_v) and sc_v > SCORE_MAX:
                            guard_ok = False
                            print(f"[GR] skip alert sym={sym_u} score={sc_v:.2f} > {SCORE_MAX} (R1)")
                except Exception:
                    pass


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
                                # R3追加: SHORT×RSI上限
                                if guard_ok and bool(item.get("is_sell", False)) and (rsi_v > RSI_SHORT_MAX):
                                    guard_ok = False
                                    print(f"[GR] skip alert sym={sym_u} side=SHORT rsi={rsi_v:.2f} > {RSI_SHORT_MAX} (R3)")
                        except Exception:
                            guard_ok = False
                            print(f"[GR] skip alert sym={sym_u} rsi=parse_failed (no-signal)")

                except Exception:
                    pass


            # 2.5) SHORT BTC下落深度ガード
            #
            # 方針（品質優先）:
            #   - BTC_Mode=Range のときだけ「浅い下落SHORT」を抑止する（乱発防止）
            #   - BTC_Mode=Up/Down では shallow dip を適用しない（勝てるSHORTを潰さないため）
            #
            # ※ is_sell=True（SHORT）のときだけ対象
            
            if guard_ok and bool(item.get("is_sell", False)):
            
                try:
                    _btc_mode_v = str(item.get("btc_mode", "")).strip()
                    _btc_1h_v = item.get("btc_1h_change", None)
            
                    # Range のときだけ shallow dip filter を適用
                    if _btc_mode_v == "Range" and _btc_1h_v is not None and str(_btc_1h_v).strip() != "":
            
                        _btc_1h_f = float(_btc_1h_v)
            
                        if np.isfinite(_btc_1h_f) and _btc_1h_f > float(SHORT_BTC_1H_MIN):
            
                            guard_ok = False
            
                            print(
                                f"[GR] skip alert sym={sym_u} side=SHORT"
                                f" btc_mode={_btc_mode_v}"
                                f" btc_1h={_btc_1h_f:.6f} > {SHORT_BTC_1H_MIN}"
                                f" (shallow dip filter)"
                            )
            
                except Exception:
                    pass

            # 2.55) SHORT × BTC_Mode=Down × BandWidth ブロック
            #       BTC下落トレンド中にBandWidthが広い（ボラ拡大中の）SHORTを止める
            if guard_ok and bool(item.get("is_sell", False)):
                try:
                    _btc_mode_bw = str(item.get("btc_mode", "")).strip()
                    if _btc_mode_bw == "Down":
                        _bw_raw = item.get("BandWidth", None)
                        if _bw_raw is not None and str(_bw_raw).strip() != "":
                            _bw_f = float(_bw_raw)
                            if np.isfinite(_bw_f) and _bw_f >= SHORT_DOWN_BW_BLOCK:
                                guard_ok = False
                                print(
                                    f"[GR] skip alert sym={sym_u} side=SHORT"
                                    f" btc_mode={_btc_mode_bw}"
                                    f" BandWidth={_bw_f:.6f} >= {SHORT_DOWN_BW_BLOCK}"
                                    f" (SHORT_DOWN_BW_BLOCK)"
                                )
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

        # ===== AI debug values for analysis (NO zero-fill; invalid => blank) =====
        def _to_finite_float_or_none(x):
            try:
                if x is None:
                    return None
                v = float(x)
                if not np.isfinite(v):
                    return None
                return v
            except Exception:
                return None

        # item から ai_proba_* を拾う（無効値は None）
        ai_proba_base_v = _to_finite_float_or_none(item.get("ai_proba_base", None))
        ai_proba_flip_v = _to_finite_float_or_none(item.get("ai_proba_flip", None))
        ai_proba_used_v = _to_finite_float_or_none(item.get("ai_proba_used", None))

        # ai_th_used が未定義の経路でも落ちないように AI_TH にフォールバック
        try:
            _th_src = ai_th_used if "ai_th_used" in locals() else AI_TH
            ai_th_effective = float(_th_src)
            if not np.isfinite(ai_th_effective):
                ai_th_effective = None
        except Exception:
            ai_th_effective = None

        # margin は used - effective_th（揃わないなら None のまま）
        ai_margin_v = None
        if ai_proba_used_v is not None and ai_th_effective is not None:
            ai_margin_v = ai_proba_used_v - ai_th_effective

        # 欠損理由（0埋め禁止：欠損なら空欄で残す）
        missing_reason = None
        if ai_proba_used_v is None:
            missing_reason = "missing_used"
        elif ai_th_effective is None:
            missing_reason = "missing_ai_th"

        if missing_reason is not None:
            print(
                "[AI_DEC] skip"
                f" sym={sym} side={('LONG' if item['is_buy'] else 'SHORT')}"
                f" reason={missing_reason}"
                f" base={ai_proba_base_v} flip={ai_proba_flip_v} used={ai_proba_used_v} ai_th={ai_th_effective}"
                f" ai_pass={bool(item.get('ai_pass'))}"
                f" invert_env={os.environ.get('AI_PROBA_INVERT', '')}"
            )
            # 欠損時は列には空欄を書き込む（候補行は残す）
            ai_proba_base_v = None
            ai_proba_flip_v = None
            ai_proba_used_v = None
            ai_margin_v = None
        else:
            # invert の証拠（item に入っていれば出す。無ければ空のまま）
            ai_proba_invert_env = os.environ.get("AI_PROBA_INVERT", "")
            proba_raw_v = item.get("proba_raw", "")
            invert_applied_v = item.get("invert_applied", "")

            print(
                "[AI_DEC]"
                f" sym={sym} side={('LONG' if item['is_buy'] else 'SHORT')}"
                f" base={ai_proba_base_v} flip={ai_proba_flip_v} used={ai_proba_used_v} margin={ai_margin_v}"
                f" ai_th={ai_th_effective} ai_pass={bool(item.get('ai_pass'))}"
                f" invert_env={ai_proba_invert_env} invert_applied={invert_applied_v} proba_raw={proba_raw_v}"
            )

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
            DEFAULT_LEV, "", "", bool(item["ai_pass"]), bool(BTC_CALM),
            VERSION, item["type"], "", "",
            ("STORM" if not BTC_CALM else "CALM"), btc_mode, float(btc_1h_change),
            float(item["rsi"]), note_str,

            # --- ここは「候補段階」なので基本空欄（後でDONE時に埋まる） ---
            "", "", "", "", "", "", "",

            # --- optional-but-we-want-to-always-write ---
            bw_val, bw_chg_val, vol_chg_val, btc_ret_val, btc_vol_val,
            m_ai_score, m_ai_pass, m_ai_debug,

            # --- ★追加：learn_log の末尾列（AO / AP..AS）---
            item.get("ai_debug", ""),          # AO: ai_debug（JSON文字列）
            item.get("ai_proba_base", ""),     # AP
            item.get("ai_proba_flip", ""),     # AQ
            item.get("ai_proba_used", ""),     # AR
            item.get("ai_margin", ""),         # AS
        ]

        # ai_debug 列が存在する場合はその列位置に代入（列ズレ防止）
        if ai_debug_field:
            ai_debug_idx = -1
            for i, h in enumerate(learn_fields):
                if str(h).strip().lower() == "ai_debug":
                    ai_debug_idx = i
                    break
            if ai_debug_idx >= 0:
                if len(row_out) <= ai_debug_idx:
                    row_out = row_out + [""] * (ai_debug_idx + 1 - len(row_out))
                row_out[ai_debug_idx] = ai_debug_label

        # ai_proba_* / ai_margin は列位置に代入（列ズレ防止）
        def _set_field_value(row_list, fields, field_name, value):
            try:
                idx0 = -1
                for j, h in enumerate(fields):
                    if str(h).strip().lower() == field_name.lower():
                        idx0 = j
                        break
                if idx0 < 0:
                    return row_list
                if len(row_list) <= idx0:
                    row_list = row_list + [""] * (idx0 + 1 - len(row_list))
                row_list[idx0] = ("" if value is None else value)
                return row_list
            except Exception:
                return row_list
        
        row_out = _set_field_value(row_out, learn_fields, "ai_proba_base", ai_proba_base_v)
        row_out = _set_field_value(row_out, learn_fields, "ai_proba_flip", ai_proba_flip_v)
        row_out = _set_field_value(row_out, learn_fields, "ai_proba_used", ai_proba_used_v)
        row_out = _set_field_value(row_out, learn_fields, "ai_margin", ai_margin_v)

        # ★追加：列が存在する時だけ「証拠」も書く（無ければ何もしない）
        row_out = _set_field_value(row_out, learn_fields, "proba_raw", _to_finite_float_or_none(item.get("proba_raw", None)))
        row_out = _set_field_value(row_out, learn_fields, "proba_used", _to_finite_float_or_none(item.get("proba_used", None)))
        row_out = _set_field_value(row_out, learn_fields, "invert_applied", item.get("invert_applied", ""))
        row_out = _set_field_value(row_out, learn_fields, "is_flip", item.get("is_flip", ""))
        # ★列ズレ防止：必ずヘッダー長に合わせる
        row_out = _pad_row_to_fields(row_out, learn_fields, fill="")

        # --- learn_log への書き込み安全化（列ズレ＆dict書き込み事故を潰す）---

        # 1) ai_debug が dict/list の場合は JSON 文字列にしてから書く
        #    ※ Sheets には dict のまま渡さない（失敗しやすい）
        try:
            idx_dbg = learn_fields.index("ai_debug")
            if idx_dbg < len(row_out) and isinstance(row_out[idx_dbg], (dict, list)):
                row_out[idx_dbg] = json.dumps(
                    row_out[idx_dbg],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
        except Exception:
            pass

        # 2) row_out の長さを learn_fields に合わせてパディング（末尾列が zip から落ちないように）
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
    
    # =========================
    # Phase1: Regime notify filter (案A)
    # =========================
    def _env_flag(name, default=False):
        v = os.getenv(name, "1" if default else "0")
        if v is None:
            return bool(default)
        return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")

    def _env_text(name, default=""):
        v = os.getenv(name, None)
        return default if v is None else str(v).strip()

    def _env_float(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    def _nf(v, default=None):
        try:
            if v is None:
                return default
            x = float(v)
            if not np.isfinite(x):
                return default
            return x
        except Exception:
            return default

    def _item_hour_jst(it):
        try:
            dtv = it.get("dt", None)
            if dtv is None:
                return None
            return int(dtv.hour)
        except Exception:
            return None

    # ---- Feature flags（Phase1=案A本番 / Phase2=案B shadowはOFFのまま仕込み）----
    ENABLE_REGIME_NOTIFY_FILTER = _env_flag("ENABLE_REGIME_NOTIFY_FILTER", True)
    ENABLE_SHADOW_NOTIFY = _env_flag("ENABLE_SHADOW_NOTIFY", False)  # Phase2で使用（今回は未使用）
    REGIME_FORCE_MODE = _env_text("REGIME_FORCE_MODE", "AUTO").upper()
    RULE_VERSION_TAG = _env_text("RULE_VERSION_TAG", "REGIME_NOTIFY_V1")

    # ---- Regime判定しきい値（初期値。後で環境変数で微調整可）----
    REGIME_BEAR_BTC1H_TH = _env_float("REGIME_BEAR_BTC1H_TH", -0.0060)      # -0.60%
    REGIME_REBOUND_BTC1H_TH = _env_float("REGIME_REBOUND_BTC1H_TH", 0.0040) # +0.40%

    # ---- NEUTRAL（通常）通知フィルター ----
    NEUTRAL_SHORT_VSIG_MAX = _env_float("NEUTRAL_SHORT_VSIG_MAX", 0.0035)
    NEUTRAL_LONG_VSIG_MAX = _env_float("NEUTRAL_LONG_VSIG_MAX", 0.0047)
    NEUTRAL_LONG_RSI_MAX = _env_float("NEUTRAL_LONG_RSI_MAX", 56.0)

    # ---- BEAR_AGGR（下落ショート優位）----
    BEAR_SHORT_RSI_MIN = _env_float("BEAR_SHORT_RSI_MIN", 38.0)
    # ★追加：RSIが低くてもAIが強い時はSHORTを許可（品質を落とさないため）
    BEAR_SHORT_RSI_BYPASS_AI_TH = _env_float("BEAR_SHORT_RSI_BYPASS_AI_TH", 0.60)

    BEAR_SHORT_BTC1H_MIN = _env_float("BEAR_SHORT_BTC1H_MIN", -0.0110)  # -1.10%
    BEAR_SHORT_BTCVOL_MAX = _env_float("BEAR_SHORT_BTCVOL_MAX", 0.0077)
    BEAR_ALLOW_LONG = _env_flag("BEAR_ALLOW_LONG", False)

    # ---- REBOUND_LONG（反発上昇）----
    REBOUND_SHORT_RSI_BLOCK = _env_float("REBOUND_SHORT_RSI_BLOCK", 64.0)
    REBOUND_SHORT_BTC1H_BLOCK = _env_float("REBOUND_SHORT_BTC1H_BLOCK", 0.0050)  # +0.50%
    REBOUND_LONG_STRONG_RSI = _env_float("REBOUND_LONG_STRONG_RSI", 74.0)

    BAD_HOURS_NEUTRAL = {14, 16, 17}

    def detect_regime_mode_for_notify(item):
        """
        戻り値:
          - NEUTRAL
          - BEAR_AGGR
          - REBOUND_LONG
        """
        try:
            forced = str(REGIME_FORCE_MODE).upper().strip()
            if forced in ("NEUTRAL", "BEAR_AGGR", "REBOUND_LONG"):
                return forced

            btc_mode_s = str(item.get("btc_mode", "")).strip().upper()
            btc1h = _nf(item.get("btc_1h_change", None), None)

            # mode文字列ヒント
            mode_has_down = ("DOWN" in btc_mode_s) or ("BEAR" in btc_mode_s)
            mode_has_up = ("UP" in btc_mode_s) or ("BULL" in btc_mode_s)

            # 数値 + mode の複合で判定（安全側）
            if (btc1h is not None) and (btc1h <= REGIME_BEAR_BTC1H_TH) and mode_has_down:
                return "BEAR_AGGR"

            if (btc1h is not None) and (btc1h >= REGIME_REBOUND_BTC1H_TH) and mode_has_up:
                return "REBOUND_LONG"

            # フォールバック（modeだけで明確な時）
            if mode_has_down and (btc1h is not None) and (btc1h < 0):
                return "BEAR_AGGR"

            if mode_has_up and (btc1h is not None) and (btc1h > 0):
                return "REBOUND_LONG"

            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    def should_notify_signal_phaseA(item, regime_mode):
        """
        戻り値: (notify_ok: bool, notify_tier: str, notify_reason: str)
        Phase1は案Aのみ（AI_Pass=False救済はまだ行わない）
        """
        try:
            if not ENABLE_REGIME_NOTIFY_FILTER:
                return True, "NORMAL", "filter_off"

            ai_pass = bool(item.get("ai_pass", False))
            if not ai_pass:
                # Phase1ではAI_Pass=Falseは通知しない（案Bでshadow育成）
                return False, "BLOCK", "ai_pass_false"

            side = "LONG" if bool(item.get("is_buy", False)) else "SHORT"
            hour = _item_hour_jst(item)
            rsi = _nf(item.get("rsi", None), None)
            vsig = _nf(item.get("sigma", None), None)          # item["sigma"] は実質 VolSigma
            bw = _nf(item.get("BandWidth", None), None)
            btc1h = _nf(item.get("btc_1h_change", None), None)
            btc_vol = _nf(item.get("btc_vol", None), None)

            # ---- NEUTRAL（通常）----
            if regime_mode == "NEUTRAL":
                if hour in BAD_HOURS_NEUTRAL:
                    return False, "BLOCK", f"neutral_bad_hour_{hour}"

                if side == "SHORT":
                    if (vsig is not None) and (vsig > NEUTRAL_SHORT_VSIG_MAX):
                        return False, "BLOCK", f"neutral_short_vsig>{NEUTRAL_SHORT_VSIG_MAX}"
                    return True, "NORMAL", "neutral_short_ok"

                # LONG
                if (vsig is not None) and (vsig > NEUTRAL_LONG_VSIG_MAX):
                    return False, "BLOCK", f"neutral_long_vsig>{NEUTRAL_LONG_VSIG_MAX}"

                if (rsi is not None) and (rsi > NEUTRAL_LONG_RSI_MAX):
                    return False, "BLOCK", f"neutral_long_rsi>{NEUTRAL_LONG_RSI_MAX}"

                return True, "NORMAL", "neutral_long_ok"

            # ---- BEAR_AGGR（下落ショート優位）----
            if regime_mode == "BEAR_AGGR":
                if side == "LONG":
                    if not BEAR_ALLOW_LONG:
                        return False, "BLOCK", "bear_long_blocked"
                    return True, "NORMAL", "bear_long_allowed"

                # SHORT: 基本通すが「売られすぎ最終局面」を抑制
                # ただし、AIが強い（ai_score が高い）場合は例外として許可（品質維持）
                if (rsi is not None) and (rsi < BEAR_SHORT_RSI_MIN):
                    ai_score = _nf(item.get("ai_score", None), None)  # 0〜1（usedスコア）
                    if (ai_score is None) or (ai_score < BEAR_SHORT_RSI_BYPASS_AI_TH):
                        return False, "BLOCK", f"bear_short_rsi<{BEAR_SHORT_RSI_MIN}(ai<{BEAR_SHORT_RSI_BYPASS_AI_TH})"

                if (btc1h is not None) and (btc1h < BEAR_SHORT_BTC1H_MIN):
                    return False, "BLOCK", f"bear_short_btc1h<{BEAR_SHORT_BTC1H_MIN}"

                if (btc_vol is not None) and (btc_vol >= BEAR_SHORT_BTCVOL_MAX):
                    return False, "BLOCK", f"bear_short_btcvol>={BEAR_SHORT_BTCVOL_MAX}"

                return True, "NORMAL", "bear_short_ok"

            # ---- REBOUND_LONG（反発上昇）----
            if regime_mode == "REBOUND_LONG":
                if side == "LONG":
                    if (rsi is not None) and (rsi >= REBOUND_LONG_STRONG_RSI):
                        return True, "STRONG", f"rebound_long_rsi>={REBOUND_LONG_STRONG_RSI}"
                    return True, "NORMAL", "rebound_long_ok"

                # SHORTは厳選（上昇逆行ショートをブロック）
                if (rsi is not None) and (rsi >= REBOUND_SHORT_RSI_BLOCK):
                    return False, "BLOCK", f"rebound_short_rsi>={REBOUND_SHORT_RSI_BLOCK}"

                if (btc1h is not None) and (btc1h >= REBOUND_SHORT_BTC1H_BLOCK):
                    return False, "BLOCK", f"rebound_short_btc1h>={REBOUND_SHORT_BTC1H_BLOCK}"

                # ここまで通れば「反発中でも許容SHORT」
                # （Phase1では簡易版。Phase2以降でSTRONG条件を追加）
                return True, "NORMAL", "rebound_short_ok"

            # 予期しない値は安全側でNEUTRAL相当
            return True, "NORMAL", "regime_unknown_fallback"

        except Exception as e:
            # フィルター異常で止めない（安全に既存動作寄せ）
            print(f"[NOTIFY_FILTER] error fallback pass sym={item.get('symbol','?')}: {e}")
            return True, "NORMAL", "filter_error_fallback_pass"

    # 旧V1通知は、明示的に許可したときだけ動かす
    engine_mode = str(SIGNAL_ENGINE).strip().lower()
    legacy_live_notify_enabled = bool(ENABLE_LEGACY_REGIME_NOTIFY) and (engine_mode == "v1")

    if not legacy_live_notify_enabled:
        print(
            f"[LEGACY-NOTIFY] skipped "
            f"SIGNAL_ENGINE={engine_mode} "
            f"ENABLE_LEGACY_REGIME_NOTIFY={ENABLE_LEGACY_REGIME_NOTIFY}"
        )

    # 旧通知ループに入れる候補は、legacy許可時だけ残す
    filtered = (
        sorted(pending_alerts, key=lambda x: x["score"], reverse=True)
        if legacy_live_notify_enabled
        else []
    )


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

        # --- Regime / Notify filter (Phase1: 案A) ---
        regime_mode = detect_regime_mode_for_notify(item)
        notify_ok, notify_tier, notify_reason = should_notify_signal_phaseA(item, regime_mode)

        # item にも残しておく（後続の note_compact / debug 用）
        item["_regime_mode"] = regime_mode
        item["_notify_tier"] = notify_tier
        item["_notify_reason"] = notify_reason

        if not notify_ok:
            print(
                f"[NOTIFY_FILTER] blocked "
                f"sym={sym} side={'LONG' if item.get('is_buy', False) else 'SHORT'} "
                f"regime={regime_mode} reason={notify_reason}"
            )
            continue

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

        # itemに積んだBTC情報を優先して表示（無ければ既存変数をフォールバック）
        btc_mode_disp = str(item.get("btc_mode", btc_mode))
        try:
            btc_1h_disp_val = float(item.get("btc_1h_change", btc_1h_change))
            btc_1h_disp = f"{btc_1h_disp_val:.2%}"
        except Exception:
            btc_1h_disp = "N/A"
        btc_calm_disp = bool(item.get("btc_calm", BTC_CALM))

        msg = (
            f"{icon} **{d_str}** {icon}\n"
            f"{VERSION} [{RULE_VERSION_TAG}]\n"
            f"💎 {sym} ({item['type']})\n"
            f"📈 Score:{item['score']:.2f}σ  AI:{ai_disp}\n"
            f"🧭 Regime:{regime_mode}  Tier:{notify_tier}\n"
            f"🟦 BTC:{btc_mode_disp} 1h:{btc_1h_disp}  Calm:{btc_calm_disp}  BTC_OK:{btc_ok}\n"
            f"💰 {cp:.4f}\n"
            f"🎯 TP: {tp:.4f} ({tp_pct:.2f}%) HL:{hl_tp_pct:.1f}%\n"
            f"🛑 SL: {sl:.4f} ({sl_pct:.2f}%) HL:{hl_sl_pct:.1f}%"
        )

        send_discord_message(msg)

        # ★重要：送信処理の後で記録する（送信失敗時の誤更新を避ける）
        last_alert_records[sym] = ts_ms

        count += 1

        # itemに積んだBTC情報を優先（無ければ既存変数をフォールバック）
        btc_mode_for_note = str(item.get("btc_mode", btc_mode))
        try:
            btc_1h_for_note = float(item.get("btc_1h_change", btc_1h_change))
            btc_1h_for_note_str = f"{btc_1h_for_note:.2%}"
        except Exception:
            btc_1h_for_note_str = "N/A"

        btc_calm_for_tbl = bool(item.get("btc_calm", BTC_CALM))
        regime_for_note = str(item.get("_regime_mode", ""))
        tier_for_note = str(item.get("_notify_tier", ""))
        reason_for_note = str(item.get("_notify_reason", ""))

        parts = [
            f"{item['type']}",
            f"AI:{ai_disp}",
            f"RSI:{item['rsi']:.1f}",
            fmt_opt("Chg:", item["chg_pct"], "%"),
            fmt_opt("VolR:", item["vol_ratio"]),
            f"BTC:{btc_mode_for_note}",
            f"1h:{btc_1h_for_note_str}",
            f"BTC_OK:{btc_ok}",
            (f"RG:{regime_for_note}" if regime_for_note else ""),
            (f"Tier:{tier_for_note}" if tier_for_note else ""),
            (f"NF:{reason_for_note}" if reason_for_note else ""),
        ]
        note_compact = " | ".join([p for p in parts if p])

        row_tbl = [
            dt_cell, sym, "LONG" if item["is_buy"] else "SHORT",
            float(cp), float(item["score"]), float(item["sigma"]), "AI_PASS",
            float(tp), float(sl), float(tp_pct), float(sl_pct),
            lev, float(tp_pct * lev), float(sl_pct * lev),
            btc_calm_for_tbl, True, VERSION, note_compact,
            item["chg_pct"], item["vol_ratio"], "MARKET",
        ]

        # ★列ズレ防止：TABLE_FIELDS の長さに合わせる（tableが37列でも事故らない）
        row_tbl = _pad_row_to_fields(row_tbl, TABLE_FIELDS, fill="")

        alert_rows.append(row_tbl)

        table_keys.add(k)

    if alert_rows:
        append_rows_to_sheet(MAIN_SHEET_NAME, alert_rows, TABLE_FIELDS)

    # --- V2 Shadow Mode ---
    if SIGNAL_ENGINE in ("shadow", "v2", "v2_live"):
        try:
            v2_result = v2_shadow_run(exchange, now_jst, force=force)
            print(f"[V2] {v2_result}")
        except Exception as e:
            print(f"[V2-ERR] shadow run crashed: {e}")

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

        # 最後の1本が「形成中の15m足」なら除外して、確定足だけで判定する
        if len(candles) >= 1:
            last_open_ts_ms = int(candles[-1][0])
            _log_judge_15m_bar(last_open_ts_ms, label=f"{market}-15m")

            now_ms = int(time.time() * 1000)
            forming = now_ms < (last_open_ts_ms + 15 * 60 * 1000)

            if forming:
                # forming足（今の足）を落として、直前の確定足で判定する
                if len(candles) >= 2:
                    candles = candles[:-1]
                    last_open_ts_ms = int(candles[-1][0])
                    _log_judge_15m_bar(last_open_ts_ms, label=f"{market}-15m-closed")
                else:
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
    if V1_DISABLE:
        print("[V1-DISABLED] judge_main skipped")
        return "V1 judge disabled"

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

# 診断用: AI_PROBA_INVERT の「白黒」確認（1回で確定させる）
@app.route("/ai_smoke", methods=["GET"])
def ai_smoke():
    """
    統計ではなく「1回の実行で白黒」を付けるための診断用。
    - predict を必ず走らせる入力を用意
    - proba_raw / proba_used / invert_applied を必ず返す
    """
    if ai_model is None:
        return jsonify({
            "ok": False,
            "model_loaded": False,
            "reason": "ai_model is None",
            "model_version": os.environ.get("MODEL_VERSION", ""),
        }), 200

    # グローバル定義（冒頭の AI_PROBA_INVERT）を使う
    invert_env = bool(AI_PROBA_INVERT)

    # ---- 安全な固定入力（診断用）----
    expected_cols = None
    try:
        if hasattr(ai_model, "feature_names_in_"):
            expected_cols = list(getattr(ai_model, "feature_names_in_", [])) or None
    except Exception:
        expected_cols = None

    if expected_cols:
        feats = pd.DataFrame([{c: 0.0 for c in expected_cols}])
    else:
        # 修正後（交互作用項追加）
        feats = pd.DataFrame([{
            "Sigma": 0.0,
            "BandWidth": 0.0,
            "BW Change": 0.0,
            "RSI": 50.0,
            "Vol Change": 0.0,
            "BTC Ret": 0.0,
            "BTC Vol": 0.0,
            "Score": 0.0,
            "Is Long": 0.0,
            "Long x BTC Ret": 0.0,
            "Long x RSI": 0.0,
        }])

    proba = None
    bypassed = True
    dbg = {"info": "init"}

    try:
        proba, bypassed, dbg = safe_predict_proba(ai_model, feats)
    except Exception as e:
        dbg = {"error": f"{type(e).__name__}: {e}"}
        proba = None
        bypassed = True

    # ---- raw/used を必ず作る（白黒判定用）----
    proba_raw = None
    proba_used = None
    invert_applied = False
    reason = ""

    d = dbg if isinstance(dbg, dict) else {"dbg": str(dbg)}

    if bypassed or proba is None:
        reason = d.get("reason", "bypassed_or_no_proba")
    else:
        try:
            p = np.asarray(proba, dtype=float)

            # classes_ を見て win_idx を確定（本番 _score_side と同じ思想）
            win_idx = 1
            cls_list = None
            try:
                cls = getattr(ai_model, "classes_", None)
                cls_list = list(cls) if cls is not None else []
                if cls_list and (1 in cls_list):
                    win_idx = cls_list.index(1)
            except Exception:
                win_idx = 1

            s_raw = float(p[0][win_idx])
            if not np.isfinite(s_raw):
                raise ValueError("non_finite_proba_raw")

            proba_raw = float(s_raw)

            if invert_env:
                proba_used = float(1.0 - proba_raw)
                invert_applied = True
            else:
                proba_used = float(proba_raw)
                invert_applied = False

            if not np.isfinite(proba_used):
                raise ValueError("non_finite_proba_used")

            # debug にも残す（見える化）
            d["ai_proba_invert_enabled"] = bool(invert_env)
            d["ai_proba_invert_applied"] = bool(invert_applied)
            d["proba_raw"] = float(proba_raw)
            d["proba_used"] = float(proba_used)
            d["win_index"] = int(win_idx)
            d["classes_"] = [int(x) for x in cls_list] if cls_list is not None else None

        except Exception as e:
            bypassed = True
            reason = f"parse_error: {type(e).__name__}: {e}"
            d["score_error"] = reason

    score = proba_used if (proba_used is not None and np.isfinite(float(proba_used))) else None

    return jsonify({
        "ok": True,
        "model_loaded": True,
        "invert_env": 1 if invert_env else 0,
        "invert_applied": 1 if invert_applied else 0,
        "proba_raw": proba_raw,
        "proba_used": proba_used,
        "score": score,
        "bypassed": bool(bypassed),
        "reason": reason or d.get("reason", ""),
        "debug": d,
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
    # 既存処理はそのまま

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


def _train_v2_side_process(side: str):
    """
    v2_shadow_ai から LONG / SHORT 別に学習する。
    URL例:
      /train_v2_long?lookback=20000&min_samples=80&hot_reload=1&upload=1
      /train_v2_short?lookback=20000&min_samples=80&hot_reload=1&upload=1
    """
    global ai_model_long, ai_model_short
    global AI_MODEL_VERSION_RUNTIME_LONG, AI_MODEL_SOURCE_RUNTIME_LONG
    global AI_MODEL_VERSION_RUNTIME_SHORT, AI_MODEL_SOURCE_RUNTIME_SHORT

    side_u = str(side or "").strip().upper()
    if side_u not in {"LONG", "SHORT"}:
        return jsonify({"ok": False, "error": f"invalid side: {side}", "version": VERSION}), 400

    if not TRAIN_ENABLED:
        return jsonify({"ok": False, "error": "TRAIN_ENABLED=0", "version": VERSION}), 400

    if not SKLEARN_OK:
        return jsonify({"ok": False, "error": "scikit-learn unavailable", "version": VERSION}), 500

    if not _run_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Busy (run/judge/train already in progress).", "version": VERSION}), 409

    mutex_token = ""
    try:
        ok, msg = self_heal_prerequisites()
        if not ok:
            return jsonify({"ok": False, "error": f"preflight failed: {msg}", "version": VERSION}), 500

        lock_ok, mutex_token = acquire_run_mutex()
        if not lock_ok:
            return jsonify({"ok": False, "error": "Busy (distributed mutex locked).", "version": VERSION}), 409

        lookback = int(float(request.args.get("lookback", str(TRAIN_V2_LOOKBACK_ROWS))))
        default_min_samples = TRAIN_V2_LONG_MIN_SAMPLES if side_u == "LONG" else TRAIN_V2_SHORT_MIN_SAMPLES
        min_samples = int(float(request.args.get("min_samples", str(default_min_samples))))
        hot_reload = str(request.args.get("hot_reload", "1")).strip() == "1"
        upload = str(request.args.get("upload", "1")).strip() == "1"
        out_version = str(request.args.get("version", "")).strip()

        X, y, info = _build_training_dataset_from_v2_shadow_ai(side_u, lookback)

        n = int(len(y))
        if n < int(min_samples):
            return jsonify({
                "ok": False,
                "error": f"not enough labeled samples: {n} < {min_samples}",
                "version": VERSION,
                "info": info,
            }), 400

        uniq, cnt = np.unique(y, return_counts=True)
        uniq_list = sorted(int(v) for v in uniq)

        if uniq_list != [0, 1]:
            return jsonify({
                "ok": False,
                "error": f"need both classes 0/1, got {uniq_list}",
                "version": VERSION,
                "info": info,
            }), 400

        class_count = {int(k): int(v) for k, v in zip(uniq, cnt)}
        min_class_n = min(class_count.get(0, 0), class_count.get(1, 0))

        if min_class_n < 20:
            return jsonify({
                "ok": False,
                "error": f"class imbalance too strong for training: class_count={class_count}, min_class_n={min_class_n} < 20",
                "version": VERSION,
                "info": info,
            }), 400

        test_size = float(TRAIN_TEST_SIZE)
        if not (0.05 <= test_size <= 0.50):
            test_size = 0.20

        n_total = len(X)
        n_test = max(1, int(round(n_total * test_size)))
        n_train = n_total - n_test

        if n_train < 20 or n_test < 10:
            return jsonify({
                "ok": False,
                "error": f"not enough samples for time-order split: train={n_train}, test={n_test}",
                "version": VERSION,
                "side": side_u,
            }), 400

        X_tr = X.iloc[:n_train].copy()
        X_te = X.iloc[n_train:].copy()
        y_tr = y[:n_train]
        y_te = y[n_train:]

        trainer_name = V2_CLASSIFIER_TYPE
        trainer_params: Dict[str, Any] = {}

        if V2_CLASSIFIER_TYPE == "LR":
            trainer_name = "LR"
            trainer_params = {
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 800,
            }
            model = LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=800,
            )

        elif V2_CLASSIFIER_TYPE == "RF":
            trainer_name = "RF"
            trainer_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        else:
            trainer_name = "HGB"
            trainer_params = {
                "max_iter": 300,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "learning_rate": 0.05,
                "random_state": 42,
            }
            model = HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=5,
                min_samples_leaf=20,
                learning_rate=0.05,
                random_state=42,
            )

        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        acc = float(accuracy_score(y_te, y_pred)) if accuracy_score is not None else None

        auc = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)[:, 1]
                auc = float(roc_auc_score(y_te, proba)) if roc_auc_score is not None else None
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X_te)
                auc = float(roc_auc_score(y_te, score)) if roc_auc_score is not None else None
            else:
                auc = None
        except Exception:
            auc = None

        ts_ver_raw = out_version if out_version else datetime.now(JST).strftime(f"v2_{side_u.lower()}_%Y%m%d_%H%M%S")
        ts_ver = _sanitize_version_tag(ts_ver_raw) or datetime.now(JST).strftime(f"v2_{side_u.lower()}_%Y%m%d_%H%M%S")

        tmp_local = f"/tmp/{side_u.lower()}_model_{ts_ver}.pkl"
        try:
            joblib.dump(model, tmp_local)
        except Exception as e:
            return jsonify({"ok": False, "error": f"joblib.dump failed: {e}", "version": VERSION}), 500

        out_uri = ""
        if upload:
            out_uri = _build_train_output_uri_for_side(side_u, ts_ver)
            if not out_uri:
                return jsonify({
                    "ok": False,
                    "error": "cannot determine GCS output uri for side model",
                    "version": VERSION,
                }), 500

            ok_up, up_msg = _gcs_upload_from(tmp_local, out_uri)
            if not ok_up:
                return jsonify({
                    "ok": False,
                    "error": f"GCS upload failed: {out_uri} ({up_msg})",
                    "version": VERSION,
                }), 500

        reloaded = False
        if hot_reload:
            try:
                if side_u == "LONG":
                    ai_model_long = model
                    AI_MODEL_VERSION_RUNTIME_LONG = ts_ver
                    AI_MODEL_SOURCE_RUNTIME_LONG = "trained_v2_long" if upload else "trained_v2_long(no-upload)"
                else:
                    ai_model_short = model
                    AI_MODEL_VERSION_RUNTIME_SHORT = ts_ver
                    AI_MODEL_SOURCE_RUNTIME_SHORT = "trained_v2_short" if upload else "trained_v2_short(no-upload)"
                reloaded = True
            except Exception as e:
                print(f"[TRAIN_V2_{side_u}] hot_reload failed: {e}")
                reloaded = False

        next_env_vars = {}
        if upload and side_u == "LONG":
            next_env_vars = {"LONG_MODEL_VERSION": ts_ver, "LONG_MODEL_GCS_URI": out_uri}
        elif upload and side_u == "SHORT":
            next_env_vars = {"SHORT_MODEL_VERSION": ts_ver, "SHORT_MODEL_GCS_URI": out_uri}

        return jsonify({
            "ok": True,
            "version": VERSION,
            "side": side_u,
            "trained_samples": n,
            "trainer_name": trainer_name,
            "trainer_params": trainer_params,
            "metrics": {
                "accuracy": acc,
                "auc": auc,
            },
            "info": info,
            "new_model": {
                "model_version_suggest": ts_ver,
                "model_gcs_uri": out_uri,
                "uploaded": bool(upload),
                "hot_reloaded_in_this_instance": bool(reloaded),
            },
            "next_env_vars": next_env_vars,
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "version": VERSION, "side": side_u}), 500
    finally:
        if mutex_token:
            try:
                release_run_mutex(mutex_token)
            except Exception as e:
                print(f"[WARN] /train_v2_{side_u.lower()} release_run_mutex failed: {type(e).__name__}: {e}")
        try:
            _run_lock.release()
        except RuntimeError:
            pass


@app.route("/train_v2_long", methods=["GET", "POST"])
def train_v2_long_process():
    return _train_v2_side_process("LONG")


@app.route("/train_v2_short", methods=["GET", "POST"])
def train_v2_short_process():
    return _train_v2_side_process("SHORT")


@app.route("/label_market", methods=["GET", "POST"])
def label_market_process():
    """
    Cloud Scheduler の /label_market 用エンドポイント。
    目的：404 を無くして、Scheduler が正常に叩ける状態にする。
    - /run, /judge, /train と同じ設計で 500 を避ける（Scheduler失敗扱いにしない）
    - 実処理は label_market_main() が存在すれば呼ぶ。無ければ「未実装」扱いで 200 を返す。
    """
    # 同時実行ガード（run/judge/train/label_market）
    if not _run_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Busy (run/judge/train/label_market already in progress).", "version": VERSION}), 200

    mutex_token = ""
    try:
        ok, msg = preflight_check()
        if not ok:
            err = f"Preflight NG: {msg}"
            print(f"[WARN] /label_market {err}")
            send_discord_message(f"[WARN] /label_market {err}")
            return jsonify({"ok": False, "error": err, "version": VERSION}), 200

        okm, token = acquire_run_mutex()
        if not okm:
            return jsonify({"ok": False, "error": "Busy (distributed mutex).", "version": VERSION}), 200
        mutex_token = token

        # 実処理があるなら呼ぶ（関数名はここで固定）
        try:
            fn = globals().get("label_market_main", None)
            if callable(fn):
                res = fn()
                return jsonify({"ok": True, "result": str(res), "version": VERSION}), 200
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[ERR] /label_market crashed: {err}")
            send_discord_message(f"[ERR] /label_market crashed: {err}")
            return jsonify({"ok": False, "error": err, "version": VERSION}), 200

        # 無くても 200（まず 404 を消すのが目的）
        return jsonify({
            "ok": True,
            "version": VERSION,
            "note": "label_market endpoint is alive. label_market_main() is not implemented (or not found).",
        }), 200

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /label_market outer exception: {err}")
        send_discord_message(f"[ERR] /label_market outer crashed: {err}")
        return jsonify({"ok": False, "error": err, "version": VERSION}), 200

    finally:
        if mutex_token:
            try:
                release_run_mutex(mutex_token)
            except Exception as e:
                msg = f"[WARN] /label_market release_run_mutex failed: {type(e).__name__}: {e}"
                print(msg)
                try:
                    send_discord_message(msg)
                except Exception as e2:
                    print(f"[WARN] /label_market send_discord_message failed: {type(e2).__name__}: {e2}")

        try:
            _run_lock.release()
        except RuntimeError:
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

# ==========================================
# V2 判定係 (/judge_v2)
# ==========================================
def judge_v2_sheet(
    sheet_name: str = V2_SHADOW_SHEET,
    lookback_rows: int = V2_JUDGE_LOOKBACK_ROWS,
    max_judge: int = 100,
    priority_bands: Optional[List[str]] = None,
    include_deferred: bool = True,
    oldest_first: bool = False,
) -> int:
    service = get_sheet_service()
    exchange = build_exchange()

    ok, msg = self_heal_prerequisites()
    if not ok:
        print(f"[WARN] self_heal_prerequisites failed (judge_v2): {msg}")
        return 0

    headers, _, okh = get_headers_and_len(sheet_name)
    if STRICT_HEADER_CHECK and not okh:
        print(f"[WARN] judge_v2_sheet header check failed -> skip. sheet={sheet_name}")
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
    col_time = _resolve_col_idx(hm, "Datetime_JST")
    col_entry = _resolve_col_idx(hm, "EntryPrice")
    col_dir = _resolve_col_idx(hm, "Direction")
    col_tp = _resolve_col_idx(hm, "TP_Price")
    col_sl = _resolve_col_idx(hm, "SL_Price")
    col_tp_pct = _resolve_col_idx(hm, "TP_Pct")
    col_sl_pct = _resolve_col_idx(hm, "SL_Pct")
    col_ai_pass = _resolve_col_idx(hm, "AI_Pass")
    col_ai_band = _resolve_col_idx(hm, "AI_Band")

    col_eval = _resolve_col_idx(hm, "EvalStatus")
    col_exit_time = _resolve_col_idx(hm, "ExitTime")
    col_exit_price = _resolve_col_idx(hm, "ExitPrice")
    col_exit_reason = _resolve_col_idx(hm, "ExitReason")
    col_pnl = _resolve_col_idx(hm, "PnL_Pct")
    col_winlose = _resolve_col_idx(hm, "WinLose")
    col_holdmin = _resolve_col_idx(hm, "HoldMin")

    must = [
        col_symbol, col_time, col_entry, col_dir, col_tp, col_sl,
        col_tp_pct, col_sl_pct,
        col_eval, col_exit_time, col_exit_price, col_exit_reason,
        col_pnl, col_winlose, col_holdmin,
    ]
    if any(c == -1 for c in must):
        print(f"[WARN] judge_v2_sheet missing required columns in {sheet_name}")
        return 0

    priority_band_set: Set[str] = {
        str(x).strip().upper()
        for x in (priority_bands or [])
        if str(x).strip()
    }

    updates = []
    judged = 0

    def get_cell(row, idx: int) -> str:
        if idx < 0:
            return ""
        return str(row[idx]).strip() if idx < len(row) else ""

    def put_one(row_idx_1based: int, col_idx: int, value):
        if col_idx < 0:
            return
        a1 = col_to_a1(col_idx)
        updates.append({"range": f"{sheet_name}!{a1}{row_idx_1based}", "values": [[value]]})

    def mark_invalid_market(row_idx_1based: int, market: str):
        put_one(row_idx_1based, col_eval, "INVALID_MARKET")
        put_one(row_idx_1based, col_exit_time, "")
        put_one(row_idx_1based, col_exit_price, "")
        put_one(row_idx_1based, col_exit_reason, f"BadSymbol:{market}")
        put_one(row_idx_1based, col_pnl, "")
        put_one(row_idx_1based, col_winlose, "")
        put_one(row_idx_1based, col_holdmin, "")

    def mark_invalid_data(row_idx_1based: int, reason: str):
        put_one(row_idx_1based, col_eval, "INVALID_DATA")
        put_one(row_idx_1based, col_exit_time, "")
        put_one(row_idx_1based, col_exit_price, "")
        put_one(row_idx_1based, col_exit_reason, reason)
        put_one(row_idx_1based, col_pnl, "")
        put_one(row_idx_1based, col_winlose, "")
        put_one(row_idx_1based, col_holdmin, "")

    pending_priority_offsets: List[int] = []
    open_priority_offsets: List[int] = []
    pending_deferred_offsets: List[int] = []
    open_deferred_offsets: List[int] = []

    for offset in range(len(rows)):
        row = rows[offset]
        status = get_cell(row, col_eval).upper()

        if status not in ("", "OPEN"):
            continue

        ai_pass = get_cell(row, col_ai_pass)
        ai_band = get_cell(row, col_ai_band).upper()
        is_priority = (ai_pass == "1") or (ai_band in priority_band_set)

        if is_priority:
            if status == "":
                pending_priority_offsets.append(offset)
            else:
                open_priority_offsets.append(offset)
        elif include_deferred:
            if status == "":
                pending_deferred_offsets.append(offset)
            else:
                open_deferred_offsets.append(offset)

    if oldest_first:
        target_offsets = (
            pending_priority_offsets +
            open_priority_offsets +
            pending_deferred_offsets +
            open_deferred_offsets
        )
    else:
        target_offsets = (
            list(reversed(pending_priority_offsets)) +
            list(reversed(open_priority_offsets)) +
            list(reversed(pending_deferred_offsets)) +
            list(reversed(open_deferred_offsets))
        )

    for offset in target_offsets:
        if judged >= max_judge:
            break

        row = rows[offset]
        sheet_row_idx = start_row + offset

        sym = get_cell(row, col_symbol)
        tstr = get_cell(row, col_time)
        direction = get_cell(row, col_dir).upper()
        entry = to_float(get_cell(row, col_entry), default=None)
        tp = to_float(get_cell(row, col_tp), default=None)
        sl = to_float(get_cell(row, col_sl), default=None)
        tp_pct = to_float(get_cell(row, col_tp_pct), default=None)
        sl_pct = to_float(get_cell(row, col_sl_pct), default=None)

        if not sym:
            mark_invalid_data(sheet_row_idx, "symbol_missing")
            judged += 1
            time.sleep(0.12)
            continue

        if not tstr:
            mark_invalid_data(sheet_row_idx, "time_missing")
            judged += 1
            time.sleep(0.12)
            continue

        if entry is None:
            mark_invalid_data(sheet_row_idx, "entry_missing")
            judged += 1
            time.sleep(0.12)
            continue

        if entry <= 0:
            mark_invalid_data(sheet_row_idx, "entry_nonpositive")
            judged += 1
            time.sleep(0.12)
            continue

        if tp is None:
            mark_invalid_data(sheet_row_idx, "tp_missing")
            judged += 1
            time.sleep(0.12)
            continue

        if sl is None:
            mark_invalid_data(sheet_row_idx, "sl_missing")
            judged += 1
            time.sleep(0.12)
            continue

        dt0 = parse_dt_any(tstr)
        if dt0 is None:
            mark_invalid_data(sheet_row_idx, "invalid_datetime")
            judged += 1
            time.sleep(0.12)
            continue

        if getattr(dt0, "tzinfo", None) is None:
            dt_jst = dt0.replace(tzinfo=JST)
        else:
            dt_jst = dt0.astimezone(JST)

        actual_entry_jst = dt_jst + timedelta(minutes=15)
        since_ms = int(actual_entry_jst.astimezone(timezone.utc).timestamp() * 1000)

        raw_sym = str(sym).strip().upper()
        base_sym = raw_sym.split(":", 1)[0]
        base_sym = base_sym.split("/", 1)[0].strip()
        market = f"{base_sym}/USDT"

        candles = None
        bad_symbol = False
        bad_symbol_msg = ""

        try:
            candles = fetch_ohlcv_safe(
                exchange,
                market,
                timeframe="15m",
                since=since_ms,
                limit=V2_JUDGE_MAX_BARS + 4,
            )
        except Exception as e:
            emsg = str(e)
            if "does not have market symbol" in emsg:
                bad_symbol = True
                bad_symbol_msg = emsg
            else:
                print(f"[JUDGE15M-V2]fetch exception keep-retry row={sheet_row_idx} market={market} err={emsg}")
                continue

        if bad_symbol:
            print(f"[JUDGE15M-V2]Mark invalid market row={sheet_row_idx} market={market} err={bad_symbol_msg}")
            mark_invalid_market(sheet_row_idx, market)
            judged += 1
            time.sleep(0.12)
            continue

        if not candles:
            continue

        if len(candles) >= 1:
            last_open_ts_ms = int(candles[-1][0])
            _log_judge_15m_bar(last_open_ts_ms, label=f"{market}-15m-v2")
            now_ms = int(time.time() * 1000)
            forming = now_ms < (last_open_ts_ms + 15 * 60 * 1000)
            if forming:
                if len(candles) >= 2:
                    candles = candles[:-1]
                else:
                    continue

        res_status = "OPEN"
        res_win = ""
        res_reason = ""
        res_exit_price = ""
        res_exit_time_ms = None

        side = "SHORT" if "SHORT" in direction else "LONG"

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

        if res_status == "OPEN" and len(candles) >= V2_JUDGE_MAX_BARS:
            res_status = "EXPIRED"
            res_reason = "TimeOver"

        if res_exit_time_ms is not None:
            exit_dt_jst = datetime.fromtimestamp(res_exit_time_ms / 1000, JST)
            exit_time_str = exit_dt_jst.strftime("%Y-%m-%d %H:%M:%S")
            hold_min = int((exit_dt_jst - actual_entry_jst).total_seconds() / 60)
        else:
            exit_time_str = ""
            hold_min = ""

        if res_status == "DONE":
            if res_reason == "TP":
                pnl_pct = tp_pct if tp_pct is not None else ((abs(tp - entry) / entry) * 100.0)
            else:
                pnl_pct = -(sl_pct if sl_pct is not None else ((abs(sl - entry) / entry) * 100.0))
        else:
            pnl_pct = ""

        put_one(sheet_row_idx, col_eval, res_status)
        put_one(sheet_row_idx, col_exit_time, exit_time_str)
        put_one(sheet_row_idx, col_exit_price, res_exit_price)
        put_one(sheet_row_idx, col_exit_reason, res_reason)
        put_one(sheet_row_idx, col_pnl, pnl_pct)
        put_one(sheet_row_idx, col_winlose, res_win)
        put_one(sheet_row_idx, col_holdmin, hold_min)

        judged += 1
        time.sleep(0.12)

    if updates:
        service.spreadsheets().values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"valueInputOption": "USER_ENTERED", "data": updates},
        ).execute()
        _invalidate_sheet_caches(sheet_name)

    return judged


def judge_v2_main() -> str:
    recent_total = judge_v2_sheet(
        V2_SHADOW_SHEET,
        lookback_rows=V2_JUDGE_LOOKBACK_ROWS,
        max_judge=V2_JUDGE_MAX_ROWS,
        priority_bands=V2_JUDGE_PRIORITY_BANDS,
        include_deferred=True,
        oldest_first=False,
    )

    backfill_total = 0
    if V2_JUDGE_BACKFILL_ENABLE and V2_JUDGE_BACKFILL_MAX_ROWS > 0:
        backfill_total = judge_v2_sheet(
            V2_SHADOW_SHEET,
            lookback_rows=max(V2_JUDGE_BACKFILL_LOOKBACK_ROWS, V2_JUDGE_LOOKBACK_ROWS),
            max_judge=V2_JUDGE_BACKFILL_MAX_ROWS,
            priority_bands=V2_JUDGE_PRIORITY_BANDS,
            include_deferred=True,
            oldest_first=True,
        )

    total = recent_total + backfill_total
    return (
        f"Judged recent={recent_total} "
        f"backfill={backfill_total} "
        f"total={total} rows ({V2_SHADOW_SHEET})"
    )


@app.route("/v2_report", methods=["GET", "POST"])
def v2_report_process():
    try:
        result = analyze_v2_performance()
        return result, 200
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /v2_report crashed: {err}")
        return f"ERR: {err}", 200


@app.route("/v2_guard_report", methods=["GET"])
def v2_guard_report():
    try:
        state = get_v2_guardrail_state(datetime.now(JST))
        return jsonify({
            "ok": True,
            "version": VERSION,
            "guardrail": state,
        }), 200
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /v2_guard_report crashed: {err}")
        return jsonify({"ok": False, "error": err, "version": VERSION}), 200


@app.route("/v2_retrain_report", methods=["GET"])
def v2_retrain_report():
    try:
        state = get_v2_retrain_review_state(datetime.now(JST))
        return jsonify({
            "ok": True,
            "version": VERSION,
            "retrain_review": state,
        }), 200
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /v2_retrain_report crashed: {err}")
        return jsonify({"ok": False, "error": err, "version": VERSION}), 200


@app.route("/judge_v2", methods=["GET", "POST"])
def judge_v2_process():
    if not _run_lock.acquire(blocking=False):
        return "Busy (run/judge/judge_v2 already in progress).", 200

    mutex_token = ""
    try:
        ok, msg = preflight_check()
        if not ok:
            err = f"Preflight NG: {msg}"
            print(f"[WARN] /judge_v2 {err}")
            send_discord_message(f"[WARN] /judge_v2 {err}")
            return f"NG: {err}", 200

        okm, token = acquire_run_mutex()
        if not okm:
            return "Busy (distributed mutex).", 200
        mutex_token = token

        try:
            now_utc = time.time()
            rem = now_utc % (15 * 60)
            window = V2_JUDGE_CLOSE_WINDOW_SEC
            if rem > window:
                msg = f"Skip: not 15m close window rem={rem:.1f}s window={window}s"
                print(f"[JUDGE15M-V2]{msg}")
                return msg, 200

            return str(judge_v2_main()), 200
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[ERR] /judge_v2 crashed: {err}")
            send_discord_message(f"[ERR] /judge_v2 crashed: {err}")
            return f"ERR: {err}", 200

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ERR] /judge_v2 outer exception: {err}")
        send_discord_message(f"[ERR] /judge_v2 outer crashed: {err}")
        return f"ERR: {err}", 200

    finally:
        if mutex_token:
            try:
                release_run_mutex(mutex_token)
            except Exception as e:
                msg = f"[WARN] /judge_v2 release_run_mutex failed: {type(e).__name__}: {e}"
                print(msg)
                try:
                    send_discord_message(msg)
                except Exception as e2:
                    print(f"[WARN] /judge_v2 send_discord_message failed: {type(e2).__name__}: {e2}")

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
            # 15m足の「確定直後」だけ判定を実行する（5分起動は維持）
            # 例: window=90 なら、毎時 00/15/30/45 の「最初の90秒だけ」実行し、それ以外は即スキップ
            now_utc = time.time()
            rem = now_utc % (15 * 60)
            window = int(os.environ.get("JUDGE_CLOSE_WINDOW_SEC", "90"))
            if rem > window:
                msg = f"Skip: not 15m close window rem={rem:.1f}s window={window}s"
                print(f"[JUDGE15M]{msg}")
                return msg, 200

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



















