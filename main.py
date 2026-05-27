import os
import time
import threading
import hashlib
import subprocess
import re
import json
import math
import logging
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

V2_WATCH_NOTIFY_ENABLE = str(
    os.environ.get("V2_WATCH_NOTIFY_ENABLE", "1")
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

LONG_TREND_MODEL_GCS_URI = os.environ.get("LONG_TREND_MODEL_GCS_URI", "").strip()
LONG_RECOVERY_MODEL_GCS_URI = os.environ.get("LONG_RECOVERY_MODEL_GCS_URI", "").strip()
SHORT_TREND_MODEL_GCS_URI = os.environ.get("SHORT_TREND_MODEL_GCS_URI", "").strip()
SHORT_RETRACE_MODEL_GCS_URI = os.environ.get("SHORT_RETRACE_MODEL_GCS_URI", "").strip()

LONG_TREND_MODEL_VERSION = os.environ.get("LONG_TREND_MODEL_VERSION", "").strip()
LONG_RECOVERY_MODEL_VERSION = os.environ.get("LONG_RECOVERY_MODEL_VERSION", "").strip()
SHORT_TREND_MODEL_VERSION = os.environ.get("SHORT_TREND_MODEL_VERSION", "").strip()
SHORT_RETRACE_MODEL_VERSION = os.environ.get("SHORT_RETRACE_MODEL_VERSION", "").strip()

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

# --- BTC急騰直後の追いかけLONG停止 ---
BTC_SURGE_LONG_PAUSE_ENABLE = str(
    os.environ.get("BTC_SURGE_LONG_PAUSE_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

BTC_SURGE_LONG_PAUSE_NOTIFY = str(
    os.environ.get("BTC_SURGE_LONG_PAUSE_NOTIFY", "1")
).strip().lower() in ("1", "true", "yes", "on")

# 急騰検知後、何分間 LONG 抑制を維持するか
# 「急騰直後の反落」までカバーしたいので 90分を初期値にする
BTC_SURGE_LONG_COOLDOWN_MINUTES = int(float(
    os.environ.get("BTC_SURGE_LONG_COOLDOWN_MINUTES", "90")
))

# 「レンジの少し上振れ」で誤発火しないよう、かなり厳しめの初期値にする
BTC_SURGE_LONG_PAUSE_HTF_STRENGTH_MIN = _env_float(
    "BTC_SURGE_LONG_PAUSE_HTF_STRENGTH_MIN", 0.90
)
BTC_SURGE_LONG_PAUSE_1H_MIN = _env_float(
    "BTC_SURGE_LONG_PAUSE_1H_MIN", 0.0120
)  # +1.20%
BTC_SURGE_LONG_PAUSE_15M_MIN = _env_float(
    "BTC_SURGE_LONG_PAUSE_15M_MIN", 0.0035
)  # +0.35%

# 急騰中 / 冷却時間中でも全部のLONGを止めず、追いかけ気味だけ止める
BTC_SURGE_LONG_PAUSE_RSI_MIN = _env_float(
    "BTC_SURGE_LONG_PAUSE_RSI_MIN", 58.0
)
BTC_SURGE_LONG_PAUSE_P1_MIN = _env_float(
    "BTC_SURGE_LONG_PAUSE_P1_MIN", 2.0
)

# --- 同一銘柄LONG連打ブロック ---
# 直近 window 分に同一銘柄LONG通知が max_notifies 回以上あれば止める
V2_LONG_REPEAT_BLOCK_ENABLE = str(
    os.environ.get("V2_LONG_REPEAT_BLOCK_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_LONG_REPEAT_WINDOW_MINUTES = int(float(
    os.environ.get("V2_LONG_REPEAT_WINDOW_MINUTES", "90")
))

V2_LONG_REPEAT_MAX_NOTIFIES = int(float(
    os.environ.get("V2_LONG_REPEAT_MAX_NOTIFIES", "2")
))

# --- 同一銘柄LONG連打は hard block ではなく rank penalty で扱う ---
V2_LONG_REPEAT_PENALTY_ENABLE = str(
    os.environ.get("V2_LONG_REPEAT_PENALTY_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_LONG_REPEAT_PENALTY_MIN_COUNT = int(float(
    os.environ.get("V2_LONG_REPEAT_PENALTY_MIN_COUNT", "5")
))

V2_LONG_REPEAT_PENALTY_VALUE = _env_float(
    "V2_LONG_REPEAT_PENALTY_VALUE", 0.12
)

# --- 事故パターン対策（2026-04 incident pass 1） ---
V2_LONG_REVERSAL_HARD_REJECT_ENABLE = str(
    os.environ.get("V2_LONG_REVERSAL_HARD_REJECT_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_LONG_REVERSAL_BTC_RET_MAX = _env_float(
    "V2_LONG_REVERSAL_BTC_RET_MAX", -0.0080
)
V2_LONG_REVERSAL_BTC_1H_MIN = _env_float(
    "V2_LONG_REVERSAL_BTC_1H_MIN", 0.0020
)

V2_LONG_SURGE_PENALTY_ENABLE = str(
    os.environ.get("V2_LONG_SURGE_PENALTY_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_SURGE_PENALTY_VALUE = _env_float(
    "V2_LONG_SURGE_PENALTY_VALUE", 0.12
)

V2_LONG_VOL_SHOCK_PENALTY_ENABLE = str(
    os.environ.get("V2_LONG_VOL_SHOCK_PENALTY_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")
V2_LONG_VOL_SHOCK_PENALTY_VALUE = _env_float(
    "V2_LONG_VOL_SHOCK_PENALTY_VALUE", 0.10
)

V2_SHORT_DUMP_PENALTY_ENABLE = str(
    os.environ.get("V2_SHORT_DUMP_PENALTY_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_DUMP_PENALTY_VALUE = _env_float(
    "V2_SHORT_DUMP_PENALTY_VALUE", 0.10
)

V2_SHORT_VOL_SHOCK_PENALTY_ENABLE = str(
    os.environ.get("V2_SHORT_VOL_SHOCK_PENALTY_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_VOL_SHOCK_PENALTY_VALUE = _env_float(
    "V2_SHORT_VOL_SHOCK_PENALTY_VALUE", 0.08
)

V2_SHORT_REPEAT_PENALTY_ENABLE = str(
    os.environ.get("V2_SHORT_REPEAT_PENALTY_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_REPEAT_WINDOW_MINUTES = int(float(
    os.environ.get("V2_SHORT_REPEAT_WINDOW_MINUTES", "90")
))
V2_SHORT_REPEAT_PENALTY_MIN_COUNT = int(float(
    os.environ.get("V2_SHORT_REPEAT_PENALTY_MIN_COUNT", "5")
))
V2_SHORT_REPEAT_PENALTY_VALUE = _env_float(
    "V2_SHORT_REPEAT_PENALTY_VALUE", 0.08
)

V2_INCIDENT_VOLRATIO_SHOCK_MIN = _env_float(
    "V2_INCIDENT_VOLRATIO_SHOCK_MIN", 1.80
)
V2_INCIDENT_BTC_1H_SURGE_MIN = _env_float(
    "V2_INCIDENT_BTC_1H_SURGE_MIN", 0.0120
)
V2_INCIDENT_BTC_1H_DUMP_MAX = _env_float(
    "V2_INCIDENT_BTC_1H_DUMP_MAX", -0.0120
)
V2_INCIDENT_BTC_15M_SURGE_MIN = _env_float(
    "V2_INCIDENT_BTC_15M_SURGE_MIN", 0.0035
)
V2_INCIDENT_BTC_15M_DUMP_MAX = _env_float(
    "V2_INCIDENT_BTC_15M_DUMP_MAX", -0.0035
)

# --- 当日LONG緊急ブレーキ ---
# 直近DONE済み通知LONGが lookback_n 件たまっていて、
# 勝率 <= stop_winrate かつ avg_pnl < stop_avgpnl なら LONG を止める
V2_LONG_FAST_BRAKE_ENABLE = str(
    os.environ.get("V2_LONG_FAST_BRAKE_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_LONG_FAST_BRAKE_MIN_DONE = int(float(
    os.environ.get("V2_LONG_FAST_BRAKE_MIN_DONE", "4")
))

V2_LONG_FAST_BRAKE_LOOKBACK_N = int(float(
    os.environ.get("V2_LONG_FAST_BRAKE_LOOKBACK_N", "4")
))

V2_LONG_FAST_BRAKE_STOP_WINRATE = _env_float(
    "V2_LONG_FAST_BRAKE_STOP_WINRATE", 25.0
)

V2_LONG_FAST_BRAKE_STOP_AVGPNL = _env_float(
    "V2_LONG_FAST_BRAKE_STOP_AVGPNL", 0.0
)

# ブレーキ発動後は最低この時間だけ止める
V2_LONG_FAST_BRAKE_HOLD_MINUTES = int(float(
    os.environ.get("V2_LONG_FAST_BRAKE_HOLD_MINUTES", "180")
))


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

_repeat_state_cache: Dict[str, Dict[str, Any]] = {}
REPEAT_STATE_TTL_SEC = int(float(os.environ.get("REPEAT_STATE_TTL_SEC", "120")))
V2_REPEAT_STATE_LOOKBACK_ROWS = int(float(os.environ.get("V2_REPEAT_STATE_LOOKBACK_ROWS", "2000")))
V2_REGIME_SNAPSHOT_TTL_SEC = int(float(os.environ.get("V2_REGIME_SNAPSHOT_TTL_SEC", "1200")))
V2_BRAKE_STATE_TTL_SEC = int(float(os.environ.get("V2_BRAKE_STATE_TTL_SEC", "120")))

_v2_fetch_cache: Dict[str, Dict[str, Any]] = {}
V2_FETCH_TTL_SEC = int(float(os.environ.get("V2_FETCH_TTL_SEC", "10")))
V2_FUNDING_FETCH_TTL_SEC = int(float(os.environ.get("V2_FUNDING_FETCH_TTL_SEC", "300")))

HEALTH_CHECK_TTL_SEC = int(float(os.environ.get("HEALTH_CHECK_TTL_SEC", "600")))
_health_check_cache: Dict[str, float] = {}

_DEBUG_LOG_ENABLE_RAW = str(os.environ.get("DEBUG_LOG_ENABLE", "0")).strip().lower()
DEBUG_LOG_ENABLE: bool = _DEBUG_LOG_ENABLE_RAW in ("1", "true", "yes", "on")

_TIMER_LOG_ENABLE_RAW = str(os.environ.get("TIMER_LOG_ENABLE", "0")).strip().lower()
TIMER_LOG_ENABLE: bool = _TIMER_LOG_ENABLE_RAW in ("1", "true", "yes", "on")

JST = timezone(timedelta(hours=9))

http = requests.Session()
http.headers.update({"User-Agent": "spidey-bot/v3.4.9"})

_run_lock = threading.Lock()

_exchange_cache: Dict[str, Any] = {"ex": None, "ts": 0.0}
_symbol_resolve_cache: Dict[str, str] = {}

_btc_long_pause_runtime_state: Dict[str, Any] = {
    "raw_active": False,
    "effective_active": None,
    "reason": "",
    "cooldown_until_ts": 0.0,
    "last_sent_ts": 0.0,
}
_btc_long_pause_ctx: Dict[str, Any] = {
    "active": False,
    "reason": "",
}

_v2_long_repeat_cache: Dict[str, Any] = {
    "ts": 0.0,
    "window_min": 0,
    "state": {},
}

_v2_long_fast_brake_cache: Dict[str, Any] = {
    "ts": 0.0,
    "state": {},
}

_v2_long_fast_brake_runtime: Dict[str, Any] = {
    "active": False,
    "until_ts": 0.0,
    "last_trigger_latest_done_dt": "",
    "reason": "",
}

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
def _ensure_discord_message_frame(text: str) -> str:
    """
    通常Webhook通知を見やすくするための共通ラッパー。
    - 既に上下区切り線がある本文はそのまま送る
    - 無い本文だけ、自動で上下に区切り線を付ける
    """
    s = "" if text is None else str(text).strip()
    if s == "":
        return s

    line = "━━━━━━━━━━━━━━━━━━━━"

    has_top = s.startswith(line)
    has_bottom = s.endswith(line)

    if has_top and has_bottom:
        return s

    return f"{line}\n{s}\n{line}"


def send_discord_message(text: str) -> Tuple[bool, str]:
    if (not discord_webhook_url) or ("ここに" in discord_webhook_url):
        print("[DBG] discord webhook url empty or placeholder")
        return False, "webhook_empty"

    framed_text = _ensure_discord_message_frame(text)

    chunks = [framed_text[i:i+1900] for i in range(0, len(framed_text), 1900)] or [framed_text]
    for chunk in chunks:
        try:
            r = http.post(discord_webhook_url, json={"content": chunk}, timeout=10)
            if DEBUG_LOG_ENABLE:
                print(f"[DBG] discord status={r.status_code}")
            if r.status_code >= 300:
                print(f"[DBG] discord body={r.text[:200]}")
                return False, f"http_{r.status_code}"
        except Exception as e:
            print(f"[ERR] discord webhook post: {e}")
            return False, f"exception:{type(e).__name__}"

    return True, "ok"


def send_daily_discord_message(text: str):
    """
    日次レポート専用Webhookに送る。
    DAILY_DISCORD_WEBHOOK_URL が未設定なら送らない。
    通常Webhookへは絶対にフォールバックしない。
    """
    webhook = daily_discord_webhook_url
    if (not webhook) or ("ここに" in webhook):
        print("[WARN] DAILY_DISCORD_WEBHOOK_URL empty -> skip daily report send")
        return

    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)] or [text]
    for chunk in chunks:
        try:
            r = http.post(webhook, json={"content": chunk}, timeout=10)
            if DEBUG_LOG_ENABLE:
                print(f"[DBG] daily discord status={r.status_code}")
            if r.status_code >= 300:
                print(f"[DBG] daily discord body={r.text[:200]}")
        except Exception as e:
            print(f"[ERR] daily discord webhook post: {e}")


def _detect_btc_surge_long_pause(
    btc_htf_dir: str,
    btc_htf_strength: Any,
    btc_1h_change: Any,
    btc_ret_15m: Any,
) -> Tuple[bool, str]:
    """
    BTC急騰直後の「追いかけLONG停止」判定。
    レンジ内の誤発火を避けるため、かなり厳しめに
      1) HTF方向
      2) HTF強度
      3) 1時間上昇率
      4) 直近15分上昇率
    の全部を要求する。
    """
    if not BTC_SURGE_LONG_PAUSE_ENABLE:
        return False, "feature_off"

    try:
        dir_u = str(btc_htf_dir or "").strip().upper()
        strength = float(btc_htf_strength)
        chg_1h = float(btc_1h_change)
        chg_15m = float(btc_ret_15m)

        if not np.isfinite(strength):
            return False, "missing_htf_strength"
        if not np.isfinite(chg_1h):
            return False, "missing_btc_1h_change"
        if not np.isfinite(chg_15m):
            return False, "missing_btc_15m_change"

        dir_ok = dir_u in ("LONG", "UP", "BULL")
        strength_ok = strength >= float(BTC_SURGE_LONG_PAUSE_HTF_STRENGTH_MIN)
        chg_1h_ok = chg_1h >= float(BTC_SURGE_LONG_PAUSE_1H_MIN)
        chg_15m_ok = chg_15m >= float(BTC_SURGE_LONG_PAUSE_15M_MIN)

        active = bool(dir_ok and strength_ok and chg_1h_ok and chg_15m_ok)

        reason = (
            f"dir={dir_u} "
            f"htf_strength={strength:.4f} "
            f"btc_1h_change={chg_1h:.6f} "
            f"btc_15m_change={chg_15m:.6f} "
            f"dir_ok={int(dir_ok)} "
            f"strength_ok={int(strength_ok)} "
            f"chg_1h_ok={int(chg_1h_ok)} "
            f"chg_15m_ok={int(chg_15m_ok)}"
        )
        return active, reason

    except Exception as e:
        return False, f"detect_error:{type(e).__name__}:{e}"


def _resolve_btc_surge_long_pause_runtime(
    raw_active: bool,
    raw_reason: str,
    btc_htf_dir: str,
    btc_htf_strength: Any,
    btc_15m_df: Optional[pd.DataFrame],
) -> Tuple[bool, str]:
    """
    現在の急騰判定(raw_active)に加えて、
    直近の確定15分足をさかのぼって「急騰イベントが直近 cooldown 分にあったか」を毎回再計算する。
    これにより、
      - デプロイ直後
      - instance切替直後
      - 急騰直後の反落局面
    でも、冷却時間を見失わない。
    """
    global _btc_long_pause_runtime_state

    recent_event_active = False
    recent_event_reason = "no_recent_event"

    try:
        dir_u = str(btc_htf_dir or "").strip().upper()
        strength = float(btc_htf_strength)

        dir_ok = dir_u in ("LONG", "UP", "BULL")
        strength_ok = np.isfinite(strength) and (
            strength >= float(BTC_SURGE_LONG_PAUSE_HTF_STRENGTH_MIN)
        )

        if (
            BTC_SURGE_LONG_PAUSE_ENABLE
            and dir_ok
            and strength_ok
            and isinstance(btc_15m_df, pd.DataFrame)
            and (not btc_15m_df.empty)
        ):
            work = btc_15m_df.copy()
            work["Close"] = pd.to_numeric(work["Close"], errors="coerce")
            work["Time"] = pd.to_numeric(work["Time"], errors="coerce")

            # 形成中の最後の1本は除外し、確定足だけを見る
            confirmed = work.iloc[:-1].copy() if len(work) >= 2 else work.copy()

            if len(confirmed) >= 5:
                bars_lookback = max(
                    1,
                    int(np.ceil(float(BTC_SURGE_LONG_COOLDOWN_MINUTES) / 15.0))
                )
                start_i = max(4, len(confirmed) - bars_lookback)

                for i in range(len(confirmed) - 1, start_i - 1, -1):
                    cur = float(confirmed["Close"].iloc[i])
                    prev_15m = float(confirmed["Close"].iloc[i - 1])
                    prev_1h = float(confirmed["Close"].iloc[i - 4])

                    if (
                        (not np.isfinite(cur))
                        or (not np.isfinite(prev_15m))
                        or (not np.isfinite(prev_1h))
                        or abs(prev_15m) <= 1e-12
                        or abs(prev_1h) <= 1e-12
                    ):
                        continue

                    chg_15m = (cur - prev_15m) / prev_15m
                    chg_1h = (cur - prev_1h) / prev_1h

                    if (
                        chg_1h >= float(BTC_SURGE_LONG_PAUSE_1H_MIN)
                        and chg_15m >= float(BTC_SURGE_LONG_PAUSE_15M_MIN)
                    ):
                        event_dt_text = ""
                        event_ms = confirmed["Time"].iloc[i]
                        try:
                            if np.isfinite(float(event_ms)):
                                event_dt = datetime.fromtimestamp(
                                    float(event_ms) / 1000.0,
                                    tz=timezone.utc,
                                ).astimezone(JST)
                                event_dt_text = event_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            event_dt_text = ""

                        recent_event_active = True
                        recent_event_reason = (
                            f"cooldown_after_recent_event "
                            f"event_jst={event_dt_text} "
                            f"cooldown_min={int(BTC_SURGE_LONG_COOLDOWN_MINUTES)} "
                            f"btc_1h_change={chg_1h:.6f} "
                            f"btc_15m_change={chg_15m:.6f}"
                        )
                        break
            else:
                recent_event_reason = f"insufficient_confirmed_bars n={len(confirmed)}"
        else:
            recent_event_reason = (
                f"dir={dir_u} "
                f"htf_strength={'' if not np.isfinite(strength) else format(float(strength), '.4f')} "
                f"dir_ok={int(dir_ok)} "
                f"strength_ok={int(strength_ok)}"
            )

    except Exception as e:
        recent_event_reason = f"recent_event_error:{type(e).__name__}:{e}"

    effective_active = bool(raw_active or recent_event_active)

    if raw_active:
        effective_reason = f"surge_live {raw_reason}"
    elif recent_event_active:
        effective_reason = recent_event_reason
    else:
        effective_reason = str(raw_reason or recent_event_reason or "")

    _btc_long_pause_runtime_state["raw_active"] = bool(raw_active)
    _btc_long_pause_runtime_state["effective_active"] = bool(effective_active)
    _btc_long_pause_runtime_state["reason"] = str(effective_reason or "")
    _btc_long_pause_runtime_state["cooldown_until_ts"] = 0.0

    return bool(effective_active), str(effective_reason or "")


def _notify_btc_surge_long_pause_if_changed(active: bool, reason: str):
    """
    Discord通知は状態変化時だけ送る。
    - OFF -> ON: 抑制開始通知
    - ON  -> OFF: 解除通知
    """
    global _btc_long_pause_runtime_state

    if not BTC_SURGE_LONG_PAUSE_NOTIFY:
        _btc_long_pause_runtime_state["effective_active"] = bool(active)
        _btc_long_pause_runtime_state["reason"] = str(reason or "")
        _btc_long_pause_runtime_state["last_sent_ts"] = time.time()
        return

    prev_active = _btc_long_pause_runtime_state.get("effective_active", None)
    prev_reason = str(_btc_long_pause_runtime_state.get("reason", "") or "")

    _btc_long_pause_runtime_state["effective_active"] = bool(active)
    _btc_long_pause_runtime_state["reason"] = str(reason or "")
    _btc_long_pause_runtime_state["last_sent_ts"] = time.time()

    # 初回は、active=True のときだけ通知する
    if prev_active is None:
        if active:
            send_discord_message(
                "⚠️ BTC急騰を検知したため、LONG通知を一時抑制中です\n"
                "・対象: LONGのみ\n"
                "・理由: BTC主導の急上昇直後は反落しやすいため、急騰中または冷却時間中の追いかけLONGを抑制します\n"
                "・SHORT: 通常運転\n"
                "・状態: 相場監視中。条件が落ち着けば自動で解除します\n"
                f"・詳細: {reason}"
            )
        return

    # 同じ状態なら送らない
    if bool(prev_active) == bool(active):
        return

    if active:
        send_discord_message(
            "⚠️ BTC急騰を検知したため、LONG通知を一時抑制中です\n"
            "・対象: LONGのみ\n"
            "・理由: BTC主導の急上昇直後は反落しやすいため、急騰中または冷却時間中の追いかけLONGを抑制します\n"
            "・SHORT: 通常運転\n"
            "・状態: 相場監視中。条件が落ち着けば自動で解除します\n"
            f"・詳細: {reason}"
        )
    else:
        send_discord_message(
            "✅ BTC急騰局面のLONG抑制を解除しました\n"
            "・対象: LONG通知を通常運転に戻しました\n"
            "・SHORT: 通常運転\n"
            "・状態: 通常モードへ復帰\n"
            f"・前回理由: {prev_reason}"
        )


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


def _lane_display_name(lane: str) -> str:
    lane_u = str(lane or "").strip().lower()

    if lane_u == "long_trend":
        return "LONG本線"
    if lane_u == "long_recovery":
        return "LONG回復"
    if lane_u == "short_trend":
        return "SHORT本線"
    if lane_u == "short_retrace":
        return "SHORT戻り売り"

    return ""


def _profile_env_name(profile_name: str) -> str:
    key = re.sub(r"[^A-Z0-9]+", "_", str(profile_name or "").strip().upper())
    return f"V2_FEATURE_PROFILE_{key}"


def _parse_profile_spec(raw: str) -> Dict[str, str]:
    spec: Dict[str, str] = {}
    text = str(raw or "").strip()
    if text == "":
        return spec

    for part in text.split(";"):
        part = str(part).strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        spec[str(k).strip().lower()] = str(v).strip()
    return spec


def _profile_sig_value(sig: Dict[str, Any], key: str):
    k = str(key or "").strip().lower()
    aliases = {
        "btc_mode_compat": "mode",
        "btc_mode": "mode",
        "vol_ratio": "volratio",
        "band_width": "bandwidth",
        "bwchange": "bw_change",
        "volchange": "vol_change",
        "fravailable": "fr_available",
        "vol_confirmed": "volconfirmed",
    }
    k = aliases.get(k, k)
    if k == "side":
        return str(sig.get("direction", sig.get("side", "")) or "").strip().upper()
    if k == "lane":
        lane = str(sig.get("_runtime_lane", sig.get("lane", "")) or "").strip().lower()
        if lane == "":
            try:
                side = str(sig.get("direction", sig.get("side", "")) or "").strip().upper()
                if side in ("LONG", "SHORT"):
                    lane = str(_get_runtime_lane_for_signal(sig, side) or "").strip().lower()
            except Exception:
                lane = ""
        return lane
    if k == "symbol":
        raw = str(sig.get("symbol", sig.get("Symbol", "")) or "").strip().upper()
        return raw.split("/", 1)[0]
    if k == "hour":
        return _safe_float_or_nan(
            sig.get("hour_jst", sig.get("Hour_JST", sig.get("hour", "")))
        )
    if k == "mode":
        return str(sig.get("btc_mode_compat", sig.get("btc_mode", "")) or "").strip().upper()
    if k == "p1":
        return _safe_float_or_nan(sig.get("p1_score"))
    if k == "p2":
        return _safe_float_or_nan(sig.get("p2_score"))
    if k == "p3":
        return _safe_float_or_nan(sig.get("p3_score"))
    if k == "rsi":
        return _safe_float_or_nan(sig.get("rsi"))
    if k == "volratio":
        return _safe_float_or_nan(sig.get("vol_ratio"))
    if k == "ai":
        return _safe_float_or_nan(
            sig.get("ai_prob_win", sig.get("AI_Prob_Win", sig.get("ai_proba_used", "")))
        )
    if k == "bandwidth":
        return _safe_float_or_nan(sig.get("BandWidth"))
    if k == "bw_change":
        return _safe_float_or_nan(sig.get("BW_Change"))
    if k == "vol_change":
        return _safe_float_or_nan(sig.get("Vol_Change"))
    if k == "btc_ret":
        return _safe_float_or_nan(sig.get("btc_ret"))
    if k == "btc_vol":
        return _safe_float_or_nan(sig.get("btc_vol"))
    if k == "ltf_aligned":
        v = sig.get("ltf_aligned", "")
        s = str(v or "").strip().lower()
        if s in ("1", "true", "yes", "on"):
            return 1.0
        if s in ("0", "false", "no", "off"):
            return 0.0
        return np.nan
    if k == "volconfirmed":
        v = sig.get("vol_confirmed", "")
        s = str(v or "").strip().lower()
        if s in ("1", "true", "yes", "on"):
            return 1.0
        if s in ("0", "false", "no", "off"):
            return 0.0
        return np.nan
    if k == "fr_available":
        v = sig.get("fr_available", "")
        s = str(v or "").strip().lower()
        if s in ("1", "true", "yes", "on"):
            return 1.0
        if s in ("0", "false", "no", "off"):
            return 0.0
        return np.nan
    sig["_feature_unknown_keys"] = list(
        sorted(set((sig.get("_feature_unknown_keys", []) or []) + [k]))
    )
    return None


def _profile_match_numeric(val: Any, min_v: str = "", max_v: str = "") -> bool:
    x = _safe_float_or_nan(val)
    if not np.isfinite(x):
        return False

    if str(min_v).strip() != "":
        if float(x) < float(min_v):
            return False

    if str(max_v).strip() != "":
        if float(x) >= float(max_v):
            return False

    return True


def _match_feature_profile(sig: Dict[str, Any], profile_name: str) -> bool:
    raw = os.environ.get(_profile_env_name(profile_name), "").strip()
    spec = _parse_profile_spec(raw)
    if not spec:
        return False

    # exact match
    for key, expected in spec.items():
        k = str(key).strip().lower()
        exp = str(expected).strip()

        if k.endswith("_min") or k.endswith("_max"):
            continue

        actual = _profile_sig_value(sig, k)

        if k == "lane":
            if str(actual or "").strip().lower() != exp.lower():
                return False
            continue

        if k == "symbol":
            if str(actual or "").strip().upper() != exp.upper():
                return False
            continue

        if k in ("side", "mode"):
            if str(actual or "").strip().upper() != exp.upper():
                return False
            continue

        if k == "hour_in":
            hour_val = _profile_sig_value(sig, "hour")
            x = _safe_float_or_nan(hour_val)
            if not np.isfinite(x):
                return False
            try:
                allowed_hours = {
                    int(float(h.strip()))
                    for h in str(exp).split(",")
                    if h.strip()
                }
                if int(x) not in allowed_hours:
                    return False
            except (ValueError, TypeError):
                return False
            continue

        if k in ("ltf_aligned", "volconfirmed", "fr_available"):
            ax = _safe_float_or_nan(actual)
            if not np.isfinite(ax):
                return False
            if float(ax) != float(exp):
                return False
            continue

    # numeric range match
    for base in ("p1", "p2", "p3", "rsi", "volratio", "ai", "hour", "bandwidth", "bw_change", "vol_change", "btc_ret", "btc_vol"):
        min_v = spec.get(f"{base}_min", "")
        max_v = spec.get(f"{base}_max", "")
        if str(min_v).strip() == "" and str(max_v).strip() == "":
            continue

        actual = _profile_sig_value(sig, base)
        if not _profile_match_numeric(actual, min_v, max_v):
            return False

    return True


def _apply_feature_profiles(sig: Dict[str, Any]) -> Dict[str, Any]:
    allow_hits: List[str] = []
    reject_hits: List[str] = []
    force_hits: List[str] = []
    if not V2_FEATURE_PROFILE_ENABLE:
        sig["_feature_profile_allow_hits"] = allow_hits
        sig["_feature_profile_reject_hits"] = reject_hits
        sig["_feature_force_hits"] = force_hits
        sig["_feature_profile_gate"] = ""
        sig["_feature_force_profile"] = ""
        return sig
    for name in V2_FEATURE_ALLOW_PROFILES:
        if _match_feature_profile(sig, name):
            allow_hits.append(name)
    for name in V2_FEATURE_REJECT_PROFILES:
        if _match_feature_profile(sig, name):
            reject_hits.append(name)
    bypass_names = [str(x).strip() for x in V2_FEATURE_BYPASS_PROFILES if str(x).strip()]
    if not bypass_names:
        bypass_names = [str(x).strip() for x in V2_FEATURE_ALLOW_PROFILES if str(x).strip()]
    for name in allow_hits:
        if name in bypass_names:
            force_hits.append(name)
    sig["_feature_profile_allow_hits"] = allow_hits
    sig["_feature_profile_reject_hits"] = reject_hits
    sig["_feature_force_hits"] = force_hits
    sig["_feature_force_profile"] = force_hits[0] if force_hits else ""
    if force_hits:
        sig["_feature_profile_gate"] = "force"
    elif allow_hits:
        sig["_feature_profile_gate"] = "allow"
    elif V2_FEATURE_PROFILE_REQUIRE_MATCH and V2_FEATURE_ALLOW_PROFILES:
        sig["_feature_profile_gate"] = "no_match"
    else:
        sig["_feature_profile_gate"] = ""
    return sig
def _is_feature_force_pass(sig: Dict[str, Any]) -> bool:
    gate = str(sig.get("_feature_profile_gate", "") or "").strip().lower()
    force_profile = str(sig.get("_feature_force_profile", "") or "").strip()
    return gate == "force" and force_profile != ""


def _feature_profile_label(sig: Dict[str, Any]) -> str:
    allow_hits = [str(x) for x in (sig.get("_feature_profile_allow_hits", []) or []) if str(x).strip()]
    reject_hits = [str(x) for x in (sig.get("_feature_profile_reject_hits", []) or []) if str(x).strip()]

    if reject_hits:
        return f"reject:{'|'.join(reject_hits)}"
    if allow_hits:
        return f"allow:{'|'.join(allow_hits)}"
    return ""


def _feature_profile_log_parts(sig: Dict[str, Any]) -> Dict[str, str]:
    gate = str(sig.get("_feature_profile_gate", "") or "").strip()
    allow_hits = [
        str(x).strip()
        for x in (sig.get("_feature_profile_allow_hits", []) or [])
        if str(x).strip()
    ]
    reject_hits = [
        str(x).strip()
        for x in (sig.get("_feature_profile_reject_hits", []) or [])
        if str(x).strip()
    ]
    force_hits = [
        str(x).strip()
        for x in (sig.get("_feature_force_hits", []) or [])
        if str(x).strip()
    ]
    bypass_profile = str(
        sig.get("_feature_bypass_profile", sig.get("_feature_force_profile", "")) or ""
    ).strip()

    return {
        "FEATURE_GATE": gate,
        "ALLOW_HITS": ",".join(allow_hits),
        "REJECT_HITS": ",".join(reject_hits),
        "FORCE_HITS": ",".join(force_hits),
        "BYPASS_PROFILE": bypass_profile,
    }


def _detect_signal_lane(sig: Dict[str, Any]) -> str:
    direction = str(sig.get("direction", "") or sig.get("Direction", "")).strip().upper()
    if direction not in ("LONG", "SHORT"):
        return ""

    try:
        if direction == "LONG":
            lane, _ = _classify_long_lane(sig)
            return str(lane or "").strip()

        if direction == "SHORT":
            lane, _ = _classify_short_lane(sig)
            return str(lane or "").strip()
    except Exception:
        return ""

    return ""


def _get_v2_symbol_recent_stats(symbol: str, direction: str, lookback_n: int = 20) -> Dict[str, Any]:
    """
    v2_shadow_ai から同一銘柄・同方向の直近成績を取得する（通知ベース）。
    notify_pass=1 または AI_Pass==1 のエントリのみ対象。
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

        # 通知ベースフィルター: notify_pass=1 または AI_Pass==1
        notify_mask = pd.Series([False] * len(work), index=work.index)
        if "AI_Note" in work.columns:
            note_col = work["AI_Note"].astype(str).str.lower()
            notify_mask = notify_mask | note_col.str.contains("notify_pass=1", na=False)
        if "AI_Pass" in work.columns:
            ai_pass_col = work["AI_Pass"].astype(str).str.strip().str.lower()
            notify_mask = notify_mask | ai_pass_col.isin(["1", "true", "yes"])
        work = work[notify_mask].copy()

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


def _get_v2_global_recent_stats(direction: str, lookback_n: int = 20) -> Dict[str, Any]:
    """
    v2_shadow_ai から指定方向の全銘柄直近成績を取得する（通知ベース）。
    notify_pass=1 または AI_Pass==1 のエントリのみ対象。
    """
    try:
        df = get_v2_shadow_ai_data()
        if df is None or df.empty:
            return {"n": 0, "wins": 0, "wr": np.nan}

        work = df.copy()

        if "Direction" not in work.columns:
            return {"n": 0, "wins": 0, "wr": np.nan}

        if "Datetime_JST" in work.columns:
            work["Datetime_JST"] = pd.to_datetime(
                work["Datetime_JST"].astype(str).str.strip(),
                errors="coerce",
            )
            work = work[work["Datetime_JST"].notna()].copy()
            work = work.sort_values("Datetime_JST", ascending=False)

        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()

        if "EvalStatus" in work.columns:
            work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
            work = work[work["EvalStatus"] == "DONE"].copy()

        # 通知ベースフィルター: notify_pass=1 または AI_Pass==1
        notify_mask = pd.Series([False] * len(work), index=work.index)
        if "AI_Note" in work.columns:
            note_col = work["AI_Note"].astype(str).str.lower()
            notify_mask = notify_mask | note_col.str.contains("notify_pass=1", na=False)
        if "AI_Pass" in work.columns:
            ai_pass_col = work["AI_Pass"].astype(str).str.strip().str.lower()
            notify_mask = notify_mask | ai_pass_col.isin(["1", "true", "yes"])
        work = work[notify_mask].copy()

        sub = work[work["Direction"] == str(direction).strip().upper()].copy()

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


def _get_v2_long_recent_notified_window_state(now_jst: datetime) -> Dict[str, Any]:
    """
    直近 window 分の「通知済みLONG」を銘柄別に集計する。
    目的:
      - 同一銘柄LONGの連打を止める
    """
    global _v2_long_repeat_cache

    now_ts = time.time()
    cached_state = _v2_long_repeat_cache.get("state", {}) or {}
    cached_ts = float(_v2_long_repeat_cache.get("ts", 0.0) or 0.0)
    cached_window = int(_v2_long_repeat_cache.get("window_min", 0) or 0)

    if (
        cached_state
        and (now_ts - cached_ts) <= 60
        and cached_window == int(V2_LONG_REPEAT_WINDOW_MINUTES)
    ):
        return dict(cached_state)

    out: Dict[str, Any] = {
        "counts": {},
        "last_dt": {},
        "reason": "ok",
    }

    if not V2_LONG_REPEAT_BLOCK_ENABLE:
        _v2_long_repeat_cache["ts"] = now_ts
        _v2_long_repeat_cache["window_min"] = int(V2_LONG_REPEAT_WINDOW_MINUTES)
        _v2_long_repeat_cache["state"] = dict(out)
        return out

    try:
        df = get_v2_shadow_ai_recent_data_for_repeat()
        if df is None or df.empty:
            out["reason"] = "empty"
            _v2_long_repeat_cache["ts"] = now_ts
            _v2_long_repeat_cache["window_min"] = int(V2_LONG_REPEAT_WINDOW_MINUTES)
            _v2_long_repeat_cache["state"] = dict(out)
            return out

        work = df.copy()

        required_cols = {"Datetime_JST", "Symbol", "Direction"}
        if not required_cols.issubset(set(work.columns)):
            out["reason"] = f"missing_cols:{sorted(required_cols - set(work.columns))}"
            _v2_long_repeat_cache["ts"] = now_ts
            _v2_long_repeat_cache["window_min"] = int(V2_LONG_REPEAT_WINDOW_MINUTES)
            _v2_long_repeat_cache["state"] = dict(out)
            return out

        work["Datetime_JST"] = pd.to_datetime(
            work["Datetime_JST"].astype(str).str.strip(),
            errors="coerce",
        )
        work = work[work["Datetime_JST"].notna()].copy()

        work["Symbol"] = work["Symbol"].astype(str).str.strip().str.upper()
        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
        work = work[work["Direction"] == "LONG"].copy()

        notify_mask = pd.Series([False] * len(work), index=work.index)
        if "AI_Note" in work.columns:
            note_col = work["AI_Note"].astype(str).str.lower()
            notify_mask = notify_mask | note_col.str.contains("notify_pass=1", na=False)
        if "AI_Pass" in work.columns:
            ai_pass_col = work["AI_Pass"].astype(str).str.strip().str.lower()
            notify_mask = notify_mask | ai_pass_col.isin(["1", "true", "yes"])

        work = work[notify_mask].copy()

        cutoff = now_jst - timedelta(minutes=int(V2_LONG_REPEAT_WINDOW_MINUTES))
        work = work[work["Datetime_JST"] >= cutoff].copy()

        if work.empty:
            out["reason"] = "no_recent_notified_longs"
            _v2_long_repeat_cache["ts"] = now_ts
            _v2_long_repeat_cache["window_min"] = int(V2_LONG_REPEAT_WINDOW_MINUTES)
            _v2_long_repeat_cache["state"] = dict(out)
            return out

        counts = work.groupby("Symbol").size().to_dict()
        last_dt = work.groupby("Symbol")["Datetime_JST"].max().to_dict()

        out["counts"] = {
            str(k).strip().upper(): int(v)
            for k, v in counts.items()
        }
        out["last_dt"] = {
            str(k).strip().upper(): v.strftime("%Y-%m-%d %H:%M:%S")
            for k, v in last_dt.items()
        }

    except Exception as e:
        out["reason"] = f"repeat_window_error:{type(e).__name__}:{e}"

    _v2_long_repeat_cache["ts"] = now_ts
    _v2_long_repeat_cache["window_min"] = int(V2_LONG_REPEAT_WINDOW_MINUTES)
    _v2_long_repeat_cache["state"] = dict(out)
    return out


def _incident_sig_float(sig: Dict[str, Any], *keys: str) -> float:
    for k in keys:
        v = _safe_float_or_nan(sig.get(k))
        if np.isfinite(v):
            return float(v)
    return float("nan")


def _incident_sig_bool(sig: Dict[str, Any], *keys: str) -> bool:
    for k in keys:
        raw = sig.get(k, None)
        s = ("" if raw is None else str(raw)).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return False


def _is_btc_surge_after_long(sig: Dict[str, Any]) -> bool:
    btc_mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    btc_1h_change = _incident_sig_float(sig, "btc_1h_change", "BTC_1h_Change")
    btc_ret = _incident_sig_float(sig, "btc_ret", "BTC_Ret")
    pause_active = bool(_btc_long_pause_ctx.get("active", False))

    if pause_active:
        return True

    if btc_mode != "UP":
        return False

    if np.isfinite(btc_1h_change) and btc_1h_change >= float(V2_INCIDENT_BTC_1H_SURGE_MIN):
        return True

    if np.isfinite(btc_ret) and btc_ret >= float(V2_INCIDENT_BTC_15M_SURGE_MIN):
        return True

    return False


def _is_btc_dump_after_short(sig: Dict[str, Any]) -> bool:
    btc_mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    btc_1h_change = _incident_sig_float(sig, "btc_1h_change", "BTC_1h_Change")
    btc_ret = _incident_sig_float(sig, "btc_ret", "BTC_Ret")

    if btc_mode not in {"DOWN", "RANGE"}:
        return False

    if np.isfinite(btc_1h_change) and btc_1h_change <= float(V2_INCIDENT_BTC_1H_DUMP_MAX):
        return True

    if np.isfinite(btc_ret) and btc_ret <= float(V2_INCIDENT_BTC_15M_DUMP_MAX):
        return True

    return False


def _is_long_reversal_accident(sig: Dict[str, Any]) -> bool:
    btc_mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    btc_1h_change = _incident_sig_float(sig, "btc_1h_change", "BTC_1h_Change")
    btc_ret = _incident_sig_float(sig, "btc_ret", "BTC_Ret")

    if btc_mode != "UP":
        return False

    if not np.isfinite(btc_1h_change):
        return False
    if not np.isfinite(btc_ret):
        return False

    return (
        btc_1h_change >= float(V2_LONG_REVERSAL_BTC_1H_MIN)
        and btc_ret <= float(V2_LONG_REVERSAL_BTC_RET_MAX)
    )


def _is_vol_shock_context(sig: Dict[str, Any]) -> bool:
    vol_ratio = _incident_sig_float(sig, "vol_ratio", "VolRatio")
    btc_calm = _incident_sig_bool(sig, "btc_calm", "BTC_Calm")
    market_tag = str(sig.get("market_tag", "") or "").strip().upper()

    if np.isfinite(vol_ratio) and vol_ratio >= float(V2_INCIDENT_VOLRATIO_SHOCK_MIN):
        return True

    if market_tag == "STORM":
        return True

    if str(sig.get("btc_calm", "")).strip() != "" and (not btc_calm):
        return True

    return False


def _long_incident_surge_penalty(sig: Dict[str, Any]) -> float:
    if not V2_LONG_SURGE_PENALTY_ENABLE:
        return 0.0
    return float(V2_LONG_SURGE_PENALTY_VALUE) if _is_btc_surge_after_long(sig) else 0.0


def _long_incident_vol_shock_penalty(sig: Dict[str, Any]) -> float:
    if not V2_LONG_VOL_SHOCK_PENALTY_ENABLE:
        return 0.0
    return float(V2_LONG_VOL_SHOCK_PENALTY_VALUE) if _is_vol_shock_context(sig) else 0.0


def _short_incident_dump_penalty(sig: Dict[str, Any]) -> float:
    if not V2_SHORT_DUMP_PENALTY_ENABLE:
        return 0.0
    return float(V2_SHORT_DUMP_PENALTY_VALUE) if _is_btc_dump_after_short(sig) else 0.0


def _short_incident_vol_shock_penalty(sig: Dict[str, Any]) -> float:
    if not V2_SHORT_VOL_SHOCK_PENALTY_ENABLE:
        return 0.0
    return float(V2_SHORT_VOL_SHOCK_PENALTY_VALUE) if _is_vol_shock_context(sig) else 0.0


def _get_v2_short_recent_notified_window_state(now_jst: datetime) -> Dict[str, Any]:
    """
    直近 window 分の「通知済みSHORT」を銘柄別に集計する。
    LONG側の repeat penalty と同じ考え方で SHORT 側も扱う。
    """
    cache_key = "short_recent_notified_window_state"
    now_ts = time.time()

    try:
        cached = _repeat_state_cache.get(cache_key, {})
        if cached and (now_ts - float(cached.get("ts", 0.0))) <= REPEAT_STATE_TTL_SEC:
            return dict(cached.get("value", {}) or {})
    except Exception:
        pass

    out: Dict[str, Any] = {
        "counts": {},
        "last_dt": {},
        "reason": "ok",
    }

    if not V2_SHORT_REPEAT_PENALTY_ENABLE:
        return out

    try:
        df = get_v2_shadow_ai_recent_data_for_repeat()
        if df is None or df.empty:
            out["reason"] = "empty"
            return out

        work = df.copy()

        required_cols = {"Datetime_JST", "Symbol", "Direction"}
        if not required_cols.issubset(set(work.columns)):
            out["reason"] = f"missing_cols:{sorted(required_cols - set(work.columns))}"
            return out

        work["Datetime_JST"] = pd.to_datetime(
            work["Datetime_JST"].astype(str).str.strip(),
            errors="coerce",
        )
        work = work[work["Datetime_JST"].notna()].copy()

        work["Symbol"] = work["Symbol"].astype(str).str.strip().str.upper()
        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
        work = work[work["Direction"] == "SHORT"].copy()

        notify_mask = pd.Series([False] * len(work), index=work.index)
        if "AI_Note" in work.columns:
            note_col = work["AI_Note"].astype(str).str.lower()
            notify_mask = notify_mask | note_col.str.contains("notify_pass=1", na=False)
        if "AI_Pass" in work.columns:
            ai_pass_col = work["AI_Pass"].astype(str).str.strip().str.lower()
            notify_mask = notify_mask | ai_pass_col.isin(["1", "true", "yes"])

        work = work[notify_mask].copy()

        cutoff = now_jst - timedelta(minutes=int(V2_SHORT_REPEAT_WINDOW_MINUTES))
        work = work[work["Datetime_JST"] >= cutoff].copy()

        if work.empty:
            out["reason"] = "no_recent_notified_shorts"
            return out

        counts = work.groupby("Symbol").size().to_dict()
        last_dt = work.groupby("Symbol")["Datetime_JST"].max().to_dict()

        out["counts"] = {
            str(k).strip().upper(): int(v)
            for k, v in counts.items()
        }
        out["last_dt"] = {
            str(k).strip().upper(): v.strftime("%Y-%m-%d %H:%M:%S")
            for k, v in last_dt.items()
        }

    except Exception as e:
        out["reason"] = f"short_repeat_window_error:{type(e).__name__}:{e}"

    try:
        _repeat_state_cache[cache_key] = {
            "ts": now_ts,
            "value": dict(out or {}),
        }
    except Exception:
        pass
    return out


def _short_repeat_rank_penalty(sig: Dict[str, Any]) -> Tuple[float, int, str]:
    """
    同一銘柄SHORT連打を rank penalty で扱う。
    """
    if not V2_SHORT_REPEAT_PENALTY_ENABLE:
        return 0.0, 0, ""

    try:
        sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()
        if not sym:
            return 0.0, 0, ""

        pref_recent_n = sig.get("_prefetched_short_repeat_recent_n", None)
        pref_last_dt = sig.get("_prefetched_short_repeat_last_dt", None)

        if pref_recent_n is None:
            repeat_state = _get_v2_short_recent_notified_window_state(datetime.now(JST))
            recent_n = int(((repeat_state.get("counts", {}) or {}).get(sym, 0)) or 0)
            last_dt = str(((repeat_state.get("last_dt", {}) or {}).get(sym, "")) or "")
        else:
            recent_n = int(pref_recent_n or 0)
            last_dt = str(pref_last_dt or "")

        if recent_n > int(V2_SHORT_REPEAT_PENALTY_MIN_COUNT):
            return float(V2_SHORT_REPEAT_PENALTY_VALUE), recent_n, last_dt

        return 0.0, recent_n, last_dt

    except Exception:
        return 0.0, 0, ""


def get_v2_long_fast_brake_state(now_jst: datetime) -> Dict[str, Any]:
    """
    直近DONE済み通知LONGの成績だけを見て、当日LONGを即停止するための高速ブレーキ。
    既存guardrailより少ない母数で動かす。
    """
    global _v2_long_fast_brake_cache

    now_ts = time.time()
    cached_state = _v2_long_fast_brake_cache.get("state", {}) or {}
    cached_ts = float(_v2_long_fast_brake_cache.get("ts", 0.0) or 0.0)

    if cached_state and (now_ts - cached_ts) <= 60:
        return dict(cached_state)

    out: Dict[str, Any] = {
        "active": False,
        "n": 0,
        "wins": 0,
        "wr": np.nan,
        "avg_pnl": np.nan,
        "reason": "ok",
        "reason_text": "",
        "latest_done_dt": "",
    }

    if not V2_LONG_FAST_BRAKE_ENABLE:
        out["reason"] = "feature_off"
        _v2_long_fast_brake_cache["ts"] = now_ts
        _v2_long_fast_brake_cache["state"] = dict(out)
        return out

    try:
        try:
            df = get_v2_shadow_ai_recent_data_for_repeat()
        except Exception:
            df = get_v2_shadow_ai_data()
        if df is None or df.empty:
            out["reason"] = "empty"
            _v2_long_fast_brake_cache["ts"] = now_ts
            _v2_long_fast_brake_cache["state"] = dict(out)
            return out

        work = df.copy()

        required_cols = {"Datetime_JST", "Direction", "WinLose", "PnL_Pct"}
        if not required_cols.issubset(set(work.columns)):
            out["reason"] = f"missing_cols:{sorted(required_cols - set(work.columns))}"
            _v2_long_fast_brake_cache["ts"] = now_ts
            _v2_long_fast_brake_cache["state"] = dict(out)
            return out

        work["Datetime_JST"] = pd.to_datetime(
            work["Datetime_JST"].astype(str).str.strip(),
            errors="coerce",
        )
        work = work[work["Datetime_JST"].notna()].copy()
        work = work.sort_values("Datetime_JST", ascending=False)

        work["Direction"] = work["Direction"].astype(str).str.strip().str.upper()
        work = work[work["Direction"] == "LONG"].copy()

        if "EvalStatus" in work.columns:
            work["EvalStatus"] = work["EvalStatus"].astype(str).str.strip().str.upper()
            work = work[work["EvalStatus"] == "DONE"].copy()

        notify_mask = pd.Series([False] * len(work), index=work.index)
        if "AI_Note" in work.columns:
            note_col = work["AI_Note"].astype(str).str.lower()
            notify_mask = notify_mask | note_col.str.contains("notify_pass=1", na=False)
        if "AI_Pass" in work.columns:
            ai_pass_col = work["AI_Pass"].astype(str).str.strip().str.lower()
            notify_mask = notify_mask | ai_pass_col.isin(["1", "true", "yes"])

        work = work[notify_mask].copy()

        if work.empty:
            out["reason"] = "no_done_notified_longs"
            _v2_long_fast_brake_cache["ts"] = now_ts
            _v2_long_fast_brake_cache["state"] = dict(out)
            return out

        latest_done_dt = work["Datetime_JST"].max()
        if pd.notna(latest_done_dt):
            out["latest_done_dt"] = latest_done_dt.strftime("%Y-%m-%d %H:%M:%S")

        work["PnL_Pct"] = pd.to_numeric(work["PnL_Pct"], errors="coerce")
        work = work.head(int(V2_LONG_FAST_BRAKE_LOOKBACK_N)).copy()

        n = int(len(work))
        wins = int(
            work["WinLose"].astype(str).str.strip().str.lower().isin(["win", "w", "1", "true"]).sum()
        )
        wr_pct = float((wins / n) * 100.0) if n > 0 else np.nan
        avg_pnl = float(work["PnL_Pct"].mean()) if n > 0 else np.nan

        out["n"] = n
        out["wins"] = wins
        out["wr"] = wr_pct
        out["avg_pnl"] = avg_pnl

        active = (
            n >= int(V2_LONG_FAST_BRAKE_MIN_DONE)
            and np.isfinite(wr_pct)
            and np.isfinite(avg_pnl)
            and float(wr_pct) <= float(V2_LONG_FAST_BRAKE_STOP_WINRATE)
            and float(avg_pnl) < float(V2_LONG_FAST_BRAKE_STOP_AVGPNL)
        )

        out["active"] = bool(active)
        out["reason"] = "active" if active else "ok"
        out["reason_text"] = (
            f"n={n} "
            f"wins={wins} "
            f"wr={'' if not np.isfinite(wr_pct) else format(float(wr_pct), '.1f')} "
            f"avg_pnl={'' if not np.isfinite(avg_pnl) else format(float(avg_pnl), '.4f')} "
            f"lookback_n={int(V2_LONG_FAST_BRAKE_LOOKBACK_N)} "
            f"latest_done_dt={out['latest_done_dt']}"
        )

    except Exception as e:
        out["active"] = False
        out["reason"] = "error"
        out["reason_text"] = f"long_fast_brake_error:{type(e).__name__}:{e}"

    _v2_long_fast_brake_cache["ts"] = now_ts
    _v2_long_fast_brake_cache["state"] = dict(out)
    return out


def _resolve_v2_long_fast_brake_runtime(now_jst: datetime) -> Dict[str, Any]:
    """
    long_fast_brake の実効状態を解決する。

    方針:
    - 発動条件を満たしたら、最低 hold 分は止める
    - hold 終了後は一度解除する
    - ただし、前回ブレーキ発動時に見ていた latest_done_dt と同じ間は再発動しない
    - 新しい DONE済み通知LONG が増えたときだけ、再度ブレーキ判定する
    """
    global _v2_long_fast_brake_runtime

    raw_state = get_v2_long_fast_brake_state(now_jst)
    now_ts = time.time()

    hold_sec = max(0, int(V2_LONG_FAST_BRAKE_HOLD_MINUTES)) * 60

    runtime_active = bool(_v2_long_fast_brake_runtime.get("active", False))
    until_ts = float(_v2_long_fast_brake_runtime.get("until_ts", 0.0) or 0.0)
    last_trigger_latest_done_dt = str(
        _v2_long_fast_brake_runtime.get("last_trigger_latest_done_dt", "") or ""
    )

    raw_active = bool(raw_state.get("active", False))
    raw_latest_done_dt = str(raw_state.get("latest_done_dt", "") or "")
    raw_reason_text = str(raw_state.get("reason_text", "") or "")

    # すでに保持中なら、期限まではそのまま止める
    if runtime_active and until_ts > now_ts:
        remain_sec = int(max(0, until_ts - now_ts))
        out = dict(raw_state)
        out["active"] = True
        out["reason"] = "holding"
        out["reason_text"] = (
            f"holding remain_sec={remain_sec} "
            f"hold_min={int(V2_LONG_FAST_BRAKE_HOLD_MINUTES)} "
            f"last_trigger_latest_done_dt={last_trigger_latest_done_dt} "
            f"{raw_reason_text}"
        )
        return out

    # 保持が終わったら一度解除
    if runtime_active and until_ts <= now_ts:
        _v2_long_fast_brake_runtime["active"] = False
        _v2_long_fast_brake_runtime["until_ts"] = 0.0
        _v2_long_fast_brake_runtime["reason"] = ""

    # 新しい DONE済み通知LONG が増えていない限り、同じ材料では再発動させない
    can_rearm = bool(
        raw_active
        and raw_latest_done_dt
        and (raw_latest_done_dt != last_trigger_latest_done_dt)
    )

    if can_rearm:
        new_until_ts = now_ts + hold_sec
        _v2_long_fast_brake_runtime["active"] = True
        _v2_long_fast_brake_runtime["until_ts"] = float(new_until_ts)
        _v2_long_fast_brake_runtime["last_trigger_latest_done_dt"] = raw_latest_done_dt
        _v2_long_fast_brake_runtime["reason"] = str(raw_reason_text or "")

        out = dict(raw_state)
        out["active"] = True
        out["reason"] = "active"
        out["reason_text"] = (
            f"active hold_min={int(V2_LONG_FAST_BRAKE_HOLD_MINUTES)} "
            f"{raw_reason_text}"
        )
        return out

    out = dict(raw_state)
    out["active"] = False
    out["reason"] = "released"
    out["reason_text"] = (
        f"released hold_min={int(V2_LONG_FAST_BRAKE_HOLD_MINUTES)} "
        f"latest_done_dt={raw_latest_done_dt} "
        f"last_trigger_latest_done_dt={last_trigger_latest_done_dt} "
        f"{raw_reason_text}"
    )
    return out


def _ai_strength_label_from_pct_text(ai_pct_text: str) -> str:
    """
    _safe_pct_str() で作った "72.1" のような文字列を受けて、
    強 / 中 / 弱 を返す。
    """
    try:
        v = float(str(ai_pct_text).replace("%", "").strip())
    except Exception:
        return ""

    if v >= 70.0:
        return "強"
    if v >= 55.0:
        return "中"
    return "弱"


def _v2_signal_strength_label(sig: Dict[str, Any]) -> str:
    total_score = _safe_float_or_nan(sig.get("total_score", sig.get("score", "")))
    ai_prob = _safe_float_or_nan(sig.get("ai_prob_win", sig.get("ai_proba_used", "")))

    points = 0

    if np.isfinite(total_score):
        if total_score >= 2.2:
            points += 2
        elif total_score >= 1.5:
            points += 1

    if np.isfinite(ai_prob):
        if ai_prob >= 0.75:
            points += 2
        elif ai_prob >= 0.60:
            points += 1

    if points >= 3:
        return "大"
    if points >= 1:
        return "中"
    return "小"


def _v2_root_cause_lines(sig: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    direction = str(sig.get("direction", "") or "").strip().upper()
    btc_mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()

    p1 = _safe_float_or_nan(sig.get("p1_score", ""))
    p2 = _safe_float_or_nan(sig.get("p2_score", ""))
    p3 = _safe_float_or_nan(sig.get("p3_score", ""))
    rsi = _safe_float_or_nan(sig.get("rsi", ""))
    vol_ratio = _safe_float_or_nan(sig.get("vol_ratio", ""))

    if direction == "LONG":
        if np.isfinite(p1) and p1 >= 2.0:
            lines.append("・15分足の押し目買い形")
        elif np.isfinite(p1) and p1 >= 1.2:
            lines.append("・15分足で上方向の形")
        else:
            lines.append("・15分足で反発監視の形")
    elif direction == "SHORT":
        if np.isfinite(p1) and p1 >= 2.0:
            lines.append("・15分足の戻り売り形")
        elif np.isfinite(p1) and p1 >= 1.2:
            lines.append("・15分足で下方向の形")
        else:
            lines.append("・15分足で失速監視の形")

    if direction == "LONG":
        if btc_mode == "UP":
            lines.append("・BTC強含み")
        elif btc_mode == "RANGE":
            lines.append("・BTCはレンジ内")
        elif btc_mode == "DOWN":
            lines.append("・BTC弱含みの中で反発監視")
    elif direction == "SHORT":
        if btc_mode == "DOWN":
            lines.append("・BTC弱含み")
        elif btc_mode == "RANGE":
            lines.append("・BTCはレンジ内で上値重い")
        elif btc_mode == "UP":
            lines.append("・BTC強含みで逆風あり")

    extra = ""

    if np.isfinite(p3):
        if p3 > 0:
            extra = "・出来高増で勢いあり"
        elif p3 < 0:
            extra = "・出来高減で上値重い" if direction == "SHORT" else "・出来高減で勢いは弱め"

    if (not extra) and np.isfinite(p2):
        if p2 > 0:
            extra = "・資金の追い風あり"
        elif p2 < 0:
            extra = "・資金面はやや逆風"

    if (not extra) and np.isfinite(rsi):
        if direction == "LONG":
            if rsi < 45:
                extra = "・RSI低めで戻し余地あり"
            else:
                extra = "・RSIは中立〜やや強め"
        else:
            if rsi > 55:
                extra = "・RSI高めで反落警戒"
            else:
                extra = "・RSIは中立〜やや弱め"

    if extra:
        lines.append(extra)

    if np.isfinite(vol_ratio):
        if vol_ratio >= 1.8:
            lines.append(f"・出来高比が高い（VolR {vol_ratio:.2f}）")
        elif vol_ratio <= 0.8:
            lines.append(f"・出来高比は低め（VolR {vol_ratio:.2f}）")

    return lines[:4]


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
    ai_strength = _ai_strength_label_from_pct_text(ai_prob) if ai_prob else ""
    strength = _v2_signal_strength_label(sig)
    rationale_lines = _v2_root_cause_lines(sig)

    recent_symbol = _get_v2_symbol_recent_stats(symbol, direction, lookback_n=20)
    recent_global = _get_v2_global_recent_stats(direction, lookback_n=20)

    if direction == "LONG":
        side_emoji = "🟢"
        tp_emoji = "🎯"
        sl_emoji = "🛑"
    else:
        side_emoji = "🔴"
        tp_emoji = "🎯"
        sl_emoji = "🛑"

    recent_symbol_line = ""
    if int(recent_symbol.get("n", 0)) >= 10 and np.isfinite(recent_symbol.get("wr", np.nan)):
        recent_symbol_line = (
            f"📊 {symbol}直近{int(recent_symbol['n'])}件勝率: "
            f"{float(recent_symbol['wr']) * 100:.0f}% "
            f"（{int(recent_symbol['wins'])}/{int(recent_symbol['n'])}）"
        )
    elif int(recent_symbol.get("n", 0)) > 0:
        recent_symbol_line = f"📊 {symbol}直近{int(recent_symbol['n'])}件: 参考不足"

    recent_global_line = ""
    if int(recent_global.get("n", 0)) >= 10 and np.isfinite(recent_global.get("wr", np.nan)):
        recent_global_line = (
            f"🌐 全体直近{int(recent_global['n'])}件勝率: "
            f"{float(recent_global['wr']) * 100:.0f}% "
            f"（{int(recent_global['wins'])}/{int(recent_global['n'])}）"
        )
    elif int(recent_global.get("n", 0)) > 0:
        recent_global_line = f"🌐 全体直近{int(recent_global['n'])}件: 参考不足"

    lane_name = _lane_display_name(_detect_signal_lane(sig))
    feature_profile_text = ""
    if V2_FEATURE_PROFILE_SHOW_IN_DISCORD:
        feature_profile_text = _feature_profile_label(sig)

    lines = [
        f"{side_emoji}【{symbol} {direction}】",
        f"🧩 パターン：{lane_name}" if lane_name else "",
        f"🧬 特徴量プロファイル：{feature_profile_text}" if feature_profile_text else "",
        f"🏷️ 強さ：{strength}",
        f"💰 Entry: {entry}" if entry else "",
        f"{tp_emoji} TP: {tp_price}" if tp_price else "",
        f"{sl_emoji} SL: {sl_price}" if sl_price else "",
        f"📌 TP入力: {tp_pct}%（x{display_lev}で {tp_lev_pct}%）" if tp_pct else "",
        f"📌 SL入力: {sl_pct}%（x{display_lev}で {sl_lev_pct}%）" if sl_pct else "",
        f"🤖 AI判定: {ai_prob}%（{ai_strength}）" if ai_prob else "",
    ]

    if rationale_lines:
        lines.append("📌 根拠")
        lines.extend(rationale_lines)

    if recent_symbol_line:
        lines.append(recent_symbol_line)
    if recent_global_line:
        lines.append(recent_global_line)
    if dt_str:
        lines.append(f"🕒 {dt_str}")

    body = "\n".join([x for x in lines if x])

    # 4項目の人間監査ブロックを追加（V2キーをV1互換形式にマップ）
    audit_item: Dict[str, Any] = dict(sig)
    audit_item["is_buy"] = (direction == "LONG")
    audit_item["btc_mode"] = sig.get("btc_mode_compat", "")
    _ai_prob_raw = sig.get("ai_prob_win", sig.get("AI_Prob_Win", sig.get("ai_proba_used", "")))
    try:
        _ai_prob_f = float(_ai_prob_raw)
        audit_item["ai_score"] = _ai_prob_f
    except Exception:
        audit_item["ai_score"] = float("nan")
    audit_block = _build_signal_audit_block(audit_item)

    return body + "\n\n" + audit_block


def _audit_float(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _audit_ok_text(ok: bool) -> str:
    return "OK" if ok else "NG"


def _build_signal_audit_block(item: Dict[str, Any]) -> str:
    is_long = bool(item.get("is_buy", False))
    side = "LONG" if is_long else "SHORT"

    btc_mode = str(item.get("btc_mode", "") or "").strip().upper()
    ai_score = _audit_float(item.get("ai_score"))
    ai_th = float(LONG_AI_TH if is_long else SHORT_AI_TH)

    rsi = _audit_float(item.get("rsi"))
    vol_ratio = _audit_float(item.get("vol_ratio"))
    btc_1h = _audit_float(item.get("btc_1h_change"))
    btc_15m = _audit_float(item.get("btc_ret"))
    p1 = _audit_float(item.get("p1_score"))

    if is_long:
        btc_mode_ok = (btc_mode == "UP")
        if btc_mode_ok:
            btc_mode_detail = "LONG想定どおり UP"
        else:
            btc_mode_detail = f"LONG想定とズレ（{btc_mode or 'N/A'}）"
    else:
        btc_mode_ok = btc_mode in ("RANGE", "DOWN")
        if btc_mode == "DOWN":
            btc_mode_detail = "SHORT想定どおり DOWN"
        elif btc_mode == "RANGE":
            btc_mode_detail = "SHORT想定どおり RANGE"
        else:
            btc_mode_detail = f"SHORT想定とズレ（{btc_mode or 'N/A'}）"

    if not np.isfinite(ai_score):
        ai_ok = False
        ai_detail = "AI不明"
    elif ai_score >= ai_th:
        ai_ok = True
        ai_detail = f"side基準を満たす（{ai_score:.1%} / TH {ai_th:.1%}）"
    else:
        ai_ok = False
        ai_detail = f"side内で弱い（{ai_score:.1%} / TH {ai_th:.1%}）"

    risk_tags: List[str] = []

    if is_long:
        if np.isfinite(btc_1h) and np.isfinite(rsi) and np.isfinite(p1):
            if (
                btc_1h >= float(BTC_SURGE_LONG_PAUSE_1H_MIN)
                and rsi >= float(BTC_SURGE_LONG_PAUSE_RSI_MIN)
                and p1 >= float(BTC_SURGE_LONG_PAUSE_P1_MIN)
            ):
                risk_tags.append("急騰直後")

        if np.isfinite(btc_15m) and np.isfinite(btc_1h):
            if (
                btc_15m <= float(V2_LONG_REVERSAL_BTC_RET_MAX)
                and btc_1h >= float(V2_LONG_REVERSAL_BTC_1H_MIN)
            ):
                risk_tags.append("reversal")
    else:
        if np.isfinite(btc_1h) and btc_1h <= float(V2_INCIDENT_BTC_1H_DUMP_MAX):
            risk_tags.append("急落直後")
        if np.isfinite(btc_15m) and btc_15m <= float(V2_INCIDENT_BTC_15M_DUMP_MAX):
            risk_tags.append("急落直後")

    if np.isfinite(vol_ratio) and vol_ratio >= float(V2_INCIDENT_VOLRATIO_SHOCK_MIN):
        risk_tags.append("vol_shock")

    risk_tags = list(dict.fromkeys(risk_tags))
    risk_ok = (len(risk_tags) == 0)

    if risk_ok:
        risk_detail = "危険タグなし"
    else:
        risk_detail = " / ".join(risk_tags)

    extreme_parts: List[str] = []
    extreme_ok = True

    if is_long:
        if np.isfinite(rsi) and rsi > float(V2_LONG_DEF_RSI_HARD_MAX):
            extreme_ok = False
            extreme_parts.append(f"RSI={rsi:.1f}")
        if np.isfinite(vol_ratio) and vol_ratio >= float(V2_INCIDENT_VOLRATIO_SHOCK_MIN):
            extreme_ok = False
            extreme_parts.append(f"VolRatio={vol_ratio:.2f}")
    else:
        if np.isfinite(rsi) and not (
            float(V2_SHORT_RSI_SWEET_MIN) <= rsi < float(V2_SHORT_RSI_SWEET_MAX)
        ):
            extreme_ok = False
            extreme_parts.append(
                f"RSI={rsi:.1f}（許容 {float(V2_SHORT_RSI_SWEET_MIN):.0f}〜{float(V2_SHORT_RSI_SWEET_MAX):.0f} 未満）"
            )
        if np.isfinite(vol_ratio) and vol_ratio > float(V2_SHORT_STRONG_VOLRATIO_MAX):
            extreme_ok = False
            extreme_parts.append(
                f"VolRatio={vol_ratio:.2f}（上限 {float(V2_SHORT_STRONG_VOLRATIO_MAX):.2f} 超え）"
            )

    if extreme_ok:
        extreme_detail = "RSI, VolRatio とも許容"
    else:
        extreme_detail = " / ".join(extreme_parts)

    review_needed = not (btc_mode_ok and ai_ok and risk_ok and extreme_ok)
    final_disp = "見送り候補" if review_needed else "通過候補"

    return (
        "🔎 監査\n"
        f"・BTC mode: {_audit_ok_text(btc_mode_ok)}（{btc_mode_detail}）\n"
        f"・AI: {_audit_ok_text(ai_ok)}（{ai_detail}）\n"
        f"・危険タグ: {_audit_ok_text(risk_ok)}（{risk_detail}）\n"
        f"・極端値: {_audit_ok_text(extreme_ok)}（{extreme_detail}）\n"
        "\n"
        f"🧠 人間監査: {final_disp}"
    )


def _watch_num(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _watch_dt_str(sig: Dict[str, Any]) -> str:
    dt_raw = sig.get("time", "") or sig.get("Datetime_JST", "") or sig.get("dt", "") or ""
    if hasattr(dt_raw, "strftime"):
        return dt_raw.strftime("%Y-%m-%d %H:%M:%S")
    dt_str = str(dt_raw).strip()
    return normalize_dt_str(dt_str) if dt_str else ""


def _watch_hours_csv(hours: Set[int]) -> str:
    return ",".join(str(int(x)) for x in sorted(hours))


def _watch_mode_label(mode: str) -> str:
    mode = str(mode or "").strip().upper()
    if mode == "UP":
        return "上向き"
    if mode == "DOWN":
        return "下向き"
    if mode == "RANGE":
        return "横ばい"
    return "不明"


def _watch_side_label(side: str) -> str:
    side = str(side or "").strip().upper()
    if side == "LONG":
        return "買い方向"
    if side == "SHORT":
        return "売り方向"
    return "方向不明"


def _watch_hours_human(hours_csv: str) -> str:
    text = str(hours_csv or "").strip()
    if not text:
        return ""
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if not parts:
        return ""
    return " / ".join(f"{p}時台" for p in parts)


def _watch_warning_label(w: str) -> str:
    s = str(w or "").strip()
    if s == "vol_shock":
        return "出来高が急増"
    if s == "reversal":
        return "反転の危険"
    if s == "急騰直後":
        return "急騰の直後"
    if s == "急落直後":
        return "急落の直後"
    if s.startswith("RSI帯外("):
        return s.replace("RSI帯外", "RSIが基準外")
    if s.startswith("VolR高すぎ("):
        return s.replace("VolR高すぎ", "出来高比が高すぎ")
    if s.startswith("通知block("):
        return "通知しない時間帯"
    if s == "直近重複":
        return "直近に同じ通知あり"
    return s


def _watch_scenario_text(side: str, btc_mode: str) -> str:
    side = str(side or "").strip().upper()
    btc_mode = str(btc_mode or "").strip().upper()

    if side == "LONG":
        if btc_mode == "UP":
            return "このまま上がる流れなら、買いシグナル候補"
        if btc_mode == "RANGE":
            return "上に抜けたら、買いシグナル候補"
        return "下げ中なので、買いはまだ様子見"
    else:
        if btc_mode == "DOWN":
            return "このまま下がる流れなら、売りシグナル候補"
        if btc_mode == "RANGE":
            return "横ばいが続くなら、売りシグナル候補"
        return "上げ中なので、売りはまだ様子見"


def _build_v2_watch_message(sig: Dict[str, Any]) -> Tuple[str, str, int]:
    """
    エントリー通知とは別に送る、情報通知専用メッセージ。
    戻り値:
      (message, kind, priority)

    kind:
      abnormal / warning / scenario
    priority:
      小さいほど優先送信
    """
    if str(sig.get("_notify_pass", "0")) == "1":
        return "", "", 99

    symbol = str(sig.get("symbol", "") or "").strip().upper()
    direction = str(sig.get("direction", "") or "").strip().upper()
    side = direction if direction in {"LONG", "SHORT"} else "UNKNOWN"

    dt_str = _watch_dt_str(sig)
    score = _watch_num(sig.get("total_score", sig.get("score", "")))
    ai_prob = _watch_num(sig.get("ai_prob_win", sig.get("ai_proba_used", "")))
    btc_mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    rsi = _watch_num(sig.get("rsi", ""))
    vol_ratio = _watch_num(sig.get("vol_ratio", ""))
    notify_reason = str(sig.get("_notify_reason", "") or "")
    ai_note = str(sig.get("ai_note", "") or "")

    risk_tags: List[str] = []

    if side == "LONG":
        if _is_btc_surge_after_long(sig):
            risk_tags.append("急騰直後")
        if _is_long_reversal_accident(sig):
            risk_tags.append("reversal")
        allowed_hours = _watch_hours_csv(V2_LONG_ALLOWED_HOURS)
        scenario = _watch_scenario_text(side, btc_mode)

    else:
        if _is_btc_dump_after_short(sig):
            risk_tags.append("急落直後")
        allowed_hours = _watch_hours_csv(V2_SHORT_ALLOWED_HOURS)
        scenario = _watch_scenario_text(side, btc_mode)

    if _is_vol_shock_context(sig):
        risk_tags.append("vol_shock")

    risk_tags = list(dict.fromkeys(risk_tags))

    warnings: List[str] = []
    if risk_tags:
        warnings.extend(risk_tags)

    if side == "LONG":
        if np.isfinite(rsi) and rsi > float(V2_LONG_DEF_RSI_HARD_MAX):
            warnings.append(f"RSI高すぎ({rsi:.1f})")
    else:
        if np.isfinite(rsi) and not (float(V2_SHORT_RSI_SWEET_MIN) <= rsi < float(V2_SHORT_RSI_SWEET_MAX)):
            warnings.append(f"RSI帯外({rsi:.1f})")
        if np.isfinite(vol_ratio) and vol_ratio > float(V2_SHORT_STRONG_VOLRATIO_MAX):
            warnings.append(f"VolR高すぎ({vol_ratio:.2f})")

    if "block_hour" in notify_reason:
        warnings.append(f"通知block({notify_reason})")
    if "runtime_dedup" in ai_note:
        warnings.append("直近重複")

    if risk_tags:
        title = "🚨 監視銘柄の異常検知"
        kind = "abnormal"
        priority = 0
    elif warnings:
        title = "⛔ 今は触らない方がいい"
        kind = "warning"
        priority = 1
    elif (str(sig.get("ai_pass", "0")) == "1") or (np.isfinite(score) and score >= 1.20):
        title = "🧭 売買シナリオの提案"
        kind = "scenario"
        priority = 2
    else:
        return "", "", 99

    score_disp = "" if not np.isfinite(score) else f"{score:.2f}"
    ai_disp = "" if not np.isfinite(ai_prob) else f"{ai_prob:.1%}"

    warning_disp = "なし"
    if warnings:
        warning_disp = " / ".join(_watch_warning_label(x) for x in warnings)

    mode_disp = _watch_mode_label(btc_mode)
    side_disp = _watch_side_label(side)
    allowed_hours_disp = _watch_hours_human(allowed_hours)

    lines = [
        title,
        f"💎 {symbol}（{side_disp}）" if symbol else "",
        f"🕒 {dt_str}" if dt_str else "",
        f"🟦 BTCの流れ: {mode_disp}",
        f"📈 総合点: {score_disp}" if score_disp else "",
        f"🤖 AIの目安: {ai_disp}" if ai_disp else "",
        f"📍 通常シグナル時間: {allowed_hours_disp}" if allowed_hours_disp else "",
        f"🧭 想定: {scenario}",
        f"⛔ 今は見送り候補: {warning_disp}",
    ]
    return "\n".join([x for x in lines if x]), kind, priority


def send_v2_watch_discord_alerts(final_signals: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    エントリー通知とは別に、非エントリー時の情報通知を送る。
    - 異常検知
    - 売買シナリオ提案
    - 今は触らない方がいい警告

    1 run あたり最大2件まで送る。
    戻り値:
      (sent_n, runtime_dedup_skipped_n)
    """
    global last_alert_records

    sent = 0
    runtime_dedup_skipped = 0
    now_ts = int(time.time())
    max_per_run = 2

    candidates: List[Tuple[int, float, Dict[str, Any], str, str]] = []

    for sig in final_signals:
        msg, kind, priority = _build_v2_watch_message(sig)
        if not msg:
            continue

        score = _watch_num(sig.get("total_score", sig.get("score", "")))
        sort_score = -float(score) if np.isfinite(score) else 9999.0
        candidates.append((priority, sort_score, sig, kind, msg))

    candidates.sort(key=lambda x: (x[0], x[1]))

    for _, _, sig, kind, msg in candidates:
        if sent >= max_per_run:
            break

        key = f"WATCH|{kind}|{_v2_notify_key(sig)}"

        if (not V2_NOTIFY_IGNORE_RUNTIME_DEDUP) and key in last_alert_records:
            runtime_dedup_skipped += 1
            continue

        ok, _ = send_discord_message(msg)
        if ok:
            last_alert_records[key] = now_ts
            sent += 1

    return sent, runtime_dedup_skipped


def send_v2_live_discord_alerts(notify_candidates: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    V2 の notify_pass=1 候補を Discord に送る。
    戻り値:
      (sent_n, runtime_dedup_skipped_n)

    重要:
    実際に送れたものだけ ai_note に
      notify_sent=1
      notified_at_jst=YYYY-MM-DD HH:MM:SS
    を追記する。
    """
    global last_alert_records

    send_start = time.time()

    def _dtm(label: str):
        if TIMER_LOG_ENABLE:
            print(f"[TIMER] send_v2_live_discord_alerts {label} sec={time.time() - send_start:.2f}")

    sent = 0
    runtime_dedup_skipped = 0
    now_ts = int(time.time())
    now_jst_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

    _dtm(f"enter n={len(notify_candidates)}")

    try:
        expire_before = now_ts - 86400
        last_alert_records = {
            k: v for k, v in last_alert_records.items()
            if int(v) >= expire_before
        }
    except Exception:
        pass
    _dtm("after_expire_cleanup")

    for sig in notify_candidates:
        sig["_notify_sent"] = "0"
        sig["_notified_at_jst"] = ""

        key = _v2_notify_key(sig)
        is_feature_force = _is_feature_force_pass(sig) or str(sig.get("_feature_bypass", "0")).strip() == "1"
        if (not is_feature_force) and (not V2_NOTIFY_IGNORE_RUNTIME_DEDUP) and key in last_alert_records:
            runtime_dedup_skipped += 1
            cur_note = str(sig.get("ai_note", "") or "")
            add_note = "notify_sent=0;notify_send_reason=runtime_dedup_skip"
            sig["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note
            print(
                f"[V2-NOTIFY-SKIP] runtime_dedup "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')} "
                f"key={key}"
            )
            continue

        msg = _build_v2_discord_message(sig)
        if not msg:
            cur_note = str(sig.get("ai_note", "") or "")
            add_note = "notify_sent=0;notify_send_reason=empty_message"
            sig["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note
            print(
                f"[V2-NOTIFY-SKIP] empty_message "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')}"
            )
            continue

        ok, send_reason = send_discord_message(msg)
        _dtm(f"after_send symbol={sig.get('symbol', '')} ok={int(bool(ok))}")

        if ok:
            last_alert_records[key] = now_ts
            sent += 1

            sig["_notify_sent"] = "1"
            sig["_notified_at_jst"] = now_jst_str

            cur_note = str(sig.get("ai_note", "") or "")
            add_note = f"notify_sent=1;notified_at_jst={now_jst_str};notify_send_reason=ok"
            sig["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note

            print(
                f"[V2-NOTIFY-SENT] "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')} "
                f"key={key}"
            )
        else:
            cur_note = str(sig.get("ai_note", "") or "")
            add_note = f"notify_sent=0;notify_send_reason={send_reason}"
            sig["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note

            print(
                f"[WARN] send_v2_live_discord_alerts not sent "
                f"symbol={sig.get('symbol', '')} "
                f"direction={sig.get('direction', '')} "
                f"key={key} "
                f"reason={send_reason}"
            )

    _dtm(f"before_return sent={sent} dedup={runtime_dedup_skipped}")
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
        if DEBUG_LOG_ENABLE:
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

        if DEBUG_LOG_ENABLE:
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
        "nan_cols": None,
        "nan_n_cols": None,
        "nan_cnt": None,
        "nan_filled": 0,
        "alias_renamed": None,
        "schema_version_model": "",
        "schema_version_runtime": str(V2_RUNTIME_FEATURE_SCHEMA_VERSION),
        "schema_hash_expected": "",
        "schema_hash_actual": "",
        "schema_enforced": bool(V2_MODEL_SCHEMA_ENFORCE),
    }

    try:
        if model is None:
            debug["action"] = "model_none_bypass"
            return None, True, debug

        if isinstance(model, dict):
            for k in ("model", "estimator", "clf", "pipeline", "sk_model"):
                if k in model:
                    model = model[k]
                    debug["action"] = "unwrapped_dict_model"
                    break

        if isinstance(model, (tuple, list)) and len(model) >= 1:
            model = model[0]
            debug["action"] = "unwrapped_list_model"

        if not hasattr(model, "predict_proba"):
            debug["action"] = "no_predict_proba_bypass"
            debug["error"] = f"model_type={type(model)} has no predict_proba"
            print(f"[AI] safe_predict_proba bypass: {debug['error']}")
            return None, True, debug

        if feats is None:
            feats = pd.DataFrame([{}])
        elif not isinstance(feats, pd.DataFrame):
            feats = pd.DataFrame(feats)

        feats = feats.replace([np.inf, -np.inf], np.nan)
        debug["input_n_features"] = int(feats.shape[1])

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

        expected_cols = _extract_feature_names(model)
        if expected_cols:
            expected_cols = [str(c) for c in list(expected_cols)]
            debug["expected_cols"] = list(expected_cols)
            debug["expected_n_features"] = int(len(expected_cols))

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

        model_meta = _extract_model_meta(model)
        if model_meta:
            model_schema_version = str(model_meta.get("feature_schema_version", "") or "").strip()
            model_feature_hash = str(model_meta.get("feature_hash", "") or "").strip()
            meta_cols = [str(c) for c in list(model_meta.get("feature_columns", []) or [])]

            debug["schema_version_model"] = model_schema_version
            debug["schema_hash_expected"] = model_feature_hash

            actual_cols = [str(c) for c in list(feats.columns)]
            actual_hash = _feature_columns_hash(actual_cols)
            debug["schema_hash_actual"] = actual_hash

            if meta_cols and actual_cols != meta_cols:
                debug["action"] = "schema_columns_mismatch_bypass"
                debug["error"] = f"schema columns mismatch: actual={actual_cols} expected={meta_cols}"
                if V2_MODEL_SCHEMA_ENFORCE:
                    print(f"[AI-SCHEMA-BYPASS] {debug['error']}")
                    return None, True, debug

            if model_feature_hash and actual_hash != model_feature_hash:
                debug["action"] = "schema_hash_mismatch_bypass"
                debug["error"] = f"schema hash mismatch: actual={actual_hash} expected={model_feature_hash}"
                if V2_MODEL_SCHEMA_ENFORCE:
                    print(f"[AI-SCHEMA-BYPASS] {debug['error']}")
                    return None, True, debug

            if V2_MODEL_SCHEMA_ENFORCE and model_schema_version and model_schema_version != str(V2_RUNTIME_FEATURE_SCHEMA_VERSION):
                debug["action"] = "schema_version_mismatch_bypass"
                debug["error"] = (
                    f"schema version mismatch: "
                    f"model={model_schema_version} runtime={V2_RUNTIME_FEATURE_SCHEMA_VERSION}"
                )
                print(f"[AI-SCHEMA-BYPASS] {debug['error']}")
                return None, True, debug

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
TRAIN_V2_LOOKBACK_DAYS = float(os.environ.get("TRAIN_V2_LOOKBACK_DAYS", "10"))
TRAIN_V2_LONG_MIN_SAMPLES = int(float(os.environ.get("TRAIN_V2_LONG_MIN_SAMPLES", "80")))
TRAIN_V2_SHORT_MIN_SAMPLES = int(float(os.environ.get("TRAIN_V2_SHORT_MIN_SAMPLES", "80")))

TRAIN_V2_LONG_GCS_PREFIX = os.environ.get("TRAIN_V2_LONG_GCS_PREFIX", "side_models/long")
TRAIN_V2_SHORT_GCS_PREFIX = os.environ.get("TRAIN_V2_SHORT_GCS_PREFIX", "side_models/short")

V2_RUNTIME_FEATURE_SCHEMA_VERSION = os.environ.get("V2_RUNTIME_FEATURE_SCHEMA_VERSION", "v2_serving_16f_v1").strip()
V2_TRAIN_FEATURE_SCHEMA_VERSION = os.environ.get("V2_TRAIN_FEATURE_SCHEMA_VERSION", "v2_shadow_ai_16f_v1").strip()
V2_MODEL_SCHEMA_ENFORCE = os.environ.get("V2_MODEL_SCHEMA_ENFORCE", "0").strip() == "1"

V2_RETRAIN_COMPARE_ENABLE = os.environ.get("V2_RETRAIN_COMPARE_ENABLE", "1").strip() == "1"
V2_RETRAIN_MAX_AUC_DROP = float(os.environ.get("V2_RETRAIN_MAX_AUC_DROP", "0.03"))
V2_RETRAIN_MIN_TEST_ROWS = int(float(os.environ.get("V2_RETRAIN_MIN_TEST_ROWS", "50")))
V2_RETRAIN_REQUIRE_SERVING_SCHEMA_MATCH = os.environ.get("V2_RETRAIN_REQUIRE_SERVING_SCHEMA_MATCH", "1").strip() == "1"

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


def _get_v2_serving_feature_columns() -> List[str]:
    return [
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
    ]


def _feature_columns_hash(cols: List[str]) -> str:
    norm = [str(c).strip() for c in list(cols or [])]
    raw = "||".join(norm)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_meta_uri_for_uri(uri: str) -> str:
    u = str(uri or "").strip()
    if not u.startswith("gs://"):
        return ""
    if u.endswith(".pkl"):
        return u[:-4] + ".meta.json"
    return u + ".meta.json"


def _write_json_file(path: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _load_model_meta_from_uri(uri: str) -> Dict[str, Any]:
    meta_uri = _model_meta_uri_for_uri(uri)
    if not meta_uri:
        return {}

    tmp_meta = _safe_tmp_path_for_uri(meta_uri)
    ok = _gcs_download_to(meta_uri, tmp_meta)
    if (not ok) or (not os.path.exists(tmp_meta)):
        return {}

    try:
        with open(tmp_meta, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[AI] model meta load failed uri={meta_uri} err={type(e).__name__}: {e}")
        return {}
    finally:
        try:
            os.remove(tmp_meta)
        except Exception:
            pass


def _attach_model_meta(model, meta: Dict[str, Any]):
    try:
        setattr(model, "_ocho_model_meta", dict(meta or {}))
    except Exception:
        pass
    return model


def _extract_model_meta(model) -> Dict[str, Any]:
    try:
        meta = getattr(model, "_ocho_model_meta", None)
        return dict(meta or {}) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _evaluate_binary_model(model, X_eval: pd.DataFrame, y_eval: np.ndarray) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "skipped": False,
        "reason": "",
        "accuracy": None,
        "auc": None,
        "n_rows": int(len(X_eval)) if X_eval is not None else 0,
        "expected_cols": None,
        "expected_n_features": None,
        "schema_version": "",
        "feature_hash": "",
    }

    if model is None:
        result["skipped"] = True
        result["reason"] = "model_none"
        return result

    meta = _extract_model_meta(model)
    result["schema_version"] = str(meta.get("feature_schema_version", "") or "").strip()
    result["feature_hash"] = str(meta.get("feature_hash", "") or "").strip()

    expected_cols = _extract_feature_names(model)
    if expected_cols:
        expected_cols = [str(c) for c in list(expected_cols)]
        result["expected_cols"] = list(expected_cols)
        result["expected_n_features"] = int(len(expected_cols))
        if list(X_eval.columns) != expected_cols:
            result["skipped"] = True
            result["reason"] = "schema_mismatch_eval"
            return result
    else:
        expected_n = _infer_expected_n_features(model)
        result["expected_n_features"] = int(expected_n)
        if expected_n > 0 and int(X_eval.shape[1]) != int(expected_n):
            result["skipped"] = True
            result["reason"] = "feature_count_mismatch_eval"
            return result

    try:
        y_pred = model.predict(X_eval)
        result["accuracy"] = float(accuracy_score(y_eval, y_pred)) if accuracy_score is not None else None

        auc = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)[:, 1]
            auc = float(roc_auc_score(y_eval, proba)) if roc_auc_score is not None else None
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_eval)
            auc = float(roc_auc_score(y_eval, score)) if roc_auc_score is not None else None

        result["auc"] = auc
        result["ok"] = True
        return result

    except Exception as e:
        result["skipped"] = True
        result["reason"] = f"{type(e).__name__}: {e}"
        return result


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


def _normalize_lane_name(lane: str) -> str:
    return str(lane or "").strip().lower()


def _get_lane_model_uri(side: str, lane: str = "") -> str:
    s = str(side or "").strip().upper()
    lane_u = _normalize_lane_name(lane)

    if lane_u == "long_trend":
        return LONG_TREND_MODEL_GCS_URI or LONG_MODEL_GCS_URI
    if lane_u == "long_recovery":
        return LONG_RECOVERY_MODEL_GCS_URI or LONG_MODEL_GCS_URI
    if lane_u == "short_trend":
        return SHORT_TREND_MODEL_GCS_URI or SHORT_MODEL_GCS_URI
    if lane_u == "short_retrace":
        return SHORT_RETRACE_MODEL_GCS_URI or SHORT_MODEL_GCS_URI

    if s == "LONG":
        return LONG_MODEL_GCS_URI
    if s == "SHORT":
        return SHORT_MODEL_GCS_URI
    return ""


def _get_lane_model_version(side: str, lane: str = "") -> str:
    s = str(side or "").strip().upper()
    lane_u = _normalize_lane_name(lane)

    if lane_u == "long_trend":
        return LONG_TREND_MODEL_VERSION or LONG_MODEL_VERSION
    if lane_u == "long_recovery":
        return LONG_RECOVERY_MODEL_VERSION or LONG_MODEL_VERSION
    if lane_u == "short_trend":
        return SHORT_TREND_MODEL_VERSION or SHORT_MODEL_VERSION
    if lane_u == "short_retrace":
        return SHORT_RETRACE_MODEL_VERSION or SHORT_MODEL_VERSION

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


def _get_runtime_lane_for_signal(sig: Dict[str, Any], side: str) -> str:
    s = str(side or "").strip().upper()
    if s == "LONG":
        lane_name, _ = _classify_long_lane(sig)
        return lane_name
    if s == "SHORT":
        lane_name, _ = _classify_short_lane(sig)
        return lane_name
    return ""


def get_ai_model_for_side(side: str, sym_code: str, lane: str = ""):
    """
    優先順位:
      1) lane専用モデル
      2) side専用モデル
      3) 既存の銘柄別モデル
      4) 既存の単一モデル
    """
    global ai_model_long, ai_model_short
    global AI_MODEL_VERSION_RUNTIME_LONG, AI_MODEL_SOURCE_RUNTIME_LONG
    global AI_MODEL_VERSION_RUNTIME_SHORT, AI_MODEL_SOURCE_RUNTIME_SHORT

    s = str(side or "").strip().upper()
    lane_u = _normalize_lane_name(lane)

    if not _is_side_ai_enabled(s):
        return None, "", "disabled"

    model_uri = _get_lane_model_uri(s, lane_u)
    model_ver = _get_lane_model_version(s, lane_u)

    if model_uri.startswith("gs://"):
        cached = _model_cache.get(model_uri)
        now_ts = time.time()

        if cached and (now_ts - float(cached.get("ts", 0.0)) <= float(MODEL_CACHE_TTL_SEC)):
            model = cached.get("model")
            meta = dict(cached.get("meta", {}) or {})
            model = _attach_model_meta(model, meta)

            ver = str(cached.get("version", model_ver))
            src = str(cached.get("source", model_uri))

            if s == "LONG":
                ai_model_long = model
                AI_MODEL_VERSION_RUNTIME_LONG = ver
                AI_MODEL_SOURCE_RUNTIME_LONG = src
            elif s == "SHORT":
                ai_model_short = model
                AI_MODEL_VERSION_RUNTIME_SHORT = ver
                AI_MODEL_SOURCE_RUNTIME_SHORT = src

            return model, ver, src

        tmp_path = _safe_tmp_path_for_uri(model_uri)
        ok_dl = _gcs_download_to(model_uri, tmp_path)
        if ok_dl and os.path.exists(tmp_path):
            model = _load_model_from_path(tmp_path)
            if model is not None:
                meta = _load_model_meta_from_uri(model_uri)
                model = _attach_model_meta(model, meta)

                src = model_uri
                ver = model_ver or ""

                _model_cache[model_uri] = {
                    "model": model,
                    "meta": dict(meta or {}),
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

                print(
                    f"[AI] Lane-model loaded side={s} lane={lane_u or 'side_default'} "
                    f"uri={model_uri} ver={ver}"
                )
                try:
                    fn = getattr(model, "feature_names_in_", None)
                    if fn is not None:
                        print(f"[AI] model_feature_names_in_({s}:{lane_u or 'side_default'})={list(fn)}")
                except Exception:
                    pass

                try:
                    if meta:
                        print(
                            f"[AI] model_meta({s}:{lane_u or 'side_default'}) "
                            f"schema_version={meta.get('feature_schema_version', '')} "
                            f"feature_hash={meta.get('feature_hash', '')}"
                        )
                except Exception:
                    pass

                return model, ver, src

    if ENABLE_MULTI_MODEL:
        try:
            model_for_sym, ver_sym, src_sym = get_ai_model_for_symbol(sym_code)
            if model_for_sym is not None:
                return model_for_sym, ver_sym, src_sym
        except Exception as e:
            print(
                f"[AI] get_ai_model_for_side symbol fallback failed "
                f"side={s} lane={lane_u} sym={sym_code} err={type(e).__name__}: {e}"
            )

    try:
        model = get_ai_model()
        return model, AI_MODEL_VERSION_RUNTIME, AI_MODEL_SOURCE_RUNTIME
    except Exception as e:
        print(f"[AI] get_ai_model_for_side fallback failed side={s} err={type(e).__name__}: {e}")
        return None, "", "error"



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
V2_DEBUG_RAW_SIG                = str(os.environ.get("V2_DEBUG_RAW_SIG", "0")).strip().lower() in ("1", "true", "yes", "on")
V2_DEBUG_FEATURE_PROBE          = str(os.environ.get("V2_DEBUG_FEATURE_PROBE", "0")).strip().lower() in ("1", "true", "yes", "on")
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

V2_SHORT_PUSH_ALLOWED_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_SHORT_PUSH_ALLOWED_HOURS", "0,1,17,18,19,20,21,22")).split(",")
    if str(x).strip()
}

V2_SHORT_NOTIFY_BLOCK_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_SHORT_NOTIFY_BLOCK_HOURS", "")).split(",")
    if str(x).strip()
}

V2_SHORT_BUCKET_BTC_MODES = {
    s.strip().upper()
    for s in str(os.environ.get("V2_SHORT_BUCKET_BTC_MODES", "DOWN,RANGE")).split(",")
    if s.strip()
}

V2_SHORT_PUSH_BUCKET_BTC_MODES = {
    s.strip().upper()
    for s in str(os.environ.get("V2_SHORT_PUSH_BUCKET_BTC_MODES", "DOWN,RANGE,UP")).split(",")
    if s.strip()
}
V2_SHORT_BUCKET_P1_MIN       = _env_float("V2_SHORT_BUCKET_P1_MIN", 1.3)
V2_SHORT_BUCKET_P1_MAX       = _env_float("V2_SHORT_BUCKET_P1_MAX", 3.0)
V2_SHORT_BUCKET_RSI_MIN      = _env_float("V2_SHORT_BUCKET_RSI_MIN", 30.0)
V2_SHORT_BUCKET_RSI_MAX      = _env_float("V2_SHORT_BUCKET_RSI_MAX", 50.0)
V2_SHORT_BUCKET_P2_MIN       = _env_float("V2_SHORT_BUCKET_P2_MIN", -0.30)
V2_SHORT_STRONG_P2_MIN       = _env_float("V2_SHORT_STRONG_P2_MIN", 0.0)
V2_SHORT_STRONG_VOLRATIO_MIN = _env_float("V2_SHORT_STRONG_VOLRATIO_MIN", 1.0)
V2_SHORT_STRONG_VOLRATIO_MAX = _env_float("V2_SHORT_STRONG_VOLRATIO_MAX", 1.5)
V2_SHORT_GOOD_HOUR_AI_TH     = _env_float("V2_SHORT_GOOD_HOUR_AI_TH", 0.25)
V2_SHORT_STRONG_AI_TH        = _env_float("V2_SHORT_STRONG_AI_TH", 0.25)

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

V2_LONG_NOTIFY_BLOCK_HOURS = {
    int(float(x.strip()))
    for x in str(os.environ.get("V2_LONG_NOTIFY_BLOCK_HOURS", "")).split(",")
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
V2_LONG_ALT_BUCKET_BTC_MODES = {
    s.strip().upper()
    for s in str(os.environ.get("V2_LONG_ALT_BUCKET_BTC_MODES", "UP")).split(",")
    if s.strip()
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

# --- Feature profile gate (env-driven) ---
V2_FEATURE_PROFILE_ENABLE = str(
    os.environ.get("V2_FEATURE_PROFILE_ENABLE", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_FEATURE_PROFILE_REQUIRE_MATCH = str(
    os.environ.get("V2_FEATURE_PROFILE_REQUIRE_MATCH", "0")
).strip().lower() in ("1", "true", "yes", "on")

V2_FEATURE_PROFILE_SHOW_IN_DISCORD = str(
    os.environ.get("V2_FEATURE_PROFILE_SHOW_IN_DISCORD", "1")
).strip().lower() in ("1", "true", "yes", "on")

V2_FEATURE_ALLOW_PROFILES = [
    s.strip()
    for s in str(os.environ.get("V2_FEATURE_ALLOW_PROFILES", "")).split(",")
    if s.strip()
]

V2_FEATURE_REJECT_PROFILES = [
    s.strip()
    for s in str(os.environ.get("V2_FEATURE_REJECT_PROFILES", "")).split(",")
    if s.strip()
]

V2_FEATURE_BYPASS_PROFILES = [
    s.strip()
    for s in str(os.environ.get("V2_FEATURE_BYPASS_PROFILES", "")).split(",")
    if s.strip()
]

def _get_feature_bypass_profile(sig: Dict[str, Any]) -> str:
    if not V2_FEATURE_PROFILE_ENABLE:
        return ""
    try:
        _apply_feature_profiles(sig)
    except Exception:
        return ""
    if _is_feature_force_pass(sig):
        return str(sig.get("_feature_force_profile", "") or "").strip()
    return ""

V2_SHORT_PAUSE_ENABLE = str(os.environ.get("V2_SHORT_PAUSE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_SHORT_PAUSE_LOOKBACK_HOURS = int(float(os.environ.get("V2_SHORT_PAUSE_LOOKBACK_HOURS", "24")))
V2_SHORT_PAUSE_MIN_DONE = int(float(os.environ.get("V2_SHORT_PAUSE_MIN_DONE", "8")))
V2_SHORT_PAUSE_STOP_WINRATE = _env_float("V2_SHORT_PAUSE_STOP_WINRATE", 0.40)
V2_SHORT_PAUSE_STOP_AVGPNL = _env_float("V2_SHORT_PAUSE_STOP_AVGPNL", 0.0)
V2_SHORT_PAUSE_CACHE_TTL_SEC = int(float(os.environ.get("V2_SHORT_PAUSE_CACHE_TTL_SEC", "300")))
V2_SHORT_PAUSE_AI_RESCUE_MIN = _env_float("V2_SHORT_PAUSE_AI_RESCUE_MIN", 1.1)

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
V2_SHADOW_WRITE_ENABLE     = str(os.environ.get("V2_SHADOW_WRITE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")

# --- V2 regime-aware notify policy ---
V2_REGIME_POLICY_ENABLE         = str(os.environ.get("V2_REGIME_POLICY_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
V2_REGIME_SHORT_MODE            = str(os.environ.get("V2_REGIME_SHORT_MODE", "DOWN")).strip().upper()
V2_REGIME_LONG_MODE             = str(os.environ.get("V2_REGIME_LONG_MODE", "UP")).strip().upper()
V2_REGIME_LOOKBACK_HOURS        = int(float(os.environ.get("V2_REGIME_LOOKBACK_HOURS", "72")))
V2_REGIME_MIN_DONE              = int(float(os.environ.get("V2_REGIME_MIN_DONE", "10")))
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
    # 4レーン分析用メタ（runtime記録）
    "Runtime_Lane", "Lane_Profile", "Lane_Stage", "Lane_Notify",
    "Notify_Pass", "Notify_Reason",
    "Feature_Gate", "Feature_Allow_Hits", "Feature_Reject_Hits",
    "Feature_Force_Hits", "Feature_Bypass_Profile",
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
        _btc15m_key = "BTC/USDT|15m|30"
        _btc15m_cached = _v2_fetch_cache.get(_btc15m_key)
        if _btc15m_cached and (time.time() - _btc15m_cached["ts"]) <= V2_FETCH_TTL_SEC:
            btc_ltf = _btc15m_cached["v"]
        else:
            btc_ltf = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe="15m", limit=30)
            if btc_ltf is not None:
                _v2_fetch_cache[_btc15m_key] = {"ts": time.time(), "v": btc_ltf}
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
    feature_probe_only: bool = False,
    forced_probe_direction: str = "",
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

    def _build_early_feature_probe_sig(probe_reason: str, forced_direction: str = "") -> Optional[Dict[str, Any]]:
        if not feature_probe_only:
            return None
        probe_direction = str(forced_direction or "").strip().upper()
        btc_dir_probe = str(btc_htf.get("direction", "NEUTRAL") or "").strip().upper()
        btc_str_probe = _safe_float_or_nan(btc_htf.get("strength"))
        if probe_direction not in ("LONG", "SHORT"):
            if btc_dir_probe in ("LONG", "SHORT"):
                probe_direction = btc_dir_probe
            elif btc_dir_probe == "UP":
                probe_direction = "LONG"
            elif btc_dir_probe == "DOWN":
                probe_direction = "SHORT"
            else:
                probe_direction = "LONG"
        btc_mode_probe = {
            "LONG": "Up",
            "SHORT": "Down",
            "UP": "Up",
            "DOWN": "Down",
            "NEUTRAL": "Range",
            "RANGE": "Range",
        }.get(btc_dir_probe, "Range")
        entry_probe = np.nan
        sigma_probe = np.nan
        bw_probe_val = np.nan
        bw_change_probe_val = np.nan
        vol_change_probe_val = np.nan
        rsi_probe = np.nan
        dt_probe = now_jst
        hour_probe = now_jst.hour
        time_ms_probe = np.nan
        try:
            if 'ltf_df' in dir() and isinstance(ltf_df, pd.DataFrame) and len(ltf_df) >= 2:
                close_probe = pd.to_numeric(ltf_df["Close"], errors="coerce")
                vol_probe = pd.to_numeric(ltf_df["Volume"], errors="coerce")
                pct_probe = close_probe.pct_change(fill_method=None)
                sigma_s = pct_probe.rolling(20).std()
                ma20_probe = close_probe.rolling(20).mean()
                std20_probe = close_probe.rolling(20).std()
                bb_upper_probe = ma20_probe + (2.0 * std20_probe)
                bb_lower_probe = ma20_probe - (2.0 * std20_probe)
                bw_s = ((bb_upper_probe - bb_lower_probe) / ma20_probe.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                bw_change_s = bw_s.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                vol_change_s = vol_probe.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                rsi_s = calculate_rsi(close_probe).replace([np.inf, -np.inf], np.nan)
                entry_probe = _safe_float_or_nan(close_probe.iloc[-2])
                sigma_probe = _safe_float_or_nan(sigma_s.iloc[-2]) if len(sigma_s) >= 2 else np.nan
                bw_probe_val = _safe_float_or_nan(bw_s.iloc[-2]) if len(bw_s) >= 2 else np.nan
                bw_change_probe_val = _safe_float_or_nan(bw_change_s.iloc[-2]) if len(bw_change_s) >= 2 else np.nan
                vol_change_probe_val = _safe_float_or_nan(vol_change_s.iloc[-2]) if len(vol_change_s) >= 2 else np.nan
                rsi_probe = _safe_float_or_nan(rsi_s.iloc[-2]) if len(rsi_s) >= 2 else np.nan
                try:
                    time_ms_probe = int(ltf_df["Time"].iloc[-2])
                    dt_probe = datetime.fromtimestamp(time_ms_probe / 1000, JST)
                    hour_probe = dt_probe.hour
                except Exception:
                    pass
        except Exception:
            pass
        return {
            "symbol": sym,
            "direction": probe_direction,
            "entry_price": entry_probe,
            "total_score": 0.0,
            "score": 0.0,
            "p1_score": np.nan,
            "p2_score": 0.0,
            "p3_score": 0.0,
            "sym_htf_dir": "NEUTRAL",
            "sym_htf_strength": 0.0,
            "btc_htf_dir": btc_dir_probe,
            "btc_htf_strength": btc_str_probe if np.isfinite(btc_str_probe) else 0.0,
            "ltf_aligned": False,
            "ltf_reasons": [],
            "funding_rate": np.nan,
            "fr_available": False,
            "vol_ratio": np.nan,
            "vol_confirmed": False,
            "tp": np.nan,
            "sl": np.nan,
            "tp_pct": np.nan,
            "sl_pct": np.nan,
            "atr": np.nan,
            "rsi": rsi_probe,
            "hour": hour_probe,
            "btc_mode_compat": btc_mode_probe,
            "sigma": sigma_probe,
            "BandWidth": bw_probe_val,
            "BW_Change": bw_change_probe_val,
            "Vol_Change": vol_change_probe_val,
            "btc_ret": np.nan,
            "btc_vol": np.nan,
            "time_ms": time_ms_probe,
            "dt": dt_probe,
            "long_raw_rescue": True,
            "short_raw_rescue": True,
            "v2_version": "V2-FeatureProbe",
            "note": "",
            "ai_note": f"feature_probe_only=1;probe_reason={probe_reason}",
        }
    # ---- データ取得 ----
    _ltf_key = f"{symbol}|15m|60"
    _ltf_cached = _v2_fetch_cache.get(_ltf_key)
    if _ltf_cached and (time.time() - _ltf_cached["ts"]) <= V2_FETCH_TTL_SEC:
        ltf = _ltf_cached["v"]
    else:
        ltf = fetch_ohlcv_safe(exchange, symbol, timeframe="15m", limit=60)
        if ltf is not None:
            _v2_fetch_cache[_ltf_key] = {"ts": time.time(), "v": ltf}
    ltf_df = pd.DataFrame(ltf, columns=["Time", "Open", "High", "Low", "Close", "Volume"]) if ltf else pd.DataFrame(columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    if not ltf or len(ltf) < 30:
        _v2_reject("ltf_insufficient", f"len={0 if not ltf else len(ltf)} need=30")
        early_sig = _build_early_feature_probe_sig("ltf_insufficient")
        if early_sig is not None:
            return early_sig
        return None
    try:
        closed_bar_time_ms = int(ltf_df["Time"].iloc[-2])
        closed_bar_dt = datetime.fromtimestamp(closed_bar_time_ms / 1000, JST)
        hour = closed_bar_dt.hour
    except Exception as e:
        _v2_reject("closed_bar_time_invalid", f"e={e}")
        early_sig = _build_early_feature_probe_sig("closed_bar_time_invalid")
        if early_sig is not None:
            return early_sig
        return None
    _htf_key = f"{symbol}|{V2_HTF_TIMEFRAME}|{V2_HTF_LIMIT}"
    _htf_cached = _v2_fetch_cache.get(_htf_key)
    if _htf_cached and (time.time() - _htf_cached["ts"]) <= V2_FETCH_TTL_SEC:
        htf = _htf_cached["v"]
    else:
        htf = fetch_ohlcv_safe(exchange, symbol, timeframe=V2_HTF_TIMEFRAME, limit=V2_HTF_LIMIT)
        if htf is not None:
            _v2_fetch_cache[_htf_key] = {"ts": time.time(), "v": htf}
    htf_df = pd.DataFrame(htf, columns=["Time", "Open", "High", "Low", "Close", "Volume"]) if htf else pd.DataFrame(columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    if not htf or len(htf) < V2_EMA_SLOW + 5:
        _v2_reject("htf_insufficient", f"len={0 if not htf else len(htf)} need={V2_EMA_SLOW + 5}")
        early_sig = _build_early_feature_probe_sig("htf_insufficient")
        if early_sig is not None:
            return early_sig
        return None

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
    # 通常ルートは従来どおり。
    # ただし feature_probe_only のときは LONG 側も hidden filter を越えるため、
    # 救済扱いを有効にする。
    if feature_probe_only and direction == "LONG":
        long_rescue = True
        long_rescue_reason = "feature_probe_only_force_route"
    else:
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

    forced_probe_direction = str(forced_probe_direction or "").strip().upper()

    if feature_probe_only and forced_probe_direction in ("LONG", "SHORT"):
        direction = forced_probe_direction
        if direction == "LONG":
            long_rescue = True
            long_rescue_reason = "forced_probe_direction"
        elif direction == "SHORT":
            short_rescue = True
            short_rescue_reason = "forced_probe_direction"

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
        elif feature_probe_only:
            long_ltf_score = _safe_float_or_nan(ltf_eval_long.get("score"))
            short_ltf_score = _safe_float_or_nan(ltf_eval_short.get("score"))
            if np.isfinite(long_ltf_score) and np.isfinite(short_ltf_score):
                direction = "LONG" if long_ltf_score >= short_ltf_score else "SHORT"
            elif btc_dir in ("LONG", "UP"):
                direction = "LONG"
            elif btc_dir in ("SHORT", "DOWN"):
                direction = "SHORT"
            else:
                direction = "LONG"
            _v2_reject(
                "sym_htf_neutral_force_probe",
                f"forced_direction={direction} sym_htf_dir={sym_htf.get('direction')} sym_htf_strength={sym_htf_strength}"
            )
        else:
            _v2_reject(
                "sym_htf_neutral",
                f"sym_htf_dir={sym_htf.get('direction')} sym_htf_strength={sym_htf_strength}"
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

    # ---- Pillar2: ファンディングレート ----
    _fr_key = f"funding|{symbol}"
    _fr_cached = _v2_fetch_cache.get(_fr_key)
    if _fr_cached and (time.time() - _fr_cached["ts"]) <= V2_FUNDING_FETCH_TTL_SEC:
        fr = _fr_cached["v"]
    else:
        fr = fetch_funding_rate_safe(exchange, symbol)
        if fr is not None:
            _v2_fetch_cache[_fr_key] = {"ts": time.time(), "v": fr}
    p2 = assess_funding(fr, direction)
    p2_score = p2["score"]

    # ---- Pillar3: 出来高 ----
    p3 = assess_volume(ltf_df, direction)
    p3_score = p3["score"]

    # ---- 合計 ----
    total = p1_score + p2_score + p3_score

    # ---- TP/SL ----
    entry = float(ltf_df["Close"].iloc[-2])
    tpsl = calc_atr_tp_sl(ltf_df, direction, entry, total)

    btc_mode_compat = {"LONG": "Up", "SHORT": "Down", "NEUTRAL": "Range"}.get(btc_dir, "Range")

    # クロージャ用プレースホルダー。外側ループが sig["btc_ret"]/sig["btc_vol"] を実値で上書きする。
    btc_ret = np.nan
    btc_vol = np.nan

    def _build_feature_probe_sig(probe_reason: str) -> Optional[Dict[str, Any]]:
        if not feature_probe_only:
            return None

        close_probe = pd.to_numeric(ltf_df["Close"], errors="coerce")
        vol_probe = pd.to_numeric(ltf_df["Volume"], errors="coerce")

        sigma_s = close_probe.pct_change(fill_method=None).rolling(20).std()

        ma20_probe = close_probe.rolling(20).mean()
        std20_probe = close_probe.rolling(20).std()
        bb_upper_probe = ma20_probe + (2.0 * std20_probe)
        bb_lower_probe = ma20_probe - (2.0 * std20_probe)

        bw_probe = ((bb_upper_probe - bb_lower_probe) / ma20_probe.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        bw_change_probe = bw_probe.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol_change_probe = vol_probe.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

        sigma_val = _safe_float_or_nan(sigma_s.iloc[-2]) if len(sigma_s) >= 2 else np.nan
        bw_val = _safe_float_or_nan(bw_probe.iloc[-2]) if len(bw_probe) >= 2 else np.nan
        bw_change_val = _safe_float_or_nan(bw_change_probe.iloc[-2]) if len(bw_change_probe) >= 2 else np.nan
        vol_change_val = _safe_float_or_nan(vol_change_probe.iloc[-2]) if len(vol_change_probe) >= 2 else np.nan

        return {
            "symbol": sym,
            "direction": direction,
            "entry_price": entry,
            "total_score": float(total),
            "score": float(total),
            "p1_score": float(p1_score) if np.isfinite(p1_score) else np.nan,
            "p2_score": float(p2_score) if np.isfinite(p2_score) else np.nan,
            "p3_score": float(p3_score) if np.isfinite(p3_score) else np.nan,
            "sym_htf_dir": sym_htf["direction"],
            "sym_htf_strength": sym_htf["strength"],
            "btc_htf_dir": btc_dir,
            "btc_htf_strength": btc_str,
            "ltf_aligned": bool(ltf_eval["score"] >= 0.5),
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
            "sigma": sigma_val,
            "BandWidth": bw_val,
            "BW_Change": bw_change_val,
            "Vol_Change": vol_change_val,
            "btc_ret": _safe_float_or_nan(btc_ret),
            "btc_vol": _safe_float_or_nan(btc_vol),
            "time_ms": closed_bar_time_ms,
            "dt": closed_bar_dt,
            "long_raw_rescue": bool(long_rescue),
            "short_raw_rescue": bool(short_rescue),
            "v2_version": "V2-FeatureProbe",
            "note": "",
            "ai_note": f"feature_probe_only=1;probe_reason={probe_reason}",
        }

    # side別シンボル blocklist
    long_block = set(V2_LONG_SYMBOL_BLOCKLIST or [])
    short_block = set(V2_SHORT_SYMBOL_BLOCKLIST or [])

    if direction == "LONG" and sym in long_block:
        _v2_reject("long_symbol_block", f"sym={sym} blocklist={sorted(long_block)}")
        return _build_feature_probe_sig("long_symbol_block")

    if direction == "SHORT" and sym in short_block:
        _v2_reject("short_symbol_block", f"sym={sym} blocklist={sorted(short_block)}")
        return _build_feature_probe_sig("short_symbol_block")

    if btc_dir != "NEUTRAL" and btc_dir != direction and btc_str >= V2_BTC_CONFLICT_STRONG_TH:
        if not (direction == "LONG" and long_rescue):
            _v2_reject(
                "btc_conflict_block",
                f"direction={direction} btc_dir={btc_dir} btc_str={btc_str} min={V2_BTC_CONFLICT_STRONG_TH}"
            )
            return _build_feature_probe_sig("btc_conflict_block")

    if V2_SHORT_REGIME_CONFLICT_BLOCK_ENABLE and short_regime_conflict and direction == "SHORT":
        _v2_reject(
            "regime_conflict_block",
            f"direction={direction} btc_dir={btc_dir} short_regime_conflict=True"
        )
        return _build_feature_probe_sig("short_regime_conflict_block")

    if long_regime_conflict and direction == "LONG":
        _v2_reject(
            "regime_conflict_block",
            f"direction={direction} btc_dir={btc_dir} long_regime_conflict=True"
        )
        return _build_feature_probe_sig("long_regime_conflict_block")

    if p1_score < V2_PILLAR1_MIN:
        _v2_reject(
            "pillar1_below_min",
            f"direction={direction} p1={p1_score:.4f} min={V2_PILLAR1_MIN} "
            f"sym_htf_strength={sym_htf['strength']:.4f} ltf_score={ltf_eval['score']:.4f} "
            f"btc_dir={btc_dir} btc_str={btc_str:.4f}"
        )
        return _build_feature_probe_sig("pillar1_below_min")

    if total < V2_MIN_SCORE:
        _v2_reject(
            "total_below_min",
            f"direction={direction} total={total:.4f} min={V2_MIN_SCORE} "
            f"p1={p1_score:.4f} p2={p2_score:.4f} p3={p3_score:.4f}"
        )
        return _build_feature_probe_sig("total_below_min")

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

def _is_ai_pass_true(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return False


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


def _get_short_ai_score(sig: Dict[str, Any]) -> float:
    """sig から SHORT AI スコアを取得する。見つからなければ nan を返す。"""
    cand_keys = ["ai_score", "ai_proba_used", "ai_proba", "proba_used",
                 "market_ai_score", "short_ai_score", "score_used"]
    for key in cand_keys:
        try:
            v = sig.get(key, np.nan)
            fv = float(v)
            if np.isfinite(fv):
                return float(fv)
        except Exception:
            pass
    dbg = sig.get("ai_debug", None)
    if isinstance(dbg, dict):
        for key in ("short_ai_score", "proba_used", "score_used", "chosen_score"):
            try:
                v = dbg.get(key, np.nan)
                fv = float(v)
                if np.isfinite(fv):
                    return float(fv)
            except Exception:
                pass
    return float("nan")


def _check_short_win_bucket(sig: Dict[str, Any]) -> Tuple[bool, str]:
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    hour = _sig_hour_jst(sig)

    if not _is_fr_available_flag(sig):
        return False, "short_bucket_fr_unavailable"

    if hour is None:
        return False, "short_bucket_hour_missing"

    if hour not in V2_SHORT_ALLOWED_HOURS:
        return False, (
            f"short_bucket_hour_miss "
            f"hour={hour} "
            f"allowed={sorted(list(V2_SHORT_ALLOWED_HOURS))}"
        )

    if btc_mode_compat not in V2_SHORT_BUCKET_BTC_MODES:
        return False, (
            f"short_bucket_btc_mode_miss "
            f"btc_mode={btc_mode_compat} "
            f"allowed={sorted(list(V2_SHORT_BUCKET_BTC_MODES))}"
        )

    if not np.isfinite(p1):
        return False, "short_bucket_missing_p1"

    if not np.isfinite(p2):
        return False, "short_bucket_missing_p2"

    if not np.isfinite(rsi):
        return False, "short_bucket_missing_rsi"

    if not (float(V2_SHORT_BUCKET_P1_MIN) <= float(p1) < float(V2_SHORT_BUCKET_P1_MAX)):
        return False, (
            f"short_bucket_p1_out_of_range "
            f"p1={float(p1):.4f} "
            f"min={float(V2_SHORT_BUCKET_P1_MIN):.4f} "
            f"max={float(V2_SHORT_BUCKET_P1_MAX):.4f}"
        )

    if not (float(V2_SHORT_RSI_SWEET_MIN) <= float(rsi) < float(V2_SHORT_RSI_SWEET_MAX)):
        return False, (
            f"short_bucket_rsi_out_of_range "
            f"rsi={float(rsi):.4f} "
            f"min={float(V2_SHORT_RSI_SWEET_MIN):.4f} "
            f"max={float(V2_SHORT_RSI_SWEET_MAX):.4f}"
        )

    p2_min_eff = max(0.0, float(V2_SHORT_BUCKET_P2_MIN))
    if float(p2) < p2_min_eff:
        return False, (
            f"short_bucket_p2_below_min "
            f"p2={float(p2):.4f} "
            f"min={float(p2_min_eff):.4f}"
        )

    ai_score = _get_short_ai_score(sig)
    if np.isfinite(ai_score) and float(ai_score) < float(SHORT_AI_TH):
        return False, (
            f"short_bucket_ai_below_min "
            f"ai={float(ai_score):.6f} "
            f"min={float(SHORT_AI_TH):.6f}"
        )

    return True, "short_bucket_pass"


def _is_short_win_bucket(sig: Dict[str, Any]) -> bool:
    ok, _ = _check_short_win_bucket(sig)
    return ok


def _is_short_strong_hour_bucket(sig: Dict[str, Any]) -> bool:
    """
    SHORT_PUSH / pause bypass 用の「強いSHORT」判定。
    通常の _is_short_win_bucket(sig) に依存させず、
    push 専用の hour / btc_mode で判定する。
    """
    p1 = _safe_float_or_nan(sig.get("p1_score"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    rsi = _safe_float_or_nan(sig.get("rsi"))
    volratio = _safe_float_or_nan(sig.get("vol_ratio"))
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    hour = _sig_hour_jst(sig)

    if not _is_fr_available_flag(sig):
        return False

    if hour is None:
        return False
    if hour not in V2_SHORT_PUSH_ALLOWED_HOURS:
        return False

    if btc_mode_compat not in V2_SHORT_PUSH_BUCKET_BTC_MODES:
        return False

    if not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(rsi) or not np.isfinite(volratio):
        return False

    if not (float(V2_SHORT_BUCKET_P1_MIN) <= float(p1) < float(V2_SHORT_BUCKET_P1_MAX)):
        return False

    # RSI sweet band (広帯を外し、勝率の高い 40-55 帯のみ通す)
    if not (float(V2_SHORT_RSI_SWEET_MIN) <= float(rsi) < float(V2_SHORT_RSI_SWEET_MAX)):
        return False

    if float(p2) < float(V2_SHORT_STRONG_P2_MIN):
        return False

    # volratio: 下限 + 上限 (過熱ゾーンも弾く)
    if float(volratio) < float(V2_SHORT_STRONG_VOLRATIO_MIN):
        return False
    if float(volratio) > float(V2_SHORT_STRONG_VOLRATIO_MAX):
        return False

    # AI スコア hard gate: max(SHORT_AI_TH, V2_SHORT_STRONG_AI_TH) (値がない場合は通す)
    ai_score = _get_short_ai_score(sig)
    if np.isfinite(ai_score):
        ai_th = max(float(SHORT_AI_TH), float(V2_SHORT_STRONG_AI_TH))
        if float(ai_score) < ai_th:
            return False

    return True


def _classify_long_lane(sig: Dict[str, Any]) -> Tuple[str, str]:
    """
    LONG を 2レーンに分ける。
    - long_trend    : 順張りLONG（今の strong）
    - long_recovery : 回復LONG（今の alt）
    """
    trend_ok, trend_reason = _check_long_strong_bucket(sig)
    if trend_ok:
        return "long_trend", str(trend_reason)

    recovery_ok = _is_long_alt_win_bucket(sig)
    if recovery_ok:
        return "long_recovery", "long_recovery_pass"

    return "", str(trend_reason)


def _classify_short_lane(sig: Dict[str, Any]) -> Tuple[str, str]:
    """
    SHORT を 2レーンに分ける。
    - short_retrace : 戻り売りSHORT（push/strong hour — 先に判定）
    - short_trend   : 下げ継続SHORT（今の core）
    """
    retrace_ok = _is_short_strong_hour_bucket(sig)
    if retrace_ok:
        return "short_retrace", "short_retrace_pass"

    trend_ok, trend_reason = _check_short_win_bucket(sig)
    if trend_ok:
        return "short_trend", str(trend_reason)

    return "", str(trend_reason)


PROFILE_KEYMAP = {
    "p1": "p1_score",
    "p2": "p2_score",
    "p3": "p3_score",
    "rsi": "rsi",
    "vol": "vol_ratio",
    "ai": "ai_prob_win",
    "volconfirmed": "vol_confirmed",
}


def _profile_raw_value(feats, name):
    key = PROFILE_KEYMAP[name]
    return feats.get(key)


def _profile_float(feats, name):
    value = _profile_raw_value(feats, name)
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value):
        return None
    return value


def _profile_bool(feats, name):
    value = _profile_raw_value(feats, name)
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "TRUE"):
        return True
    if value in (0, "0", "false", "False", "FALSE"):
        return False
    return None


def _peq(value, target, tol=1e-9):
    if value is None:
        return False
    return abs(value - target) <= tol


def _plt(value, target):
    return value is not None and value < target


def _ple(value, target):
    return value is not None and value <= target


def _pge(value, target):
    return value is not None and value >= target


def _pband(value, low=None, high=None):
    if value is None:
        return False
    if low is not None and value < low:
        return False
    if high is not None and value >= high:
        return False
    return True


def _pis_missing(value):
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def _pcsv_env(name, default_value):
    raw = os.getenv(name, default_value).strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


LANE_NOTIFY_STAGES = {
    "long_trend": _pcsv_env("LANE_LONG_TREND_NOTIFY_STAGES", "stable,mid"),
    "long_recovery": _pcsv_env("LANE_LONG_RECOVERY_NOTIFY_STAGES", ""),
    "short_trend": _pcsv_env("LANE_SHORT_TREND_NOTIFY_STAGES", "stable,mid"),
    "short_retrace": _pcsv_env("LANE_SHORT_RETRACE_NOTIFY_STAGES", ""),
}


def _notify_stage_enabled(lane, stage):
    return stage in LANE_NOTIFY_STAGES.get(lane, set())


def _classify_lane_profile(side: str, lane: str, feats: Dict[str, Any]) -> Dict[str, Any]:
    p1 = _profile_float(feats, "p1")
    p2 = _profile_float(feats, "p2")
    p3 = _profile_float(feats, "p3")
    rsi = _profile_float(feats, "rsi")
    vol = _profile_float(feats, "vol")
    ai_raw = _profile_raw_value(feats, "ai")
    ai = _profile_float(feats, "ai")
    volconfirmed = _profile_bool(feats, "volconfirmed")

    if side == "LONG" and lane == "long_trend":
        if _pband(p1, 2.0, 2.5) and _peq(p2, 0.0) and _ple(p3, -0.1) and _pband(ai, 0.45, 0.8):
            return {"profile": "lt_stable_p1_20_25_p2_0_p3_le_m01_ai_045_08", "stage": "stable", "notify": _notify_stage_enabled("long_trend", "stable")}
        if _peq(p2, 0.0) and _pband(rsi, 40.0, 45.0) and _ple(vol, 0.5):
            return {"profile": "lt_stable_p2_0_rsi_40_45_vol_le_05", "stage": "stable", "notify": _notify_stage_enabled("long_trend", "stable")}
        if _pband(p1, 1.5, 2.0) and _ple(vol, 0.5) and _ple(p3, -0.1):
            return {"profile": "lt_mid_p1_15_20_vol_le_05_p3_le_m01", "stage": "mid", "notify": _notify_stage_enabled("long_trend", "mid")}
        if _pband(p1, 1.5, 2.0) and _pband(rsi, 40.0, 45.0) and _pband(vol, 0.5, 1.0):
            return {"profile": "lt_reject_p1_15_20_rsi_40_45_vol_05_10", "stage": "reject", "notify": False}
        if _pband(p1, 2.0, 2.5) and _pband(ai, 0.2, 0.45):
            return {"profile": "lt_reject_p1_20_25_ai_02_045", "stage": "reject", "notify": False}
        return {"profile": "lt_other", "stage": "research", "notify": False}

    if side == "LONG" and lane == "long_recovery":
        if _pband(p1, 2.0, 2.5) and _peq(p2, 0.0) and _ple(p3, -0.1) and _pband(rsi, 55.0, 60.0):
            return {"profile": "lr_research_p1_20_25_p2_0_p3_le_m01_rsi_55_60", "stage": "research", "notify": False}
        if _pband(p1, 2.0, 2.5) and _ple(p3, -0.1) and _pband(rsi, 55.0, 60.0):
            return {"profile": "lr_research_p1_20_25_p3_le_m01_rsi_55_60", "stage": "research", "notify": False}
        if _pge(p1, 2.5) and _ple(vol, 0.5):
            return {"profile": "lr_reject_p1_ge_25_vol_le_05", "stage": "reject", "notify": False}
        return {"profile": "lr_other", "stage": "research", "notify": False}

    if side == "SHORT" and lane == "short_trend":
        if _peq(p2, 0.0) and _plt(rsi, 40.0):
            return {"profile": "st_stable_p2_0_rsi_lt_40", "stage": "stable", "notify": _notify_stage_enabled("short_trend", "stable")}
        if _pband(p1, 1.5, 2.0) and _pis_missing(ai_raw) and _pband(vol, 0.5, 1.0):
            return {"profile": "st_stable_p1_15_20_ai_nan_vol_05_10", "stage": "stable", "notify": _notify_stage_enabled("short_trend", "stable")}
        if _pband(p1, 1.5, 2.0) and _pge(p3, 0.1) and _pband(vol, 1.0, 2.0) and volconfirmed is True:
            return {"profile": "st_mid_p1_15_20_p3_ge_01_vol_10_20_volconfirmed_1", "stage": "mid", "notify": _notify_stage_enabled("short_trend", "mid")}
        if _pband(rsi, 45.0, 50.0) and volconfirmed is False:
            return {"profile": "st_reject_rsi_45_50_volconfirmed_0", "stage": "reject", "notify": False}
        if _peq(p2, 0.0) and _pband(rsi, 45.0, 50.0):
            return {"profile": "st_reject_p2_0_rsi_45_50", "stage": "reject", "notify": False}
        return {"profile": "st_other", "stage": "research", "notify": False}

    if side == "SHORT" and lane == "short_retrace":
        if _pband(vol, 1.0, 2.0) and _pband(p1, 1.5, 2.0) and _plt(rsi, 45.0) and _pis_missing(ai_raw):
            return {"profile": "sr_research_vol_10_20_p1_15_20_rsi_lt_45_ai_nan", "stage": "research", "notify": False}
        return {"profile": "sr_other", "stage": "research", "notify": False}

    return {"profile": "unknown_lane", "stage": "reject", "notify": False}


def _classify_train_lane_from_row(row: Dict[str, Any]) -> str:
    """
    v2_shadow_ai の学習用 row を 4レーン分類する。
    優先順位:
      1) runtime の lane 判定をそのまま使う
      2) runtime で lane が取れない場合は、学習用フォールバックで分類する
    """
    def _pick(*keys, default=""):
        for key in keys:
            try:
                v = row.get(key, None)
            except Exception:
                v = None

            if v is None:
                continue

            if isinstance(v, str):
                if v.strip() == "":
                    continue
                return v

            return v

        return default

    def _to_num(v):
        try:
            if v is None:
                return np.nan
            if isinstance(v, str) and v.strip() == "":
                return np.nan
            return float(v)
        except Exception:
            return np.nan

    def _to_bool01(v):
        try:
            return _normalize_bool01(v)
        except Exception:
            s = str(v or "").strip().lower()
            if s in ("1", "true", "yes", "on"):
                return 1.0
            if s in ("0", "false", "no", "off"):
                return 0.0
            return np.nan

    side = str(_pick("Direction", "direction")).strip().upper()
    if side not in ("LONG", "SHORT"):
        return ""

    sig: Dict[str, Any] = {
        "symbol": str(_pick("Symbol", "symbol")).strip(),
        "direction": side,
        "time": _pick("Datetime_JST", "Time", "time"),
        "datetime_jst": _pick("Datetime_JST", "datetime_jst"),
        "hour_jst": _pick("Hour_JST", "hour_jst"),
        "total_score": _pick("TotalScore", "total_score"),
        "p1_score": _pick("P1_TrendScore", "p1_score"),
        "p2_score": _pick("P2_FundingScore", "p2_score"),
        "p3_score": _pick("P3_VolumeScore", "p3_score"),
        "sym_htf_strength": _pick("SymHTF_Strength", "sym_htf_strength"),
        "btc_htf_strength": _pick("BTC_HTF_Strength", "btc_htf_strength"),
        "vol_ratio": _pick("VolRatio", "vol_ratio"),
        "atr": _pick("ATR", "atr"),
        "rsi": _pick("RSI", "rsi"),
        "ltf_aligned": _pick("LTF_Aligned", "ltf_aligned"),
        "fr_available": _pick("FR_Available", "fr_available"),
        "vol_confirmed": _pick("VolConfirmed", "vol_confirmed"),
        "btc_mode_compat": str(_pick("BTC_Mode_Compat", "btc_mode_compat")).strip().upper(),
        "long_ai_score": _pick(
            "long_ai_score",
            "LONG_AI_SCORE",
            "ai_proba_used",
            "AI_Proba_Used",
            "proba_used",
            "Proba_Used",
            "score_used",
            "Score_Used",
            default=np.nan,
        ),
        "short_ai_score": _pick(
            "short_ai_score",
            "SHORT_AI_SCORE",
            "ai_proba_used",
            "AI_Proba_Used",
            "proba_used",
            "Proba_Used",
            "score_used",
            "Score_Used",
            default=np.nan,
        ),
        "ai_proba_used": _pick(
            "ai_proba_used",
            "AI_Proba_Used",
            "proba_used",
            "Proba_Used",
            "score_used",
            "Score_Used",
            default=np.nan,
        ),
        "proba_used": _pick(
            "proba_used",
            "Proba_Used",
            "ai_proba_used",
            "AI_Proba_Used",
            "score_used",
            "Score_Used",
            default=np.nan,
        ),
        "score_used": _pick(
            "score_used",
            "Score_Used",
            "ai_proba_used",
            "AI_Proba_Used",
            "proba_used",
            "Proba_Used",
            default=np.nan,
        ),
    }

    raw_ai_debug = _pick("ai_debug", "AI_Debug", "market_ai_debug", default=None)
    if raw_ai_debug not in (None, ""):
        if isinstance(raw_ai_debug, dict):
            sig["ai_debug"] = raw_ai_debug
        else:
            try:
                sig["ai_debug"] = json.loads(str(raw_ai_debug))
            except Exception:
                pass

    try:
        if side == "LONG":
            lane, _ = _classify_long_lane(sig)
            if str(lane or "").strip():
                return str(lane).strip()

        if side == "SHORT":
            lane, _ = _classify_short_lane(sig)
            if str(lane or "").strip():
                return str(lane).strip()
    except Exception:
        pass

    btc_mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    p1 = _to_num(sig.get("p1_score"))
    p2 = _to_num(sig.get("p2_score"))
    rsi = _to_num(sig.get("rsi"))
    volratio = _to_num(sig.get("vol_ratio"))
    hour_raw = _to_num(sig.get("hour_jst"))
    fr_available = _to_bool01(sig.get("fr_available"))

    hour = None
    if np.isfinite(hour_raw):
        hour = int(hour_raw)

    if side == "LONG":
        strong_hour_ok = (hour is not None) and (hour in V2_LONG_STRONG_ALLOWED_HOURS)
        strong_mode_ok = btc_mode in V2_LONG_STRONG_BUCKET_BTC_MODES
        strong_p1_ok = np.isfinite(p1) and (float(V2_LONG_STRONG_BUCKET_P1_MIN) <= float(p1) < float(V2_LONG_STRONG_BUCKET_P1_MAX))
        strong_rsi_ok = np.isfinite(rsi) and (float(V2_LONG_STRONG_BUCKET_RSI_MIN) <= float(rsi) < float(V2_LONG_STRONG_BUCKET_RSI_MAX))
        strong_p2_ok = np.isfinite(p2) and (float(p2) >= float(V2_LONG_STRONG_BUCKET_P2_MIN))
        strong_fr_ok = (not np.isfinite(fr_available)) or (float(fr_available) == 1.0)

        if strong_hour_ok and strong_mode_ok and strong_p1_ok and strong_rsi_ok and strong_p2_ok and strong_fr_ok:
            return "long_trend"

        alt_hour_ok = (hour is not None) and (hour in V2_LONG_ALT_ALLOWED_HOURS)
        alt_mode_ok = btc_mode in V2_LONG_ALT_BUCKET_BTC_MODES
        alt_p1_ok = np.isfinite(p1) and (float(V2_LONG_ALT_BUCKET_P1_MIN) <= float(p1) < float(V2_LONG_ALT_BUCKET_P1_MAX))
        alt_rsi_ok = np.isfinite(rsi) and (float(V2_LONG_ALT_BUCKET_RSI_MIN) <= float(rsi) < float(V2_LONG_ALT_BUCKET_RSI_MAX))
        alt_p2_ok = np.isfinite(p2) and (float(p2) >= float(V2_LONG_ALT_BUCKET_P2_MIN))
        alt_fr_ok = (not np.isfinite(fr_available)) or (float(fr_available) == 1.0)

        if alt_hour_ok and alt_mode_ok and alt_p1_ok and alt_rsi_ok and alt_p2_ok and alt_fr_ok:
            return "long_recovery"

        return ""

    if side == "SHORT":
        strong_hour_ok = (hour is not None) and (hour in V2_SHORT_STRONG_HOURS)
        strong_p1_ok = np.isfinite(p1) and (float(V2_SHORT_BUCKET_P1_MIN) <= float(p1) < float(V2_SHORT_BUCKET_P1_MAX))
        strong_rsi_ok = np.isfinite(rsi) and (float(V2_SHORT_RSI_SWEET_MIN) <= float(rsi) < float(V2_SHORT_RSI_SWEET_MAX))
        strong_p2_ok = np.isfinite(p2) and (float(p2) >= float(V2_SHORT_STRONG_P2_MIN))
        strong_vol_ok = (
            np.isfinite(volratio)
            and (float(volratio) >= float(V2_SHORT_STRONG_VOLRATIO_MIN))
            and (float(volratio) <= float(V2_SHORT_STRONG_VOLRATIO_MAX))
        )

        if strong_hour_ok and strong_p1_ok and strong_rsi_ok and strong_p2_ok and strong_vol_ok:
            return "short_retrace"

        return "short_trend"

    return ""


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
    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()

    ai_score = np.nan
    for key in ("long_ai_score", "ai_proba_used", "ai_proba", "proba_used", "score_used"):
        try:
            v = sig.get(key, np.nan)
            fv = float(v)
            if np.isfinite(fv):
                ai_score = float(fv)
                break
        except Exception:
            pass

    sig["long_alt_shadow_candidate"] = False

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

    if np.isfinite(ai_score) and float(ai_score) < float(V2_LONG_ALT_AI_TH):
        return False

    # RANGE はまだ live で通さない。
    # 条件に合うものだけ shadow 候補として印を付ける。
    if btc_mode_compat == "RANGE":
        sig["long_alt_shadow_candidate"] = True
        return False

    if btc_mode_compat not in V2_LONG_ALT_BUCKET_BTC_MODES:
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
    if _is_feature_force_pass(sig):
        return True, "feature_force_pass_short"

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

    short_core_ok, short_core_reason = _check_short_win_bucket(sig)
    short_push_ok = _is_short_strong_hour_bucket(sig)

    if not short_core_ok:
        if V2_SHORT_PAUSE_ALLOW_STRONG_BUCKET and short_push_ok:
            pass
        else:
            return False, (
                f"{short_core_reason} "
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
      - 急落直後 / ボラ急拡大は penalty で扱う
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

    dump_penalty = _short_incident_dump_penalty(sig)
    vol_shock_penalty = _short_incident_vol_shock_penalty(sig)

    sig["short_dump_penalty"] = float(dump_penalty)
    sig["short_vol_shock_penalty"] = float(vol_shock_penalty)

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
        (btc_over_penalty) -
        (dump_penalty) -
        (vol_shock_penalty)
    )
    return float(qs)


def rank_and_select(short_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage C:
    SHORT候補を ai_prob_win 降順に並べる。
    - SHORT AI が入っていれば ai_prob_win は AI確率
    - bypass/fallback のときは RULE fallback の値
    - SHORT brake は従来通り適用
    - SHORT repeat は hard block ではなく rank penalty で扱う
    """
    if not short_signals:
        return []

    for sig in short_signals:
        sig["rule_qs"] = calc_quality_score(sig)

    def _short_rank_score(sig: Dict[str, Any]) -> float:
        repeat_penalty, repeat_recent_n, repeat_last_dt = _short_repeat_rank_penalty(sig)

        sig["short_repeat_recent_n"] = int(repeat_recent_n)
        sig["short_repeat_last_dt"] = str(repeat_last_dt or "")
        sig["short_repeat_penalty"] = float(repeat_penalty)

        base_rule_qs = float(sig.get("rule_qs", 0.0) or 0.0)
        return float(base_rule_qs - repeat_penalty)

    ranked = sorted(
        short_signals,
        key=lambda x: (
            float(x.get("ai_prob_win", 0.0) or 0.0),
            _short_rank_score(x),
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
                f"top_n=ALL;selected=true;rank_disabled=true;brake_mode={brake_mode};"
                f"repeat_recent_n={int(sig.get('short_repeat_recent_n', 0) or 0)};"
                f"repeat_penalty={float(sig.get('short_repeat_penalty', 0.0) or 0.0):.6f};"
                f"repeat_last_dt={str(sig.get('short_repeat_last_dt', '') or '')}"
            )
        return ranked

    slot_total = len(ranked)

    for rank_idx, sig in enumerate(ranked, start=1):
        current_note = str(sig.get("ai_note", "") or "")
        rank_score = _short_rank_score(sig)

        if brake_mode == "STOP":
            sig["ai_pass"] = "0"
            if rank_idx <= probe_top_n:
                sig["ai_band"] = "BRAKE_STOP_OBSERVE"
                sig["ai_note"] = (
                    f"{current_note};rank={rank_idx};slot_total={slot_total};"
                    f"top_n={probe_top_n};selected=false;brake_mode=STOP;observe_only=true;"
                    f"rank_score={rank_score:.6f};"
                    f"repeat_recent_n={int(sig.get('short_repeat_recent_n', 0) or 0)};"
                    f"repeat_penalty={float(sig.get('short_repeat_penalty', 0.0) or 0.0):.6f};"
                    f"repeat_last_dt={str(sig.get('short_repeat_last_dt', '') or '')}"
                )
            else:
                sig["ai_band"] = "BRAKE_STOP_DROP"
                sig["ai_note"] = (
                    f"{current_note};rank={rank_idx};slot_total={slot_total};"
                    f"top_n={probe_top_n};selected=false;brake_mode=STOP;observe_only=false;"
                    f"rank_score={rank_score:.6f};"
                    f"repeat_recent_n={int(sig.get('short_repeat_recent_n', 0) or 0)};"
                    f"repeat_penalty={float(sig.get('short_repeat_penalty', 0.0) or 0.0):.6f};"
                    f"repeat_last_dt={str(sig.get('short_repeat_last_dt', '') or '')}"
                )
            continue

        if rank_idx <= effective_top_n:
            sig["ai_pass"] = "1"
            sig["ai_band"] = _v2_rank_selected_band(sig)
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n={effective_top_n};selected=true;brake_mode={brake_mode};"
                f"rank_score={rank_score:.6f};"
                f"repeat_recent_n={int(sig.get('short_repeat_recent_n', 0) or 0)};"
                f"repeat_penalty={float(sig.get('short_repeat_penalty', 0.0) or 0.0):.6f};"
                f"repeat_last_dt={str(sig.get('short_repeat_last_dt', '') or '')}"
            )
        else:
            sig["ai_pass"] = "0"
            sig["ai_band"] = _v2_rank_reject_band(sig)
            sig["ai_note"] = (
                f"{current_note};rank={rank_idx};slot_total={slot_total};"
                f"top_n={effective_top_n};selected=false;reason=rank_exceeded;brake_mode={brake_mode};"
                f"rank_score={rank_score:.6f};"
                f"repeat_recent_n={int(sig.get('short_repeat_recent_n', 0) or 0)};"
                f"repeat_penalty={float(sig.get('short_repeat_penalty', 0.0) or 0.0):.6f};"
                f"repeat_last_dt={str(sig.get('short_repeat_last_dt', '') or '')}"
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
      - 急反転事故だけは hard reject にする
    """
    direction = str(sig.get("direction", "")).strip().upper()
    if direction != "LONG":
        return True, "not_long"
    if _is_feature_force_pass(sig):
        return True, "feature_force_pass_long"

    if not V2_LONG_DEF_ENABLE:
        return True, "long_def_disabled"

    btc_mode_compat = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    long_raw_rescue = str(sig.get("long_raw_rescue", "")).strip().lower() in ("1", "true", "yes", "on")
    sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()

    now_jst = datetime.now(JST)

    # 当日LONG緊急ブレーキ
    fast_brake_state = _resolve_v2_long_fast_brake_runtime(now_jst)
    if bool(fast_brake_state.get("active", False)):
        return False, (
            f"long_fast_brake "
            f"{str(fast_brake_state.get('reason_text', '') or '')}"
        )

    # 同一銘柄LONGの連打は、ここでは reject せず rank penalty 側で扱う
    if V2_LONG_REPEAT_BLOCK_ENABLE and sym:
        pref_recent_n = sig.get("_prefetched_long_repeat_recent_n", None)
        pref_last_dt = sig.get("_prefetched_long_repeat_last_dt", None)

        if pref_recent_n is None:
            repeat_state = _get_v2_long_recent_notified_window_state(now_jst)
            recent_n = int(((repeat_state.get("counts", {}) or {}).get(sym, 0)) or 0)
            last_dt = str(((repeat_state.get("last_dt", {}) or {}).get(sym, "")) or "")
        else:
            recent_n = int(pref_recent_n or 0)
            last_dt = str(pref_last_dt or "")

        sig["long_repeat_recent_n"] = recent_n
        sig["long_repeat_last_dt"] = last_dt

    # BTC急騰中 / 急騰直後の追いかけLONGだけ止める
    # ここでは RANGE では発火させない。UP のときだけ見る。
    btc_surge_pause_active = bool(_btc_long_pause_ctx.get("active", False))
    btc_surge_pause_reason = str(_btc_long_pause_ctx.get("reason", "") or "")

    if btc_surge_pause_active and btc_mode_compat == "UP":
        pause_rsi = _safe_float_or_nan(sig.get("rsi"))
        pause_p1 = _safe_float_or_nan(sig.get("p1_score"))

        hit_rsi = bool(
            np.isfinite(pause_rsi) and
            float(pause_rsi) >= float(BTC_SURGE_LONG_PAUSE_RSI_MIN)
        )
        hit_p1 = bool(
            np.isfinite(pause_p1) and
            float(pause_p1) >= float(BTC_SURGE_LONG_PAUSE_P1_MIN)
        )

        if hit_rsi or hit_p1:
            return False, (
                f"btc_surge_long_pause "
                f"btc_mode={btc_mode_compat} "
                f"sym={sym} "
                f"rsi={_fmt_sig_num_or_blank(sig, 'rsi')} "
                f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')} "
                f"hit_rsi={int(hit_rsi)} "
                f"hit_p1={int(hit_p1)} "
                f"reason={btc_surge_pause_reason}"
            )

    # 事故分析で最も悪かった LONG 急反転は hard reject
    if V2_LONG_REVERSAL_HARD_REJECT_ENABLE and _is_long_reversal_accident(sig):
        btc_1h_change = _incident_sig_float(sig, "btc_1h_change", "BTC_1h_Change")
        btc_ret = _incident_sig_float(sig, "btc_ret", "BTC_Ret")
        return False, (
            f"long_reversal_hard_reject "
            f"btc_mode={btc_mode_compat} "
            f"sym={sym} "
            f"btc_1h_change={'' if not np.isfinite(btc_1h_change) else format(float(btc_1h_change), '.6f')} "
            f"btc_ret={'' if not np.isfinite(btc_ret) else format(float(btc_ret), '.6f')}"
        )

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
    追加方針:
      - BTC急騰直後は penalty
      - ボラ急拡大は penalty
      - LONG repeat は既存 rank penalty を使う
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

    surge_penalty = _long_incident_surge_penalty(sig)
    vol_shock_penalty = _long_incident_vol_shock_penalty(sig)

    sig["long_surge_penalty"] = float(surge_penalty)
    sig["long_vol_shock_penalty"] = float(vol_shock_penalty)

    qs = (
        (p1 * V2_LONG_QS_W_P1) +
        (rsi_bonus * V2_LONG_QS_W_RSI) +
        (p3_bonus * V2_LONG_QS_W_P3) +
        (btc_bonus * V2_LONG_QS_W_BTC) +
        (mode_bonus * V2_LONG_QS_W_MODE) +
        (other_score * V2_LONG_QS_W_OTHER) +
        (bad_symbol_flag * V2_LONG_QS_W_BAD_SYMBOL) -
        (surge_penalty) -
        (vol_shock_penalty)
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

    shadow_flag = sig.get("long_alt_shadow_candidate", False)
    shadow_on = bool(shadow_flag) or str(shadow_flag).strip().lower() in ("1", "true", "yes", "on")
    if shadow_on:
        tags.append("LONG_ALT_SHADOW_RANGE")

    return tags


def _long_notify_gate(sig: Dict[str, Any]) -> Tuple[bool, str]:
    if _is_feature_force_pass(sig) or str(sig.get("_feature_bypass", "0")).strip() == "1":
        return True, "feature_force_pass_long_notify"
    if not V2_LONG_NOTIFY_GATE_ENABLE:
        return True, "notify_gate_disabled"

    mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
    sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()
    rsi = _safe_float_or_nan(sig.get("rsi"))
    p2 = _safe_float_or_nan(sig.get("p2_score"))
    p3 = _safe_float_or_nan(sig.get("p3_score"))
    research_tags = set(_get_long_research_tags(sig))

    block_syms = {s.strip().upper() for s in V2_LONG_NOTIFY_GATE_BLOCK_SYMBOLS if str(s).strip()}
    if sym in block_syms:
        return False, f"notify_gate_symbol_block sym={sym}"

    if "LONG_DANGER_RSI70" in research_tags or "LONG_DANGER_RANGE" in research_tags:
        return False, f"notify_gate_danger_tag tags={sorted(research_tags)}"

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
    本番の V2 LONG/SHORT AI 推論用特徴量を、train 側と同じ16列で作る。

    期待列:
      TotalScore
      P1_TrendScore
      P2_FundingScore
      P3_VolumeScore
      SymHTF_Strength
      BTC_HTF_Strength
      VolRatio
      ATR
      RSI
      Hour_JST
      LTF_Aligned
      FR_Available
      VolConfirmed
      BTC_Mode_UP
      BTC_Mode_RANGE
      BTC_Mode_DOWN

    重要:
      - 固定値埋めはしない
      - 欠損があれば None を返し、上位で bypass させる
      - train 側 _build_training_matrix_from_v2_shadow_ai() と同じ列順にそろえる
    """

    total_score_val = _safe_float_or_nan(
        sig.get("total_score", sig.get("score"))
    )
    p1_score_val = _safe_float_or_nan(
        sig.get("p1_score", sig.get("P1_TrendScore"))
    )
    p2_score_val = _safe_float_or_nan(
        sig.get("p2_score", sig.get("P2_FundingScore"))
    )
    p3_score_val = _safe_float_or_nan(
        sig.get("p3_score", sig.get("P3_VolumeScore"))
    )

    sym_htf_strength_val = _safe_float_or_nan(
        sig.get("sym_htf_strength", sig.get("SymHTF_Strength"))
    )
    btc_htf_strength_val = _safe_float_or_nan(
        sig.get("btc_htf_strength", sig.get("BTC_HTF_Strength"))
    )

    vol_ratio_val = _safe_float_or_nan(
        sig.get("vol_ratio", sig.get("VolRatio"))
    )
    atr_val = _safe_float_or_nan(
        sig.get("atr", sig.get("ATR"))
    )
    rsi_val = _safe_float_or_nan(
        sig.get("rsi", sig.get("RSI"))
    )
    hour_val = _safe_float_or_nan(
        sig.get("hour", sig.get("Hour_JST"))
    )

    ltf_aligned_val = _v2_bool01(
        sig.get("ltf_aligned", sig.get("LTF_Aligned"))
    )
    fr_available_val = _v2_bool01(
        sig.get("fr_available", sig.get("FR_Available"))
    )
    vol_confirmed_val = _v2_bool01(
        sig.get("vol_confirmed", sig.get("VolConfirmed"))
    )

    btc_mode_raw = sig.get("btc_mode_compat", sig.get("BTC_Mode_Compat"))
    btc_mode_up, btc_mode_range, btc_mode_down = _v2_mode_onehot(btc_mode_raw)

    feats = pd.DataFrame([{
        "TotalScore": total_score_val,
        "P1_TrendScore": p1_score_val,
        "P2_FundingScore": p2_score_val,
        "P3_VolumeScore": p3_score_val,
        "SymHTF_Strength": sym_htf_strength_val,
        "BTC_HTF_Strength": btc_htf_strength_val,
        "VolRatio": vol_ratio_val,
        "ATR": atr_val,
        "RSI": rsi_val,
        "Hour_JST": hour_val,
        "LTF_Aligned": ltf_aligned_val,
        "FR_Available": fr_available_val,
        "VolConfirmed": vol_confirmed_val,
        "BTC_Mode_UP": btc_mode_up,
        "BTC_Mode_RANGE": btc_mode_range,
        "BTC_Mode_DOWN": btc_mode_down,
    }]).replace([np.inf, -np.inf], np.nan)

    mask = feats.isna()
    if mask.any().any():
        bad_cols = [c for c in feats.columns if mask[c].any()]
        try:
            print(
                f"[V2-AI-FEAT-SKIP] sym={sig.get('symbol', '')} "
                f"side={str(sig.get('side', '') or '').strip().upper()} "
                f"bad_cols={bad_cols} "
                f"total_score={sig.get('total_score', sig.get('score', ''))} "
                f"p1={sig.get('p1_score', sig.get('P1_TrendScore', ''))} "
                f"p2={sig.get('p2_score', sig.get('P2_FundingScore', ''))} "
                f"p3={sig.get('p3_score', sig.get('P3_VolumeScore', ''))} "
                f"sym_htf_strength={sig.get('sym_htf_strength', sig.get('SymHTF_Strength', ''))} "
                f"btc_htf_strength={sig.get('btc_htf_strength', sig.get('BTC_HTF_Strength', ''))} "
                f"vol_ratio={sig.get('vol_ratio', sig.get('VolRatio', ''))} "
                f"atr={sig.get('atr', sig.get('ATR', ''))} "
                f"rsi={sig.get('rsi', sig.get('RSI', ''))} "
                f"hour={sig.get('hour', sig.get('Hour_JST', ''))} "
                f"ltf_aligned={sig.get('ltf_aligned', sig.get('LTF_Aligned', ''))} "
                f"fr_available={sig.get('fr_available', sig.get('FR_Available', ''))} "
                f"vol_confirmed={sig.get('vol_confirmed', sig.get('VolConfirmed', ''))} "
                f"btc_mode_compat={sig.get('btc_mode_compat', sig.get('BTC_Mode_Compat', ''))}"
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

    lane_name = _get_runtime_lane_for_signal(sig, side_u)
    model_for_side, model_ver_side, model_src_side = get_ai_model_for_side(
        side_u, sym_code, lane=lane_name
    )
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


def _long_repeat_rank_penalty(sig: Dict[str, Any]) -> Tuple[float, int, str]:
    """
    同一銘柄LONG連打を hard block ではなく rank penalty で扱う。
    直近通知済みLONG件数を見て、閾値超過なら penalty を返す。
    """
    if not V2_LONG_REPEAT_PENALTY_ENABLE:
        return 0.0, 0, ""

    try:
        sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()
        if not sym:
            return 0.0, 0, ""

        pref_recent_n = sig.get("_prefetched_long_repeat_recent_n", None)
        pref_last_dt = sig.get("_prefetched_long_repeat_last_dt", None)

        if pref_recent_n is None:
            repeat_state = _get_v2_long_recent_notified_window_state(datetime.now(JST))
            recent_n = int(((repeat_state.get("counts", {}) or {}).get(sym, 0)) or 0)
            last_dt = str(((repeat_state.get("last_dt", {}) or {}).get(sym, "")) or "")
        else:
            recent_n = int(pref_recent_n or 0)
            last_dt = str(pref_last_dt or "")

        if recent_n > int(V2_LONG_REPEAT_PENALTY_MIN_COUNT):
            return float(V2_LONG_REPEAT_PENALTY_VALUE), recent_n, last_dt

        return 0.0, recent_n, last_dt

    except Exception:
        return 0.0, 0, ""


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

        repeat_penalty, repeat_recent_n, repeat_last_dt = _long_repeat_rank_penalty(sig)
        penalty += float(repeat_penalty)

        sig["long_repeat_recent_n"] = int(repeat_recent_n)
        sig["long_repeat_last_dt"] = str(repeat_last_dt or "")
        sig["long_repeat_penalty"] = float(repeat_penalty)

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
                f"repeat_recent_n={int(sig.get('long_repeat_recent_n', 0) or 0)};"
                f"repeat_penalty={float(sig.get('long_repeat_penalty', 0.0) or 0.0):.6f};"
                f"repeat_last_dt={str(sig.get('long_repeat_last_dt', '') or '')};"
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
                f"repeat_recent_n={int(sig.get('long_repeat_recent_n', 0) or 0)};"
                f"repeat_penalty={float(sig.get('long_repeat_penalty', 0.0) or 0.0):.6f};"
                f"repeat_last_dt={str(sig.get('long_repeat_last_dt', '') or '')};"
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
    sel_start = time.time()

    def _seltm(label: str):
        if TIMER_LOG_ENABLE:
            print(f"[TIMER] v2_selection_pipeline {label} sec={time.time() - sel_start:.2f}")

    def _timlog(msg: str) -> None:
        if TIMER_LOG_ENABLE:
            print(msg)

    if not signals:
        return []

    output: List[Dict[str, Any]] = []
    short_passed: List[Dict[str, Any]] = []
    long_passed: List[Dict[str, Any]] = []

    _seltm(f"enter n={len(signals)}")

    now_jst = datetime.now(JST)

    t_prefetch_start = time.time()

    _has_long_signals = any(
        str(s.get("direction", "")).strip().upper() == "LONG" for s in signals
    )

    t_long_repeat_start = time.time()
    pref_long_repeat_state = (
        _get_v2_long_recent_notified_window_state(now_jst)
        if (V2_LONG_REPEAT_BLOCK_ENABLE or V2_LONG_REPEAT_PENALTY_ENABLE) and _has_long_signals
        else {"counts": {}, "last_dt": {}}
    )
    if TIMER_LOG_ENABLE:
        print(
            "[TIMER] v2_selection_pipeline "
            f"prefetch_long_repeat sec={time.time() - t_long_repeat_start:.2f}"
            f" skipped={not _has_long_signals}"
        )

    t_short_repeat_start = time.time()
    pref_short_repeat_state = (
        _get_v2_short_recent_notified_window_state(now_jst)
        if V2_SHORT_REPEAT_PENALTY_ENABLE
        else {"counts": {}, "last_dt": {}}
    )
    if TIMER_LOG_ENABLE:
        print(
            "[TIMER] v2_selection_pipeline "
            f"prefetch_short_repeat sec={time.time() - t_short_repeat_start:.2f}"
        )

    if TIMER_LOG_ENABLE:
        print(
            "[TIMER] v2_selection_pipeline "
            f"prefetch_total sec={time.time() - t_prefetch_start:.2f}"
        )

    loop_start = time.time()

    for i, sig in enumerate(signals, start=1):
        direction = str(sig.get("direction", "")).strip().upper()
        one_start = time.time()
        sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()

        if sym:
            sig["_prefetched_long_repeat_recent_n"] = int(
                ((pref_long_repeat_state.get("counts", {}) or {}).get(sym, 0)) or 0
            )
            sig["_prefetched_long_repeat_last_dt"] = str(
                ((pref_long_repeat_state.get("last_dt", {}) or {}).get(sym, "")) or ""
            )
            sig["_prefetched_short_repeat_recent_n"] = int(
                ((pref_short_repeat_state.get("counts", {}) or {}).get(sym, 0)) or 0
            )
            sig["_prefetched_short_repeat_last_dt"] = str(
                ((pref_short_repeat_state.get("last_dt", {}) or {}).get(sym, "")) or ""
            )

        if direction == "SHORT":
            bypass_profile = _get_feature_bypass_profile(sig)
            if bypass_profile:
                cur_note = str(sig.get("ai_note", "") or "")
                bypass_note = f"feature_profile_bypass={bypass_profile}"
    
                sig["_feature_bypass"] = "1"
                sig["_feature_bypass_profile"] = bypass_profile
                sig["ai_pass"] = "1"
                sig["ai_band"] = "FEATURE_PROFILE_BYPASS"
                sig["ai_model_version"] = "feature_profile"
                sig["ai_model_type"] = "SHORT_FEATURE_BYPASS"
    
                if bypass_note not in cur_note:
                    sig["ai_note"] = (
                        f"{cur_note};{bypass_note}"
                        if cur_note else
                        bypass_note
                    )
    
                short_passed.append(sig)
                print(
                    f"[FEATURE_BYPASS] sym={sig.get('symbol')} side=SHORT profile={bypass_profile}",
                    flush=True,
                )
                _timlog(
                    f"[TIMER] v2_selection_pipeline short_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={time.time() - one_start:.2f} "
                    f"score_sec=0.00 total_sec={time.time() - one_start:.2f} "
                    f"passed=1 bypass=1"
                )
                continue
    
            # pause rescue: short_recent_pause発動中でもAI高スコアなら通す
            # V2_SHORT_PAUSE_AI_RESCUE_MIN <= 1.0 のときだけ有効（default 1.1 = 無効）
            # 注意: defensive_filter内の_is_short_strong_hour_bucketが既存挙動のまま
            #       AIスコアを参照しないよう、sigには書き込まずローカル変数だけで判定する
            _rescue_th = float(V2_SHORT_PAUSE_AI_RESCUE_MIN)
            _pre_ai_score = float("nan")
            if _rescue_th <= 1.0:
                _pre_ai_res = _predict_v2_ai_score(sig, "SHORT", float("nan"))
                _pre_ai_score = float(_pre_ai_res.get("score", float("nan")))

            ok, reason = defensive_filter(sig)
            t_after_filter = time.time()

            if not ok:
                _is_pause_block = "short_recent_pause" in str(reason)
                _can_rescue = (
                    _rescue_th <= 1.0
                    and _is_pause_block
                    and np.isfinite(_pre_ai_score)
                    and _pre_ai_score >= _rescue_th
                )
                _sym_r = str(sig.get("symbol", ""))
                if _is_pause_block and _rescue_th <= 1.0 and np.isfinite(_pre_ai_score):
                    print(
                        f"[V2-PAUSE-AI-RESCUE] sym={_sym_r} "
                        f"ai_prob={_pre_ai_score:.4f} th={_rescue_th:.4f} "
                        f"pause_reason={reason} pass={'1' if _can_rescue else '0'}",
                        flush=True,
                    )
                if _can_rescue:
                    sig["_short_pause_ai_rescue"] = "1"
                    ok = True
                    reason = f"ai_rescue_from_pause;ai_prob={_pre_ai_score:.4f};orig={reason}"
                else:
                    sig["ai_prob_win"] = ""
                    sig["ai_pass"] = "0"
                    sig["ai_band"] = "REJECTED"
                    sig["ai_model_version"] = "QS_v1"
                    sig["ai_model_type"] = "SHORT_RULE_RANK"
                    sig["ai_note"] = f"REJECTED:{reason}"
                    output.append(sig)
                    print(f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=SHORT reason={reason}")
                    _timlog(
                        f"[TIMER] v2_selection_pipeline short_one "
                        f"i={i} symbol={sig.get('symbol','')} "
                        f"filter_sec={t_after_filter - one_start:.2f} "
                        f"score_sec={time.time() - t_after_filter:.2f} "
                        f"total_sec={time.time() - one_start:.2f} "
                        f"passed=0"
                    )
                    continue
    
            qs = calc_quality_score(sig)
            t_after_score = time.time()
    
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
                _timlog(
                    f"[TIMER] v2_selection_pipeline short_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={t_after_score - t_after_filter:.2f} "
                    f"total_sec={t_after_score - one_start:.2f} "
                    f"passed=0"
                )
                continue
    
            ai_res = _predict_v2_ai_score(sig, "SHORT", qs)
    
            print(
                f"[SHORT-AI-RES] sym={sig.get('symbol')} "
                f"ok={bool(ai_res.get('ok'))} "
                f"score={ai_res.get('score', '')} "
                f"note={ai_res.get('note', '')} "
                f"model_type={ai_res.get('model_type', '')} "
                f"model_version={ai_res.get('model_version', '')}"
            )
    
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
                _timlog(
                    f"[TIMER] v2_selection_pipeline short_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={t_after_score - t_after_filter:.2f} "
                    f"total_sec={time.time() - one_start:.2f} "
                    f"passed=0"
                )
                continue
    
            ai_score = float(ai_res["score"])
            sig["ai_prob_win"] = round(ai_score, 6)
            sig["ai_pass"] = "1"
            sig["ai_band"] = (
                "SHORT_AI_RESCUE" if sig.get("_short_pause_ai_rescue") == "1"
                else ("SHORT_AI_SCORE" if not ai_res.get("used_fallback") else "SHORT_AI_FALLBACK_SCORE")
            )
            sig["ai_model_version"] = str(ai_res.get("model_version", "") or "")
            sig["ai_model_type"] = str(ai_res.get("model_type", "SHORT_AI") or "SHORT_AI")
            _rescue_note_pfx = (
                f"pause_ai_rescue=1;rescue_th={_rescue_th:.4f};"
                if sig.get("_short_pause_ai_rescue") == "1" else ""
            )
            sig["ai_note"] = (
                f"{_rescue_note_pfx}{ai_res.get('note', '')};"
                f"qs={qs:.4f};"
                f"p1={_fmt_sig_num_or_blank(sig, 'p1_score')};"
                f"total={_fmt_sig_num_or_blank(sig, 'total_score')};"
                f"btcstr={_fmt_sig_num_or_blank(sig, 'btc_htf_strength')};"
                f"rsi={sig.get('rsi', '')}"
            )
    
            short_passed.append(sig)
            _timlog(
                f"[TIMER] v2_selection_pipeline short_one "
                f"i={i} symbol={sig.get('symbol','')} "
                f"filter_sec={t_after_filter - one_start:.2f} "
                f"score_sec={t_after_score - t_after_filter:.2f} "
                f"total_sec={time.time() - one_start:.2f} "
                f"passed=1"
            )
            continue
    
        if direction == "LONG":
            bypass_profile = _get_feature_bypass_profile(sig)
            if bypass_profile:
                cur_note = str(sig.get("ai_note", "") or "")
                bypass_note = f"feature_profile_bypass={bypass_profile}"
    
                sig["_feature_bypass"] = "1"
                sig["_feature_bypass_profile"] = bypass_profile
                sig["ai_pass"] = "1"
                sig["ai_band"] = "FEATURE_PROFILE_BYPASS"
                sig["ai_model_version"] = "feature_profile"
                sig["ai_model_type"] = "LONG_FEATURE_BYPASS"
    
                if bypass_note not in cur_note:
                    sig["ai_note"] = (
                        f"{cur_note};{bypass_note}"
                        if cur_note else
                        bypass_note
                    )
    
                long_passed.append(sig)
                print(
                    f"[FEATURE_BYPASS] sym={sig.get('symbol')} side=LONG profile={bypass_profile}",
                    flush=True,
                )
                _timlog(
                    f"[TIMER] v2_selection_pipeline long_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={time.time() - one_start:.2f} "
                    f"score_sec=0.00 total_sec={time.time() - one_start:.2f} "
                    f"passed=1 bypass=1"
                )
                continue
    
            _t_rt = time.time()
            research_tags = _get_long_research_tags(sig)
            research_note = "|".join(research_tags)
            if TIMER_LOG_ENABLE:
                print(f"[TIMER] long_filter_research_tags sym={sig.get('symbol','')} sec={time.time()-_t_rt:.2f}")
            _t_p1 = time.time()
            rescued_p1 = _allow_long_p1_rescue(sig)
            if TIMER_LOG_ENABLE:
                print(f"[TIMER] long_filter_p1_rescue sym={sig.get('symbol','')} sec={time.time()-_t_p1:.2f}")
            _t_def = time.time()
            ok, reason = defensive_filter_long(sig)
            t_after_filter = time.time()
            if TIMER_LOG_ENABLE:
                print(f"[TIMER] long_filter_defensive sym={sig.get('symbol','')} sec={t_after_filter-_t_def:.2f}")
    
            if not ok:
                sig["ai_prob_win"] = ""
                sig["ai_pass"] = "0"
                sig["ai_band"] = "REJECTED"
                sig["ai_model_version"] = "LONG_QS_v2"
                sig["ai_model_type"] = "LONG_RULE_RANK"
                sig["ai_note"] = f"REJECTED:{reason};rescued_p1={rescued_p1};research_tags={research_note}"
                output.append(sig)
                print(f"[V2-SELECT-REJECT] sym={sig.get('symbol')} side=LONG reason={reason}")
                _timlog(
                    f"[TIMER] v2_selection_pipeline long_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={time.time() - t_after_filter:.2f} "
                    f"total_sec={time.time() - one_start:.2f} "
                    f"passed=0"
                )
                continue
    
            qs = calc_long_quality_score(sig)
            t_after_score = time.time()
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
                _timlog(
                    f"[TIMER] v2_selection_pipeline long_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={t_after_score - t_after_filter:.2f} "
                    f"total_sec={t_after_score - one_start:.2f} "
                    f"passed=0"
                )
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
                _timlog(
                    f"[TIMER] v2_selection_pipeline long_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={t_after_score - t_after_filter:.2f} "
                    f"total_sec={t_after_score - one_start:.2f} "
                    f"passed=0"
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
                _timlog(
                    f"[TIMER] v2_selection_pipeline long_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={t_after_score - t_after_filter:.2f} "
                    f"total_sec={time.time() - one_start:.2f} "
                    f"passed=0"
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
                _timlog(
                    f"[TIMER] v2_selection_pipeline long_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={t_after_score - t_after_filter:.2f} "
                    f"total_sec={time.time() - one_start:.2f} "
                    f"passed=0"
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
                _timlog(
                    f"[TIMER] v2_selection_pipeline long_one "
                    f"i={i} symbol={sig.get('symbol','')} "
                    f"filter_sec={t_after_filter - one_start:.2f} "
                    f"score_sec={t_after_score - t_after_filter:.2f} "
                    f"total_sec={time.time() - one_start:.2f} "
                    f"passed=0"
                )
                continue

            print(
                f"[V2-LONG-AI] sym={sig.get('symbol')} "
                f"ai_prob_win={sig.get('ai_prob_win', '')} "
                f"band={sig.get('ai_band', '')} "
                f"{bad_regime_note}"
            )
    
            long_passed.append(sig)
            _timlog(
                f"[TIMER] v2_selection_pipeline long_one "
                f"i={i} symbol={sig.get('symbol','')} "
                f"filter_sec={t_after_filter - one_start:.2f} "
                f"score_sec={t_after_score - t_after_filter:.2f} "
                f"total_sec={time.time() - one_start:.2f} "
                f"passed=1"
            )
            continue
    
        sig["ai_prob_win"] = ""
        sig["ai_pass"] = ""
        sig["ai_band"] = "COMMON"
        sig["ai_model_version"] = ""
        sig["ai_model_type"] = "COMMON"
        sig["ai_note"] = ""
        output.append(sig)
    
    if TIMER_LOG_ENABLE:
        print(f"[TIMER] v2_selection_pipeline loop_done sec={time.time() - loop_start:.2f}")

    _seltm(f"after_direction_filters short_passed={len(short_passed)} long_passed={len(long_passed)} output={len(output)}")

    ranked_shorts = rank_and_select(short_passed)
    _seltm(f"after_short_rank short_ranked={len(ranked_shorts)}")
    ranked_longs = rank_and_select_long(long_passed)
    _seltm(f"after_long_rank long_ranked={len(ranked_longs)}")

    output.extend(ranked_shorts)
    output.extend(ranked_longs)

    if V2_REGIME_POLICY_ENABLE:
        snapshot = get_v2_regime_health_snapshot(datetime.now(JST))
    else:
        snapshot = {}
        print("[V2-REGIME-SNAPSHOT-SKIP] policy_disabled")

    output = v2_apply_regime_notify_policy(output, snapshot)

    if V2_REGIME_POLICY_ENABLE:
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

    _seltm(f"before_return output={len(output)}")
    return output


# ==========================================
# V2 Shadow シートへの書き込み
# ==========================================

def v2_build_shadow_row(sig: Dict) -> List[Any]:
    """
    V2専用ヘッダーの行を構築。
    将来の V2-AI / LONG-AI / SHORT-AI を載せられるように AI列も保持する。
    """
    fp = _feature_profile_log_parts(sig)
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
        # 4レーン分析用メタ（runtime記録）
        sig.get("_runtime_lane", sig.get("lane", "")),
        sig.get("_lane_profile", ""),
        sig.get("_lane_profile_stage", ""),
        str(sig.get("_lane_profile_notify", "")),
        str(sig.get("_notify_pass", "")),
        str(sig.get("_notify_reason", "") or ""),
        fp["FEATURE_GATE"],
        fp["ALLOW_HITS"],
        fp["REJECT_HITS"],
        fp["FORCE_HITS"],
        fp["BYPASS_PROFILE"],
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
    cache_key = "v2_shadow_ai_data"
    now_ts = time.time()

    try:
        cached = _repeat_state_cache.get(cache_key, {})
        if cached and (now_ts - float(cached.get("ts", 0.0))) <= REPEAT_STATE_TTL_SEC:
            cached_df = cached.get("value")
            if isinstance(cached_df, pd.DataFrame):
                return cached_df.copy()
    except Exception:
        pass

    service = get_sheet_service()

    hdr_res = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A1:{HEADER_COL_END}1",
    ).execute()
    raw_hdr = (hdr_res.get("values", [[]]) or [[]])[0]
    headers = _normalize_headers(raw_hdr)

    if not headers:
        df = pd.DataFrame()
        try:
            _repeat_state_cache[cache_key] = {"ts": now_ts, "value": df.copy()}
        except Exception:
            pass
        return df

    data_res = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A2:{HEADER_COL_END}",
    ).execute()
    rows = data_res.get("values", []) or []

    if not rows:
        df = pd.DataFrame(columns=headers)
        try:
            _repeat_state_cache[cache_key] = {"ts": now_ts, "value": df.copy()}
        except Exception:
            pass
        return df

    n_cols = len(headers)
    fixed = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
    df = pd.DataFrame(fixed, columns=headers)

    try:
        _repeat_state_cache[cache_key] = {"ts": now_ts, "value": df.copy()}
    except Exception:
        pass

    return df


def get_v2_shadow_ai_recent_data_for_repeat() -> pd.DataFrame:
    """repeat state用: v2_shadow_ai の直近 V2_REPEAT_STATE_LOOKBACK_ROWS 行だけを返す。
    失敗時は get_v2_shadow_ai_data() にフォールバックする。"""
    cache_key = "v2_shadow_ai_repeat_recent"
    now_ts = time.time()

    try:
        cached = _repeat_state_cache.get(cache_key, {})
        if cached and (now_ts - float(cached.get("ts", 0.0))) <= REPEAT_STATE_TTL_SEC:
            cached_df = cached.get("value")
            if isinstance(cached_df, pd.DataFrame):
                return cached_df.copy()
    except Exception:
        pass

    fallback_used = False
    try:
        service = get_sheet_service()

        hdr_res = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{V2_SHADOW_SHEET}!A1:{HEADER_COL_END}1",
        ).execute()
        raw_hdr = (hdr_res.get("values", [[]]) or [[]])[0]
        headers = _normalize_headers(raw_hdr)

        if not headers:
            df = pd.DataFrame()
            _repeat_state_cache[cache_key] = {"ts": now_ts, "value": df.copy()}
            return df

        col_a_res = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{V2_SHADOW_SHEET}!A:A",
        ).execute()
        col_a_values = col_a_res.get("values", []) or []
        total_rows = len(col_a_values)

        if total_rows <= 1:
            df = pd.DataFrame(columns=headers)
            _repeat_state_cache[cache_key] = {"ts": now_ts, "value": df.copy()}
            return df

        lookback = max(1, V2_REPEAT_STATE_LOOKBACK_ROWS)
        start_row = max(2, total_rows - lookback + 1)
        end_row = total_rows

        if TIMER_LOG_ENABLE:
            print(
                f"[TIMER] repeat_recent_rows "
                f"start_row={start_row} end_row={end_row} "
                f"rows={end_row - start_row + 1} fallback=False"
            )

        data_res = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{V2_SHADOW_SHEET}!A{start_row}:{HEADER_COL_END}{end_row}",
        ).execute()
        rows = data_res.get("values", []) or []

        if not rows:
            df = pd.DataFrame(columns=headers)
            _repeat_state_cache[cache_key] = {"ts": now_ts, "value": df.copy()}
            return df

        n_cols = len(headers)
        fixed = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
        df = pd.DataFrame(fixed, columns=headers)

        _repeat_state_cache[cache_key] = {"ts": now_ts, "value": df.copy()}
        return df

    except Exception as e:
        fallback_used = True
        if TIMER_LOG_ENABLE:
            print(f"[TIMER] repeat_recent_rows fallback=True err={e}")
        try:
            return get_v2_shadow_ai_data()
        except Exception:
            return pd.DataFrame()


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
    _rsnap_ts = time.time()
    _rsnap_cached = _repeat_state_cache.get("v2_regime_health_snapshot", {})
    if _rsnap_cached and (_rsnap_ts - float(_rsnap_cached.get("ts", 0.0))) <= V2_REGIME_SNAPSHOT_TTL_SEC:
        return dict(_rsnap_cached["v"])
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

        _repeat_state_cache["v2_regime_health_snapshot"] = {"ts": _rsnap_ts, "v": dict(out)}
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
        df = get_v2_shadow_ai_recent_data_for_repeat()
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
    now_jst = datetime.now(JST)
    guard_state = None          # lazy: fetched only when regime policy is active
    pref_long_repeat_state = None  # lazy: fetched only when LONG repeat block fires

    for sig in signals:
        side = str(sig.get("direction", "")).strip().upper()
        mode = str(sig.get("btc_mode_compat", "") or "").strip().upper()
        ai_pass = str(sig.get("ai_pass", "")).strip()
        sym = str(sig.get("symbol", "")).split("/", 1)[0].strip().upper()
        hour = _sig_hour_jst(sig)

        notify_pass = "0"
        notify_reason = "init"

        if ai_pass != "1":
            notify_pass = "0"
            notify_reason = "ai_pass_0"
        elif side not in {"SHORT", "LONG"}:
            notify_pass = "0"
            notify_reason = "unsupported_side"
        else:
            if side == "LONG" and hour is not None and hour in V2_LONG_NOTIFY_BLOCK_HOURS:
                notify_pass = "0"
                notify_reason = f"long_notify_block_hour:{hour}"
            elif side == "SHORT" and hour is not None and hour in V2_SHORT_NOTIFY_BLOCK_HOURS:
                notify_pass = "0"
                notify_reason = f"short_notify_block_hour:{hour}"
            elif side == "LONG" and V2_LONG_REPEAT_BLOCK_ENABLE and sym:
                if pref_long_repeat_state is None:
                    pref_long_repeat_state = (
                        _get_v2_long_recent_notified_window_state(now_jst)
                        if V2_LONG_REPEAT_BLOCK_ENABLE
                        else {"counts": {}, "last_dt": {}}
                    )
                recent_n = int(((pref_long_repeat_state.get("counts", {}) or {}).get(sym, 0)) or 0)
                if recent_n >= int(V2_LONG_REPEAT_MAX_NOTIFIES):
                    notify_pass = "0"
                    notify_reason = (
                        f"long_repeat_hard_block:sym={sym};"
                        f"recent_n={recent_n};"
                        f"window_min={int(V2_LONG_REPEAT_WINDOW_MINUTES)}"
                    )

            if notify_reason != "init":
                pass
            elif not V2_REGIME_POLICY_ENABLE:
                notify_pass = "1"
                notify_reason = "regime_policy_disabled"
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
                        rescue_hour = _get_long_sig_hour(sig)
                        blocked_hours = _parse_hour_csv_to_set(V2_LONG_RESCUE_NOTIFY_BLOCK_HOURS)

                        if guard_state is None:
                            guard_state = get_v2_guardrail_state(now_jst)
                        if guard_state.get("mode") == "STOP":
                            notify_pass = "0"
                            notify_reason = f"guardrail_stop:{guard_state.get('reason', '')}"
                        elif rescue_hour is not None and rescue_hour in blocked_hours:
                            notify_pass = "0"
                            notify_reason = f"long_rescue_blocked_bad_hour:{rescue_hour}"
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
        "feature_schema_version": str(V2_TRAIN_FEATURE_SCHEMA_VERSION),
        "feature_hash": "",
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

    info["feature_hash"] = _feature_columns_hash(info["feature_columns"])
    info["rows_used"] = int(len(X))
    return X, y, info


def _build_training_dataset_from_v2_shadow_ai(
    side: str,
    lookback_rows: int,
    lookback_days: Optional[float] = None,
    lane: str = "",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    v2_shadow_ai から LONG/SHORT 別の学習用 X, y を作るラッパー。
    追加方針:
      - SHORT: VolConfirmed=True 優先、SHORT blocklist除外、mode選別
      - LONG : VolConfirmed=True 優先、LONG blocklist除外、UP中心
      - lane 指定時は __lane__ で絞り込む
    """
    side_u = str(side or "").strip().upper()
    lane_u = _normalize_lane_name(lane)

    meta: Dict[str, Any] = {
        "rows_scanned": 0,
        "rows_used": 0,
        "class_balance": {},
        "side": side_u,
        "lane": lane_u,
        "train_filters": {},
    }

    df = get_v2_shadow_ai_data()
    if df is None or df.empty:
        raise RuntimeError("v2_shadow_ai is empty")

    applied_window: Dict[str, Any] = {
        "mode": "all",
        "lookback_rows": None,
        "lookback_days": None,
        "datetime_column": "",
        "cutoff_jst": "",
    }

    dt_col = ""
    if "Datetime_JST" in df.columns:
        dt_col = "Datetime_JST"
    elif "Time" in df.columns:
        dt_col = "Time"

    if lookback_days is not None and float(lookback_days) > 0 and dt_col:
        dt_series = pd.to_datetime(df[dt_col].astype(str).str.strip(), errors="coerce")
        cutoff = datetime.now(JST).replace(tzinfo=None) - timedelta(days=float(lookback_days))
        df = df.loc[dt_series.notna() & (dt_series >= cutoff)].copy()

        applied_window = {
            "mode": "days",
            "lookback_rows": None,
            "lookback_days": float(lookback_days),
            "datetime_column": dt_col,
            "cutoff_jst": cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        }

    elif lookback_rows is not None and int(lookback_rows) > 0:
        df = df.tail(int(lookback_rows)).copy()

        applied_window = {
            "mode": "rows",
            "lookback_rows": int(lookback_rows),
            "lookback_days": None,
            "datetime_column": "",
            "cutoff_jst": "",
        }

    meta["rows_scanned"] = int(len(df))
    meta["window"] = applied_window

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

    work["__lane__"] = work.apply(
        lambda row: _classify_train_lane_from_row(row.to_dict()),
        axis=1,
    )

    lane_counts_before_filter = (
        work["__lane__"]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    meta["lane_counts_before_filter"] = {
        str(k): int(v) for k, v in lane_counts_before_filter.items()
    }

    if lane_u:
        work = work[work["__lane__"].astype(str).str.strip().str.lower() == lane_u].copy()

    meta["rows_after_lane_filter"] = int(len(work))

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
            "lane": lane_u,
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
            "lane": lane_u,
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

    meta["feature_schema_version"] = str(info.get("feature_schema_version", "") or "")
    meta["feature_hash"] = str(info.get("feature_hash", "") or "")
    meta["feature_columns"] = list(info.get("feature_columns", []) or [])
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
    _brake_ts = time.time()
    _brake_cached = _repeat_state_cache.get("v2_short_brake_state", {})
    if _brake_cached and (_brake_ts - float(_brake_cached.get("ts", 0.0))) <= V2_BRAKE_STATE_TTL_SEC:
        return dict(_brake_cached["v"])
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
        _brake_result = {
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
        _repeat_state_cache["v2_short_brake_state"] = {"ts": _brake_ts, "v": dict(_brake_result)}
        return _brake_result
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
    _lbrake_ts = time.time()
    _lbrake_cached = _repeat_state_cache.get("v2_long_brake_state", {})
    if _lbrake_cached and (_lbrake_ts - float(_lbrake_cached.get("ts", 0.0))) <= V2_BRAKE_STATE_TTL_SEC:
        return dict(_lbrake_cached["v"])
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

        _lbrake_result = {
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
        _repeat_state_cache["v2_long_brake_state"] = {"ts": _lbrake_ts, "v": dict(_lbrake_result)}
        return _lbrake_result

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


def _fmt_report_wr(v: Any) -> str:
    try:
        x = float(v)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:.1f}%"
    except Exception:
        return "N/A"


def _fmt_report_avg_pnl(v: Any) -> str:
    try:
        x = float(v)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:+.2f}%"
    except Exception:
        return "N/A"


def _build_symbol_ranking_lines(
    df: pd.DataFrame,
    title: str,
    top_n: int = 5,
    sort_bad: bool = False,
    min_n: int = 1,
) -> list:
    """
    銘柄別ランキングを daily 用に出す。
    sort_bad=False: 良い順
    sort_bad=True : 悪い順
    min_n=1 なので、1件でも表示する。
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
        lines.append("・対象なし")
        return lines

    rank_df = pd.DataFrame(rows)

    wr_has_value = bool(np.isfinite(rank_df["wr"]).any()) if "wr" in rank_df.columns else False
    pnl_has_value = bool(np.isfinite(rank_df["avg_pnl"]).any()) if "avg_pnl" in rank_df.columns else False
    if (not wr_has_value) and (not pnl_has_value):
        lines.append("・まだ判定未了のため集計保留")
        return lines

    if sort_bad:
        rank_df = rank_df.sort_values(
            ["avg_pnl", "wr", "n"],
            ascending=[True, True, False],
            na_position="last",
        ).head(int(top_n))
    else:
        rank_df = rank_df.sort_values(
            ["avg_pnl", "wr", "n"],
            ascending=[False, False, False],
            na_position="last",
        ).head(int(top_n))

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

    for i, row in enumerate(rank_df.itertuples(index=False), start=1):
        medal = medals[i - 1] if i - 1 < len(medals) else f"{i}."
        lines.append(
            f"{medal} {row.symbol}: {row.n}件 / 勝率 {_fmt_report_wr(row.wr)} / 平均 {_fmt_report_avg_pnl(row.avg_pnl)}"
        )

    return lines


def _build_hour_ranking_lines(
    df: pd.DataFrame,
    title: str,
    top_n: int = 8,
    min_n: int = 1,
) -> list:
    """
    時間帯別ランキングを daily 用に出す。
    優先順:
      1) NotifiedAt_JST 列
      2) _notified_at_jst 列
      3) AI_Note / ai_note 内の notified_at_jst=YYYY-MM-DD HH:MM:SS
      4) Datetime_JST / _dt / Hour_JST / hour
    """
    lines = [f"【{title}】"]

    if df is None or df.empty:
        lines.append("・データなし")
        return lines

    work = df.copy()
    work["_hour"] = pd.Series([np.nan] * len(work), index=work.index, dtype="float64")

    # 1) 実送信時刻の専用列があれば最優先
    if "NotifiedAt_JST" in work.columns:
        dt = pd.to_datetime(work["NotifiedAt_JST"].astype(str).str.strip(), errors="coerce")
        work["_hour"] = dt.dt.hour

    elif "_notified_at_jst" in work.columns:
        dt = pd.to_datetime(work["_notified_at_jst"].astype(str).str.strip(), errors="coerce")
        work["_hour"] = dt.dt.hour

    else:
        # 2) AI_Note / ai_note から notified_at_jst=... を抜く
        note_col = None
        for c in ["AI_Note", "ai_note"]:
            if c in work.columns:
                note_col = c
                break

        if note_col is not None:
            note_text = work[note_col].astype(str)
            extracted = note_text.str.extract(r"notified_at_jst=([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})", expand=False)
            dt = pd.to_datetime(extracted, errors="coerce")
            work["_hour"] = dt.dt.hour

        # 3) まだ取れていない行だけ、従来の列へフォールバック
        missing_mask = work["_hour"].isna()

        if missing_mask.any() and "_dt" in work.columns:
            dt = pd.to_datetime(work.loc[missing_mask, "_dt"], errors="coerce")
            work.loc[missing_mask, "_hour"] = dt.dt.hour

        missing_mask = work["_hour"].isna()
        if missing_mask.any() and "Datetime_JST" in work.columns:
            dt = pd.to_datetime(
                work.loc[missing_mask, "Datetime_JST"].astype(str).str.strip(),
                errors="coerce",
            )
            work.loc[missing_mask, "_hour"] = dt.dt.hour

        missing_mask = work["_hour"].isna()
        if missing_mask.any() and "Hour_JST" in work.columns:
            work.loc[missing_mask, "_hour"] = pd.to_numeric(
                work.loc[missing_mask, "Hour_JST"], errors="coerce"
            )

        missing_mask = work["_hour"].isna()
        if missing_mask.any() and "hour" in work.columns:
            work.loc[missing_mask, "_hour"] = pd.to_numeric(
                work.loc[missing_mask, "hour"], errors="coerce"
            )

    work = work[work["_hour"].notna()].copy()
    if work.empty:
        lines.append("・データなし")
        return lines

    work["_hour"] = work["_hour"].astype(int)

    rows = []
    for hour, sub in work.groupby("_hour"):
        n = len(sub)
        if n < int(min_n):
            continue

        w = int(sub["_win"].sum()) if "_win" in sub.columns else 0
        lo = int(sub["_lose"].sum()) if "_lose" in sub.columns else 0
        wr = (w / (w + lo) * 100.0) if (w + lo) > 0 else float("nan")
        avg_pnl = sub["_pnl"].mean() if "_pnl" in sub.columns else float("nan")

        rows.append({
            "hour": int(hour),
            "n": n,
            "wr": wr,
            "avg_pnl": avg_pnl,
        })

    if not rows:
        lines.append("・対象なし")
        return lines

    rank_df = pd.DataFrame(rows)

    wr_has_value = bool(np.isfinite(rank_df["wr"]).any()) if "wr" in rank_df.columns else False
    pnl_has_value = bool(np.isfinite(rank_df["avg_pnl"]).any()) if "avg_pnl" in rank_df.columns else False
    if (not wr_has_value) and (not pnl_has_value):
        lines.append("・まだ判定未了のため集計保留")
        return lines

    rank_df = rank_df.sort_values(
        ["avg_pnl", "wr", "n"],
        ascending=[False, False, False],
        na_position="last",
    ).head(int(top_n))

    for row in rank_df.itertuples(index=False):
        lines.append(
            f"・{int(row.hour):02d}時: {row.n}件 / 勝率 {_fmt_report_wr(row.wr)} / 平均 {_fmt_report_avg_pnl(row.avg_pnl)}"
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

    if "EvalStatus" in df_all.columns:
        status_all = df_all["EvalStatus"].astype(str).str.strip().str.upper()
        n_open = int((status_all == "OPEN").sum())
    else:
        n_open = 0

    now_jst = datetime.now(JST)
    now_str = now_jst.strftime("%Y-%m-%d %H:%M JST")
    since_jst = now_jst - timedelta(hours=24)

    # 1) DONEベースの集計用
    df_done = get_v2_done_records(df_all).copy()
    n_done = len(df_done)

    # 2) 実送信ベースの集計用（OPENも含める）
    df_report_all = df_all.copy()
    if "Datetime_JST" in df_report_all.columns:
        df_report_all["Datetime_JST"] = pd.to_datetime(
            df_report_all["Datetime_JST"].astype(str).str.strip(),
            errors="coerce",
        )
        df_report_all = df_report_all[df_report_all["Datetime_JST"].notna()].copy()
        df_report_all = df_report_all[df_report_all["Datetime_JST"] >= pd.Timestamp(since_jst.replace(tzinfo=None))].copy()
    else:
        df_report_all = pd.DataFrame(columns=df_all.columns)

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

    # 3) 従来の daily 本文の DONE は維持
    if "Datetime_JST" in df_done.columns:
        df_done["Datetime_JST"] = pd.to_datetime(
            df_done["Datetime_JST"].astype(str).str.strip(),
            errors="coerce",
        )
        df_done = df_done[df_done["Datetime_JST"].notna()].copy()
        df_24h = df_done[df_done["Datetime_JST"] >= pd.Timestamp(since_jst.replace(tzinfo=None))].copy()
    else:
        df_24h = pd.DataFrame(columns=df_done.columns)

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

    ai_pass_series = pd.Series([False] * len(df_24h), index=df_24h.index)
    if "AI_Pass" in df_24h.columns:
        ai_pass_text = df_24h["AI_Pass"].astype(str).str.strip().str.lower()
        ai_pass_series = ai_pass_text.isin(["1", "true", "yes", "on"])

    if note_col:
        note = df_24h[note_col].astype(str)

        has_notify_sent_tag = note.str.contains("notify_sent=", na=False)
        has_notify_pass_tag = note.str.contains("notify_pass=1", na=False)

        # 採用判定:
        # AI_Note の notify_pass=1 を基本にしつつ、Excel互換で AI_Pass=1 も含める
        df_24h["_notify_1"] = has_notify_pass_tag | ai_pass_series

        # 実送信:
        # 1) notify_sent=1 があれば最優先
        # 2) notify_sent タグが無い旧データは、AI_Pass=1 を実送信扱いにする
        # 3) さらに後方互換として notify_pass=1 も残す
        df_24h["_notify_sent_1"] = (
            note.str.contains("notify_sent=1", na=False)
            | (~has_notify_sent_tag & ai_pass_series)
            | (~has_notify_sent_tag & has_notify_pass_tag)
        )
    else:
        # AI_Note が無い場合も、Excel上の AI_Pass=1 を基準に集計する
        df_24h["_notify_1"] = ai_pass_series
        df_24h["_notify_sent_1"] = ai_pass_series

    if "AI_Band" in df_24h.columns:
        df_24h["AI_Band"] = df_24h["AI_Band"].astype(str).str.strip().str.upper()
    else:
        df_24h["AI_Band"] = ""

    notify_df = df_24h[df_24h["_notify_1"]].copy()

    # 実送信は DONE に限定しない
    sent_df = df_report_all.copy()

    note_col_sent = None
    for c in ["AI_Note", "ai_note"]:
        if c in sent_df.columns:
            note_col_sent = c
            break

    sent_ai_pass_series = pd.Series([False] * len(sent_df), index=sent_df.index)
    if "AI_Pass" in sent_df.columns:
        sent_ai_pass_text = sent_df["AI_Pass"].astype(str).str.strip().str.lower()
        sent_ai_pass_series = sent_ai_pass_text.isin(["1", "true", "yes", "on"])

    if note_col_sent:
        sent_note = sent_df[note_col_sent].astype(str)
        sent_has_notify_sent_tag = sent_note.str.contains("notify_sent=", na=False)
        sent_has_notify_pass_tag = sent_note.str.contains("notify_pass=1", na=False)

        sent_df["_notify_sent_1"] = (
            sent_note.str.contains("notify_sent=1", na=False)
            | (~sent_has_notify_sent_tag & sent_ai_pass_series)
            | (~sent_has_notify_sent_tag & sent_has_notify_pass_tag)
        )
    else:
        sent_df["_notify_sent_1"] = sent_ai_pass_series

    sent_df = sent_df[sent_df["_notify_sent_1"]].copy()

    rank_reject_df = df_24h[df_24h["AI_Band"] == "RANK_REJECT"].copy()
    rejected_df = df_24h[df_24h["AI_Band"] == "REJECTED"].copy()

    done_long_n = 0
    done_short_n = 0
    if "Direction" in df_24h.columns:
        dir_done = df_24h["Direction"].astype(str).str.strip().str.upper()
        done_long_n = int((dir_done == "LONG").sum())
        done_short_n = int((dir_done == "SHORT").sum())

    notify_long_n = 0
    notify_short_n = 0
    if "Direction" in notify_df.columns:
        dir_notify = notify_df["Direction"].astype(str).str.strip().str.upper()
        notify_long_n = int((dir_notify == "LONG").sum())
        notify_short_n = int((dir_notify == "SHORT").sum())

    sent_long_n = 0
    sent_short_n = 0
    if "Direction" in sent_df.columns:
        dir_sent = sent_df["Direction"].astype(str).str.strip().str.upper()
        sent_long_n = int((dir_sent == "LONG").sum())
        sent_short_n = int((dir_sent == "SHORT").sum())

    sent_long_df = pd.DataFrame()
    sent_short_df = pd.DataFrame()
    if "Direction" in sent_df.columns:
        dir_sent = sent_df["Direction"].astype(str).str.strip().str.upper()
        sent_long_df = sent_df[dir_sent == "LONG"].copy()
        sent_short_df = sent_df[dir_sent == "SHORT"].copy()

    lines = [
        "📘【V2 デイリーレポート】",
        f"🕒 {now_str}",
        f"📦 直近24h DONE {len(df_24h)}件 / OPEN {n_open}件",
        f"📊 DONE内訳: 🟢LONG {done_long_n}件 / 🔴SHORT {done_short_n}件",
        f"🤖 採用判定内訳: 🟢LONG {notify_long_n}件 / 🔴SHORT {notify_short_n}件",
        f"📣 実送信内訳: 🟢LONG {sent_long_n}件 / 🔴SHORT {sent_short_n}件",
        "",
        _summarize_simple(notify_df, "採用判定", "🤖"),
        _summarize_simple(sent_df, "実送信", "📣"),
        _summarize_simple(rank_reject_df, "惜しくも落選", "🟡"),
        _summarize_simple(rejected_df, "除外", "🔴"),
        "",
    ]

    lines += _build_symbol_ranking_lines(
        sent_df,
        "良い銘柄ランキング（実送信）",
        top_n=5,
        sort_bad=False,
        min_n=1,
    )
    lines += [""]
    lines += _build_symbol_ranking_lines(
        sent_df,
        "悪い銘柄ランキング（実送信）",
        top_n=5,
        sort_bad=True,
        min_n=1,
    )
    lines += [""]
    lines += _build_symbol_ranking_lines(
        rank_reject_df,
        "惜しくも落選ランキング",
        top_n=5,
        sort_bad=False,
        min_n=1,
    )
    lines += [""]
    lines += _build_hour_ranking_lines(
        sent_long_df,
        "実送信時間帯ランキング（LONG）",
        top_n=8,
        min_n=1,
    )
    lines += [""]
    lines += _build_hour_ranking_lines(
        sent_short_df,
        "実送信時間帯ランキング（SHORT）",
        top_n=8,
        min_n=1,
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
    v2_start = time.time()

    def _v2tm(label: str):
        if TIMER_LOG_ENABLE:
            print(f"[TIMER] v2_shadow_run {label} sec={time.time() - v2_start:.2f}")

    engine_mode = str(SIGNAL_ENGINE).strip().lower()
    _v2tm("enter")

    # 15分足確定待ち
    if (not force) and ((now_jst.minute % 15) < 10):
        return "V2: waiting"

    # シート準備（HEALTH_CHECK_TTL_SEC 以内に成功済みならスキップ）
    _vs_key = "v2_ensure_shadow_sheet_ok"
    _vs_last = _health_check_cache.get(_vs_key, 0.0)
    if time.time() - _vs_last > HEALTH_CHECK_TTL_SEC:
        try:
            v2_ensure_shadow_sheet(get_sheet_service(), SPREADSHEET_ID)
            _health_check_cache[_vs_key] = time.time()
            _v2tm("after_ensure_shadow_sheet")
        except Exception as e:
            print(f"[V2-ERR] sheet setup: {e}")
            return f"V2: sheet error {e}"
    else:
        _v2tm("after_ensure_shadow_sheet")

    # BTC 上位足トレンド
    btc_htf_ohlcv = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe=V2_HTF_TIMEFRAME, limit=V2_HTF_LIMIT)
    if not btc_htf_ohlcv or len(btc_htf_ohlcv) < V2_EMA_SLOW + 5:
        return "V2: BTC HTF insufficient"

    btc_htf_df = pd.DataFrame(btc_htf_ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    btc_htf = assess_htf_trend(btc_htf_df)
    _v2tm("after_btc_htf")

    # BTC 15m Pct_Change / Vol (AI 特徴量用)
    btc_ret = np.nan
    btc_vol = np.nan
    btc_1h_change_v2 = np.nan
    btc_15m_df = None
    try:
        _btc15m_key = "BTC/USDT|15m|30"
        _btc15m_cached = _v2_fetch_cache.get(_btc15m_key)
        if _btc15m_cached and (time.time() - _btc15m_cached["ts"]) <= V2_FETCH_TTL_SEC:
            btc_15m_ohlcv = _btc15m_cached["v"]
        else:
            btc_15m_ohlcv = fetch_ohlcv_safe(exchange, "BTC/USDT", timeframe="15m", limit=30)
            if btc_15m_ohlcv is not None:
                _v2_fetch_cache[_btc15m_key] = {"ts": time.time(), "v": btc_15m_ohlcv}
        if btc_15m_ohlcv and len(btc_15m_ohlcv) >= 6:
            btc_15m_df = pd.DataFrame(
                btc_15m_ohlcv,
                columns=["Time", "Open", "High", "Low", "Close", "Volume"]
            )
            btc_15m_df["Pct_Change"] = pd.to_numeric(
                btc_15m_df["Close"], errors="coerce"
            ).pct_change(fill_method=None)

            btc_ret = float(btc_15m_df["Pct_Change"].iloc[-2])
            btc_vol = abs(btc_ret)

            btc_current_15m = float(btc_15m_df["Close"].iloc[-2])
            btc_1h_ago_15m = float(btc_15m_df["Close"].iloc[-6])
            if abs(btc_1h_ago_15m) > 1e-12:
                btc_1h_change_v2 = (btc_current_15m - btc_1h_ago_15m) / btc_1h_ago_15m

    except Exception as e:
        print(f"[V2-WARN] BTC 15m fetch for AI feats failed: {e}")
    _v2tm("after_btc_15m_features")

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

    btc_surge_raw_active, btc_surge_raw_reason = _detect_btc_surge_long_pause(
        btc_htf_dir=btc_htf.get("direction", ""),
        btc_htf_strength=btc_htf.get("strength", np.nan),
        btc_1h_change=btc_1h_change_v2,
        btc_ret_15m=btc_ret,
    )

    btc_surge_long_pause_active, btc_surge_long_pause_reason = _resolve_btc_surge_long_pause_runtime(
        raw_active=btc_surge_raw_active,
        raw_reason=btc_surge_raw_reason,
        btc_htf_dir=btc_htf.get("direction", ""),
        btc_htf_strength=btc_htf.get("strength", np.nan),
        btc_15m_df=btc_15m_df,
    )

    _btc_long_pause_ctx["active"] = bool(btc_surge_long_pause_active)
    _btc_long_pause_ctx["reason"] = str(btc_surge_long_pause_reason or "")

    print(
        f"[V2-BTC-SURGE-LONG-PAUSE] "
        f"raw_active={int(bool(btc_surge_raw_active))} "
        f"effective_active={int(bool(btc_surge_long_pause_active))} "
        f"reason={btc_surge_long_pause_reason}"
    )

    _notify_btc_surge_long_pause_if_changed(
        active=btc_surge_long_pause_active,
        reason=btc_surge_long_pause_reason,
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

    def _build_feature_probe_candidates(
        symbol: str,
        btc_htf_probe: Dict[str, Any],
        now_jst_probe: datetime,
        regime_info_probe: Optional[Dict[str, Any]],
        note_suffix: str,
        include_bad_col: str = "",
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        # bypass/allow profile に side 定義がある方向だけ probe（無駄な fetch を省く）
        _bypass_names = V2_FEATURE_BYPASS_PROFILES or V2_FEATURE_ALLOW_PROFILES
        active_probe_sides: set = set()
        for _pname in _bypass_names:
            _praw = str(os.environ.get(_profile_env_name(_pname), "")).strip()
            _pspec = _parse_profile_spec(_praw)
            _pside = str(_pspec.get("side", "")).strip().upper()
            if _pside in ("LONG", "SHORT"):
                active_probe_sides.add(_pside)
            elif not _pside:
                active_probe_sides.update(("LONG", "SHORT"))
        if not active_probe_sides:
            active_probe_sides = {"LONG", "SHORT"}

        for forced_side in ("LONG", "SHORT"):
            if forced_side not in active_probe_sides:
                continue
            probe_sig = None
            try:
                probe_sig = v2_generate_signal(
                    exchange,
                    symbol,
                    btc_htf_probe,
                    now_jst_probe,
                    regime_info=regime_info_probe,
                    feature_probe_only=True,
                    forced_probe_direction=forced_side,
                )
            except Exception as e:
                print(
                    f"[V2] FEATURE-PROBE-BUILD-ERR sym={symbol} side={forced_side} "
                    f"src={note_suffix} err={e}"
                )
                continue

            if probe_sig is None:
                continue

            _apply_feature_profiles(probe_sig)

            if not _is_feature_force_pass(probe_sig):
                continue

            probe_sig["_feature_bypass"] = "1"
            probe_sig["_feature_bypass_profile"] = str(
                probe_sig.get("_feature_force_profile", "") or ""
            )

            cur_note = str(probe_sig.get("ai_note", "") or "")
            add_note = f"feature_probe_source={note_suffix}"
            if include_bad_col:
                add_note += f";bad_col={include_bad_col}"
            probe_sig["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note

            if not str(probe_sig.get("v2_version", "")).strip():
                probe_sig["v2_version"] = "V2-FeatureProbe"

            probe_sig["note"] = ""
            try:
                probe_sig["btc_ret"] = _safe_float_or_nan(btc_ret)
            except Exception:
                pass
            try:
                probe_sig["btc_vol"] = _safe_float_or_nan(btc_vol)
            except Exception:
                pass

            out.append(probe_sig)

        return out

    raw_signals: List[Dict[str, Any]] = []

    for symbol in symbols:
        try:
            sig = v2_generate_signal(
                exchange,
                symbol,
                btc_htf,
                now_jst,
                regime_info=regime,
                feature_probe_only=False,
            )

            if sig is None:
                probe_candidates = _build_feature_probe_candidates(
                    symbol=symbol,
                    btc_htf_probe=btc_htf,
                    now_jst_probe=now_jst,
                    regime_info_probe=regime,
                    note_suffix="post_reject_fallback",
                )

                _disabled_fb_routes = {
                    r.strip().lower()
                    for r in os.environ.get(
                        "V2_FEATURE_POST_REJECT_FALLBACK_DISABLED_ROUTES", ""
                    ).split(",")
                    if r.strip()
                }
                if _disabled_fb_routes:
                    probe_candidates = [
                        c for c in probe_candidates
                        if str(c.get("_feature_bypass_profile", "")).strip().lower()
                        not in _disabled_fb_routes
                    ]

                if probe_candidates:
                    sig = probe_candidates[0]

                    if V2_DEBUG_FEATURE_PROBE:
                        print(
                            f"[V2] FEATURE-PROBE: {sig['symbol']} {sig['direction']} "
                            f"total={sig['total_score']:.2f} "
                            f"P1={sig['p1_score']:.2f} P2={sig['p2_score']:.2f} P3={sig['p3_score']:.2f} "
                            f"RSI={sig.get('rsi', '')} BTCstr={sig.get('btc_htf_strength', '')} "
                            f"NOTE={sig.get('ai_note', '')}"
                        )

            if sig is not None:
                if not str(sig.get("v2_version", "")).strip():
                    sig["v2_version"] = "V2-Shadow"

                sig["note"] = ""
                sig["btc_ret"] = _safe_float_or_nan(btc_ret)
                sig["btc_vol"] = _safe_float_or_nan(btc_vol)

                raw_signals.append(sig)

                if V2_DEBUG_RAW_SIG:
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

    for sig in final_signals:
        # 記録専用ブロック: 全 final_signals にレーン情報を記録
        # 通知ロジック・副作用は後段の既存ゲート内ブロックに任せる
        try:
            _rec_dir = str(sig.get("direction", "")).strip().upper()
            if _rec_dir in ("LONG", "SHORT"):
                _rec_lane = _get_runtime_lane_for_signal(sig, _rec_dir)
                sig["_runtime_lane"] = _rec_lane
                _rec_profile = _classify_lane_profile(_rec_dir, _rec_lane, sig)
                sig["_lane_profile"] = _rec_profile.get("profile", "")
                sig["_lane_profile_stage"] = _rec_profile.get("stage", "")
                sig["_lane_profile_notify"] = _rec_profile.get("notify", False)
        except Exception as _rec_err:
            logging.warning(
                f"[LANE-PROFILE-RECORD-ERR] sym={sig.get('symbol','')} err={_rec_err}"
            )

        _apply_feature_profiles(sig)

        cur_note = str(sig.get("ai_note", "") or "")
        profile_label = _feature_profile_label(sig)
        if profile_label:
            sig["ai_note"] = f"{cur_note};matched_profile={profile_label}" if cur_note else f"matched_profile={profile_label}"

        gate = str(sig.get("_feature_profile_gate", "") or "").strip().lower()

        force_hits = sig.get("_feature_force_hits", []) or []
        bypass_profile = str(sig.get("_feature_bypass_profile", sig.get("_feature_force_profile", "")) or "")

        print(
            "[FEATURE_PROFILE] "
            f"sym={sig.get('symbol','')} "
            f"side={sig.get('direction','')} "
            f"gate={gate} "
            f"allow_hits={sig.get('_feature_profile_allow_hits', [])} "
            f"reject_hits={sig.get('_feature_profile_reject_hits', [])} "
            f"force_hits={force_hits} "
            f"bypass_profile={bypass_profile}",
            flush=True,
        )

        if gate == "reject":
            cur_note2 = str(sig.get("ai_note", "") or "")
            sig["ai_pass"] = "0"
            sig["ai_band"] = "FEATURE_PROFILE_REJECT"
            sig["ai_note"] = (
                f"{cur_note2};selected=false;reason=feature_profile_reject"
                if cur_note2 else
                "selected=false;reason=feature_profile_reject"
            )
        elif gate == "no_match":
            cur_note2 = str(sig.get("ai_note", "") or "")
            sig["ai_pass"] = "0"
            sig["ai_band"] = "FEATURE_PROFILE_NO_MATCH"
            sig["ai_note"] = (
                f"{cur_note2};selected=false;reason=feature_profile_no_match"
                if cur_note2 else
                "selected=false;reason=feature_profile_no_match"
            )
        else:
            allow_hits = [
                str(x).strip()
                for x in sig.get("_feature_profile_allow_hits", [])
                if str(x).strip()
            ]
            bypass_hits = [
                x for x in allow_hits
                if x in V2_FEATURE_BYPASS_PROFILES
            ]

            if bypass_hits:
                selected_bypass = bypass_hits[0]
                cur_note2 = str(sig.get("ai_note", "") or "")

                sig["_feature_bypass"] = "1"
                sig["_feature_bypass_profile"] = selected_bypass
                sig["ai_pass"] = "1"
                sig["_notify_pass"] = "1"

                if not str(sig.get("ai_band", "")).strip():
                    sig["ai_band"] = "FEATURE_PROFILE_BYPASS"

                bypass_note = f"feature_profile_bypass={selected_bypass}"
                if bypass_note not in cur_note2:
                    sig["ai_note"] = (
                        f"{cur_note2};{bypass_note}"
                        if cur_note2 else
                        bypass_note
                    )

                print(
                    "[FEATURE_BYPASS] "
                    f"sym={sig.get('symbol','')} "
                    f"side={sig.get('direction','')} "
                    f"profile={selected_bypass}",
                    flush=True,
                )

        # レーンプロファイル判定（ai_pass="1" の通過シグナルのみ）
        # ただし feature bypass 通過シグナル / SHORT AI ランク通過シグナルは lane profile で落とさない
        # SHORT_AI_RANK はAIが既に採点・選抜済みのため lane profile チェックをスキップする
        try:
            _lane_chk_skip = (
                str(sig.get("_feature_bypass", "0")) == "1"
                or str(sig.get("ai_band", "")) in {"SHORT_AI_RANK", "SHORT_AI_FALLBACK_RANK"}
            )
            if str(sig.get("ai_pass", "")) == "1" and not _lane_chk_skip:
                _sig_direction = str(sig.get("direction", "")).strip().upper()
                if _sig_direction in ("LONG", "SHORT"):
                    _lane = _get_runtime_lane_for_signal(sig, _sig_direction)
                    profile_decision = _classify_lane_profile(_sig_direction, _lane, sig)
                    profile_name = profile_decision["profile"]
                    profile_stage = profile_decision["stage"]
                    profile_notify = profile_decision["notify"]

                    sig["_lane_profile"] = profile_name
                    sig["_lane_profile_stage"] = profile_stage
                    sig["_lane_profile_notify"] = profile_notify

                    if profile_stage == "reject":
                        sig["ai_pass"] = "0"
                        sig["_notify_pass"] = "0"
                        cur_note3 = str(sig.get("ai_note", "") or "")
                        sig["ai_note"] = (
                            f"{cur_note3};lane_profile_reject={profile_name}"
                            if cur_note3 else f"lane_profile_reject={profile_name}"
                        )
                    elif profile_stage == "research":
                        sig["_notify_pass"] = "0"
                        cur_note3 = str(sig.get("ai_note", "") or "")
                        sig["ai_note"] = (
                            f"{cur_note3};lane_profile_research={profile_name}"
                            if cur_note3 else f"lane_profile_research={profile_name}"
                        )
                    else:
                        if not profile_notify:
                            sig["_notify_pass"] = "0"
                            cur_note3 = str(sig.get("ai_note", "") or "")
                            sig["ai_note"] = (
                                f"{cur_note3};lane_profile_no_notify={profile_name}"
                                if cur_note3 else f"lane_profile_no_notify={profile_name}"
                            )
        except Exception as _lp_err:
            logging.warning(f"[LANE-PROFILE-ERR] sym={sig.get('symbol','')} err={_lp_err}")

        try:
            _ai_prob_dbg = sig.get("ai_prob", sig.get("ai_prob_win", sig.get("AI_Prob_Win", "")))
            _score_dbg = sig.get("score", sig.get("total_score", sig.get("TotalScore", "")))
            _dir_dbg = str(sig.get("direction", "")).upper()
            _lane_dbg = sig.get("_runtime_lane", sig.get("lane", ""))
            _lp_dbg = sig.get("_lane_profile", "")
            _lps_dbg = sig.get("_lane_profile_stage", "")
            _sym_dbg = sig.get("symbol", sig.get("Symbol", ""))
            _ai_pass_dbg = str(sig.get("ai_pass", ""))
            _notify_dbg = str(sig.get("_notify_pass", "0"))
            _th_dbg = LONG_AI_TH if _dir_dbg == "LONG" else SHORT_AI_TH if _dir_dbg == "SHORT" else AI_TH

            logging.warning(
                f"[AI-TRACE] side={_dir_dbg} symbol={_sym_dbg} "
                f"score={_score_dbg} ai_prob={_ai_prob_dbg} th={_th_dbg} "
                f"ai_pass={_ai_pass_dbg} notify={_notify_dbg} "
                f"lane={_lane_dbg} lane_profile={_lp_dbg} lane_stage={_lps_dbg}"
            )
        except Exception as e:
            logging.warning(f"[AI-TRACE-ERR] {e}")

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

        fp = _feature_profile_log_parts(sig)
        feature_gate = fp["FEATURE_GATE"]
        allow_hits = fp["ALLOW_HITS"]
        reject_hits = fp["REJECT_HITS"]
        force_hits = fp["FORCE_HITS"]
        bypass_profile = fp["BYPASS_PROFILE"]

        if direction == "SHORT" and ai_pass == "1":
            print(
                f"[V2] SHORT-SELECTED: {sig['symbol']} {sig['direction']} "
                f"total={sig['total_score']:.2f} "
                f"P1={sig['p1_score']:.2f} "
                f"QS={sig.get('ai_prob_win', '')} "
                f"TYPE={sig.get('ai_model_type', '')} "
                f"NOTIFY={notify_pass} "
                f"NOTIFY_REASON={notify_reason} "
                f"FEATURE_GATE={feature_gate} "
                f"ALLOW_HITS={allow_hits} "
                f"REJECT_HITS={reject_hits} "
                f"FORCE_HITS={force_hits} "
                f"BYPASS_PROFILE={bypass_profile}"
            )
        elif direction == "SHORT" and ai_pass == "0":
            print(
                f"[V2] SHORT-REJECTED: {sig['symbol']} {sig['direction']} "
                f"BAND={sig.get('ai_band', '')} "
                f"FEATURE_GATE={feature_gate} "
                f"ALLOW_HITS={allow_hits} "
                f"REJECT_HITS={reject_hits} "
                f"FORCE_HITS={force_hits} "
                f"BYPASS_PROFILE={bypass_profile} "
                f"NOTE={sig.get('ai_note', '')}"
            )
        elif direction == "LONG" and ai_pass == "1":
            print(
                f"[V2] LONG-SELECTED: {sig['symbol']} {sig['direction']} "
                f"total={sig['total_score']:.2f} "
                f"P1={sig['p1_score']:.2f} "
                f"QS={sig.get('ai_prob_win', '')} "
                f"TYPE={sig.get('ai_model_type', '')} "
                f"NOTIFY={notify_pass} "
                f"NOTIFY_REASON={notify_reason} "
                f"FEATURE_GATE={feature_gate} "
                f"ALLOW_HITS={allow_hits} "
                f"REJECT_HITS={reject_hits} "
                f"FORCE_HITS={force_hits} "
                f"BYPASS_PROFILE={bypass_profile}"
            )
        elif direction == "LONG" and ai_pass == "0":
            print(
                f"[V2] LONG-REJECTED: {sig['symbol']} {sig['direction']} "
                f"BAND={sig.get('ai_band', '')} "
                f"FEATURE_GATE={feature_gate} "
                f"ALLOW_HITS={allow_hits} "
                f"REJECT_HITS={reject_hits} "
                f"FORCE_HITS={force_hits} "
                f"BYPASS_PROFILE={bypass_profile} "
                f"NOTE={sig.get('ai_note', '')}"
            )

    _v2tm("after_signal_selection")
    if engine_mode == "v2_live":
        # 特徴量の掛け合わせ条件に一致したものだけを通知候補にする。
        # ただし、feature_probe_only / rank_exceeded / selected=false は本番通知しない。
        # 目的:
        #   - feature profile route は維持する
        #   - v2_core はまだ本番復活させない
        #   - probe専用候補やrank落ち候補が notify_sent=1 に混ざるのを防ぐ
        def _v2_feature_live_guard_reason(sig: Dict[str, Any]) -> str:
            note_raw = str(sig.get("ai_note", "") or "")
            note = note_raw.lower().replace(" ", "")

            feature_probe_flag = str(sig.get("feature_probe_only", "") or "").strip().lower()
            if (
                "feature_probe_only=1" in note
                or feature_probe_flag in ("1", "true", "yes", "on")
            ):
                return "feature_probe_only"

            if "reason=rank_exceeded" in note or "rank_exceeded" in note:
                return "rank_exceeded"

            if "selected=false" in note or "selected=0" in note:
                return "selected_false"

            if "selected=true" not in note and "selected=1" not in note:
                return "selected_missing"

            return ""

        feature_only_candidates = []
        feature_guard_blocked_n = 0

        for x in final_signals:
            is_feature_hit = str(x.get("_feature_bypass", "0")) == "1"

            if is_feature_hit:
                guard_reason = _v2_feature_live_guard_reason(x)

                if guard_reason == "":
                    x["_notify_pass"] = "1"
                    x["_notify_reason"] = "feature_profile_only_selected"
                    feature_only_candidates.append(x)
                else:
                    feature_guard_blocked_n += 1
                    x["_notify_pass"] = "0"
                    x["_notify_reason"] = f"feature_profile_selected_guard:{guard_reason}"
                    cur_note = str(x.get("ai_note", "") or "")
                    add_note = (
                        "notify_sent=0;"
                        f"notify_send_reason=feature_profile_selected_guard;"
                        f"selected_guard_reason={guard_reason}"
                    )
                    x["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note
            else:
                x["_notify_pass"] = "0"
                cur_note = str(x.get("ai_note", "") or "")
                add_note = "notify_sent=0;notify_send_reason=feature_profile_only_no_match"
                x["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note

        notify_candidates = feature_only_candidates

        print(
            f"[V2] feature_selected_guard "
            f"feature_candidates={len(feature_only_candidates)} "
            f"feature_guard_blocked={feature_guard_blocked_n} "
            f"final_signals={len(final_signals)}"
        )

        sent_n, runtime_dedup_skipped_n = send_v2_live_discord_alerts(notify_candidates)
        _v2tm("after_live_discord")

        if V2_WATCH_NOTIFY_ENABLE:
            watch_sent_n, watch_runtime_dedup_skipped_n = send_v2_watch_discord_alerts(final_signals)
        else:
            watch_sent_n, watch_runtime_dedup_skipped_n = 0, 0
        _v2tm("after_watch_discord")

        print(
            f"[V2] engine_mode=v2_live "
            f"notify_candidates={len(notify_candidates)} "
            f"discord_sent={sent_n} "
            f"discord_runtime_dedup_skipped={runtime_dedup_skipped_n} "
            f"watch_discord_sent={watch_sent_n} "
            f"watch_runtime_dedup_skipped={watch_runtime_dedup_skipped_n} "
            f"notify_ignore_runtime_dedup={int(V2_NOTIFY_IGNORE_RUNTIME_DEDUP)} "
            f"watch_notify_enable={int(V2_WATCH_NOTIFY_ENABLE)} "
            f"feature_only_mode=1"
        )
    else:
        sent_n, runtime_dedup_skipped_n = 0, 0
        watch_sent_n, watch_runtime_dedup_skipped_n = 0, 0
        for sig in final_signals:
            sig["_notify_sent"] = "0"
            sig["_notified_at_jst"] = ""
            cur_note = str(sig.get("ai_note", "") or "")
            add_note = "notify_sent=0;notify_send_reason=engine_not_v2_live"
            sig["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note

    # シート書き込み（選抜通過もREJECTも全部書く）
    # 重要: 送信結果を ai_note に反映したあとで書く
    if V2_SHADOW_WRITE_ENABLE:
        rows = [v2_build_shadow_row(sig) for sig in final_signals]
        v2_write_shadow_rows(rows)
    else:
        print("[V2] shadow write skipped by V2_SHADOW_WRITE_ENABLE=0")

    _v2tm("before_return")
    return f"V2-shadow: final={len(final_signals)} (short_pass={n_short_pass} short_reject={n_short_reject} long_pass={n_long_pass} long_reject={n_long_reject}) raw={len(raw_signals)}"




# ==========================================
# ロジック本体 (/run)
# ==========================================
def logic_main(force: bool = False):
    global last_alert_records, last_candidate_records, ai_model

    start = time.time()
    now_jst = datetime.now(JST)
    print(f"[RUN] start {now_jst.isoformat()}  VERSION={VERSION} force={force}")

    def _tm(label: str):
        if TIMER_LOG_ENABLE:
            print(f"[TIMER] logic_main {label} sec={time.time() - start:.2f}")

    _tm("enter")

    # self heal（HEALTH_CHECK_TTL_SEC 以内に成功済みならスキップ）
    _hc_key = "self_heal_ok"
    _hc_last = _health_check_cache.get(_hc_key, 0.0)
    if time.time() - _hc_last > HEALTH_CHECK_TTL_SEC:
        ok, msg = self_heal_prerequisites()
        _tm("after_self_heal")
        if not ok:
            send_discord_message(f"[WARN] self_heal_prerequisites failed: {msg}")
            return f"SelfHealFailed: {msg}"
        _health_check_cache[_hc_key] = time.time()
    else:
        _tm("after_self_heal")

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
    _tm("after_get_ai_model")

    # 15分足の確定直後は取引所側の反映遅れがあるため、通常は「各15分の10分以降」に実行する
    # ...以下、BTCデータの取得へ続く

    # ただし /run?force=1 のときはこの待機をスキップする
    if (not force) and ((now_jst.minute % 15) < 10):
        print(f"[RUN] skip (waiting window) minute={now_jst.minute}")
        return "Waiting..."

    exchange = build_exchange()
    _tm("after_build_exchange")

    if SIGNAL_ENGINE in ("v2_live", "v2", "shadow"):
        try:
            v2_result = v2_shadow_run(exchange, now_jst, force=force)
            print(f"[V2] {v2_result}")
        except Exception as e:
            print(f"[V2-ERR] shadow run crashed: {e}")

        _tm("after_v2_shadow_run_v2_only")

        elapsed = time.time() - start
        print(f"[RUN] done alerts=0 candidates=0 elapsed={elapsed:.2f}s")
        return f"V2OnlyDone: elapsed={elapsed:.2f}s"

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

    def _build_feature_probe_candidates(
        symbol: str,
        btc_htf_probe: Dict[str, Any],
        now_jst_probe: datetime,
        regime_info_probe: Optional[Dict[str, Any]],
        note_suffix: str,
        include_bad_col: str = "",
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for forced_side in ("LONG", "SHORT"):
            probe_sig = None
            try:
                probe_sig = v2_generate_signal(
                    exchange,
                    symbol,
                    btc_htf_probe,
                    now_jst_probe,
                    regime_info=regime_info_probe,
                    feature_probe_only=True,
                    forced_probe_direction=forced_side,
                )
            except Exception as e:
                print(
                    f"[V2] FEATURE-PROBE-BUILD-ERR sym={symbol} side={forced_side} "
                    f"src={note_suffix} err={e}"
                )
                continue

            if probe_sig is None:
                continue

            _apply_feature_profiles(probe_sig)

            if not _is_feature_force_pass(probe_sig):
                continue

            probe_sig["_feature_bypass"] = "1"
            probe_sig["_feature_bypass_profile"] = str(
                probe_sig.get("_feature_force_profile", "") or ""
            )

            cur_note = str(probe_sig.get("ai_note", "") or "")
            add_note = f"feature_probe_source={note_suffix}"
            if include_bad_col:
                add_note += f";bad_col={include_bad_col}"
            probe_sig["ai_note"] = f"{cur_note};{add_note}" if cur_note else add_note

            if not str(probe_sig.get("v2_version", "")).strip():
                probe_sig["v2_version"] = "V2-FeatureProbe"

            probe_sig["note"] = ""
            try:
                probe_sig["btc_ret"] = _safe_float_or_nan(btc_ret)
            except Exception:
                pass
            try:
                probe_sig["btc_vol"] = _safe_float_or_nan(btc_vol)
            except Exception:
                pass

            out.append(probe_sig)

        return out

    raw_signals: List[Dict[str, Any]] = []

    # CAND-SKIP probe 用に btc_htf を V1 BTC データから構築する
    _btc_dir_probe = {"Up": "LONG", "Down": "SHORT"}.get(btc_mode, "NEUTRAL")
    _btc_str_probe = abs(float(btc_1h_change)) if np.isfinite(float(btc_1h_change)) else 0.0
    btc_htf_for_probe: Dict[str, Any] = {
        "direction": _btc_dir_probe,
        "strength": _btc_str_probe,
        "ema_fast": np.nan,
        "ema_slow": np.nan,
        "close": np.nan,
    }

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

                probe_candidates = _build_feature_probe_candidates(
                    symbol=symbol,
                    btc_htf_probe=btc_htf_for_probe,
                    now_jst_probe=now_jst,
                    regime_info_probe=None,
                    note_suffix="pre_candidate_skip",
                    include_bad_col=bad_col,
                )

                for probe_sig in probe_candidates:
                    raw_signals.append(probe_sig)

                    if V2_DEBUG_FEATURE_PROBE:
                        print(
                            f"[V2] FEATURE-PROBE: {probe_sig['symbol']} {probe_sig['direction']} "
                            f"src=pre_candidate_skip bad_col={bad_col} "
                            f"total={probe_sig['total_score']:.2f} "
                            f"P1={probe_sig['p1_score']:.2f} P2={probe_sig['p2_score']:.2f} P3={probe_sig['p3_score']:.2f} "
                            f"RSI={probe_sig.get('rsi', '')} BTCstr={probe_sig.get('btc_htf_strength', '')} "
                            f"NOTE={probe_sig.get('ai_note', '')}"
                        )

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
                probe_candidates = _build_feature_probe_candidates(
                    symbol=symbol,
                    btc_htf_probe=btc_htf_for_probe,
                    now_jst_probe=now_jst,
                    regime_info_probe=None,
                    note_suffix="signal_type_skip",
                )

                for probe_sig in probe_candidates:
                    raw_signals.append(probe_sig)
                    if V2_DEBUG_FEATURE_PROBE:
                        print(
                            f"[V2] FEATURE-PROBE: {probe_sig['symbol']} {probe_sig['direction']} "
                            f"src=signal_type_skip "
                            f"total={probe_sig['total_score']:.2f} "
                            f"P1={probe_sig['p1_score']:.2f} P2={probe_sig['p2_score']:.2f} P3={probe_sig['p3_score']:.2f} "
                            f"RSI={probe_sig.get('rsi', '')} BTCstr={probe_sig.get('btc_htf_strength', '')} "
                            f"NOTE={probe_sig.get('ai_note', '')}"
                        )
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
                side_u = str(side).strip().upper()

                total_score = _safe_float_or_nan(sig_score)
                p1_score = _safe_float_or_nan(sig_score)
                p2_score = 0.0
                p3_score = 0.0

                sym_htf_strength = _safe_float_or_nan(sig_score)

                btc_htf_strength = 0.0
                try:
                    if "BTC_1h_Change" in row and pd.notna(row["BTC_1h_Change"]):
                        btc_htf_strength = abs(float(row["BTC_1h_Change"]))
                except Exception:
                    btc_htf_strength = 0.0

                vol_ratio = 1.0

                atr_val = 0.0
                try:
                    if "Dynamic_Sigma" in row and pd.notna(row["Dynamic_Sigma"]):
                        atr_val = abs(float(row["Dynamic_Sigma"]))
                except Exception:
                    atr_val = 0.0

                rsi_val = _safe_float_or_nan(float(row["RSI"])) if "RSI" in row and pd.notna(row["RSI"]) else np.nan

                time_ms_val = row.get("Time", row.get("time", None))
                hour_jst_val = np.nan
                try:
                    if time_ms_val is not None and str(time_ms_val).strip() != "":
                        hour_jst_val = datetime.fromtimestamp(float(time_ms_val) / 1000.0, JST).hour
                except Exception:
                    hour_jst_val = np.nan

                ltf_aligned_val = 0.0
                fr_available_val = 0.0
                vol_confirmed_val = 0.0

                btc_mode_u = str(btc_mode).strip().upper()
                btc_mode_up = 1.0 if btc_mode_u == "UP" else 0.0
                btc_mode_range = 1.0 if btc_mode_u == "RANGE" else 0.0
                btc_mode_down = 1.0 if btc_mode_u == "DOWN" else 0.0

                feats = pd.DataFrame([{
                    "TotalScore": total_score,
                    "P1_TrendScore": p1_score,
                    "P2_FundingScore": p2_score,
                    "P3_VolumeScore": p3_score,
                    "SymHTF_Strength": sym_htf_strength,
                    "BTC_HTF_Strength": btc_htf_strength,
                    "VolRatio": vol_ratio,
                    "ATR": atr_val,
                    "RSI": rsi_val,
                    "Hour_JST": hour_jst_val,
                    "LTF_Aligned": ltf_aligned_val,
                    "FR_Available": fr_available_val,
                    "VolConfirmed": vol_confirmed_val,
                    "BTC_Mode_UP": btc_mode_up,
                    "BTC_Mode_RANGE": btc_mode_range,
                    "BTC_Mode_DOWN": btc_mode_down,
                }])

                feats = feats.replace([np.inf, -np.inf], np.nan)

                mask = feats.isna()
                if mask.any().any():
                    bad_cols = [c for c in feats.columns if mask[c].any()]
                    try:
                        print(
                            f"[AI-FEAT-SKIP] sym={symbol} side={side_u} "
                            f"bad_cols={bad_cols} "
                            f"btc_mode={btc_mode} "
                            f"time_ms={time_ms_val} "
                            f"rsi={row.get('RSI', np.nan)} "
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
                btc_mode_compat_lane = btc_mode.upper()
                hour_jst_lane = now_jst.hour
                pseudo_sig = {
                    "symbol": symbol,
                    "direction": side,
                    "btc_mode_compat": btc_mode_compat_lane,
                    "rsi": float(row["RSI"]),
                    "p1_score": float(sig_score),
                    "p2_score": 0.0,
                    "hour_jst": hour_jst_lane,
                    "note": "",
                    "ai_note": "",
                }
                lane_name = _get_runtime_lane_for_signal(pseudo_sig, side)

                model_for_side, model_ver_side, model_src_side = get_ai_model_for_side(
                    side, sym_code, lane=lane_name
                )
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

                        _time_ms_for_long_ai = row.get("Time", row.get("time", None))
                        _hour_fallback_for_long_ai = np.nan

                        try:
                            if _time_ms_for_long_ai is not None and str(_time_ms_for_long_ai).strip() != "":
                                _hour_fallback_for_long_ai = datetime.fromtimestamp(
                                    float(_time_ms_for_long_ai) / 1000.0,
                                    JST,
                                ).hour
                        except Exception:
                            _hour_fallback_for_long_ai = np.nan

                        long_ai_sig = {
                            "symbol": symbol.replace("/USDT", ""),
                            "p1_score": np.nan,
                            "p2_score": np.nan,
                            "rsi": float(row["RSI"]) if "RSI" in row and pd.notna(row["RSI"]) else np.nan,
                            "btc_mode_compat": str(btc_mode),
                            "hour_jst": _hour_fallback_for_long_ai,
                            "fr_available": "",
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

                        short_core_ok, short_core_reason = _check_short_win_bucket(short_ai_sig)
                        short_push_ok = _is_short_strong_hour_bucket(short_ai_sig)

                        # 強いSHORTは AI hard gate を bypass
                        ai_th_effective = 0.0
                        if short_push_ok and V2_SHORT_PAUSE_ALLOW_STRONG_BUCKET:
                            ai_pass = True
                        else:
                            ai_pass = bool(short_core_ok)

                        dbg["short_ai_mode"] = "soft"
                        dbg["short_ai_core_ok"] = bool(short_core_ok)
                        dbg["short_ai_core_reason"] = str(short_core_reason)
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
                proba_raw_val = ai_proba_used_val if ai_proba_used_val != "" else ""
                proba_used_val = ai_proba_used_val if ai_proba_used_val != "" else ""
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
                "proba_used": proba_used_val if proba_used_val != "" else ai_proba_used_val,
                "invert_applied": invert_applied_val,
                "is_flip": bool(flipped),

                "chg_pct": chg_pct_val,
                "vol_ratio": vol_ratio_val,

                # learn_log(AP..AS) 用
                "ai_proba_base": ai_proba_base_val,
                "ai_proba_flip": ai_proba_flip_val,
                "ai_proba_used": ai_proba_used_val if ai_proba_used_val != "" else proba_used_val,
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

                # --- lane profile 用の runtime 特徴量を保持 ---
                "p1_score": row.get("P1_TrendScore", row.get("p1_score", np.nan)),
                "p2_score": row.get("P2_FundingScore", row.get("p2_score", np.nan)),
                "p3_score": row.get("P3_VolumeScore", row.get("p3_score", np.nan)),
                "hour_jst": row.get("Hour_JST", row.get("hour_jst", now_jst.hour)),
                "fr_available": row.get("FR_Available", row.get("fr_available", "")),
                "vol_confirmed": row.get("VolConfirmed", row.get("vol_confirmed", "")),
                "btc_mode_compat": row.get("BTC_Mode_Compat", row.get("btc_mode_compat", btc_mode)),
                "ai_prob_win": ai_proba_used_val if ai_proba_used_val != "" else ai_score,

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
                f" ai_pass={_is_ai_pass_true(item.get('ai_pass'))}"
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
                f" ai_th={ai_th_effective} ai_pass={_is_ai_pass_true(item.get('ai_pass'))}"
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
            DEFAULT_LEV, "", "", _is_ai_pass_true(item.get("ai_pass")), bool(BTC_CALM),
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

        audit_block = _build_signal_audit_block(item)

        msg = (
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{icon} **{d_str}** {icon}\n"
            f"{VERSION} [{RULE_VERSION_TAG}]\n"
            f"💎 {sym} ({item['type']})\n"
            f"📈 Score:{item['score']:.2f}σ  AI:{ai_disp}\n"
            f"🧭 Regime:{regime_mode}  Tier:{notify_tier}\n"
            f"🟦 BTC:{btc_mode_disp} 1h:{btc_1h_disp}  Calm:{btc_calm_disp}  BTC_OK:{btc_ok}\n"
            f"💰 {cp:.4f}\n"
            f"🎯 TP: {tp:.4f} ({tp_pct:.2f}%) HL:{hl_tp_pct:.1f}%\n"
            f"🛑 SL: {sl:.4f} ({sl_pct:.2f}%) HL:{hl_sl_pct:.1f}%\n"
            f"\n"
            f"{audit_block}\n"
            "━━━━━━━━━━━━━━━━━━━━"
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
        _tm("after_v2_shadow_run")

    elapsed = time.time() - start
    print(f"[RUN] done alerts={count} candidates={len(candidate_rows)} elapsed={elapsed:.2f}s")
    print(f"[DBG] pending_candidates={len(pending_candidates)} pending_alerts={len(pending_alerts)} "
          f"BTC_CALM={BTC_CALM} btc_mode={btc_mode} median_sigma={median_sigma} btc_ok={btc_ok}")
    _tm("before_return")
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
    global ai_model_long, ai_model_short
    global AI_MODEL_VERSION_RUNTIME_LONG, AI_MODEL_SOURCE_RUNTIME_LONG
    global AI_MODEL_VERSION_RUNTIME_SHORT, AI_MODEL_SOURCE_RUNTIME_SHORT

    side_u = str(side or "").strip().upper()

    if side_u not in ("LONG", "SHORT"):
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

        lookback_arg = str(request.args.get("lookback", str(TRAIN_V2_LOOKBACK_ROWS))).strip()
        lookback = int(float(lookback_arg)) if lookback_arg else int(TRAIN_V2_LOOKBACK_ROWS)

        lookback_days_arg = str(request.args.get("lookback_days", str(TRAIN_V2_LOOKBACK_DAYS))).strip()
        lookback_days: Optional[float] = None
        if lookback_days_arg not in ("", "0", "0.0"):
            lookback_days = float(lookback_days_arg)

        lane = str(request.args.get("lane", "")).strip().lower()
        lane_u = _normalize_lane_name(lane)

        default_min_samples = TRAIN_V2_LONG_MIN_SAMPLES if side_u == "LONG" else TRAIN_V2_SHORT_MIN_SAMPLES
        min_samples = int(float(request.args.get("min_samples", str(default_min_samples))))
        hot_reload = str(request.args.get("hot_reload", "1")).strip() == "1"
        upload = str(request.args.get("upload", "1")).strip() == "1"
        out_version = str(request.args.get("version", "")).strip()

        X, y, info = _build_training_dataset_from_v2_shadow_ai(
            side_u,
            lookback,
            lookback_days=lookback_days,
            lane=lane_u,
        )

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

        new_eval = _evaluate_binary_model(model, X_te, y_te)

        train_feature_columns = [str(c) for c in list(X.columns)]
        train_feature_hash = _feature_columns_hash(train_feature_columns)
        serving_feature_columns = _get_v2_serving_feature_columns()
        serving_feature_hash = _feature_columns_hash(serving_feature_columns)
        serving_schema_match = (train_feature_columns == serving_feature_columns)

        old_model, old_ver, old_src = get_ai_model_for_side(side_u, "")
        old_eval = {
            "ok": False,
            "skipped": True,
            "reason": "compare_disabled_or_not_enough_test_rows",
            "accuracy": None,
            "auc": None,
            "n_rows": int(len(X_te)),
            "expected_cols": None,
            "expected_n_features": None,
            "schema_version": "",
            "feature_hash": "",
        }

        if V2_RETRAIN_COMPARE_ENABLE and int(len(X_te)) >= int(V2_RETRAIN_MIN_TEST_ROWS):
            old_eval = _evaluate_binary_model(old_model, X_te, y_te)

        acc = new_eval.get("accuracy")
        auc = new_eval.get("auc")

        ts_ver_raw = out_version if out_version else datetime.now(JST).strftime(f"v2_{side_u.lower()}_%Y%m%d_%H%M%S")
        ts_ver = _sanitize_version_tag(ts_ver_raw) or datetime.now(JST).strftime(f"v2_{side_u.lower()}_%Y%m%d_%H%M%S")

        model_meta = {
            "side": side_u,
            "model_version": ts_ver,
            "trainer_name": trainer_name,
            "trainer_params": trainer_params,
            "feature_schema_version": str(V2_TRAIN_FEATURE_SCHEMA_VERSION),
            "feature_columns": list(train_feature_columns),
            "feature_hash": str(train_feature_hash),
            "trained_samples": int(n),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "class_balance": dict(info.get("class_balance", {}) or {}),
            "train_filters": dict(info.get("train_filters", {}) or {}),
            "build_info": dict(info.get("build_info", {}) or {}),
            "serving_compatibility": {
                "runtime_feature_schema_version": str(V2_RUNTIME_FEATURE_SCHEMA_VERSION),
                "runtime_feature_columns": list(serving_feature_columns),
                "runtime_feature_hash": str(serving_feature_hash),
                "serving_schema_match": bool(serving_schema_match),
            },
            "created_at_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
        }

        decision: Dict[str, Any] = {
            "promote": True,
            "reason": "ok",
            "compare_enabled": bool(V2_RETRAIN_COMPARE_ENABLE),
            "serving_schema_match": bool(serving_schema_match),
            "train_feature_schema_version": str(V2_TRAIN_FEATURE_SCHEMA_VERSION),
            "runtime_feature_schema_version": str(V2_RUNTIME_FEATURE_SCHEMA_VERSION),
            "train_feature_hash": str(train_feature_hash),
            "runtime_feature_hash": str(serving_feature_hash),
            "auc_diff_vs_old": None,
            "blocked_by_schema": False,
        }

        if V2_RETRAIN_REQUIRE_SERVING_SCHEMA_MATCH and not serving_schema_match:
            decision["promote"] = False
            decision["blocked_by_schema"] = True
            decision["reason"] = "training_schema_mismatch_with_runtime_serving"

        elif not bool(new_eval.get("ok")):
            decision["promote"] = False
            decision["reason"] = f"new_model_eval_failed:{new_eval.get('reason', '')}"

        elif V2_RETRAIN_COMPARE_ENABLE and int(len(X_te)) >= int(V2_RETRAIN_MIN_TEST_ROWS):
            old_auc = old_eval.get("auc")
            new_auc = new_eval.get("auc")

            if bool(old_eval.get("ok")) and (old_auc is not None) and (new_auc is not None):
                auc_diff = float(new_auc) - float(old_auc)
                decision["auc_diff_vs_old"] = auc_diff
                if auc_diff < (-1.0 * float(V2_RETRAIN_MAX_AUC_DROP)):
                    decision["promote"] = False
                    decision["reason"] = f"auc_regression:{auc_diff:.6f}"
            elif bool(old_eval.get("skipped")):
                decision["reason"] = f"compare_skipped:{old_eval.get('reason', '')}"

        tmp_local = f"/tmp/{side_u.lower()}_model_{ts_ver}.pkl"
        tmp_meta_local = f"/tmp/{side_u.lower()}_model_{ts_ver}.meta.json"

        out_uri = ""
        meta_uri = ""
        uploaded = False
        meta_uploaded = False
        reloaded = False

        if decision["promote"] and upload:
            try:
                joblib.dump(model, tmp_local)
            except Exception as e:
                return jsonify({"ok": False, "error": f"joblib.dump failed: {e}", "version": VERSION}), 500

            ok_json, json_msg = _write_json_file(tmp_meta_local, model_meta)
            if not ok_json:
                return jsonify({"ok": False, "error": f"meta json write failed: {json_msg}", "version": VERSION}), 500

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

            meta_uri = _model_meta_uri_for_uri(out_uri)
            ok_meta_up, meta_up_msg = _gcs_upload_from(tmp_meta_local, meta_uri)
            if not ok_meta_up:
                return jsonify({
                    "ok": False,
                    "error": f"GCS meta upload failed: {meta_uri} ({meta_up_msg})",
                    "version": VERSION,
                }), 500

            uploaded = True
            meta_uploaded = True

        model = _attach_model_meta(model, model_meta)

        if decision["promote"] and hot_reload:
            try:
                if side_u == "LONG":
                    ai_model_long = model
                    AI_MODEL_VERSION_RUNTIME_LONG = ts_ver
                    AI_MODEL_SOURCE_RUNTIME_LONG = "trained_v2_long" if uploaded else "trained_v2_long(no-upload)"
                else:
                    ai_model_short = model
                    AI_MODEL_VERSION_RUNTIME_SHORT = ts_ver
                    AI_MODEL_SOURCE_RUNTIME_SHORT = "trained_v2_short" if uploaded else "trained_v2_short(no-upload)"
                reloaded = True
            except Exception as e:
                print(f"[TRAIN_V2_{side_u}] hot_reload failed: {e}")
                reloaded = False

        next_env_vars = {}
        if decision["promote"] and uploaded:
            if lane == "long_trend":
                next_env_vars = {
                    "LONG_TREND_MODEL_VERSION": ts_ver,
                    "LONG_TREND_MODEL_GCS_URI": out_uri,
                }
            elif lane == "long_recovery":
                next_env_vars = {
                    "LONG_RECOVERY_MODEL_VERSION": ts_ver,
                    "LONG_RECOVERY_MODEL_GCS_URI": out_uri,
                }
            elif lane == "short_trend":
                next_env_vars = {
                    "SHORT_TREND_MODEL_VERSION": ts_ver,
                    "SHORT_TREND_MODEL_GCS_URI": out_uri,
                }
            elif lane == "short_retrace":
                next_env_vars = {
                    "SHORT_RETRACE_MODEL_VERSION": ts_ver,
                    "SHORT_RETRACE_MODEL_GCS_URI": out_uri,
                }
            elif side_u == "LONG":
                next_env_vars = {
                    "LONG_MODEL_VERSION": ts_ver,
                    "LONG_MODEL_GCS_URI": out_uri,
                }
            elif side_u == "SHORT":
                next_env_vars = {
                    "SHORT_MODEL_VERSION": ts_ver,
                    "SHORT_MODEL_GCS_URI": out_uri,
                }

        try:
            msg_lines = [
                f"[TRAIN_V2_{side_u}:{lane or 'side_default'}] decision={decision['reason']}",
                f"promote={decision['promote']}",
                f"serving_schema_match={decision['serving_schema_match']}",
                f"train_schema={V2_TRAIN_FEATURE_SCHEMA_VERSION}",
                f"runtime_schema={V2_RUNTIME_FEATURE_SCHEMA_VERSION}",
                f"new_auc={new_eval.get('auc')}",
                f"old_auc={old_eval.get('auc')}",
                f"auc_diff={decision.get('auc_diff_vs_old')}",
                f"uploaded={uploaded}",
                f"hot_reloaded={reloaded}",
                f"version={ts_ver}",
            ]
            send_discord_message("\n".join(msg_lines))
        except Exception as e:
            print(f"[TRAIN_V2_{side_u}] discord notify failed: {type(e).__name__}: {e}")

        return jsonify({
            "ok": True,
            "version": VERSION,
            "side": side_u,
            "lane": lane,
            "trained_samples": n,
            "trainer_name": trainer_name,
            "trainer_params": trainer_params,
            "metrics": {
                "accuracy": acc,
                "auc": auc,
            },
            "info": info,
            "old_model_eval": old_eval,
            "new_model_eval": new_eval,
            "decision": decision,
            "new_model": {
                "model_version_suggest": ts_ver,
                "model_gcs_uri": out_uri,
                "model_meta_gcs_uri": meta_uri,
                "uploaded": bool(uploaded),
                "meta_uploaded": bool(meta_uploaded),
                "hot_reloaded_in_this_instance": bool(reloaded),
                "feature_schema_version": str(model_meta.get("feature_schema_version", "")),
                "feature_hash": str(model_meta.get("feature_hash", "")),
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


def judge_v2_main(run_backfill: bool = False) -> str:
    recent_total = judge_v2_sheet(
        V2_SHADOW_SHEET,
        lookback_rows=V2_JUDGE_LOOKBACK_ROWS,
        max_judge=V2_JUDGE_MAX_ROWS,
        priority_bands=V2_JUDGE_PRIORITY_BANDS,
        include_deferred=True,
        oldest_first=False,
    )

    backfill_total = 0
    if run_backfill and V2_JUDGE_BACKFILL_ENABLE and V2_JUDGE_BACKFILL_MAX_ROWS > 0:
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

            run_backfill = str(request.args.get("backfill", "0")).strip() == "1"
            return str(judge_v2_main(run_backfill=run_backfill)), 200
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



















