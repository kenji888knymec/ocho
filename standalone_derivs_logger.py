#!/usr/bin/env python3
"""
standalone_derivs_logger.py
============================
【完全独立・本番非接続・record-only ロガー】

OKX無料APIから Funding / OI / LSR / 価格 をスナップショット取得し
ローカルCSVに追記するだけ。

実行:
  python3 standalone_derivs_logger.py once   # launchd向け・推奨（1回スナップして終了）
  python3 standalone_derivs_logger.py loop   # 手動デバッグ用（1時間ごとにループ）

保存先 : ~/Downloads/derivs_log/derivs_log.csv  （追記）
ログ   : ~/Downloads/derivs_log/derivs_log.err  （エラー追記）

禁止事項（厳守）:
  - main.py を import しない / 参照しない
  - Discord 通知なし
  - Entry 判定なし
  - スプレッドシート書き込みなし
  - 本番変更なし / deploy なし / merge なし
  - 欠損値は必ず空欄（0補完・平均補完・推測値補完 絶対禁止）

launchd 設定（手動テスト確認後にセットアップ）:
  → このファイル末尾のコメント参照
"""
import sys, ssl, json, time, csv
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ─── 定数 ────────────────────────────────────────────────────────────────
BASE = "https://www.okx.com"
CTX  = ssl.create_default_context()
HDRS = {"User-Agent": "Mozilla/5.0"}

# (instId, ccy for LSR, instId for index ticker)
SYMBOLS = [
    ("BTC-USDT-SWAP",  "BTC",  "BTC-USDT"),
    ("ETH-USDT-SWAP",  "ETH",  "ETH-USDT"),
    ("SOL-USDT-SWAP",  "SOL",  "SOL-USDT"),
    ("DOGE-USDT-SWAP", "DOGE", "DOGE-USDT"),
    ("BONK-USDT-SWAP", "BONK", "BONK-USDT"),
]

LOG_DIR  = Path.home() / "Downloads" / "derivs_log"
CSV_PATH = LOG_DIR / "derivs_log.csv"
ERR_PATH = LOG_DIR / "derivs_log.err"

HEADER = [
    "snapshot_utc", "snapshot_jst", "hour_bucket_utc", "symbol",
    "last_price", "mark_price", "index_price",
    "funding_rate", "next_funding_time_utc", "predicted_funding_rate",
    "open_interest", "open_interest_ccy",
    "lsr_account",
]


# ─── ユーティリティ ───────────────────────────────────────────────────────
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _floor_hour(dt: datetime) -> str:
    """YYYY-MM-DDTHH:00:00Z 形式の1h床値"""
    return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00:00Z")


def _log_err(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
    with ERR_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"  [ERR] {msg}", file=sys.stderr)


def _get(path: str, params: dict, retries: int = 3):
    """OKX API GET。成功→(data list, None)、失敗→(None, エラー文字列)"""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}{path}?{qs}" if qs else f"{BASE}{path}"
    last_err = "未実行"
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                d = json.loads(r.read().decode())
            if str(d.get("code")) != "0":
                return None, f"code={d.get('code')} msg={d.get('msg')}"
            return d.get("data", []), None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if i < retries - 1:
                time.sleep(1.5 ** i)
    return None, last_err


def _sf(v) -> str:
    """数値文字列化。None/空/変換不可 → ''（0補完・推測値補完は絶対にしない）"""
    if v is None or v == "":
        return ""
    try:
        return str(float(v))
    except Exception:
        return ""


def _ms_to_utc(ms_str) -> str:
    try:
        dt = datetime.fromtimestamp(int(ms_str) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


# ─── 各API取得関数 ────────────────────────────────────────────────────────
def fetch_prices(inst_id: str, idx_inst_id: str) -> dict:
    r = {"last_price": "", "mark_price": "", "index_price": ""}

    d, e = _get("/api/v5/market/ticker", {"instId": inst_id})
    if d:
        r["last_price"] = _sf(d[0].get("last"))
    else:
        _log_err(f"ticker {inst_id}: {e}")
    time.sleep(0.12)

    d, e = _get("/api/v5/public/mark-price", {"instId": inst_id})
    if d:
        r["mark_price"] = _sf(d[0].get("markPx"))
    else:
        _log_err(f"mark-price {inst_id}: {e}")
    time.sleep(0.12)

    # index_price: BONKなど非対応銘柄は空欄になる（エラーを記録するが止めない）
    d, e = _get("/api/v5/market/index-tickers", {"instId": idx_inst_id})
    if d:
        r["index_price"] = _sf(d[0].get("idxPx"))
    else:
        _log_err(f"index-tickers {idx_inst_id}: {e}")
    time.sleep(0.12)

    return r


def fetch_funding(inst_id: str) -> dict:
    r = {"funding_rate": "", "next_funding_time_utc": "", "predicted_funding_rate": ""}
    d, e = _get("/api/v5/public/funding-rate", {"instId": inst_id})
    if d:
        r["funding_rate"]           = _sf(d[0].get("fundingRate"))
        r["next_funding_time_utc"]  = _ms_to_utc(d[0].get("nextFundingTime", ""))
        r["predicted_funding_rate"] = _sf(d[0].get("nextFundingRate"))
    else:
        _log_err(f"funding-rate {inst_id}: {e}")
    time.sleep(0.12)
    return r


def fetch_oi(inst_id: str) -> dict:
    r = {"open_interest": "", "open_interest_ccy": ""}
    d, e = _get("/api/v5/public/open-interest", {"instId": inst_id})
    if d:
        r["open_interest"]     = _sf(d[0].get("oi"))
        r["open_interest_ccy"] = _sf(d[0].get("oiCcy"))
    else:
        _log_err(f"open-interest {inst_id}: {e}")
    time.sleep(0.12)
    return r


def fetch_lsr(ccy: str) -> dict:
    r = {"lsr_account": ""}
    d, e = _get("/api/v5/rubik/stat/contracts/long-short-account-ratio",
                {"ccy": ccy, "period": "5m"})
    if d:
        row = d[0]
        # OKX返り値は [timestamp_ms, ratio] のリスト形式
        if isinstance(row, list) and len(row) >= 2:
            r["lsr_account"] = _sf(row[1])
        elif isinstance(row, dict):
            r["lsr_account"] = _sf(row.get("longShortRatio") or row.get("ratio") or "")
    else:
        _log_err(f"lsr {ccy}: {e or 'empty'}")
    time.sleep(0.12)
    return r


# ─── 重複チェック ─────────────────────────────────────────────────────────
def already_recorded(hour_bucket: str, symbol: str) -> bool:
    """同一 hour_bucket × symbol が既にCSVに存在するか（末尾300行を走査）"""
    if not CSV_PATH.exists():
        return False
    try:
        with CSV_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-300:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and parts[2] == hour_bucket and parts[3] == symbol:
                return True
    except Exception:
        pass
    return False


# ─── スナップショット本体 ─────────────────────────────────────────────────
def take_snapshot() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    now      = _now_utc()
    snap_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    snap_jst = (now + timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
    hour_bkt = _floor_hour(now)
    need_hdr = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    written  = 0

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_hdr:
            w.writerow(HEADER)
            print(f"  → ヘッダー書き込み: {CSV_PATH}")

        for inst_id, ccy, idx_id in SYMBOLS:
            if already_recorded(hour_bkt, inst_id):
                print(f"  [skip] {inst_id} ({hour_bkt} 既記録)")
                continue

            print(f"  {inst_id} ...", end=" ", flush=True)
            prices  = fetch_prices(inst_id, idx_id)
            funding = fetch_funding(inst_id)
            oi      = fetch_oi(inst_id)
            lsr     = fetch_lsr(ccy)

            row = [
                snap_utc, snap_jst, hour_bkt, inst_id,
                prices["last_price"],
                prices["mark_price"],
                prices["index_price"],
                funding["funding_rate"],
                funding["next_funding_time_utc"],
                funding["predicted_funding_rate"],
                oi["open_interest"],
                oi["open_interest_ccy"],
                lsr["lsr_account"],
            ]
            w.writerow(row)
            f.flush()
            written += 1
            print(
                f"last={prices['last_price']}, "
                f"fund={funding['funding_rate']}, "
                f"oi={oi['open_interest']}, "
                f"lsr={lsr['lsr_account']}"
            )

    print(f"\n  完了: {written}行追記 → {CSV_PATH}")
    return written


# ─── エントリポイント ─────────────────────────────────────────────────────
def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "once")

    if mode not in ("once", "loop"):
        print(f"使い方: python3 {Path(__file__).name} [once|loop]")
        print("  once : 1回スナップして終了（launchd向け・推奨）")
        print("  loop : 1時間ごとにループ（手動デバッグ用）")
        sys.exit(1)

    if mode == "loop":
        print("loop モード — 1時間ごとに実行します (Ctrl+C で停止)")
        while True:
            print(f"\n[{_now_utc().strftime('%Y-%m-%dT%H:%M:%SZ')}] スナップ開始")
            try:
                take_snapshot()
            except Exception as e:
                _log_err(f"loop snapshot error: {e}")
            now    = _now_utc()
            next_h = (now.replace(minute=0, second=0, microsecond=0)
                      + timedelta(hours=1, seconds=30))
            wait   = max(0.0, (next_h - _now_utc()).total_seconds())
            print(f"  次回: {next_h.strftime('%H:%M:%S')} UTC (待機 {wait:.0f}秒)")
            time.sleep(wait)

    else:  # once
        print(f"[{_now_utc().strftime('%Y-%m-%dT%H:%M:%SZ')}] スナップ (once)")
        try:
            take_snapshot()
        except Exception as e:
            _log_err(f"once snapshot error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# launchd セットアップ手順（手動テストが成功してから実施）
# ═══════════════════════════════════════════════════════════════════════════
#
# ① まず手動テスト
#   python3 ~/Downloads/standalone_derivs_logger.py once
#   → ~/Downloads/derivs_log/derivs_log.csv に1行追記されることを確認
#   → ~/Downloads/derivs_log/derivs_log.err が空か確認
#
# ② plist ファイルを作成
#   nano ~/Library/LaunchAgents/com.derivs.logger.plist
#
#   ---- ここから ----
#   <?xml version="1.0" encoding="UTF-8"?>
#   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
#     "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
#   <plist version="1.0">
#   <dict>
#     <key>Label</key>
#     <string>com.derivs.logger</string>
#
#     <key>ProgramArguments</key>
#     <array>
#       <string>/usr/bin/python3</string>
#       <string>/Users/kenjinakamura/Downloads/standalone_derivs_logger.py</string>
#       <string>once</string>
#     </array>
#
#     <key>StartInterval</key>
#     <integer>3600</integer>
#
#     <key>RunAtLoad</key>
#     <false/>
#
#     <key>StandardOutPath</key>
#     <string>/Users/kenjinakamura/Downloads/derivs_log/launchd_stdout.log</string>
#
#     <key>StandardErrorPath</key>
#     <string>/Users/kenjinakamura/Downloads/derivs_log/launchd_stderr.log</string>
#   </dict>
#   </plist>
#   ---- ここまで ----
#
# ③ 登録・開始
#   launchctl load ~/Library/LaunchAgents/com.derivs.logger.plist
#   launchctl start com.derivs.logger   # 即時テスト実行
#
# ④ 停止（不要になったとき）
#   launchctl unload ~/Library/LaunchAgents/com.derivs.logger.plist
#
# 注意:
#   - StartInterval=3600 は「3600秒ごとに起動」。Macがスリープ中は飛ぶが許容。
#   - 欠損行は 0補完しない。後の分析で空欄スキップが前提。
#   - python3 のパスは which python3 で確認して変更する。
#   - RunAtLoad=false にしているので load直後は実行しない（start で明示的に実行）。
# ═══════════════════════════════════════════════════════════════════════════
