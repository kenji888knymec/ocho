"""
fetch_okx_ohlcv.py
==================
OKX swap 1h/15m OHLCVを28銘柄分取得してCSVに保存するスクリプト。
このスクリプトはローカル（OKXにアクセスできる環境）で実行すること。
Claude Code sandboxからはOKX APIへの接続がネットワークポリシーでブロックされている。

依存: pip install ccxt pandas
出力: /tmp/ohlcv_okx/{SYMBOL}_1h.csv と {SYMBOL}_15m.csv (28銘柄×2=56ファイル)

実行後: build_v2_okx.py → tf_thr_validate_okx.py の順で検証する
"""
import ccxt
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timezone

OUT_DIR = Path("/tmp/ohlcv_okx")
OUT_DIR.mkdir(exist_ok=True)

# 28銘柄
SYMBOLS = ["AAVE","ADA","APT","ARB","ATOM","AVAX","BNB","BONK","BTC","DOGE",
           "DOT","ETH","FET","HBAR","INJ","LINK","LTC","NEAR","POL","SEI",
           "SHIB","SOL","STX","SUI","TRX","UNI","XLM","XRP"]

# OKX swap シンボルマッピング（1000x系に注意）
# OKXでは BONK/SHIB は 1000BONK/USDT:USDT / 1000SHIB/USDT:USDT として取引される
# 価格は 1000単位建て（取得後に 1/1000 補正は不要 → パーセンテージは同じ）
# POLはPolygon(旧MATIC)のOKX上の名称を要確認
OKX_SYMBOL_MAP = {
    "AAVE": "AAVE/USDT:USDT",
    "ADA":  "ADA/USDT:USDT",
    "APT":  "APT/USDT:USDT",
    "ARB":  "ARB/USDT:USDT",
    "ATOM": "ATOM/USDT:USDT",
    "AVAX": "AVAX/USDT:USDT",
    "BNB":  "BNB/USDT:USDT",
    "BONK": "1000BONK/USDT:USDT",  # 1000x建て → 価格は 1000BONK/USDT
    "BTC":  "BTC/USDT:USDT",
    "DOGE": "DOGE/USDT:USDT",
    "DOT":  "DOT/USDT:USDT",
    "ETH":  "ETH/USDT:USDT",
    "FET":  "FET/USDT:USDT",        # OKXに存在しない場合は下記ALTを参照
    "HBAR": "HBAR/USDT:USDT",
    "INJ":  "INJ/USDT:USDT",
    "LINK": "LINK/USDT:USDT",
    "LTC":  "LTC/USDT:USDT",
    "NEAR": "NEAR/USDT:USDT",
    "POL":  "POL/USDT:USDT",        # OKXにない場合 "MATIC/USDT:USDT" を試す
    "SEI":  "SEI/USDT:USDT",
    "SHIB": "1000SHIB/USDT:USDT",   # 1000x建て
    "SOL":  "SOL/USDT:USDT",
    "STX":  "STX/USDT:USDT",
    "SUI":  "SUI/USDT:USDT",
    "TRX":  "TRX/USDT:USDT",
    "UNI":  "UNI/USDT:USDT",
    "XLM":  "XLM/USDT:USDT",
    "XRP":  "XRP/USDT:USDT",
}

# フォールバック候補（メインシンボルが存在しない場合）
OKX_FALLBACK = {
    "FET":  "FET/USDT:USDT",   # ASI rebranding前後で変わる場合あり
    "POL":  "MATIC/USDT:USDT", # OKXでPOLとして上場していない場合
}

# 取得開始・終了 (UTC)
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026, 6, 21, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def create_exchange():
    """OKX swap 接続（APIキー不要・public OHLCV）"""
    return ccxt.okx({
        "options": {"defaultType": "swap"},
        "rateLimit": 200,
        "enableRateLimit": True,
    })


def fetch_ohlcv_full(exchange, okx_sym, timeframe, start_ms, end_ms, symbol_label):
    """ページネーションして全期間のOHLCVを取得"""
    limit = 300  # OKXの最大はtimeframeにより異なる（1h:1440, 15m:1440）
    since = start_ms
    all_rows = []

    while since < end_ms:
        try:
            bars = exchange.fetch_ohlcv(okx_sym, timeframe, since=since, limit=limit)
            if not bars:
                break
            all_rows.extend(bars)
            last_ts = bars[-1][0]
            if last_ts >= end_ms or len(bars) < limit:
                break
            since = last_ts + 1
            time.sleep(0.25)  # rate limit
        except Exception as ex:
            print(f"  ERROR {symbol_label} {timeframe} @ {since}: {ex}")
            time.sleep(2)
            break

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[df["timestamp"] <= end_ms].drop_duplicates("timestamp").sort_values("timestamp")
    df.insert(0, "symbol", symbol_label)
    df.insert(1, "source_symbol", okx_sym)
    return df


def check_and_resolve(exchange, sym):
    """シンボルが存在するか確認し、フォールバック適用"""
    primary = OKX_SYMBOL_MAP[sym]
    try:
        exchange.load_markets()
        if primary in exchange.markets:
            return primary, None
    except Exception:
        pass

    fallback = OKX_FALLBACK.get(sym)
    if fallback and fallback in exchange.markets:
        print(f"  [{sym}] フォールバック使用: {fallback}")
        return fallback, f"fallback={fallback}"
    return primary, None  # tryして失敗したら後で判明


def main():
    print(f"OKX swap OHLCV取得開始: {len(SYMBOLS)}銘柄, 2024-01-01 〜 2026-06-21")
    exchange = create_exchange()

    # マーケット一覧を一度だけロード（エラー確認用）
    try:
        markets = exchange.load_markets()
        print(f"OKX markets loaded: {len(markets)} total")
    except Exception as ex:
        print(f"WARNING: マーケット読込失敗: {ex}")
        markets = {}

    success, failed = [], []
    for sym in SYMBOLS:
        okx_sym = OKX_SYMBOL_MAP[sym]

        # フォールバック確認
        if okx_sym not in markets and sym in OKX_FALLBACK:
            alt = OKX_FALLBACK[sym]
            if alt in markets:
                okx_sym = alt
                print(f"  [{sym}] フォールバック: {okx_sym}")

        print(f"\n[{sym}] {okx_sym}")

        for tf in ["1h", "15m"]:
            out_path = OUT_DIR / f"{sym}_{tf}.csv"
            if out_path.exists():
                existing = pd.read_csv(out_path)
                print(f"  {tf}: 既存 {len(existing)}行 → スキップ")
                continue

            df = fetch_ohlcv_full(exchange, okx_sym, tf, START_MS, END_MS, sym)
            if df is None or len(df) == 0:
                print(f"  {tf}: 取得失敗 → スキップ")
                failed.append(f"{sym}_{tf}")
                continue

            df.to_csv(out_path, index=False)
            print(f"  {tf}: {len(df)}行 保存 → {out_path}")

        success.append(sym)

    print(f"\n=== 完了 ===")
    print(f"成功: {len(success)}/{len(SYMBOLS)} 銘柄")
    if failed:
        print(f"失敗: {failed}")
    print(f"出力先: {OUT_DIR}")
    print(f"\n次のステップ:")
    print(f"  python3 /tmp/build_v2_okx.py")
    print(f"  python3 /tmp/tf_thr_validate_okx.py")


if __name__ == "__main__":
    main()
