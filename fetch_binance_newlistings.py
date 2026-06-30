#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_binance_newlistings.py  —  【Macローカルで実行】Binance新規上場ユニバース取得（無料・キー不要）
==============================================================================
CEX新規上場後リターン検証（事前登録 verify_newlisting_PREREG.md）のデータ取得。
Binance公開API（無料・キー不要・カード不要）から:
  1. 全Spot USDTペアを列挙
  2. 各ペアの "最初のkline" で上場日を判定 → 2024-01-01以降の新規上場を抽出
  3. 新規上場ペアの「上場後31日分の1h kline」を取得
を行い、CSV2本に保存する。

★賢治さんのMac（Binanceに接続できる環境）で実行。リモートはプロキシで403。

使い方（Macのターミナル・1行ずつ）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_binance_newlistings.py?v=1" -o fetch_binance_newlistings.py
    head -3 fetch_binance_newlistings.py
    python3 fetch_binance_newlistings.py

  → ~/Downloads に
      newlisting_universe.csv  （列: symbol, listing_date_utc, first_day_quote_vol_usd）
      newlisting_klines_1h.csv （列: symbol, open_time_utc, open, high, low, close, volume, quote_vol）
    が出来る。2本をチャットに貼れば、こちらで事前登録どおり分析する。
    （klinesは重い場合あり。重ければ universe だけでも先に貼ってOK）

依存: 標準ライブラリのみ（pip不要・キー不要）。売買・送金・Bot接続は一切なし。取得・保存のみ。
レート制限に配慮し待機を入れる。失敗銘柄はスキップして続行。
"""

from __future__ import annotations
import urllib.request
import urllib.error
import json
import time
import datetime as dt

BASE = "https://api.binance.com"
EXCHANGE_INFO = f"{BASE}/api/v3/exchangeInfo"
KLINES = f"{BASE}/api/v3/klines"

LISTING_FROM = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
POST_DAYS    = 31           # 上場後31日分の1h足
MIN_FIRST_DAY_QUOTE_VOL = 1_000_000.0   # 初日quote出来高 < $1M は除外
# 除外: ステーブル/法定 as-base、レバレッジトークン
STABLE_BASES = {"USDC","FDUSD","TUSD","DAI","USDP","BUSD","EUR","GBP","AEUR","USD1","USDe","PYUSD"}
LEV_MARKERS  = ("UP", "DOWN", "BULL", "BEAR")


def get_json(url: str, tries: int = 5):
    backoffs = [3, 6, 12, 24]
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) and a < len(backoffs):
                print(f"    {e.code} → {backoffs[a]}秒待機")
                time.sleep(backoffs[a]); continue
            raise
    raise RuntimeError("retries exhausted")


def base_asset(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def is_excluded(symbol: str) -> bool:
    b = base_asset(symbol)
    if b in STABLE_BASES:
        return True
    if any(m in b for m in LEV_MARKERS):   # レバトークン（BTCUP, ETHDOWN 等）
        return True
    return False


def earliest_listing(symbol: str):
    """最古の1d klineで上場日を判定。(listing_dt, first_day_quote_vol) or None。"""
    url = f"{KLINES}?symbol={symbol}&interval=1d&startTime=0&limit=1"
    data = get_json(url)
    if not data:
        return None
    k = data[0]
    open_ms = int(k[0]); quote_vol = float(k[7])
    listing_dt = dt.datetime.utcfromtimestamp(open_ms/1000).replace(tzinfo=dt.timezone.utc)
    return listing_dt, quote_vol


def fetch_1h(symbol: str, start_dt: dt.datetime):
    """上場後POST_DAYS分の1h klineを取得（1000本上限内に収まる: 31d*24=744）。"""
    start_ms = int(start_dt.timestamp()*1000)
    url = f"{KLINES}?symbol={symbol}&interval=1h&startTime={start_ms}&limit={POST_DAYS*24}"
    return get_json(url)


def main():
    print("="*64)
    print("Binance 新規上場ユニバース取得  検証(CEX新規上場後リターン)  無料・キー不要")
    print("="*64)
    now = dt.datetime.now(dt.timezone.utc)
    cutoff_listed_before = now - dt.timedelta(days=POST_DAYS-1)  # 30日後データが揃うもの

    print("\n■ exchangeInfo 取得...")
    info = get_json(EXCHANGE_INFO)
    syms = [s["symbol"] for s in info.get("symbols", [])
            if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"
            and s.get("isSpotTradingAllowed", True) and not is_excluded(s["symbol"])]
    print(f"  USDT Spot 候補（除外後）: {len(syms)}銘柄")

    print("\n■ 各銘柄の上場日を判定中（最古kline）... 時間がかかります")
    universe = []  # (symbol, listing_dt, first_day_quote_vol)
    for i, sym in enumerate(syms):
        try:
            r = earliest_listing(sym)
        except Exception as e:
            print(f"  {sym}: 上場日取得失敗 {e}")
            time.sleep(0.3); continue
        if r is None:
            time.sleep(0.3); continue
        listing_dt, qv = r
        if listing_dt >= LISTING_FROM and listing_dt <= cutoff_listed_before and qv >= MIN_FIRST_DAY_QUOTE_VOL:
            universe.append((sym, listing_dt, qv))
            print(f"  ★新規 {sym}: 上場 {listing_dt.date()}  初日quote_vol ${qv/1e6:.1f}M  (累計{len(universe)})")
        if (i+1) % 50 == 0:
            print(f"    ...{i+1}/{len(syms)} 走査済み")
        time.sleep(0.25)   # レート制限配慮

    # universe 保存
    with open("newlisting_universe.csv", "w") as f:
        f.write("symbol,listing_date_utc,first_day_quote_vol_usd\n")
        for sym, ld, qv in universe:
            f.write(f"{sym},{ld.strftime('%Y-%m-%d %H:%M:%S')},{qv}\n")
    print(f"\n✅ newlisting_universe.csv: {len(universe)}銘柄")

    # 各新規上場の上場後1h kline
    print("\n■ 新規上場銘柄の上場後1h kline取得...")
    rows = 0
    with open("newlisting_klines_1h.csv", "w") as f:
        f.write("symbol,open_time_utc,open,high,low,close,volume,quote_vol\n")
        for j, (sym, ld, qv) in enumerate(universe):
            try:
                kl = fetch_1h(sym, ld)
            except Exception as e:
                print(f"  {sym}: kline取得失敗 {e}")
                time.sleep(0.4); continue
            for k in kl:
                ot = dt.datetime.utcfromtimestamp(int(k[0])/1000).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{sym},{ot},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]}\n")
                rows += 1
            print(f"  ✅ {sym}: {len(kl)}本  ({j+1}/{len(universe)})")
            time.sleep(0.35)
    print(f"\n✅ newlisting_klines_1h.csv: {rows}行")
    print("\n" + "="*64)
    print("完了。newlisting_universe.csv と newlisting_klines_1h.csv をチャットに貼ってください。")
    print("（klinesが重ければ universe だけ先に貼ってもOK）")
    print("="*64)


if __name__ == "__main__":
    main()
