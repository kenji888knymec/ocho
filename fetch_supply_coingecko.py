#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_supply_coingecko.py  —  【Macローカルで実行】循環供給データ取得（検証B）
==============================================================================
CoinGecko 無料APIから、対象28銘柄の price / market_cap 日次履歴を取得して保存する。
循環供給は後でリモート側で circ_supply = market_cap / price として復元し、
≥1% の単日ジャンプ＝アンロックイベントとして検出する（事前登録 §2・§4）。

★賢治さんのMac（CoinGeckoに接続できる環境）で実行する。
  リモート（Claude側）はプロキシで api.coingecko.com が403。

使い方（Macのターミナル）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_supply_coingecko.py" -o fetch_supply_coingecko.py
    python3 fetch_supply_coingecko.py

  → ~/Downloads に supply_coingecko.csv が出来る。
    （列: symbol,date_utc,price,market_cap）
    このファイルをチャットに貼れば、リモートで供給復元＋イベント検出＋反応分析を回す。

依存: 標準ライブラリのみ（pip不要）。売買・送金・Bot接続は一切なし。取得して保存するだけ。
レート制限に配慮し、各銘柄の取得後に少し待つ。失敗銘柄はスキップして続行。
"""

from __future__ import annotations
import urllib.request
import json
import time
import datetime as dt

# 対象28銘柄（既存OHLCVと一致）→ CoinGecko ID（事前固定）
SYMBOL_TO_ID = {
    "AAVE": "aave",
    "ADA":  "cardano",
    "APT":  "aptos",
    "ARB":  "arbitrum",
    "ATOM": "cosmos",
    "AVAX": "avalanche-2",
    "BNB":  "binancecoin",
    "BONK": "bonk",
    "BTC":  "bitcoin",
    "DOGE": "dogecoin",
    "DOT":  "polkadot",
    "ETH":  "ethereum",
    "FET":  "fetch-ai",
    "HBAR": "hedera-hashgraph",
    "INJ":  "injective-protocol",
    "LINK": "chainlink",
    "LTC":  "litecoin",
    "NEAR": "near",
    "POL":  "polygon-ecosystem-token",
    "SEI":  "sei-network",
    "SHIB": "shiba-inu",
    "SOL":  "solana",
    "STX":  "blockstack",
    "SUI":  "sui",
    "TRX":  "tron",
    "UNI":  "uniswap",
    "XLM":  "stellar",
    "XRP":  "ripple",
}

BASE = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
# days=max を使うと長期は日次粒度で返る（無料tier）。interval指定はしない（無料tierで弾かれる為）。


def fetch_market_chart(coin_id: str):
    url = f"{BASE.format(id=coin_id)}?vs_currency=usd&days=max"
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    print("=" * 60)
    print("CoinGecko 循環供給（price/market_cap）取得  検証B")
    print("=" * 60)

    out_rows = []  # (symbol, date_utc, price, market_cap)
    ok, ng = [], []
    for sym, cid in SYMBOL_TO_ID.items():
        print(f"\n■ {sym} ({cid}) 取得中...")
        try:
            j = fetch_market_chart(cid)
            prices = j.get("prices", [])
            mcaps  = j.get("market_caps", [])
            # ts(ms) → 日次。同日複数は最後を採用。priceとmcapをtsで突き合わせ。
            price_by_day, mcap_by_day = {}, {}
            for ts, v in prices:
                day = dt.datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                price_by_day[day] = v
            for ts, v in mcaps:
                day = dt.datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                mcap_by_day[day] = v
            days = sorted(set(price_by_day) & set(mcap_by_day))
            if not days:
                print(f"  ⚠ {sym}: price/mcap の共通日なし。スキップ")
                ng.append(sym)
                continue
            for day in days:
                out_rows.append((sym, day, price_by_day[day], mcap_by_day[day]))
            print(f"  ✅ {sym}: {len(days)}日 ({days[0]}〜{days[-1]})")
            ok.append(sym)
        except Exception as e:
            print(f"  ❌ {sym}: 取得失敗 {e}（スキップ）")
            ng.append(sym)
        time.sleep(3.0)  # 無料tierのレート制限に配慮

    path = "supply_coingecko.csv"
    with open(path, "w") as f:
        f.write("symbol,date_utc,price,market_cap\n")
        for sym, day, price, mcap in out_rows:
            f.write(f"{sym},{day},{price},{mcap}\n")
    print("\n" + "=" * 60)
    print(f"✅ {path} 保存: {len(out_rows)}行  成功{len(ok)}銘柄 / 失敗{len(ng)}銘柄")
    if ng:
        print(f"   失敗（要確認）: {ng}")
    print("→ supply_coingecko.csv をチャットに貼ってください。")
    print("   リモートで circ_supply=mcap/price を復元し、≥1%ジャンプを検出して反応分析します。")
    print("=" * 60)


if __name__ == "__main__":
    main()
