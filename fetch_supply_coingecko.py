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

★無料Demoキーが必要（2026-06以降、CoinGeckoはキーレスを401/429で封鎖）:
    1. https://www.coingecko.com にサインアップ（無料）
    2. ダッシュボード → Developer → 「Demo API Key」を作成（課金なし・$0プラン）
    3. ターミナルでキーを環境変数に入れてから実行:
         export COINGECKO_API_KEY="CG-xxxxxxxxxxxxxxxx"
         python3 fetch_supply_coingecko.py
  → キーはチャットにもgitにも貼らないこと（環境変数だけで渡す）。
  → 無料Demoプランは履歴が直近365日まで。検証期間はその範囲に絞られる（窓・基準は不変）。

依存: 標準ライブラリのみ（pip不要）。売買・送金・Bot接続は一切なし。取得して保存するだけ。
429（レート制限）は指数バックオフでリトライ。失敗銘柄はスキップして続行。
"""

from __future__ import annotations
import os
import urllib.request
import urllib.error
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
# 無料Demoプランは履歴365日まで。days=365 を要求（max超過要求はエラーになる為）。
DAYS = "365"
API_KEY = os.environ.get("COINGECKO_API_KEY", "").strip()


def fetch_market_chart(coin_id: str):
    """429は指数バックオフでリトライ（5,10,20,40秒）。
    Demoキーは CoinGecko docs 準拠でクエリパラメータ x_cg_demo_api_key= で渡す
    （ヘッダ x-cg-demo-api-key も併用）。401時は本文を表示して切り分け。"""
    url = f"{BASE.format(id=coin_id)}?vs_currency=usd&days={DAYS}"
    if API_KEY:
        url += f"&x_cg_demo_api_key={API_KEY}"   # ← docs準拠のクエリ方式（主）
    headers = {"User-Agent": "research/1.0"}
    if API_KEY:
        headers["x-cg-demo-api-key"] = API_KEY    # ← ヘッダ方式も併用（保険）
    backoffs = [5, 10, 20, 40]
    last_err = None
    for attempt in range(len(backoffs) + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 and attempt < len(backoffs):
                wait = backoffs[attempt]
                print(f"    429 → {wait}秒待ってリトライ ({attempt+1}/{len(backoffs)})")
                time.sleep(wait)
                continue
            # 401等はCoinGeckoのエラー本文を表示（キーは含まれない）
            try:
                body = e.read().decode()[:200]
                print(f"    HTTP {e.code} 本文: {body}")
            except Exception:
                pass
            raise
    raise last_err


def main():
    print("=" * 60)
    print("CoinGecko 循環供給（price/market_cap）取得  検証B")
    print("=" * 60)
    if not API_KEY:
        print("\n❌ 環境変数 COINGECKO_API_KEY が未設定です。")
        print("   無料Demoキーを作成し、以下を実行してから再度走らせてください:")
        print('     export COINGECKO_API_KEY="CG-xxxxxxxxxxxxxxxx"')
        print("     python3 fetch_supply_coingecko.py")
        print("   （キーはチャットにもgitにも貼らないこと）")
        return
    print(f"  Demoキー: 検出OK（長さ={len(API_KEY)}文字・末尾4文字 ...{API_KEY[-4:]}） / 履歴={DAYS}日")
    if not API_KEY.startswith("CG-") or len(API_KEY) < 20:
        print("  ⚠ 警告: 本物のDemoキーは 'CG-' 始まり・約25文字。"
              "短い/形式が違う場合はキーが不完全な可能性（Dashboardで全文コピーを確認）。")

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
