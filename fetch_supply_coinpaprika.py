#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_supply_coinpaprika.py  —  【Macローカルで実行】供給データ取得（検証B・無料代替）
==============================================================================
Coinpaprika 無料API（★APIキー不要・カード登録不要）から、対象28銘柄の
price / market_cap 日次履歴を取得して保存する。
循環供給は後でリモート側で circ_supply = market_cap / price として復元し、
≥1% の単日ジャンプ＝アンロックイベントとして検出する（事前登録 §2・§4）。

★背景: DeFiLlama emissions=402（有料化）、CoinGecko=Demoキーが有料Basic $29/moに
  誘導されたため、無料・キー不要の Coinpaprika に切り替え。

使い方（Macのターミナル・1行ずつ）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_supply_coinpaprika.py?v=1" -o fetch_supply_coinpaprika.py
    head -3 fetch_supply_coinpaprika.py        # 先頭が #!/usr/bin/env python3 ならOK
    python3 fetch_supply_coinpaprika.py

  → ~/Downloads に supply_coingecko.csv が出来る（列 symbol,date_utc,price,market_cap）。
    ※ファイル名は分析コードと揃えるため supply_coingecko.csv のまま。
    このファイルをチャットに貼れば、リモートで供給復元＋イベント検出＋反応分析を回す。

依存: 標準ライブラリのみ（pip不要・キー不要）。売買・送金・Bot接続は一切なし。取得して保存するだけ。
無料枠の履歴は約1年。失敗銘柄はスキップして続行。
"""

from __future__ import annotations
import urllib.request
import urllib.error
import json
import time
import datetime as dt

# 対象28銘柄（既存OHLCVと一致）。Coinpaprika IDは /v1/coins から自動解決するので
# ここではシンボルだけ持つ（IDハードコードの取り違えを避ける）。
TARGET_SYMBOLS = [
    "AAVE", "ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BONK", "BTC", "DOGE",
    "DOT", "ETH", "FET", "HBAR", "INJ", "LINK", "LTC", "NEAR", "POL", "SEI",
    "SHIB", "SOL", "STX", "SUI", "TRX", "UNI", "XLM", "XRP",
]

COINS_URL = "https://api.coinpaprika.com/v1/coins"
HIST_URL  = "https://api.coinpaprika.com/v1/tickers/{cid}/historical?start={start}&interval=1d&limit=366"


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def resolve_ids(symbols):
    """/v1/coins から symbol→coin_id を解決。is_active かつ rank最小（最上位）を採用。"""
    print("■ 銘柄ID解決中（/v1/coins 取得）...")
    coins = get_json(COINS_URL)
    want = {s.upper() for s in symbols}
    best = {}  # symbol -> (rank, id, name)
    for c in coins:
        sym = str(c.get("symbol", "")).upper()
        if sym not in want:
            continue
        if not c.get("is_active", False):
            continue
        if c.get("type") and c.get("type") != "coin" and c.get("type") != "token":
            continue
        rank = c.get("rank", 0) or 0
        rank_key = rank if rank > 0 else 10**9  # rank0(未ランク)は最後尾扱い
        if sym not in best or rank_key < best[sym][0]:
            best[sym] = (rank_key, c["id"], c.get("name", ""))
    resolved = {}
    for s in symbols:
        u = s.upper()
        if u in best:
            resolved[s] = best[u][1]
            print(f"   {s:5s} → {best[u][1]} ({best[u][2]}, rank={best[u][0] if best[u][0]<10**9 else 'NA'})")
        else:
            print(f"   {s:5s} → ❌ 解決できず（スキップ）")
    return resolved


def fetch_hist(cid: str, start: str):
    url = HIST_URL.format(cid=cid, start=start)
    return get_json(url)


def main():
    print("=" * 60)
    print("Coinpaprika 供給（price/market_cap）取得  検証B（無料・キー不要）")
    print("=" * 60)

    # 約1年前から（無料枠の履歴上限に合わせる）
    today = dt.date.today()
    start = (today - dt.timedelta(days=364)).isoformat()
    print(f"  取得期間: {start} 〜 {today.isoformat()}（無料枠・約1年）")

    try:
        ids = resolve_ids(TARGET_SYMBOLS)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        print(f"\n❌ /v1/coins 取得失敗 HTTP {e.code}: {body}")
        print("   → Coinpaprikaも無料で叩けない可能性。Bは『データ取得コストあり候補』として保留に。")
        return
    time.sleep(1.0)

    out_rows = []
    ok, ng = [], []
    for sym, cid in ids.items():
        print(f"\n■ {sym} ({cid}) 履歴取得中...")
        try:
            hist = fetch_hist(cid, start)
            if not isinstance(hist, list) or not hist:
                print(f"  ⚠ {sym}: 履歴0件。スキップ")
                ng.append(sym)
                time.sleep(1.5)
                continue
            cnt = 0
            for row in hist:
                ts = row.get("timestamp", "")
                price = row.get("price", None)
                mcap  = row.get("market_cap", None)
                if not ts or price is None or mcap is None:
                    continue
                day = ts[:10]  # YYYY-MM-DD
                out_rows.append((sym, day, price, mcap))
                cnt += 1
            print(f"  ✅ {sym}: {cnt}日")
            ok.append(sym)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:160]
            except Exception:
                pass
            print(f"  ❌ {sym}: HTTP {e.code}: {body}（スキップ）")
            ng.append(sym)
        except Exception as e:
            print(f"  ❌ {sym}: 取得失敗 {e}（スキップ）")
            ng.append(sym)
        time.sleep(1.5)  # 無料枠のレート制限に配慮

    path = "supply_coingecko.csv"  # 分析コードと同じファイル名
    with open(path, "w") as f:
        f.write("symbol,date_utc,price,market_cap\n")
        for sym, day, price, mcap in out_rows:
            f.write(f"{sym},{day},{price},{mcap}\n")
    print("\n" + "=" * 60)
    print(f"✅ {path} 保存: {len(out_rows)}行  成功{len(ok)}銘柄 / 失敗{len(ng)}銘柄")
    if ng:
        print(f"   失敗（要確認）: {ng}")
    print("→ supply_coingecko.csv をチャットに貼ってください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
