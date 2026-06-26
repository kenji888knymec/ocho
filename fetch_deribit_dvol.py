#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_deribit_dvol.py  —  【Macローカルで実行するデータ取得スクリプト】
====================================================================
Deribit の DVOL（インプライドボラティリティ指数 = BTC版のVIX）の
日次履歴を取得して CSV に保存する。

★このスクリプトは賢治さんのMac（Deribitに接続できる環境）で実行する。
  リモート環境（Claude Code側）はプロキシでDeribitがブロックされているため。

使い方（Macのターミナル）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_deribit_dvol.py" -o fetch_deribit_dvol.py
    python3 fetch_deribit_dvol.py

  → ~/Downloads に dvol_BTC.csv と dvol_ETH.csv が出来る。
    この2ファイルをチャットに貼れば、こちらでVRP検証を回す。

依存: 標準ライブラリのみ（pip不要）。
出力CSV列: date_utc, dvol_open, dvol_high, dvol_low, dvol_close
"""

from __future__ import annotations
import urllib.request
import json
import time
import datetime as dt

# DVOL指数の開始（BTCは2021年頃から。広めに2021-01-01から取得を試みる）
START = "2021-01-01"
CURRENCIES = ["BTC", "ETH"]
RESOLUTION = "43200"   # 秒。43200=12h。日次相当に後で集約。1Dが通らない環境向けに12hで取得
API = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def to_ms(date_str: str) -> int:
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def fetch_chunk(currency: str, start_ms: int, end_ms: int):
    url = (f"{API}?currency={currency}&start_timestamp={start_ms}"
           f"&end_timestamp={end_ms}&resolution={RESOLUTION}")
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read().decode())
    return j.get("result", {}).get("data", [])


def fetch_all(currency: str):
    start_ms = to_ms(START)
    end_ms = int(time.time() * 1000)
    rows = []
    # 30日ずつチャンクで取得（API上限対策）
    chunk = 30 * 24 * 3600 * 1000
    cur = start_ms
    while cur < end_ms:
        nxt = min(cur + chunk, end_ms)
        try:
            data = fetch_chunk(currency, cur, nxt)
            rows.extend(data)
            print(f"  {currency}: {dt.datetime.utcfromtimestamp(cur/1000).date()} "
                  f"〜 {dt.datetime.utcfromtimestamp(nxt/1000).date()}  +{len(data)}本")
        except Exception as e:
            print(f"  {currency}: チャンク取得失敗 {e}（スキップ）")
        cur = nxt
        time.sleep(0.2)  # レート制限に配慮
    return rows


def save_csv(currency: str, rows):
    # rows: [[timestamp_ms, open, high, low, close], ...]
    seen = {}
    for row in rows:
        ts = int(row[0])
        day = dt.datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        # 同日複数本（12h×2）→ 最後の足を採用（日次終値相当）
        seen[day] = row
    days = sorted(seen.keys())
    path = f"dvol_{currency}.csv"
    with open(path, "w") as f:
        f.write("date_utc,dvol_open,dvol_high,dvol_low,dvol_close\n")
        for day in days:
            _, o, h, l, c = seen[day]
            f.write(f"{day},{o},{h},{l},{c}\n")
    print(f"✅ {path} 保存: {len(days)}日 ({days[0]}〜{days[-1]})")


def main():
    print("=" * 60)
    print("Deribit DVOL（インプライドボラ指数）取得")
    print("=" * 60)
    for cur in CURRENCIES:
        print(f"\n■ {cur} 取得中...")
        rows = fetch_all(cur)
        if rows:
            save_csv(cur, rows)
        else:
            print(f"❌ {cur}: データ0件。期間やAPIを確認。")
    print("\n完了。dvol_BTC.csv / dvol_ETH.csv をチャットに貼ってください。")


if __name__ == "__main__":
    main()
