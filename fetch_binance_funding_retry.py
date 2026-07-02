#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_binance_funding_retry.py  —  【Macローカルで実行】403で落ちたfundingだけ再取得
==============================================================================
fetch_binance_perp_check.py の実行末尾でレート制限(403)により取得失敗した銘柄の
funding履歴だけを、ゆっくり（2.5秒間隔）再取得して newlisting_funding.csv に追記する。

使い方（Macのターミナル・~/Downloads で・1行ずつ）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_binance_funding_retry.py?v=1" -o fetch_binance_funding_retry.py
    python3 fetch_binance_funding_retry.py

前提: newlisting_perp_check.csv と newlisting_funding.csv が同ディレクトリにあること。
依存: 標準ライブラリのみ。売買なし。取得・追記のみ。
"""

from __future__ import annotations
import urllib.request
import urllib.error
import json
import time
import datetime as dt
import csv
import os

FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
PERP_CSV = "newlisting_perp_check.csv"
FUND_CSV = "newlisting_funding.csv"
POST_DAYS = 31
SLEEP = 2.5


def get_json(url: str, tries: int = 6):
    backoffs = [10, 20, 40, 80, 120]
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (403, 418, 429) and a < len(backoffs):
                print(f"    {e.code} → {backoffs[a]}秒待機してリトライ")
                time.sleep(backoffs[a]); continue
            raise
    raise RuntimeError("retries exhausted")


def main():
    if not (os.path.exists(PERP_CSV) and os.path.exists(FUND_CSV)):
        print("❌ newlisting_perp_check.csv / newlisting_funding.csv が必要です。")
        return
    have_fund = set()
    with open(FUND_CSV) as f:
        for row in csv.DictReader(f):
            have_fund.add(row["perp_symbol"])
    targets = []
    with open(PERP_CSV) as f:
        for row in csv.DictReader(f):
            if row["perp_symbol"] and row["perp_symbol"] not in have_fund:
                targets.append((row["symbol"], row["perp_symbol"], row["spot_listing_utc"]))
    print(f"funding未取得のperp: {len(targets)}銘柄 → ゆっくり再取得（{SLEEP}s間隔）")
    if not targets:
        print("✅ 再取得の必要なし。")
        return

    added = 0
    with open(FUND_CSV, "a") as f:
        for i, (sym, psym, ld_str) in enumerate(targets):
            ld = dt.datetime.strptime(ld_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            start_ms = int(ld.timestamp() * 1000)
            end_ms = start_ms + POST_DAYS * 24 * 3600 * 1000
            try:
                data = get_json(f"{FUNDING}?symbol={psym}&startTime={start_ms}&endTime={end_ms}&limit=1000")
            except Exception as e:
                print(f"  ❌ {psym}: {e}")
                time.sleep(SLEEP); continue
            for d in data:
                ft = dt.datetime.fromtimestamp(int(d["fundingTime"]) / 1000, tz=dt.timezone.utc)
                f.write(f"{psym},{ft.strftime('%Y-%m-%d %H:%M:%S')},{d['fundingRate']}\n")
                added += 1
            print(f"  ✅ {psym}: {len(data)}行  ({i+1}/{len(targets)})")
            time.sleep(SLEEP)
    print(f"\n✅ 追記完了: +{added}行 → newlisting_funding.csv")
    print("→ newlisting_perp_check.csv と newlisting_funding.csv をチャットに貼ってください。")


if __name__ == "__main__":
    main()
