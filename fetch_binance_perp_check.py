#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_binance_perp_check.py  —  【Macローカルで実行】Track A Step-2 執行可能性データ取得
==============================================================================
新規上場ショートの最終関門「執行可能性」を白黒つけるためのデータ取得。
Binance USDM先物 公開API（無料・キー不要・カード不要）から:
  1. 全perpの上場日(onboardDate) → 177銘柄のSpot上場日と突き合わせ
  2. 各perpの「Spot上場後30日分」のfunding履歴（8時間ごと）
を取得してCSV2本に保存する。

★事前に newlisting_universe.csv が ~/Downloads に必要（前回作成済みのもの）。

使い方（Macのターミナル・1行ずつ）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_binance_perp_check.py?v=1" -o fetch_binance_perp_check.py
    head -3 fetch_binance_perp_check.py
    python3 fetch_binance_perp_check.py

  → ~/Downloads に
      newlisting_perp_check.csv （symbol, perp_symbol, perp_onboard_utc, spot_listing_utc, perp_minus_spot_hours）
      newlisting_funding.csv    （perp_symbol, funding_time_utc, funding_rate）
    が出来る。2本をチャットに貼れば、ロック済みの判定スクリプトでGO/NO-GOを出す。

依存: 標準ライブラリのみ。売買・送金・Bot接続は一切なし。取得・保存のみ。
"""

from __future__ import annotations
import urllib.request
import urllib.error
import json
import time
import datetime as dt
import csv
import os

FAPI = "https://fapi.binance.com"
EXINFO = f"{FAPI}/fapi/v1/exchangeInfo"
FUNDING = f"{FAPI}/fapi/v1/fundingRate"
UNIVERSE = "newlisting_universe.csv"
POST_DAYS = 31


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


def base_of(spot_symbol: str) -> str:
    return spot_symbol[:-4] if spot_symbol.endswith("USDT") else spot_symbol


def build_perp_index():
    """perp一覧: base → (perp_symbol, onboard_ms)。PERPETUAL/USDTのみ。"""
    info = get_json(EXINFO)
    idx = {}
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
            continue
        sym = s["symbol"]
        ob = s.get("onboardDate")
        if ob is None:
            continue
        b = base_of(sym)
        # 同一baseが複数あれば最も早いonboardを採用
        if b not in idx or ob < idx[b][1]:
            idx[b] = (sym, int(ob))
    return idx


def match_perp(spot_base: str, idx: dict):
    """spot base → perp。完全一致 → 1000プレフィクス → 1000/1M剥がし の順で探す。"""
    if spot_base in idx:
        return idx[spot_base]
    if ("1000" + spot_base) in idx:
        return idx["1000" + spot_base]
    for pre in ("1000", "1M"):
        if spot_base.startswith(pre) and spot_base[len(pre):] in idx:
            return idx[spot_base[len(pre):]]
    return None


def fetch_funding(perp_symbol: str, start_ms: int, end_ms: int):
    url = f"{FUNDING}?symbol={perp_symbol}&startTime={start_ms}&endTime={end_ms}&limit=1000"
    return get_json(url)


def main():
    print("=" * 66)
    print("Track A Step-2: Binance perp 存在・funding 取得（無料・キー不要）")
    print("=" * 66)
    if not os.path.exists(UNIVERSE):
        print(f"❌ {UNIVERSE} が見つかりません。~/Downloads で実行してください。")
        return

    spot = []
    with open(UNIVERSE) as f:
        for row in csv.DictReader(f):
            spot.append((row["symbol"], row["listing_date_utc"]))
    print(f"Spot新規上場: {len(spot)}銘柄")

    print("\n■ perp一覧取得（exchangeInfo）...")
    idx = build_perp_index()
    print(f"  USDT PERPETUAL: {len(idx)}銘柄")

    # 突き合わせ
    rows = []
    matched = []
    for sym, ld_str in spot:
        ld = dt.datetime.strptime(ld_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        m = match_perp(base_of(sym), idx)
        if m is None:
            rows.append((sym, "", "", ld_str, ""))
            continue
        perp_sym, ob_ms = m
        ob = dt.datetime.fromtimestamp(ob_ms / 1000, tz=dt.timezone.utc)
        diff_h = (ob - ld).total_seconds() / 3600.0
        rows.append((sym, perp_sym, ob.strftime("%Y-%m-%d %H:%M:%S"), ld_str, f"{diff_h:.1f}"))
        matched.append((sym, perp_sym, ld))

    with open("newlisting_perp_check.csv", "w") as f:
        f.write("symbol,perp_symbol,perp_onboard_utc,spot_listing_utc,perp_minus_spot_hours\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    n_perp = len(matched)
    n_before = sum(1 for r in rows if r[4] != "" and float(r[4]) <= 4.0)
    print(f"\n✅ newlisting_perp_check.csv: perpあり {n_perp}/{len(spot)}")
    print(f"   うち Spot上場+4h以内にperp取引可能: {n_before}銘柄（E4エントリー可能数の目安）")

    # funding履歴（Spot上場後31日分）
    print("\n■ funding履歴取得（Spot上場後31日・8hごと）...")
    cnt = 0
    with open("newlisting_funding.csv", "w") as f:
        f.write("perp_symbol,funding_time_utc,funding_rate\n")
        for j, (sym, perp_sym, ld) in enumerate(matched):
            start_ms = int(ld.timestamp() * 1000)
            end_ms = start_ms + POST_DAYS * 24 * 3600 * 1000
            try:
                data = fetch_funding(perp_sym, start_ms, end_ms)
            except Exception as e:
                print(f"  {perp_sym}: funding取得失敗 {e}")
                time.sleep(0.4); continue
            for d in data:
                ft = dt.datetime.fromtimestamp(int(d["fundingTime"]) / 1000, tz=dt.timezone.utc)
                f.write(f"{perp_sym},{ft.strftime('%Y-%m-%d %H:%M:%S')},{d['fundingRate']}\n")
                cnt += 1
            if (j + 1) % 20 == 0:
                print(f"  ...{j+1}/{n_perp}")
            time.sleep(0.3)
    print(f"\n✅ newlisting_funding.csv: {cnt}行")
    print("\n" + "=" * 66)
    print("完了。newlisting_perp_check.csv と newlisting_funding.csv をチャットに貼ってください。")
    print("=" * 66)


if __name__ == "__main__":
    main()
