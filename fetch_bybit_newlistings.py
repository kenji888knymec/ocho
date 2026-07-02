#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_bybit_newlistings.py  —  【Macローカルで実行】Track A Step-5 Bybit新規上場データ取得
==============================================================================
Binanceで確認した「新規上場ショート」をBybitでクロス検証するためのデータ取得。
Bybit公開API v5（無料・キー不要・カード不要・公開GETのみ）から:
  1. Spot USDTペア全件 → 最古klineで上場日判定 → 2024-01-01以降の新規上場を抽出
  2. 各新規上場の「上場後31日分の1h kline」
  3. linear perp の launchTime（perp上場日）と Spot上場日の突き合わせ
  4. 各perpの funding履歴（Spot上場後31日分）
を取得し、CSV4本に保存する（Binance版とスキーマ互換・bybit_プレフィクス）。

★安全性: 公開マーケットデータのGETのみ。APIキー不要・注文/送金/口座アクセス一切なし。
  依存は標準ライブラリのみ（pip不要）。

使い方（Macのターミナル・1行ずつ）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_bybit_newlistings.py?v=1" -o fetch_bybit_newlistings.py
    head -3 fetch_bybit_newlistings.py
    python3 fetch_bybit_newlistings.py

  → ~/Downloads に以下4本が出来る:
      bybit_newlisting_universe.csv / bybit_newlisting_klines_1h.csv
      bybit_newlisting_perp_check.csv / bybit_newlisting_funding.csv
    4本をチャットに貼る（klinesが重ければ他3本を先でもOK）。
  所要: 全Spot走査があるため10〜20分程度。403/429は自動バックオフ。
"""

from __future__ import annotations
import urllib.request
import urllib.error
import urllib.parse
import json
import time
import datetime as dt

BASE = "https://api.bybit.com"
LISTING_FROM = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
POST_DAYS = 31
MIN_FIRST_DAY_QUOTE_VOL = 1_000_000.0
STABLE_BASES = {"USDC","FDUSD","TUSD","DAI","USDP","BUSD","EUR","GBP","AEUR","USD1","USDE","PYUSD","USTC","USDD","USDY","XUSD","RLUSD","USDS"}
LT_SUFFIX = ("2L","2S","3L","3S","4L","4S","5L","5S")   # Bybitレバレッジトークン


def get_json(url: str, tries: int = 6):
    backoffs = [3, 6, 12, 24, 48]
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                j = json.loads(r.read().decode())
            if j.get("retCode") != 0:
                raise RuntimeError(f"retCode={j.get('retCode')} {j.get('retMsg')}")
            return j["result"]
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            if a < len(backoffs):
                time.sleep(backoffs[a]); continue
            raise
    raise RuntimeError("retries exhausted")


def paged_instruments(category: str):
    out, cursor = [], ""
    while True:
        url = f"{BASE}/v5/market/instruments-info?category={category}&limit=1000"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        res = get_json(url)
        out.extend(res.get("list", []))
        cursor = res.get("nextPageCursor", "")
        if not cursor:
            break
        time.sleep(0.15)
    return out


def kline(category: str, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    url = (f"{BASE}/v5/market/kline?category={category}&symbol={symbol}"
           f"&interval={interval}&start={start_ms}&end={end_ms}&limit={limit}")
    res = get_json(url)
    rows = res.get("list", [])
    rows.sort(key=lambda r: int(r[0]))   # Bybitは新しい順で返る → 昇順に
    return rows


def earliest_listing(symbol: str):
    """月足→日足のドリルダウンで最古の日足を特定。(listing_day_dt, first_day_turnover) or None"""
    now_ms = int(time.time() * 1000)
    months = kline("spot", symbol, "M", 0, now_ms, 1000)
    if not months:
        return None
    m0 = int(months[0][0])
    days = kline("spot", symbol, "D", m0, m0 + 40 * 86400_000, 1000)
    if not days:
        return None
    d0 = days[0]
    listing = dt.datetime.fromtimestamp(int(d0[0]) / 1000, tz=dt.timezone.utc)
    turnover = float(d0[6]) if len(d0) > 6 and d0[6] else 0.0
    return listing, turnover


def fetch_1h(symbol: str, start_dt: dt.datetime):
    """上場後POST_DAYS分の1h kline（チャンク取得・limit上限差異に対応）"""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = start_ms + POST_DAYS * 24 * 3600_000
    out, cur = [], start_ms
    while cur < end_ms:
        chunk_end = min(cur + 200 * 3600_000, end_ms)   # 200本ずつ
        rows = kline("spot", symbol, "60", cur, chunk_end - 1, 1000)
        out.extend(rows)
        cur = chunk_end
        time.sleep(0.12)
    # 重複除去・昇順
    seen, uniq = set(), []
    for r in out:
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    uniq.sort(key=lambda r: int(r[0]))
    return uniq


def fetch_funding(perp_symbol: str, start_ms: int, end_ms: int):
    out, cur = [], start_ms
    while cur < end_ms:
        chunk_end = min(cur + 60 * 86400_000, end_ms)
        url = (f"{BASE}/v5/market/funding/history?category=linear&symbol={perp_symbol}"
               f"&startTime={cur}&endTime={chunk_end}&limit=200")
        res = get_json(url)
        out.extend(res.get("list", []))
        cur = chunk_end
        time.sleep(0.12)
    return out


def main():
    print("=" * 66)
    print("Track A Step-5: Bybit 新規上場データ取得（無料・キー不要・GETのみ）")
    print("=" * 66)
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=POST_DAYS - 1)

    print("\n■ Spot銘柄一覧...")
    spot = paged_instruments("spot")
    syms = []
    for s in spot:
        if s.get("quoteCoin") != "USDT" or s.get("status") != "Trading":
            continue
        b = s.get("baseCoin", "")
        if b in STABLE_BASES or any(b.endswith(sfx) for sfx in LT_SUFFIX):
            continue
        syms.append(s["symbol"])
    print(f"  USDT Spot 候補（除外後）: {len(syms)}銘柄")

    print("\n■ 上場日判定（月足→日足ドリルダウン）... 10分前後かかります")
    universe = []
    for i, sym in enumerate(syms):
        try:
            r = earliest_listing(sym)
        except Exception as e:
            print(f"  {sym}: 失敗 {e}")
            time.sleep(0.3); continue
        if r:
            ld, qv = r
            if LISTING_FROM <= ld <= cutoff and qv >= MIN_FIRST_DAY_QUOTE_VOL:
                universe.append((sym, ld, qv))
                print(f"  ★新規 {sym}: {ld.date()}  初日quote_vol ${qv/1e6:.1f}M  (累計{len(universe)})")
        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{len(syms)}")
        time.sleep(0.15)

    with open("bybit_newlisting_universe.csv", "w") as f:
        f.write("symbol,listing_date_utc,first_day_quote_vol_usd\n")
        for sym, ld, qv in universe:
            f.write(f"{sym},{ld.strftime('%Y-%m-%d %H:%M:%S')},{qv}\n")
    print(f"\n✅ bybit_newlisting_universe.csv: {len(universe)}銘柄")

    print("\n■ 上場後1h kline取得...")
    rows = 0
    with open("bybit_newlisting_klines_1h.csv", "w") as f:
        f.write("symbol,open_time_utc,open,high,low,close,volume,quote_vol\n")
        for j, (sym, ld, qv) in enumerate(universe):
            try:
                kl = fetch_1h(sym, ld)
            except Exception as e:
                print(f"  {sym}: kline失敗 {e}"); time.sleep(0.4); continue
            for k in kl:
                ot = dt.datetime.fromtimestamp(int(k[0]) / 1000, tz=dt.timezone.utc)
                f.write(f"{sym},{ot.strftime('%Y-%m-%d %H:%M:%S')},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[6]}\n")
                rows += 1
            print(f"  ✅ {sym}: {len(kl)}本  ({j+1}/{len(universe)})")
    print(f"✅ bybit_newlisting_klines_1h.csv: {rows}行")

    print("\n■ perp一覧（linear）と突き合わせ...")
    linear = paged_instruments("linear")
    idx = {}
    for s in linear:
        if s.get("quoteCoin") != "USDT" or s.get("contractType") not in (None, "LinearPerpetual"):
            continue
        b = s.get("baseCoin", "")
        lt = s.get("launchTime")
        if not lt:
            continue
        lt = int(lt)
        if b not in idx or lt < idx[b][1]:
            idx[b] = (s["symbol"], lt)

    def match(base):
        if base in idx: return idx[base]
        if ("1000" + base) in idx: return idx["1000" + base]
        for pre in ("1000", "1M"):
            if base.startswith(pre) and base[len(pre):] in idx:
                return idx[base[len(pre):]]
        return None

    matched = []
    with open("bybit_newlisting_perp_check.csv", "w") as f:
        f.write("symbol,perp_symbol,perp_onboard_utc,spot_listing_utc,perp_minus_spot_hours\n")
        for sym, ld, qv in universe:
            base = sym[:-4]
            m = match(base)
            if m is None:
                f.write(f"{sym},,,{ld.strftime('%Y-%m-%d %H:%M:%S')},\n")
                continue
            psym, lt = m
            ob = dt.datetime.fromtimestamp(lt / 1000, tz=dt.timezone.utc)
            diff_h = (ob - ld).total_seconds() / 3600.0
            f.write(f"{sym},{psym},{ob.strftime('%Y-%m-%d %H:%M:%S')},{ld.strftime('%Y-%m-%d %H:%M:%S')},{diff_h:.1f}\n")
            matched.append((sym, psym, ld))
    n4 = sum(1 for _, _, _ in matched)  # 詳細は分析側で
    print(f"✅ bybit_newlisting_perp_check.csv: perpあり {len(matched)}/{len(universe)}")

    print("\n■ funding履歴取得...")
    cnt = 0
    with open("bybit_newlisting_funding.csv", "w") as f:
        f.write("perp_symbol,funding_time_utc,funding_rate\n")
        for j, (sym, psym, ld) in enumerate(matched):
            start_ms = int(ld.timestamp() * 1000)
            end_ms = start_ms + POST_DAYS * 24 * 3600_000
            try:
                data = fetch_funding(psym, start_ms, end_ms)
            except Exception as e:
                print(f"  {psym}: funding失敗 {e}"); time.sleep(0.4); continue
            for d in data:
                ft = dt.datetime.fromtimestamp(int(d["fundingRateTimestamp"]) / 1000, tz=dt.timezone.utc)
                f.write(f"{psym},{ft.strftime('%Y-%m-%d %H:%M:%S')},{d['fundingRate']}\n")
                cnt += 1
            if (j + 1) % 20 == 0:
                print(f"  ...{j+1}/{len(matched)}")
            time.sleep(0.15)
    print(f"✅ bybit_newlisting_funding.csv: {cnt}行")

    print("\n" + "=" * 66)
    print("完了。4本のCSVをチャットに貼ってください（klinesが重ければ他3本先でもOK）。")
    print("=" * 66)


if __name__ == "__main__":
    main()
