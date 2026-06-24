"""
verify_funding_extreme.py — Funding Extreme Contrarian 仮説検証
=================================================================

【事前登録済み仮説】
  方向: HIGH_FUNDING (funding 極大) → SHORT期待 / LOW_FUNDING (funding 極小) → LONG期待
  理由: funding 極大はLONG過密（クラウディング）= 多数派がポジション整理で値下がりしやすい
       funding 極小はSHORT過密（クラウディング）= SHORT整理で値上がりしやすい

【設計】
  - 対象: BTCUSDT / ETHUSDT Binance perpetual (data.binance.vision)
  - fundingRate: 8時間毎（1日3回: 00:00/08:00/16:00 UTC）
  - 極値判定: 過去 ROLL_DAYS 日間の 分位数 P_HIGH / P_LOW
    HIGH = funding >= rolling P_HIGH 分位 → SHORT期待
    LOW  = funding <= rolling P_LOW  分位 → LONG期待
  - 評価ウィンドウ: 発火後 FWD_H 時間後の価格リターン (SHORT/LONG それぞれ方向を反転)
  - コスト: COST_RT (往復, one-side funding込み) を 1 エントリー毎に差引
  - train: TRAIN_START 〜 TRAIN_END  (完全非接触で仮説策定済み)
  - test : TEST_START  〜 TEST_END

【合格基準 (事前固定・変更禁止)】
  PASS_AVG      avg PnL_Net > 0%
  PASS_WR       勝率 WR >= 50%
  PASS_MONTHS   active_months >= 4
  PASS_SIGN     train/test で avg 符号一致
  PASS_BOTH     BTC と ETH 両方合格
  PASS_YEAR_TOP top1年の集中度 < 70%

【実行方法 (ローカルMac)】
  python3 verify_funding_extreme.py [--download]  # --download で Binance からDL
  python3 verify_funding_extreme.py               # キャッシュ済みCSVがあればそのまま

  データキャッシュ先: ~/Downloads/binance_funding/
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 事前登録済みパラメータ (変更禁止)
# ---------------------------------------------------------------------------
SYMBOLS     = ["BTCUSDT", "ETHUSDT"]
ROLL_DAYS   = 30        # rolling 窓 (日)
P_HIGH      = 0.90      # HIGH_FUNDING 判定パーセンタイル
P_LOW       = 0.10      # LOW_FUNDING  判定パーセンタイル
FWD_H       = 24        # forward return 評価時間 (h)
COST_RT     = 0.0015    # round-trip コスト (片道 0.075% × 2)
TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-05-31"

PASS_AVG      =  0.00
PASS_WR       = 50.0
PASS_MONTHS   =  4
PASS_YEAR_TOP = 70.0

# Binance public data base URL
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/"
KLINE_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1h/"
CACHE_DIR  = Path.home() / "Downloads" / "binance_funding"

# ---------------------------------------------------------------------------
# データ取得 (ローカルキャッシュ優先)
# ---------------------------------------------------------------------------

def download_zip(url: str, dest: Path) -> bool:
    """1ファイルをダウンロードして保存 (失敗時 False)。"""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  SKIP {url}: {e}", file=sys.stderr)
        return False


def load_funding(sym: str, do_download: bool) -> pd.DataFrame:
    """fundingRate CSV を読み込む。キャッシュになければ DL。"""
    cache = CACHE_DIR / sym / "funding"
    cache.mkdir(parents=True, exist_ok=True)

    dfs = []
    from datetime import date
    cur = date.fromisoformat(TRAIN_START[:7] + "-01")
    end = date.fromisoformat(TEST_END[:7] + "-01")

    import calendar
    while cur <= end:
        ym = cur.strftime("%Y-%m")
        csv_path = cache / f"{sym}-fundingRate-{ym}.csv"
        if not csv_path.exists():
            if not do_download:
                cur = date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
                continue
            zip_name = f"{sym}-fundingRate-{ym}.zip"
            url = BASE_URL.format(sym=sym) + zip_name
            zip_path = cache / zip_name
            if download_zip(url, zip_path):
                with zipfile.ZipFile(zip_path) as z:
                    names = [n for n in z.namelist() if n.endswith(".csv")]
                    if names:
                        csv_path.write_bytes(z.read(names[0]))
                zip_path.unlink(missing_ok=True)
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=None,
                             names=["calc_time", "sym", "funding", "mark_price"],
                             usecols=["calc_time", "funding"])
            dfs.append(df)

        y, m = cur.year, cur.month
        m += 1
        if m > 12:
            m = 1; y += 1
        cur = date(y, m, 1)

    if not dfs:
        return pd.DataFrame(columns=["ts", "funding"])
    out = pd.concat(dfs, ignore_index=True)
    out["ts"] = pd.to_datetime(out["calc_time"], unit="ms", utc=True)
    out["funding"] = pd.to_numeric(out["funding"], errors="coerce")
    out = out.dropna(subset=["funding"]).sort_values("ts").reset_index(drop=True)
    out["ts"] = out["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    return out[["ts", "funding"]]


def load_klines(sym: str, do_download: bool) -> pd.DataFrame:
    """1h klines を読み込む (forward return 計算用)。"""
    cache = CACHE_DIR / sym / "klines_1h"
    cache.mkdir(parents=True, exist_ok=True)

    dfs = []
    from datetime import date
    cur = date.fromisoformat(TRAIN_START[:7] + "-01")
    end = date.fromisoformat(TEST_END[:7] + "-01")

    while cur <= end:
        ym = cur.strftime("%Y-%m")
        csv_path = cache / f"{sym}-1h-{ym}.csv"
        if not csv_path.exists():
            if not do_download:
                y, m = cur.year, cur.month
                m += 1
                if m > 12:
                    m = 1; y += 1
                cur = date(y, m, 1)
                continue
            zip_name = f"{sym}-1h-{ym}.zip"
            url = KLINE_URL.format(sym=sym) + zip_name
            zip_path = cache / zip_name
            if download_zip(url, zip_path):
                with zipfile.ZipFile(zip_path) as z:
                    names = [n for n in z.namelist() if n.endswith(".csv")]
                    if names:
                        csv_path.write_bytes(z.read(names[0]))
                zip_path.unlink(missing_ok=True)
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=None,
                             names=["open_time","open","high","low","close","vol",
                                    "close_time","quote_vol","n_trades",
                                    "taker_buy_base","taker_buy_quote","_"],
                             usecols=["open_time","close"])
            dfs.append(df)

        y, m = cur.year, cur.month
        m += 1
        if m > 12:
            m = 1; y += 1
        cur = date(y, m, 1)

    if not dfs:
        return pd.DataFrame(columns=["ts", "close"])
    out = pd.concat(dfs, ignore_index=True)
    out["ts"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)
    out["ts"] = out["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    return out[["ts", "close"]]


# ---------------------------------------------------------------------------
# シグナル構築
# ---------------------------------------------------------------------------

def build_signals(funding: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    """
    funding 毎に rolling ROLL_DAYS 日間の分位数でシグナルを立てる。
    forward return = klines の FWD_H 時間後の価格変化率。
    """
    if funding.empty or klines.empty:
        return pd.DataFrame()

    f = funding.copy().sort_values("ts").reset_index(drop=True)
    k = klines.copy().sort_values("ts").reset_index(drop=True)

    # rolling 分位数 (as-of: row i の分位数は row i-1 までで計算)
    roll_n = ROLL_DAYS * 3  # 1日3回の funding
    f["roll_hi"] = f["funding"].shift(1).rolling(roll_n, min_periods=roll_n // 2).quantile(P_HIGH)
    f["roll_lo"] = f["funding"].shift(1).rolling(roll_n, min_periods=roll_n // 2).quantile(P_LOW)
    f["sig_short"] = f["funding"] >= f["roll_hi"]   # HIGH_FUNDING → SHORT
    f["sig_long"]  = f["funding"] <= f["roll_lo"]   # LOW_FUNDING  → LONG

    # forward return: FWD_H 時間後の close / エントリー時の close - 1
    # funding の ts で nearest kline close を取る → merge_asof
    f_sorted = f.sort_values("ts")
    k_sorted = k.sort_values("ts")
    merged = pd.merge_asof(f_sorted, k_sorted.rename(columns={"close": "entry_close"}),
                           on="ts", direction="backward")
    # FWD_H 時間後の close
    k_fwd = k_sorted.copy()
    k_fwd["ts_fwd"] = k_fwd["ts"] - pd.Timedelta(hours=FWD_H)
    k_fwd2 = k_fwd.rename(columns={"ts_fwd": "ts", "close": "exit_close"})
    merged = pd.merge_asof(merged.sort_values("ts"),
                           k_fwd2[["ts", "exit_close"]].sort_values("ts"),
                           on="ts", direction="forward")

    merged = merged.dropna(subset=["entry_close", "exit_close", "roll_hi", "roll_lo"])
    # raw return
    merged["raw_ret"] = merged["exit_close"] / merged["entry_close"] - 1.0

    # SHORT signal: 値下がり = 利益 → pnl = -raw_ret - cost
    # LONG  signal: 値上がり = 利益 → pnl = +raw_ret - cost
    rows = []
    for _, row in merged.iterrows():
        if row["sig_short"] and not row["sig_long"]:
            rows.append({"ts": row["ts"], "direction": "SHORT",
                         "funding": row["funding"],
                         "pnl_net": -row["raw_ret"] - COST_RT})
        elif row["sig_long"] and not row["sig_short"]:
            rows.append({"ts": row["ts"], "direction": "LONG",
                         "funding": row["funding"],
                         "pnl_net": row["raw_ret"] - COST_RT})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, label: str, sym: str) -> dict:
    if df.empty:
        return {"sym": sym, "label": label, "n": 0}

    df = df.copy()
    df["win"] = df["pnl_net"] > 0
    df["year"] = df["ts"].dt.year
    df["ym"] = df["ts"].dt.to_period("M")

    n       = len(df)
    wr      = df["win"].mean() * 100
    avg     = df["pnl_net"].mean() * 100
    med     = df["pnl_net"].median() * 100
    active_months = df["ym"].nunique()
    total   = df["pnl_net"].sum() * 100

    year_groups = df.groupby("year")["pnl_net"].sum() * 100
    top1_year   = year_groups.abs().nlargest(1).values[0] if len(year_groups) > 0 else 0
    year_conc   = (top1_year / abs(total) * 100) if total != 0 else 0

    # maxDD (cumulative)
    cum = df["pnl_net"].cumsum()
    roll_max = cum.cummax()
    dd = (cum - roll_max) * 100
    max_dd = dd.min()

    return {
        "sym": sym, "label": label, "n": n, "WR": wr, "avg": avg,
        "median": med, "active_months": active_months, "total": total,
        "year_conc": year_conc, "maxDD": max_dd,
    }


def check_pass(r: dict, train_avg: float | None = None) -> list:
    fails = []
    if r.get("n", 0) == 0:
        fails.append("n=0"); return fails
    if r["avg"] <= PASS_AVG:
        fails.append(f"avg {r['avg']:.3f}% ≤ {PASS_AVG}")
    if r["WR"] < PASS_WR:
        fails.append(f"WR {r['WR']:.1f}% < {PASS_WR}")
    if r["active_months"] < PASS_MONTHS:
        fails.append(f"active_months {r['active_months']} < {PASS_MONTHS}")
    if train_avg is not None and (train_avg > 0) != (r["avg"] > 0):
        fails.append(f"train/test sign reversal (train_avg={train_avg:.3f}%)")
    if r["year_conc"] >= PASS_YEAR_TOP:
        fails.append(f"year_conc {r['year_conc']:.1f}% ≥ {PASS_YEAR_TOP}")
    return fails


def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ym"] = df["ts"].dt.to_period("M")
    return (df.groupby("ym")["pnl_net"]
              .agg(n="count", avg=lambda x: x.mean() * 100, WR=lambda x: (x > 0).mean() * 100)
              .reset_index())


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Binance public data から DL (ローカルMac実行時)")
    args = parser.parse_args()
    do_dl = args.download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("verify_funding_extreme.py — Funding Extreme Contrarian 仮説検証")
    print("=" * 70)
    print(f"ROLL_DAYS={ROLL_DAYS}d | P_HIGH={P_HIGH} | P_LOW={P_LOW}")
    print(f"FWD_H={FWD_H}h | COST_RT={COST_RT*100:.3f}%")
    print(f"train={TRAIN_START}〜{TRAIN_END} / test={TEST_START}〜{TEST_END}")
    print()

    all_results = {}
    for sym in SYMBOLS:
        print(f"--- {sym} ---")
        funding = load_funding(sym, do_dl)
        klines  = load_klines(sym, do_dl)
        print(f"  funding rows: {len(funding)}, kline rows: {len(klines)}")
        if funding.empty:
            print(f"  [SKIP] funding data not found. Run with --download")
            continue

        sigs = build_signals(funding, klines)
        if sigs.empty:
            print(f"  [SKIP] no signals generated"); continue

        sigs["ts"] = pd.to_datetime(sigs["ts"])
        train = sigs[(sigs["ts"] >= TRAIN_START) & (sigs["ts"] <= TRAIN_END)]
        test  = sigs[(sigs["ts"] >= TEST_START)  & (sigs["ts"] <= TEST_END)]

        for direction in ["SHORT", "LONG"]:
            tr = evaluate(train[train["direction"] == direction], "TRAIN", sym)
            te = evaluate(test [test ["direction"] == direction], "TEST",  sym)
            train_avg = tr.get("avg")
            fails = check_pass(te, train_avg=train_avg)
            ok = "✅ PASS" if not fails else "❌ FAIL"

            print(f"  {direction}:")
            print(f"    TRAIN: n={tr.get('n',0):4d} WR={tr.get('WR',0):.1f}% avg={tr.get('avg',0):.3f}% med={tr.get('median',0):.3f}%")
            print(f"    TEST:  n={te.get('n',0):4d} WR={te.get('WR',0):.1f}% avg={te.get('avg',0):.3f}% med={te.get('median',0):.3f}%  {ok}")
            if fails:
                for f in fails:
                    print(f"      → {f}")

            # 月次テーブル (TEST のみ)
            mt = monthly_table(test[test["direction"] == direction])
            if not mt.empty:
                print(f"    月次 (TEST):")
                for _, row in mt.iterrows():
                    bar = "+" if row["avg"] >= 0 else "-"
                    print(f"      {row['ym']}: n={int(row['n']):3d} WR={row['WR']:.0f}% avg={row['avg']:+.3f}%  {bar}")

            all_results[f"{sym}_{direction}"] = (tr, te, fails)
        print()

    # 総合判定
    print("=" * 70)
    print("【総合判定】")
    judged_btc = all_results.get("BTCUSDT_SHORT"), all_results.get("BTCUSDT_LONG")
    judged_eth = all_results.get("ETHUSDT_SHORT"), all_results.get("ETHUSDT_LONG")

    any_pass = False
    for sym, pairs in [("BTC", judged_btc), ("ETH", judged_eth)]:
        for direction, pair in zip(["SHORT", "LONG"], pairs):
            if pair is None:
                print(f"  {sym} {direction}: データ不足")
                continue
            _, te, fails = pair
            if not fails:
                print(f"  {sym} {direction}: ✅ PASS (avg={te['avg']:.3f}%, WR={te['WR']:.1f}%)")
                any_pass = True
            else:
                print(f"  {sym} {direction}: ❌ ({'; '.join(fails)})")

    if not any_pass:
        print()
        print("  → BTC/ETH いずれも合格基準未達。Funding Extreme 仮説: 不採用。")
    else:
        print()
        print("  → 合格あり。本番採用前に必ず leave-one-year-out / 感度分析を実施。")
    print()
    print("【合格基準 (事前固定)】")
    print(f"  avg>{PASS_AVG}% / WR≥{PASS_WR}% / active_months≥{PASS_MONTHS}")
    print(f"  train/test 符号一致 / top1年集中<{PASS_YEAR_TOP}%")
    print(f"  BTC+ETH 両方合格で仮説採用 (片方だけは不採用)")


if __name__ == "__main__":
    main()
