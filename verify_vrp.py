#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_vrp.py
=============
【検証A】BTC/ETH VRP（ボラティリティリスクプレミアム）構造検証

仮説:
  BTC/ETHのインプライドボラ(DVOL)は、その後30日に実現したボラ(RV)より
  構造的に高い。= VRP = DVOL - RV_forward > 0 が平均的に成立。

合格基準（事前固定・5つ）:
  ① 全期間で VRP=(DVOL-RV_forward) の平均 > 0
  ② VRP>0 の日が60%以上（恒常的に割高）
  ③ train/test 両方で平均>0（符号一貫）
  ④ BTC・ETH 両方で成立
  ⑤ 簡易ボラ売りPnL（年率, コスト2%控除後）がプラス

データ:
  DVOL  : dvol_BTC.csv / dvol_ETH.csv（同ディレクトリに配置）
           Deribit get_volatility_index_data API から取得（%年率）
  OHLCV : /tmp/ohlcv_long/ohlcv_2024_2026/{BTC,ETH}_1h.csv
           重複期間: OHLCV が 2024-01-01 からのため 2024-01 以降のみ有効

注意:
  RV_forward は t+1〜t+30 日の事後値 = 検証専用（リアルタイム未来情報）。
  「VRPが過去に存在したか」の構造確認が目的。実運用設計は別途。

本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

# ───────── 事前固定パラメータ（変更禁止） ─────────
OHLCV_DIR   = "/tmp/ohlcv_long/ohlcv_2024_2026"
RV_WINDOW   = 30            # 日。DVOLの参照期間（30日相当）と揃える
ANNUALIZE   = np.sqrt(365)  # 日次ボラ → 年率換算
COST_PER_YR = 2.0           # 年率コスト%（bid-ask+margin+諸費用の保守的見積もり）
TRAIN_START = "2024-01-01"
TRAIN_END   = "2025-03-31"
TEST_START  = "2025-04-01"
TEST_END    = "2026-12-31"
CURRENCIES  = ["BTC", "ETH"]
# ──────────────────────────────────────────────


def load_dvol(currency: str, script_dir: str) -> pd.Series:
    path = os.path.join(script_dir, f"dvol_{currency}.csv")
    if not os.path.exists(path):
        print(f"\n❌ {path} が見つかりません。")
        print(f"   Macで fetch_deribit_dvol.py を実行し、dvol_{currency}.csv を")
        print(f"   {script_dir}/ に配置してください。")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["date_utc"])
    df = df.sort_values("date_utc").reset_index(drop=True)
    s = df.set_index("date_utc")["dvol_close"]
    s.index.name = "date"
    s = s.rename(f"dvol_{currency}")
    print(f"  DVOL {currency}: {s.index[0].date()} 〜 {s.index[-1].date()}  "
          f"({len(s)}日, avg={s.mean():.1f}%, range=[{s.min():.0f}〜{s.max():.0f}])")
    return s


def load_daily_close(currency: str) -> pd.Series:
    path = f"{OHLCV_DIR}/{currency}_1h.csv"
    if not os.path.exists(path):
        print(f"\n❌ {path} が見つかりません。")
        sys.exit(1)
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("dt")
    df["date"] = df["dt"].dt.date
    daily = df.groupby("date")["close"].last()
    daily.index = pd.to_datetime(daily.index)
    daily.name = f"close_{currency}"
    print(f"  OHLCV {currency}: {daily.index[0].date()} 〜 {daily.index[-1].date()}  ({len(daily)}日)")
    return daily


def compute_rv_forward(close: pd.Series) -> pd.Series:
    """
    t日の「翌 RV_WINDOW 日間」実現ボラ（年率%、事後値・検証専用）。
    RV[t] = std(log_ret[t+1..t+RV_WINDOW]) * sqrt(365) * 100
    DVOLの単位（%年率）と直接比較できるよう annualize×100。
    shift(-1)でt+1日以降の収益を使い、t日の足は含めない（as-of原則との対比で明示）。
    """
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.shift(-1).rolling(RV_WINDOW).std() * ANNUALIZE * 100
    rv.name = f"rv_fwd{RV_WINDOW}d"
    return rv


def period_stats(vrp: pd.Series, label: str) -> dict:
    s = vrp.dropna()
    if len(s) == 0:
        print(f"  [{label}] データなし（この期間はスキップ）")
        return {"n": 0, "avg": np.nan, "pos_pct": np.nan, "med": np.nan}
    avg     = s.mean()
    med     = s.median()
    pos_pct = (s > 0).mean() * 100
    print(f"  [{label:38s}]  n={len(s):4d}日  "
          f"avg={avg:+6.2f}%pt  med={med:+6.2f}%pt  VRP>0={pos_pct:.1f}%")
    return {"n": len(s), "avg": avg, "pos_pct": pos_pct, "med": med}


def verify_one(currency: str, script_dir: str) -> dict:
    print(f"\n{'='*65}")
    print(f"  {currency}")
    print(f"{'='*65}")
    dvol  = load_dvol(currency, script_dir)
    close = load_daily_close(currency)
    rv    = compute_rv_forward(close)

    d = pd.DataFrame({"dvol": dvol, "rv_fwd": rv})
    d = d.dropna()
    if len(d) == 0:
        print("  ❌ DVOL と OHLCV の重複期間がゼロ。確認してください。")
        sys.exit(1)
    d["vrp"] = d["dvol"] - d["rv_fwd"]
    print(f"\n  重複期間: {d.index[0].date()} 〜 {d.index[-1].date()}  ({len(d)}日)")
    print(f"  DVOL平均={d['dvol'].mean():.1f}%年率   RV_fwd平均={d['rv_fwd'].mean():.1f}%年率")

    tm = (d.index >= TRAIN_START) & (d.index <= TRAIN_END)
    te = (d.index >= TEST_START)  & (d.index <= TEST_END)

    print()
    r_all   = period_stats(d["vrp"],           "全期間")
    r_train = period_stats(d.loc[tm, "vrp"],   f"TRAIN({TRAIN_START}〜{TRAIN_END})")
    r_test  = period_stats(d.loc[te, "vrp"],   f"TEST({TEST_START}〜{TEST_END})")

    # ⑤ 簡易ボラ売りP&L
    # 毎日 DVOL で売り → 30日後 RV で実質決着。線形近似: PnL_d ≈ (VRP/365) - cost_d
    cost_daily = COST_PER_YR / 365
    d["pnl_d"] = d["vrp"] / 365 - cost_daily
    pnl_all   = d["pnl_d"].mean() * 365
    pnl_train = d.loc[tm, "pnl_d"].mean() * 365 if tm.sum() > 0 else np.nan
    pnl_test  = d.loc[te, "pnl_d"].mean() * 365 if te.sum() > 0 else np.nan
    print(f"\n  ⑤ 簡易ボラ売りPnL（コスト{COST_PER_YR:.0f}%/年控除後）:")
    print(f"     全期間={pnl_all:+.2f}%/年", end="")
    if not np.isnan(pnl_train):
        print(f"   TRAIN={pnl_train:+.2f}%/年", end="")
    if not np.isnan(pnl_test):
        print(f"   TEST={pnl_test:+.2f}%/年", end="")
    print()

    # 月次サマリー（leave-one-month感覚で）
    d["ym"] = d.index.to_period("M")
    monthly_avg = d.groupby("ym")["vrp"].mean()
    pos_months  = (monthly_avg > 0).sum()
    print(f"\n  月次: {len(monthly_avg)}ヶ月中 {pos_months}ヶ月でVRP>0")
    if len(monthly_avg) <= 36:
        for ym, v in monthly_avg.items():
            print(f"     {ym}: {v:+.1f}%pt {'✅' if v > 0 else '❌'}")

    return {
        "all": r_all, "train": r_train, "test": r_test,
        "pnl_all": pnl_all, "pnl_train": pnl_train, "pnl_test": pnl_test,
        "monthly_pos": pos_months, "monthly_total": len(monthly_avg),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("="*70)
    print("【検証A】VRP（ボラティリティリスクプレミアム）構造検証")
    print(f"  RVウィンドウ={RV_WINDOW}日  コスト={COST_PER_YR:.0f}%/年（保守的）")
    print(f"  TRAIN={TRAIN_START}〜{TRAIN_END}  TEST={TEST_START}〜{TEST_END}")
    print("="*70)

    results = {}
    for cur in CURRENCIES:
        results[cur] = verify_one(cur, script_dir)

    print("\n" + "="*70)
    print("■ 合格基準チェック（事前固定・5つ全て満たすこと）")
    print("="*70)

    t = lambda b: "✅" if b else "❌"
    ok = {}
    for cur in CURRENCIES:
        r = results[cur]
        c1    = bool(r["all"]["avg"] > 0)   if r["all"]["n"]   > 0 else False
        c2    = bool(r["all"]["pos_pct"] >= 60) if r["all"]["n"] > 0 else False
        c3_tr = bool(r["train"]["avg"] > 0) if r["train"]["n"] > 0 else False
        c3_te = bool(r["test"]["avg"]  > 0) if r["test"]["n"]  > 0 else False
        c3    = c3_tr and c3_te
        c5    = bool(r["pnl_all"] > 0)
        ok[cur] = {"c1": c1, "c2": c2, "c3": c3, "c5": c5}
        tr_avg = r["train"]["avg"] if r["train"]["n"] > 0 else float("nan")
        te_avg = r["test"]["avg"]  if r["test"]["n"]  > 0 else float("nan")
        print(f"\n  {cur}:")
        print(f"    ① 全期間 VRP平均>0     : {t(c1)}  ({r['all']['avg']:+.2f}%pt)")
        print(f"    ② VRP>0が60%以上       : {t(c2)}  ({r['all']['pos_pct']:.1f}%)")
        print(f"    ③ train/test 両方>0    : {t(c3)}  (train={tr_avg:+.2f}%pt / test={te_avg:+.2f}%pt)")
        print(f"    ⑤ ボラ売りPnL>0        : {t(c5)}  ({r['pnl_all']:+.2f}%/年)")

    c4 = all(ok[c]["c1"] for c in CURRENCIES)
    print(f"\n    ④ BTC/ETH 両方で①成立  : {t(c4)}")

    all_pass = c4 and all(
        ok[c]["c1"] and ok[c]["c2"] and ok[c]["c3"] and ok[c]["c5"]
        for c in CURRENCIES
    )
    print("\n" + "="*70)
    if all_pass:
        print("■ 総合判定: ✅ 合格（VRP構造が実在 → A路線として研究継続）")
    else:
        print("■ 総合判定: ❌ 不合格")
    print("="*70)


if __name__ == "__main__":
    main()
