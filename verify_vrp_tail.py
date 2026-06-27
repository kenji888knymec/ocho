#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_vrp_tail.py
==================
【検証A・テール可視化（1回限り）】BTC VRP の最大の弱点だけを見る

目的:
  検証A（verify_vrp.py）で BTC は IV>RV の構造が明確に出た。
  ただしボラ売りは「平常時に小さく勝ち、暴落・IV急騰時に一発で壊れる」型。
  本番採用ではなく「監視トラックに残すか」を判断するため、最大の弱点＝
  テールリスクだけを可視化する。

★これは合格基準ではない。閾値も条件も作らない。後出しで合格化もしない。
  事前固定の総合判定は「BTC/ETH両方が5基準」＝不合格のまま固定する。

見る項目（これだけ・BTCのみ）:
  1. VRPが大きくマイナスだった最悪日（RV_fwd が DVOL を大きく上回った局面）
  2. 月次VRPの最悪月
  3. 簡易ボラ売りPnLの最大ドローダウン・最長水中期間
  4. 月次PnLの最悪月
  5. DVOL水準/変化で見た「IV急騰局面」での VRP の壊れ方（記述のみ・売買ルールではない）

データ: dvol_BTC.csv + /tmp/ohlcv_long/ohlcv_2024_2026/BTC_1h.csv（verify_vrp.py と同じ）
本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from verify_vrp import (
    load_dvol, load_daily_close, compute_rv_forward,
    RV_WINDOW, COST_PER_YR,
)


def build_btc() -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dvol  = load_dvol("BTC", script_dir)
    close = load_daily_close("BTC")
    rv    = compute_rv_forward(close)
    d = pd.DataFrame({"dvol": dvol, "rv_fwd": rv}).dropna()
    d["vrp"] = d["dvol"] - d["rv_fwd"]
    # 簡易ボラ売りPnL（verify_vrp.py と完全に同じ定義・線形近似）
    d["pnl_d"] = d["vrp"] / 365 - (COST_PER_YR / 365)
    # 参考: DVOLの前日比（IV急騰の記述用）
    d["dvol_chg"] = d["dvol"].diff()
    return d


def max_drawdown_curve(daily_pnl: pd.Series):
    """累積PnL（%・単純加算）のドローダウン。最大DDと最長水中日数を返す。"""
    eq = daily_pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min()
    # 最長水中期間
    underwater = dd < 0
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return max_dd, longest, eq, dd


def main():
    print("="*70)
    print("【検証A・テール可視化】BTC VRP の最大の弱点だけを見る（1回限り）")
    print("  ※合格基準ではない。閾値も条件も作らない。総合判定は不合格のまま固定。")
    print("="*70)

    d = build_btc()
    print(f"\n重複期間: {d.index[0].date()} 〜 {d.index[-1].date()}  ({len(d)}日)")
    print(f"VRP: avg={d['vrp'].mean():+.2f}  med={d['vrp'].median():+.2f}  "
          f"std={d['vrp'].std():.2f}  min={d['vrp'].min():+.2f}  max={d['vrp'].max():+.2f} (vol pt)")

    # ── 1. VRP最悪日（RV_fwd が DVOL を大きく上回った日） ──
    print("\n" + "─"*70)
    print("① VRP最悪日 ワースト15（負け＝RV_fwdがDVOLを上回り、ボラ売りが踏まれた局面）")
    print("   date         DVOL   RV_fwd    VRP(vol pt)")
    worst = d.nsmallest(15, "vrp")
    for dt, row in worst.iterrows():
        print(f"   {dt.date()}   {row['dvol']:5.1f}  {row['rv_fwd']:6.1f}   {row['vrp']:+7.1f}")
    # VRP分布の下側パーセンタイル
    print("\n   VRP下側パーセンタイル（vol pt）:")
    for p in [1, 5, 10, 25]:
        print(f"     p{p:2d} = {np.percentile(d['vrp'], p):+.1f}")
    neg_days = (d["vrp"] < 0).mean() * 100
    big_neg = (d["vrp"] < -10).mean() * 100
    print(f"   VRP<0 の日: {neg_days:.1f}%   VRP<-10 の日: {big_neg:.1f}%")

    # ── 2. 月次VRP最悪月 ──
    print("\n" + "─"*70)
    print("② 月次VRP（平均）ワースト8")
    d["ym"] = d.index.to_period("M")
    m_vrp = d.groupby("ym")["vrp"].mean().sort_values()
    for ym, v in m_vrp.head(8).items():
        print(f"   {ym}: {v:+6.1f} vol pt")

    # ── 3. 簡易ボラ売りPnL 最大DD・最長水中 ──
    print("\n" + "─"*70)
    print("③ 簡易ボラ売りPnL（線形近似・verify_vrp.pyと同定義）のドローダウン")
    max_dd, longest, eq, dd = max_drawdown_curve(d["pnl_d"])
    total = eq.iloc[-1]
    print(f"   累積PnL（単純加算, 期間合計）: {total:+.2f}%")
    print(f"   最大ドローダウン           : {max_dd:+.2f}%")
    print(f"   最長水中期間               : {longest}日")
    dd_date = dd.idxmin()
    print(f"   最大DD到達日               : {dd_date.date()}")

    # ── 4. 月次PnL最悪月 ──
    print("\n" + "─"*70)
    print("④ 月次PnL（線形近似）ワースト8")
    m_pnl = d.groupby("ym")["pnl_d"].sum().sort_values()
    for ym, v in m_pnl.head(8).items():
        print(f"   {ym}: {v:+6.2f}%")

    # ── 5. IV急騰局面での壊れ方（記述のみ・売買ルールではない） ──
    print("\n" + "─"*70)
    print("⑤ DVOL前日比（IV変化）で5分割した翌VRP（記述のみ・売買ルールではない）")
    print("   ※『IVが急騰した日にボラを売っていたら、その後どうなったか』の事後記述")
    d2 = d.dropna(subset=["dvol_chg"]).copy()
    d2["bucket"] = pd.qcut(d2["dvol_chg"], 5, labels=["急低下","低下","横ばい","上昇","急上昇"])
    g = d2.groupby("bucket", observed=True)["vrp"].agg(["mean", "median", "count"])
    print("   DVOL変化帯   VRP平均   VRP中央   n")
    for b, row in g.iterrows():
        print(f"   {str(b):8s}   {row['mean']:+6.1f}   {row['median']:+6.1f}   {int(row['count']):4d}")

    # ── テールリスクの正直な注記 ──
    print("\n" + "="*70)
    print("■ 正直な注記（誇張しないため）")
    print("="*70)
    print("""  ・上記PnLは VRP/365 の線形近似。実際のオプション売りは vega損・gamma損・
    証拠金増加・強制ロスカットを伴い、IV急騰時の真の損失はこの近似より大きい。
  ・よって③の最大DDは『下限の楽観値』。実運用ではこれより悪くなりうる。
  ・①の最悪日のVRP（vol pt）こそが、ショートボラが一発で踏まれる規模感を示す。
  ・この可視化は採否判定ではなく、監視トラックに残すか否かの材料。""")


if __name__ == "__main__":
    main()
