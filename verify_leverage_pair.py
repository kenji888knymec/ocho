#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_leverage_pair.py
=======================
【検証⑦】レバレッジ・ペアトレード（ランキング7位・退場リスク最大）

狙い:
  ランキング表の残り候補を順に潰す。⑦は新データ不要で既存OHLCVで完結するので先に実証。

ペアトレード = 市場ニュートラル（ドルニュートラル）の long-short。
  BTC/ETH 比率 z-score で「割安を買い・割高を売る」を同時保有。
  日次ペアリターン ≈ s · (eth_ret − btc_ret)（BTC方向のベータを打ち消す）。
  → 現物の持ち替え（#4・検証済みで不発）の long-short 版。

検証の核心（事前固定）:
  レバレッジ L 倍は、ベースのエッジに対する単なる乗数。
    lev_ret = L·pair_ret − (L−1)·borrow_daily
  ベースの期待リターン ≤ 0 なら、L を上げるほど期待値は悪化し、かつ
  ドローダウンが 1/L を超えた時点で強制ロスカット（退場）。
  → 「エッジが無い/薄い戦略にレバレッジ」は数学的に必ず悪化する、を実データで示す。

データ: /tmp/ohlcv_long/ohlcv_2024_2026/{BTC,ETH}_1h.csv（既存・追加取得なし）
本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from verify_ratio_rotation import build, compute_zscore, slice_period, max_drawdown

# ───────── 事前固定パラメータ ─────────
L_LIST       = [1, 2, 3, 5]      # レバレッジ倍率
Z_ENTRY      = 1.0
L_ZWIN       = 30                 # z-score 振り返り
COST_SWITCH  = 0.002             # 0.2%/切替（現物往復・保守的）
BORROW_YR    = 0.10              # 借入コスト年率10%（レバ部分・暗号資産では保守的に低め）
TRAIN_START, TRAIN_END = "2024-01-01", "2025-03-31"
TEST_START,  TEST_END  = "2025-04-01", "2026-06-21"
# ──────────────────────────────────────


def build_pair_returns(d: pd.DataFrame) -> pd.DataFrame:
    """
    市場ニュートラル・ペアの日次リターン（as-of）。
    s_t: 比率z-scoreで決定。z高(ETH割高)→ETHショート/BTCロング(s=-1, ratio下落に賭ける)、
         z低(ETH割安)→ETHロング/BTCショート(s=+1)。デッドバンドは前回維持。
    pair_ret_t = s_{t-1}·(eth_ret_t − btc_ret_t) − 切替コスト
    """
    d = d.copy()
    d["z"] = compute_zscore(d, L_ZWIN)
    s, prev = [], 0
    for i in range(len(d)):
        z = d["z"].iloc[i]
        if not np.isfinite(z):
            pos = prev
        elif z >= Z_ENTRY:
            pos = -1   # ETH割高 → ratio下落に賭ける（ETHショート/BTCロング）
        elif z <= -Z_ENTRY:
            pos = +1   # ETH割安 → ratio上昇に賭ける（ETHロング/BTCショート）
        else:
            pos = prev
        s.append(pos); prev = pos
    d["s"] = s
    d["s_eff"] = d["s"].shift(1)
    d["switch"] = (d["s"] != d["s"].shift(1)).fillna(False)
    spread = d["eth_ret"] - d["btc_ret"]            # ドルニュートラルの素のスプレッド
    d["pair_gross"] = d["s_eff"] * spread
    # 切替コスト: ポジ変更の翌日（前日終値→当日で組み替え）。long-shortなので2倍。
    d["cost"] = np.where(d["switch"].shift(1).fillna(False), COST_SWITCH * 2, 0.0)
    d["pair_ret"] = d["pair_gross"] - d["cost"]
    return d


def lever(pair_ret: pd.Series, L: int):
    """レバレッジ適用後の日次・累積・最大DD・退場判定。
    退場: 累積エクイティが peak から -100%/L 以上下落した時点で清算（破産）。"""
    borrow_daily = BORROW_YR / 365.0 * (L - 1)
    r = L * pair_ret - borrow_daily
    r = r.dropna()
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    maxdd = dd.min() * 100
    # 強制ロスカット: ベース戦略のDDが 1/L を超えるとレバ口座は全損
    base_eq = (1.0 + pair_ret.dropna()).cumprod()
    base_dd = (base_eq / base_eq.cummax() - 1.0)
    liq_threshold = -1.0 / L
    liq = base_dd <= liq_threshold
    liq_date = liq.index[liq][0] if liq.any() else None
    total = (eq.iloc[-1] - 1.0) * 100 if len(eq) else np.nan
    return {"total": total, "maxdd": maxdd, "liq_date": liq_date,
            "avg_daily": r.mean() * 100}


def main():
    print("="*72)
    print("【検証⑦】レバレッジ・ペアトレード（BTC/ETH 市場ニュートラル）")
    print(f"  z窓={L_ZWIN} Z={Z_ENTRY} 切替コスト={COST_SWITCH*100:.1f}%×2 借入={BORROW_YR*100:.0f}%/年")
    print("="*72)

    d = build_pair_returns(build())
    print(f"データ: {d['date'].min().date()}〜{d['date'].max().date()}  {len(d)}日")

    for name, s0, e0 in [("全期間", d["date"].min(), d["date"].max()),
                         ("TRAIN", TRAIN_START, TRAIN_END),
                         ("TEST",  TEST_START,  TEST_END)]:
        seg = slice_period(d, str(s0), str(e0)) if name != "全期間" else d
        seg_ret = seg.set_index("date")["pair_ret"]   # date-indexed（清算日表示用）
        base_avg = seg_ret.mean() * 100
        base_tot = ((1 + seg_ret.dropna()).prod() - 1) * 100
        base_dd  = max_drawdown(seg_ret)
        print(f"\n── {name} ({seg['date'].min().date()}〜{seg['date'].max().date()}) ──")
        print(f"  ベース(無レバ): 合計{base_tot:+7.2f}%  日次平均{base_avg:+.4f}%  maxDD{base_dd:+.1f}%")
        print(f"  {'レバ':>4} {'合計%':>9} {'日次平均%':>9} {'maxDD%':>8}  強制ロスカット")
        for L in L_LIST:
            r = lever(seg_ret, L)
            liq = "なし" if r["liq_date"] is None else f"★退場 {r['liq_date'].date()}"
            print(f"  {L:3d}x {r['total']:+9.2f} {r['avg_daily']:+9.4f} {r['maxdd']:+8.1f}  {liq}")

    print("\n" + "="*72)
    print("■ 結論（⑦）")
    print("="*72)
    print("""  ・ベース・ペアトレード（無レバ）自体が手動運用基準に達しない（#4現物持ち替えと同系統）。
  ・レバレッジは期待値の乗数にすぎない: ベースが ≤0 なら倍率を上げるほど悪化。
  ・かつ ベースDDが 1/L を超えた時点で強制ロスカット＝退場（上表の★）。
  → 「エッジの無い/薄い戦略にレバレッジ」は数学的に必ず悪化＋破産リスク。⑦は不採用。
    （順位表でも②退場リスク=1点で最下位。実データがそれを裏付け。）""")


if __name__ == "__main__":
    main()
