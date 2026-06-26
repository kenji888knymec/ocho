#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_xsec_momentum.py
=======================
【検証C''】28銘柄 クロスセクション相対強弱モメンタム（ロングオンリー・現物・退場なし）

仮説:
  「相対強弱モメンタム（今強い銘柄を持つ）」は、BTC/ETHの1ペアの偶然ではなく
  銘柄群全体でも効く。

設計（現物のみ・ショートなし・強制決済なし・常に100%ロング）:
  - 28銘柄を「過去L日リターン」でランク付け（as-of・完了足のみ）
  - 上位N銘柄を等金額で保有、H日ごとにリバランス
  - 切替コスト 0.4%/往復（C'の0.2%から倍にストレス・保守的固定）
  - 選定は t 日終値で確定 → t+1 日以降のリターンを得る（未来情報なし）

★ベンチマーク（最初から固定・後出ししない）:
  「28銘柄 等金額 毎日リバランス」= 受動の市場平均。これに勝てなければ無意味。

合格基準（事前固定・5つ全て）:
  ① test で「28銘柄等金額hold」に勝つ
  ② train/test 両方で勝つ（符号一貫）
  ③ L(20/30/45/60)・N(3/5) すべてで勝つ
  ④ LOMO（1ヶ月除外）で優位が残る
  ⑤ 最大DDが等金額holdより極端に悪くない（-10pt以内）

データ: /tmp/ohlcv_long/ohlcv_2024_2026/*_1h.csv
本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。
"""

from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

# ───────── 事前固定パラメータ（変更禁止） ─────────
DATA_DIR    = "/tmp/ohlcv_long/ohlcv_2024_2026"
L_PRIMARY   = 30
N_PRIMARY   = 3
H_REBAL     = 7              # リバランス間隔（日）
COST_RT     = 0.004          # 0.4%/往復（保守的・倍ストレス）
L_GRID      = [20, 30, 45, 60]
N_GRID      = [3, 5]
TRAIN_START = "2024-01-01"
TRAIN_END   = "2025-03-31"
TEST_START  = "2025-04-01"
TEST_END    = "2026-06-21"
SUSPECT     = ["POL", "SUI", "SEI", "BONK"]  # 合成疑いトークン（除外頑健性用）
# ──────────────────────────────────────────────


def load_daily_close(sym: str) -> pd.Series:
    df = pd.read_csv(f"{DATA_DIR}/{sym}_1h.csv")
    df["dt"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("dt")
    df["date"] = df["dt"].dt.date
    daily = df.groupby("date")["close"].last()
    daily.index = pd.to_datetime(daily.index)
    daily.name = sym
    return daily


def build_panel(symbols) -> pd.DataFrame:
    series = [load_daily_close(s) for s in symbols]
    panel = pd.concat(series, axis=1).sort_index()
    return panel  # index=date, columns=symbols, values=close


def simulate(panel: pd.DataFrame, L: int, N: int, H: int):
    """
    クロスセクション相対強弱・ロングオンリー。
    戻り値: DataFrame(date, strat_ret, bench_ret)
    """
    close = panel.copy()
    rets = close.pct_change()  # 日次リターン（t-1→t）
    dates = close.index
    n_days = len(dates)

    # ベンチマーク: 28銘柄等金額・毎日リバランス = 各日の有効銘柄の平均リターン
    bench_ret = rets.mean(axis=1, skipna=True)

    active = None           # 現在保有中の銘柄リスト（前日までに決定済み）
    pending = None          # i日終値で決めた次のセット（i+1日から有効）
    prev_set = set()
    strat_ret = pd.Series(index=dates, dtype=float)

    for i in range(n_days):
        # ── ① 昨日終値で決めた pending を今日から適用（as-of: 先読みなし） ──
        cost_today = 0.0
        if pending is not None:
            new_set = set(pending)
            changed = len(new_set.symmetric_difference(prev_set)) / 2 if prev_set else len(new_set)
            cost_today = (changed / len(pending)) * COST_RT  # 入替分のみ往復コスト
            active = pending
            prev_set = new_set
            pending = None

        # ── ② 今日の戦略リターン（active = 前日までに確定したセット） ──
        if active is not None:
            day_ret = rets.iloc[i][active].mean(skipna=True)
            strat_ret.iloc[i] = day_ret - cost_today

        # ── ③ 今日の終値でリバランス判定 → pending に入れ、翌日から有効 ──
        if i >= L and (i - L) % H == 0:
            mom = close.iloc[i] / close.iloc[i - L] - 1.0  # 過去L日リターン（完了足）
            mom = mom.dropna()
            if len(mom) >= N:
                pending = list(mom.sort_values(ascending=False).head(N).index)

    out = pd.DataFrame({
        "date": dates,
        "strat_ret": strat_ret.values,
        "bench_ret": bench_ret.values,
    })
    return out


def cum(r: pd.Series) -> float:
    r = r.dropna()
    return ((1 + r).prod() - 1) * 100 if len(r) else np.nan


def maxdd(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    eq = (1 + r).cumprod()
    return ((eq / eq.cummax()) - 1).min() * 100


def slice_p(d: pd.DataFrame, s: str, e: str) -> pd.DataFrame:
    m = (d["date"] >= pd.Timestamp(s)) & (d["date"] <= pd.Timestamp(e))
    return d[m].reset_index(drop=True)


def report(name: str, d: pd.DataFrame):
    st = cum(d["strat_ret"]); bm = cum(d["bench_ret"])
    dds = maxdd(d["strat_ret"]); ddb = maxdd(d["bench_ret"])
    win = "✅勝ち" if st > bm else "❌負け"
    print(f"\n=== {name} ({d['date'].min().date()}〜{d['date'].max().date()}, {len(d)}日) ===")
    print(f"  戦略         : {st:+8.2f}%  (maxDD {dds:+.1f}%)")
    print(f"  等金額hold   : {bm:+8.2f}%  (maxDD {ddb:+.1f}%)")
    print(f"  → 差 {st-bm:+.2f}pt  {win}")
    return {"strat": st, "bench": bm, "beat": st > bm, "dds": dds, "ddb": ddb}


def main():
    print("="*70)
    print("【検証C''】28銘柄 クロスセクション相対強弱モメンタム（ロングオンリー）")
    print("="*70)
    symbols = sorted([os.path.basename(f).replace("_1h.csv", "")
                      for f in glob.glob(f"{DATA_DIR}/*_1h.csv")])
    print(f"銘柄数: {len(symbols)}")
    print(f"事前固定: L={L_PRIMARY}, N={N_PRIMARY}, H={H_REBAL}日, コスト={COST_RT*100:.1f}%/往復")
    print(f"ベンチマーク: 28銘柄等金額・毎日リバランス（最初から固定）")

    panel = build_panel(symbols)

    # ── 主パラメータ ──
    d = simulate(panel, L_PRIMARY, N_PRIMARY, H_REBAL)
    tr = slice_p(d, TRAIN_START, TRAIN_END)
    te = slice_p(d, TEST_START, TEST_END)
    print("\n" + "─"*70)
    print(f"■ 主パラメータ L={L_PRIMARY}, N={N_PRIMARY}")
    r_tr = report("TRAIN", tr)
    r_te = report("TEST", te)

    print("\n" + "─"*70)
    print("■ 合格基準チェック")
    c1 = r_te["beat"]
    c2 = r_tr["beat"] and r_te["beat"]
    print(f"  ① test で等金額holdに勝つ        : {'✅' if c1 else '❌'}")
    print(f"  ② train/test 両方で勝つ          : {'✅' if c2 else '❌'}")

    # ── ③ L×N グリッド ──
    print("\n  ③ L×N グリッド頑健性（test期間）:")
    grid = []
    for L in L_GRID:
        for N in N_GRID:
            dLN = simulate(panel, L, N, H_REBAL)
            teLN = slice_p(dLN, TEST_START, TEST_END)
            st = cum(teLN["strat_ret"]); bm = cum(teLN["bench_ret"])
            beat = st > bm; grid.append(beat)
            print(f"     L={L:3d} N={N}: 戦略 {st:+8.2f}% vs hold {bm:+8.2f}%  {'✅' if beat else '❌'}")
    c3 = all(grid)
    print(f"  ③ 全L×Nで勝つ                    : {'✅' if c3 else '❌'}")

    # ── ④ LOMO ──
    print("\n  ④ LOMO（test内・1ヶ月除外）:")
    te2 = te.copy(); te2["ym"] = te2["date"].dt.to_period("M")
    lomo = []
    for m in sorted(te2["ym"].unique()):
        sub = te2[te2["ym"] != m]
        st = cum(sub["strat_ret"]); bm = cum(sub["bench_ret"])
        beat = st > bm; lomo.append(beat)
        print(f"     {str(m)}除外: 戦略 {st:+8.2f}% vs hold {bm:+8.2f}%  {'✅' if beat else '❌'}")
    c4 = all(lomo)
    print(f"  ④ 全月除外で勝つ                 : {'✅' if c4 else '❌'}")

    # ── ⑤ DD ──
    c5 = r_te["dds"] >= r_te["ddb"] - 10.0
    print(f"\n  ⑤ DDが等金額holdより極端に悪くない: {'✅' if c5 else '❌'}"
          f"  (戦略{r_te['dds']:+.1f}% vs hold{r_te['ddb']:+.1f}%)")

    # ── 総合 ──
    print("\n" + "="*70)
    allpass = c1 and c2 and c3 and c4 and c5
    print(f"■ 総合判定: {'✅ 合格（本物の候補に格上げ）' if allpass else '❌ 不合格'}")
    print(f"   ①{'✅' if c1 else '❌'} ②{'✅' if c2 else '❌'} ③{'✅' if c3 else '❌'} "
          f"④{'✅' if c4 else '❌'} ⑤{'✅' if c5 else '❌'}")
    print("="*70)

    # ── 合成疑い銘柄を除外した頑健性（参考・合否には使わない） ──
    print("\n■ 参考: 合成疑い銘柄除外（POL/SUI/SEI/BONK）の頑健性チェック")
    clean = [s for s in symbols if s not in SUSPECT]
    panel_c = build_panel(clean)
    dc = simulate(panel_c, L_PRIMARY, N_PRIMARY, H_REBAL)
    report("TEST(除外版)", slice_p(dc, TEST_START, TEST_END))


if __name__ == "__main__":
    main()
