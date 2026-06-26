#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ratio_rotation.py
========================
【検証C】現物BTC/ETH 比率ローテーション（ショートなし・退場なし）

仮説:
  ETH/BTC 比率は平均回帰する。
    比率が安値（ETH割安）→ ETHを保有
    比率が高値（ETH割高）→ BTCを保有
  → 単純に1銘柄を持ち続ける(buy&hold)より総資産が増えるか。

設計（現物のみ・ショートなし・強制決済なし）:
  - 常に100%仮想通貨を保有（BTCかETHのどちらか）
  - ETH/BTC 比率の z-score で持ち替える
  - 全特徴量は as-of（完了足のみ）。シグナルは t 日終値で確定し、t+1 日に適用
  - 切替コストは 0.2%/回（現物の往復・保守的）

合格基準（事前固定・結果を見てから変えない）:
  ① test期間で「BTC hold」「ETH hold」両方に勝つ
  ② train/test 両方で最良 buy&hold に勝つ（符号一貫）
  ③ L(z-score振り返り日数) を 20/30/45/60 で変えても最良 hold に勝つ
  ④ LOMO(1ヶ月除外)で優位が残る
  ⑤ 最大DDが「持ちっぱなし」より極端に悪くない

データ: /tmp/ohlcv_long/ohlcv_2024_2026/{BTC,ETH}_1h.csv
        列: symbol,source_symbol,timestamp,open,high,low,close,volume,datetime_utc

本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ───────── 事前固定パラメータ（変更禁止） ─────────
DATA_DIR    = "/tmp/ohlcv_long/ohlcv_2024_2026"
L_PRIMARY   = 30        # z-score 振り返り日数（主）
L_GRID      = [20, 30, 45, 60]  # 頑健性チェック用
Z_ENTRY     = 1.0       # 持ち替え閾値
COST_SWITCH = 0.002     # 0.2%/回（現物往復・保守的）
TRAIN_START = "2024-01-01"
TRAIN_END   = "2025-03-31"
TEST_START  = "2025-04-01"
TEST_END    = "2026-06-21"
# ──────────────────────────────────────────────


def load_daily(sym: str) -> pd.DataFrame:
    """1h CSV を日次終値に集約。as-of: 日次は『その日の最終終値』。"""
    df = pd.read_csv(f"{DATA_DIR}/{sym}_1h.csv")
    df["dt"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["date"] = df["dt"].dt.date
    daily = df.groupby("date").agg(close=("close", "last")).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily[["date", "close"]]


def build() -> pd.DataFrame:
    btc = load_daily("BTC").rename(columns={"close": "btc"})
    eth = load_daily("ETH").rename(columns={"close": "eth"})
    d = pd.merge(btc, eth, on="date", how="inner").sort_values("date").reset_index(drop=True)
    d["ratio"] = d["eth"] / d["btc"]
    # 日次リターン（t-1 終値 → t 終値）
    d["btc_ret"] = d["btc"].pct_change()
    d["eth_ret"] = d["eth"].pct_change()
    return d


def compute_zscore(d: pd.DataFrame, L: int) -> pd.Series:
    """as-of z-score: t 日の z は t-1..t-L の平均/標準偏差で計算（未来情報なし）。"""
    r = d["ratio"]
    mean = r.shift(1).rolling(L).mean()
    std  = r.shift(1).rolling(L).std()
    z = (r - mean) / std
    return z


def run_strategy(d: pd.DataFrame, L: int) -> pd.DataFrame:
    """
    ポジション決定: t 日終値で z を見て position[t] を確定。
    実現リターンは t→t+1 で position[t] を適用（as-of・未来情報なし）。
    position: 'ETH' or 'BTC'
    """
    d = d.copy()
    d["z"] = compute_zscore(d, L)

    positions = []
    prev = "BTC"  # 初期はBTC（シグナル確定前）
    for i in range(len(d)):
        z = d.at[i, "z"]
        if not np.isfinite(z):
            pos = prev  # ウォームアップ中は前回維持
        elif z <= -Z_ENTRY:
            pos = "ETH"     # ETH割安 → ETH保有（比率が戻る=ETH優位に賭ける）
        elif z >= Z_ENTRY:
            pos = "BTC"     # ETH割高 → BTC保有
        else:
            pos = prev      # デッドバンド: 前回維持（ヒステリシス）
        positions.append(pos)
        prev = pos
    d["pos"] = positions

    # t 日のポジを t+1 日リターンに適用
    d["pos_eff"] = d["pos"].shift(1)  # 前日終値で決めたポジを当日リターンに当てる
    d["switch"] = (d["pos"] != d["pos"].shift(1)).fillna(False)

    # 戦略リターン
    def strat_ret(row):
        if not isinstance(row["pos_eff"], str):
            return np.nan
        return row["eth_ret"] if row["pos_eff"] == "ETH" else row["btc_ret"]
    d["strat_ret_gross"] = d.apply(strat_ret, axis=1)

    # 切替コスト: ポジが変わった「当日」に発生（前日終値→当日始値で乗り換える想定）
    d["cost"] = np.where(d["switch"].shift(1).fillna(False), COST_SWITCH, 0.0)
    d["strat_ret"] = d["strat_ret_gross"] - d["cost"]
    return d


def cum_return(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) == 0:
        return np.nan
    return (1.0 + r).prod() - 1.0


def max_drawdown(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) == 0:
        return np.nan
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    return dd.min() * 100.0  # %


def n_switches(d: pd.DataFrame) -> int:
    return int(d["switch"].sum())


def slice_period(d: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))
    return d[m].reset_index(drop=True)


def report_period(name: str, d: pd.DataFrame):
    strat = cum_return(d["strat_ret"]) * 100
    bh_btc = cum_return(d["btc_ret"]) * 100
    bh_eth = cum_return(d["eth_ret"]) * 100
    best_hold = max(bh_btc, bh_eth)
    dd_strat = max_drawdown(d["strat_ret"])
    dd_btc = max_drawdown(d["btc_ret"])
    dd_eth = max_drawdown(d["eth_ret"])
    sw = n_switches(d)
    win = "✅勝ち" if strat > best_hold else "❌負け"
    print(f"\n=== {name} ({d['date'].min().date()} 〜 {d['date'].max().date()}, {len(d)}日) ===")
    print(f"  戦略         : {strat:+8.2f}%   (切替 {sw}回, maxDD {dd_strat:+.1f}%)")
    print(f"  BTC hold     : {bh_btc:+8.2f}%   (maxDD {dd_btc:+.1f}%)")
    print(f"  ETH hold     : {bh_eth:+8.2f}%   (maxDD {dd_eth:+.1f}%)")
    print(f"  最良hold     : {best_hold:+8.2f}%")
    print(f"  → 戦略 vs 最良hold: {strat - best_hold:+.2f}pt  {win}")
    return {"strat": strat, "best_hold": best_hold, "beat": strat > best_hold,
            "dd_strat": dd_strat, "dd_btc": dd_btc, "dd_eth": dd_eth}


def main():
    print("="*70)
    print("【検証C】現物 BTC/ETH 比率ローテーション（ショートなし・退場なし）")
    print("="*70)
    d_all = build()
    print(f"データ: {d_all['date'].min().date()} 〜 {d_all['date'].max().date()}  {len(d_all)}日")
    print(f"事前固定: L={L_PRIMARY}, Z_ENTRY={Z_ENTRY}, 切替コスト={COST_SWITCH*100:.1f}%/回")

    # ── 主パラメータ run ──
    d = run_strategy(d_all, L_PRIMARY)
    tr = slice_period(d, TRAIN_START, TRAIN_END)
    te = slice_period(d, TEST_START, TEST_END)

    print("\n" + "─"*70)
    print(f"■ 主パラメータ L={L_PRIMARY}")
    r_train = report_period("TRAIN", tr)
    r_test  = report_period("TEST", te)

    # ── 合格判定 ①② ──
    print("\n" + "─"*70)
    print("■ 合格基準チェック")
    c1 = r_test["beat"]
    c2 = r_train["beat"] and r_test["beat"]
    print(f"  ① test で最良holdに勝つ            : {'✅' if c1 else '❌'}")
    print(f"  ② train/test 両方で勝つ(符号一貫)    : {'✅' if c2 else '❌'}")

    # ── ③ L グリッド頑健性 ──
    print("\n  ③ L グリッド頑健性（test期間・各Lで最良holdに勝つか）:")
    grid_beats = []
    for L in L_GRID:
        dL = run_strategy(d_all, L)
        teL = slice_period(dL, TEST_START, TEST_END)
        strat = cum_return(teL["strat_ret"]) * 100
        bh = max(cum_return(teL["btc_ret"])*100, cum_return(teL["eth_ret"])*100)
        beat = strat > bh
        grid_beats.append(beat)
        print(f"     L={L:3d}: 戦略 {strat:+8.2f}% vs 最良hold {bh:+8.2f}%  {'✅' if beat else '❌'}")
    c3 = all(grid_beats)
    print(f"  ③ 全Lで勝つ                         : {'✅' if c3 else '❌'}")

    # ── ④ LOMO（1ヶ月除外・主L） ──
    print("\n  ④ LOMO（test内・1ヶ月除外しても最良holdに勝つか）:")
    te2 = te.copy()
    te2["ym"] = te2["date"].dt.to_period("M")
    months = sorted(te2["ym"].unique())
    lomo_beats = []
    for m in months:
        sub = te2[te2["ym"] != m]
        strat = cum_return(sub["strat_ret"]) * 100
        bh = max(cum_return(sub["btc_ret"])*100, cum_return(sub["eth_ret"])*100)
        beat = strat > bh
        lomo_beats.append(beat)
        print(f"     {str(m)}除外: 戦略 {strat:+8.2f}% vs 最良hold {bh:+8.2f}%  {'✅' if beat else '❌'}")
    c4 = all(lomo_beats)
    print(f"  ④ 全月除外で勝つ                    : {'✅' if c4 else '❌'}")

    # ── ⑤ DD ──
    dd_ok = r_test["dd_strat"] >= min(r_test["dd_btc"], r_test["dd_eth"]) - 5.0  # 5pt以上悪化しないか
    print(f"\n  ⑤ 最大DDが持ちっぱなしより極端に悪くない: {'✅' if dd_ok else '❌'}"
          f"  (戦略{r_test['dd_strat']:+.1f}% vs hold最悪{min(r_test['dd_btc'],r_test['dd_eth']):+.1f}%)")

    # ── 総合 ──
    print("\n" + "="*70)
    allpass = c1 and c2 and c3 and c4 and dd_ok
    print(f"■ 総合判定: {'✅ 合格（次の頑健性検証へ）' if allpass else '❌ 不合格'}")
    print(f"   ①{'✅' if c1 else '❌'} ②{'✅' if c2 else '❌'} ③{'✅' if c3 else '❌'} "
          f"④{'✅' if c4 else '❌'} ⑤{'✅' if dd_ok else '❌'}")
    print("="*70)


if __name__ == "__main__":
    main()
