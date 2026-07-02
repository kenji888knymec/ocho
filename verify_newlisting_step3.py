#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_newlisting_step3.py
==========================
【Track A Step-3】新規上場ショート 資金曲線の現実化（funding込み・E4・stop+50%固定）

★パラメータ・判定基準は CLAUDE.md 2026-07-01 ロードマップで事前固定済み。動かさない。
  - エントリー: E4（Spot上場4本目1h足close）。対象=E4時点でperp取引可能(lag≤+4h)＆funding有り
  - 保有: +7d（Step-2のGOセル）。stop: 保有中に高値が entry×1.5 到達で手仕舞い(-50%)
  - コスト: 往復1.0% / funding: 実履歴を entry〜exit(実際の手仕舞い時点)で合算（ショートは+rate受取り）
  - サイズ: 1銘柄 = 資本の5% / 10%（複利・エントリー時点の資本比）
  - 同時保有上限: 3 / 5（上限到達中の新規上場はスキップ＝機会損失として記録）
  判定基準（事前固定・全て）:
    ① 月次リターンの中央値 > 0
    ② maxDD ≥ -25%（資本比）
    ③ 最悪月 ≥ -15%
    ④ 2024・2025・2026 各年プラス（単一年依存でない）

近似の開示: 資金曲線はトレード決済時点で更新（保有中の含み損益は stop で -50%×サイズに
上限が切られているため、決済ベースDDは実態の近似。証拠金・清算は未モデル＝Step-4で確認）。

本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。欠損補完なし。
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

UNIVERSE_CSV = "newlisting_universe.csv"
KLINES_CSV   = "newlisting_klines_1h.csv"
PERP_CSV     = "newlisting_perp_check.csv"
FUND_CSV     = "newlisting_funding.csv"

# ── 事前固定（変更禁止）──
E_IDX   = 3
HOLD_H  = 168          # +7d（GOセル）
STOP    = 0.50
COST_RT = 0.010
SIZES   = [0.05, 0.10]
CAPS    = [3, 5]
PERP_LAG_MAX_H = 4.0
# ─────────────────────────


def build_events():
    perp = pd.read_csv(PERP_CSV)
    perp["lag_h"] = pd.to_numeric(perp["perp_minus_spot_hours"], errors="coerce")
    elig = perp[perp["lag_h"] <= PERP_LAG_MAX_H].dropna(subset=["lag_h"])
    perp_map = dict(zip(elig["symbol"], elig["perp_symbol"]))

    fund = pd.read_csv(FUND_CSV)
    fund["t"] = pd.to_datetime(fund["funding_time_utc"])
    fund["rate"] = pd.to_numeric(fund["funding_rate"], errors="coerce")
    fund_by = {s: g.sort_values("t") for s, g in fund.groupby("perp_symbol")}

    kdf = pd.read_csv(KLINES_CSV)
    kdf["t"] = pd.to_datetime(kdf["open_time_utc"])
    kl = {s: g.sort_values("t").reset_index(drop=True) for s, g in kdf.groupby("symbol")}

    events = []
    excluded_nofund = 0
    for sym, psym in perp_map.items():
        pg = fund_by.get(psym)
        if pg is None:
            excluded_nofund += 1
            continue
        g = kl.get(sym)
        if g is None or E_IDX + HOLD_H >= len(g):
            continue
        p_entry = g["close"].iloc[E_IDX]
        if not np.isfinite(p_entry) or p_entry <= 0:
            continue
        t_entry = g["t"].iloc[E_IDX] + pd.Timedelta(hours=1)
        # stop判定: 保有窓内で最初に high >= entry*(1+STOP) となった時間で手仕舞い
        stop_level = p_entry * (1.0 + STOP)
        exit_idx = E_IDX + HOLD_H
        stopped = False
        win = g.iloc[E_IDX+1:exit_idx+1]
        hit = win.index[win["high"] >= stop_level]
        if len(hit) > 0:
            exit_idx = int(hit[0]); stopped = True
        t_exit = g["t"].iloc[exit_idx] + pd.Timedelta(hours=1)
        if stopped:
            price_pnl = -STOP
        else:
            p_exit = g["close"].iloc[exit_idx]
            if not np.isfinite(p_exit) or p_exit <= 0:
                continue
            price_pnl = -(p_exit / p_entry - 1.0)
        m = (pg["t"] > t_entry) & (pg["t"] <= t_exit)
        fsum = pg.loc[m, "rate"].sum()
        net = price_pnl - COST_RT + fsum
        events.append({"symbol": sym, "t_entry": t_entry, "t_exit": t_exit,
                       "net": net, "stopped": stopped})
    if excluded_nofund:
        print(f"⚠ funding履歴なしで除外: {excluded_nofund}銘柄")
    ev = pd.DataFrame(events).sort_values("t_entry").reset_index(drop=True)
    return ev


def simulate(ev: pd.DataFrame, size: float, cap: int):
    """複利・同時保有上限つきの資金曲線。戻り: (equity_points[(time,equity)], taken, skipped)"""
    equity = 1.0
    open_pos = []   # (t_exit, notional, net)
    points = []
    taken = skipped = 0
    for _, e in ev.iterrows():
        # 先に、このエントリー時刻までに決済されるポジションを時系列で閉じる
        due = sorted([p for p in open_pos if p[0] <= e["t_entry"]], key=lambda p: p[0])
        for t_exit, notional, net in due:
            equity += notional * net
            points.append((t_exit, equity))
        open_pos = [p for p in open_pos if p[0] > e["t_entry"]]
        if len(open_pos) >= cap:
            skipped += 1
            continue
        notional = size * equity
        open_pos.append((e["t_exit"], notional, e["net"]))
        taken += 1
    for t_exit, notional, net in sorted(open_pos, key=lambda p: p[0]):
        equity += notional * net
        points.append((t_exit, equity))
    return points, taken, skipped


def analyze(points, label):
    eq = pd.Series([p[1] for p in points], index=pd.DatetimeIndex([p[0] for p in points]))
    total = (eq.iloc[-1] - 1.0) * 100
    peak = eq.cummax()
    mdd = ((eq / peak) - 1.0).min() * 100
    # 月次リターン（決済ベースの資本ステップから）
    m_eq = eq.resample("ME").last().ffill()
    m_ret = m_eq.pct_change().dropna() * 100
    # 初月は基準1.0から
    if len(m_eq) > 0:
        first = (m_eq.iloc[0] - 1.0) * 100
        m_ret = pd.concat([pd.Series([first], index=[m_eq.index[0]]), m_ret])
    y_ret = {}
    for y in (2024, 2025, 2026):
        ys = eq[eq.index.year == y]
        prev = eq[eq.index.year < y]
        base = prev.iloc[-1] if len(prev) else 1.0
        if len(ys):
            y_ret[y] = (ys.iloc[-1] / base - 1.0) * 100
    med_m = m_ret.median()
    worst_m = m_ret.min()
    c1 = med_m > 0
    c2 = mdd >= -25.0
    c3 = worst_m >= -15.0
    c4 = all(v > 0 for v in y_ret.values()) and len(y_ret) == 3
    ok = c1 and c2 and c3 and c4
    yr = " ".join(f"{y}:{v:+.1f}%" for y, v in y_ret.items())
    print(f"  {label}: 最終{total:+8.1f}%  maxDD{mdd:+6.1f}%  月次中央{med_m:+5.2f}%  "
          f"最悪月{worst_m:+6.1f}%  年別[{yr}]")
    print(f"    判定: ①月次中央>0{'✅' if c1 else '❌'} ②DD≥-25%{'✅' if c2 else '❌'} "
          f"③最悪月≥-15%{'✅' if c3 else '❌'} ④各年+{'✅' if c4 else '❌'}"
          f"  → {'🟢合格' if ok else '🔴不合格'}")
    return ok


def main():
    for p in (UNIVERSE_CSV, KLINES_CSV, PERP_CSV, FUND_CSV):
        if not os.path.exists(p):
            print(f"❌ {p} が無い"); sys.exit(1)
    print("="*84)
    print("【Track A Step-3】資金曲線の現実化  E4 / +7d / stop+50% / コスト1.0% / funding込み")
    print("="*84)
    ev = build_events()
    n_stop = int(ev["stopped"].sum())
    print(f"イベント: {len(ev)}件（{ev['t_entry'].min().date()}〜{ev['t_entry'].max().date()}）  "
          f"stop手仕舞い {n_stop}件({n_stop/len(ev)*100:.0f}%)")
    print(f"1トレードnet: 平均{ev['net'].mean()*100:+.2f}% 中央{ev['net'].median()*100:+.2f}% "
          f"勝率{(ev['net']>0).mean()*100:.0f}% 最悪{ev['net'].min()*100:+.1f}%")

    print(f"\n■ サイズ×同時保有上限 グリッド（複利・上限超過はスキップ）")
    results = {}
    for size in SIZES:
        for cap in CAPS:
            points, taken, skipped = simulate(ev, size, cap)
            print(f"\n[size={size*100:.0f}% × cap={cap}]  執行{taken}件 / スキップ{skipped}件")
            results[(size, cap)] = analyze(points, f"size{size*100:.0f}%/cap{cap}")

    n_pass = sum(results.values())
    print("\n" + "="*84)
    print(f"■ Step-3 総合: {n_pass}/{len(results)} 構成が合格 → "
          + ("🟢 Step-4（手動小口トライアル設計）へ" if n_pass > 0 else "🔴 資金管理で成立せず"))
    print("="*84)
    print("※合格でも Bot自動売買・通知化・deploy はしない。Step-4は紙上/最小サイズの記録から。")


if __name__ == "__main__":
    main()
