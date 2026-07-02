#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_newlisting_step5.py
==========================
【Track A Step-5】Binance+Bybit 結合クロス検証（事前登録・分析前ロック）

判定基準（CLAUDE.md 2026-07-02 事前登録・変更禁止）:
  結合資金曲線（E4 / +7d / stop+50% / コスト1.0% / funding込み / 複利 / size5-10%×cap3-5）で
  ① 月次リターン中央値 > 0
  ② maxDD ≥ -25%
  ③ 最悪月 ≥ -15%
  ④' 年間イベント n≥10 の年がすべてプラス（n<10 の年は参考報告のみ・ゲートにしない）
  S  Bybit単独でも funding調整後 +7d の 平均>0 かつ 中央値>0
     （クロス検証の目的＝Binance特殊性の検出。分析実行前の本ヘッダで登録）
  → ≥1構成が①②③④'を通過 かつ S を満たす場合のみ 🟢（Step-4設計へ）

結合ルール（分析前に固定）:
  - 両取引所のイベントを時系列で1本の資金曲線に結合
  - 同一トークン（base正規化: 先頭1000/1M除去）が両取引所に30日以内で上場した場合は
    早い方のみ採用（同一トークンへの二重エクスポージャ回避）。30日超は別イベントとして両方採用
  - 適格条件は両取引所共通: perp_onboard ≤ spot上場日00:00 + 4h（Binance版と同一規約・保守的）

本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。欠損補完なし。
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

SETS = {
    "binance": dict(universe="newlisting_universe.csv", klines="newlisting_klines_1h.csv",
                    perp="newlisting_perp_check.csv", fund="newlisting_funding.csv"),
    "bybit":   dict(universe="bybit_newlisting_universe.csv", klines="bybit_newlisting_klines_1h.csv",
                    perp="bybit_newlisting_perp_check.csv", fund="bybit_newlisting_funding.csv"),
}

# ── 事前固定（変更禁止・Step-3と同一）──
E_IDX = 3; HOLD_H = 168; STOP = 0.50; COST_RT = 0.010
SIZES = [0.05, 0.10]; CAPS = [3, 5]
PERP_LAG_MAX_H = 4.0
DEDUP_DAYS = 30
# ─────────────────────────


def norm_base(spot_symbol: str) -> str:
    b = spot_symbol[:-4] if spot_symbol.endswith("USDT") else spot_symbol
    for pre in ("1000", "1M"):
        if b.startswith(pre) and len(b) > len(pre):
            b = b[len(pre):]
    return b.upper()


def build_events(files: dict, tag: str) -> pd.DataFrame:
    perp = pd.read_csv(files["perp"])
    perp["lag_h"] = pd.to_numeric(perp["perp_minus_spot_hours"], errors="coerce")
    elig = perp[perp["lag_h"] <= PERP_LAG_MAX_H].dropna(subset=["lag_h"])
    perp_map = dict(zip(elig["symbol"], elig["perp_symbol"]))

    fund = pd.read_csv(files["fund"])
    fund["t"] = pd.to_datetime(fund["funding_time_utc"])
    fund["rate"] = pd.to_numeric(fund["funding_rate"], errors="coerce")
    fund_by = {s: g.sort_values("t") for s, g in fund.groupby("perp_symbol")}

    kdf = pd.read_csv(files["klines"])
    kdf["t"] = pd.to_datetime(kdf["open_time_utc"])
    kl = {s: g.sort_values("t").reset_index(drop=True) for s, g in kdf.groupby("symbol")}

    events, nofund = [], 0
    for sym, psym in perp_map.items():
        pg = fund_by.get(psym)
        if pg is None:
            nofund += 1; continue
        g = kl.get(sym)
        if g is None or E_IDX + HOLD_H >= len(g):
            continue
        p_entry = g["close"].iloc[E_IDX]
        if not np.isfinite(p_entry) or p_entry <= 0:
            continue
        t_entry = g["t"].iloc[E_IDX] + pd.Timedelta(hours=1)
        stop_level = p_entry * (1.0 + STOP)
        exit_idx = E_IDX + HOLD_H; stopped = False
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
        events.append({"exchange": tag, "symbol": sym, "base": norm_base(sym),
                       "t_entry": t_entry, "t_exit": t_exit,
                       "net": price_pnl - COST_RT + fsum, "stopped": stopped})
    if nofund:
        print(f"  ⚠ {tag}: funding無しで除外 {nofund}銘柄")
    return pd.DataFrame(events)


def dedup(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values("t_entry").reset_index(drop=True)
    keep, dropped = [], 0
    last_by_base = {}
    for i, r in ev.iterrows():
        prev = last_by_base.get(r["base"])
        if prev is not None and (r["t_entry"] - prev).days <= DEDUP_DAYS:
            dropped += 1
            continue
        last_by_base[r["base"]] = r["t_entry"]
        keep.append(i)
    print(f"  重複除去（同一base・{DEDUP_DAYS}日以内）: {dropped}件除外")
    return ev.loc[keep].reset_index(drop=True)


def t_stat(v):
    v = v[np.isfinite(v)]
    if len(v) < 2 or v.std(ddof=1) == 0:
        return np.nan
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


def simulate(ev, size, cap):
    equity = 1.0; open_pos = []; points = []; taken = skipped = 0
    for _, e in ev.iterrows():
        due = sorted([p for p in open_pos if p[0] <= e["t_entry"]], key=lambda p: p[0])
        for t_exit, notional, net in due:
            equity += notional * net
            points.append((t_exit, equity))
        open_pos = [p for p in open_pos if p[0] > e["t_entry"]]
        if len(open_pos) >= cap:
            skipped += 1; continue
        open_pos.append((e["t_exit"], size * equity, e["net"]))
        taken += 1
    for t_exit, notional, net in sorted(open_pos, key=lambda p: p[0]):
        equity += notional * net
        points.append((t_exit, equity))
    return points, taken, skipped


def analyze(points, year_counts, label):
    eq = pd.Series([p[1] for p in points], index=pd.DatetimeIndex([p[0] for p in points]))
    total = (eq.iloc[-1] - 1.0) * 100
    mdd = ((eq / eq.cummax()) - 1.0).min() * 100
    m_eq = eq.resample("ME").last().ffill()
    m_ret = m_eq.pct_change().dropna() * 100
    if len(m_eq) > 0:
        m_ret = pd.concat([pd.Series([(m_eq.iloc[0]-1.0)*100], index=[m_eq.index[0]]), m_ret])
    y_ret = {}
    for y in sorted(year_counts):
        ys = eq[eq.index.year == y]; prev = eq[eq.index.year < y]
        base = prev.iloc[-1] if len(prev) else 1.0
        if len(ys):
            y_ret[y] = (ys.iloc[-1] / base - 1.0) * 100
    med_m, worst_m = m_ret.median(), m_ret.min()
    c1, c2, c3 = med_m > 0, mdd >= -25.0, worst_m >= -15.0
    gate_years = [y for y, n in year_counts.items() if n >= 10]
    c4 = all(y_ret.get(y, 0) > 0 for y in gate_years)
    ok = c1 and c2 and c3 and c4
    yr = " ".join(f"{y}:{y_ret.get(y, float('nan')):+.1f}%(n={year_counts[y]}{'' if year_counts[y]>=10 else '·参考'})"
                  for y in sorted(year_counts))
    print(f"  {label}: 最終{total:+8.1f}%  maxDD{mdd:+6.1f}%  月次中央{med_m:+5.2f}%  最悪月{worst_m:+6.1f}%")
    print(f"    年別[{yr}]")
    print(f"    ①{'✅' if c1 else '❌'} ②{'✅' if c2 else '❌'} ③{'✅' if c3 else '❌'} "
          f"④'(n≥10年+){'✅' if c4 else '❌'} → {'🟢' if ok else '🔴'}")
    return ok


def main():
    for files in SETS.values():
        for p in files.values():
            if not os.path.exists(p):
                print(f"❌ {p} が無い"); sys.exit(1)
    print("="*88)
    print("【Track A Step-5】Binance+Bybit 結合クロス検証  E4/+7d/stop+50%/コスト1.0%/funding込み")
    print("="*88)

    evs = {}
    for tag, files in SETS.items():
        ev = build_events(files, tag)
        evs[tag] = ev
        yc = ev["t_entry"].dt.year.value_counts().sort_index()
        print(f"  {tag}: 適格イベント {len(ev)}件  年別 {dict(yc)}")

    # S: Bybit単独の out-of-sample 確認（トレードレベル）
    bb = evs["bybit"]["net"].values
    s_mean, s_med, s_t = bb.mean(), np.median(bb), t_stat(bb)
    s_ok = s_mean > 0 and s_med > 0
    print(f"\n■ S: Bybit単独（+7d funding調整後・n={len(bb)}）: "
          f"平均{s_mean*100:+.2f}% 中央{s_med*100:+.2f}% t{s_t:+.2f} 勝率{(bb>0).mean()*100:.0f}% "
          f"→ {'✅' if s_ok else '❌ Binance特殊性の疑い'}")

    # 結合＋重複除去
    print("\n■ 結合（時系列・同一base30日以内は早い方のみ）")
    allev = pd.concat([evs["binance"], evs["bybit"]]).sort_values("t_entry").reset_index(drop=True)
    allev = dedup(allev)
    year_counts = allev["t_entry"].dt.year.value_counts().sort_index().to_dict()
    print(f"  結合イベント: {len(allev)}件  年別 {year_counts}")
    print(f"  1トレードnet: 平均{allev['net'].mean()*100:+.2f}% 中央{allev['net'].median()*100:+.2f}% "
          f"勝率{(allev['net']>0).mean()*100:.0f}% 最悪{allev['net'].min()*100:+.1f}% "
          f"stop率{allev['stopped'].mean()*100:.0f}%")

    print("\n■ 資金曲線グリッド（複利・上限超過スキップ）")
    results = {}
    for size in SIZES:
        for cap in CAPS:
            points, taken, skipped = simulate(allev, size, cap)
            print(f"\n[size={size*100:.0f}% × cap={cap}] 執行{taken}/スキップ{skipped}")
            results[(size, cap)] = analyze(points, year_counts, f"size{size*100:.0f}%/cap{cap}")

    n_pass = sum(results.values())
    final_go = n_pass > 0 and s_ok
    print("\n" + "="*88)
    print(f"■ Step-5 総合: 曲線{n_pass}/{len(results)}構成合格・S{'✅' if s_ok else '❌'} → "
          + ("🟢 GO（Step-4 手動小口トライアル設計へ）" if final_go else "🔴 NO-GO（レジーム依存/一般化不可 → ショート化closed・『買わない』のみ残す）"))
    print("="*88)
    print("※GOでも Bot自動売買・通知化・deploy はしない。Step-4は紙上/最小サイズの記録から。")


if __name__ == "__main__":
    main()
