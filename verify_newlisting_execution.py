#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_newlisting_execution.py
==============================
【Track A Step-2 判定】新規上場ショートの執行可能性 GO/NO-GO

★このコードは「funding/perpデータを見る前」にロックする。データ投入後に基準を動かさない。
  GO/NO-GO はCLAUDE.mdの事前登録（2026-07-01 ロードマップ）に完全に従う:

  GO: E4エントリー時点（Spot上場+4h）で perp が取引可能だった銘柄が n≥30、かつ
      その部分集合の funding調整後ショートリターンが +24h または +7d で
      「平均>0 かつ 中央値>0 / |t|≥2 / LOO(最も利益の大きい1件除外)で符号維持 /
       top3(最も利益の大きい3件除外)で符号維持」を全て通過。
  NO-GO: 上記を満たさない。

  funding符号: rate>0 = ロングがショートに払う（ショートは受取り）。
  ショートnet = -raw - COST + Σ(funding_rate)   ※perpのfundingをspotリターンに近似適用。
  +30d は参考値（funding累積で死にやすい・GO判定に使わない）。
  stop+50% 変種は参考表示のみ（GO判定に使わない）。

入力（同ディレクトリ）:
  newlisting_universe.csv / newlisting_klines_1h.csv （既存）
  newlisting_perp_check.csv / newlisting_funding.csv （Mac Step-2フェッチ）

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
E_IDX    = 3                      # E4 = 4本目の1h足 close
PERP_LAG_MAX_H = 4.0              # E4時点でperp取引可能 = perp_onboard <= spot+4h
COST_RT  = 0.010                  # 往復1.0%
HORIZONS = {"+24h": 24, "+7d": 168, "+30d(参考)": 720}
GO_HORIZONS = ["+24h", "+7d"]     # GO判定に使うのはこの2つのみ
STOP_REF = 0.50                   # 参考表示のみ
# ─────────────────────────


def t_stat(v):
    v = v[np.isfinite(v)]
    if len(v) < 2 or v.std(ddof=1) == 0:
        return np.nan
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


def main():
    for p in (UNIVERSE_CSV, KLINES_CSV, PERP_CSV, FUND_CSV):
        if not os.path.exists(p):
            print(f"❌ {p} が無い。Step-2フェッチ結果を配置してください。"); sys.exit(1)

    perp = pd.read_csv(PERP_CSV)
    perp["lag_h"] = pd.to_numeric(perp["perp_minus_spot_hours"], errors="coerce")
    n_total = len(perp)
    have = perp.dropna(subset=["lag_h"])
    eligible = have[have["lag_h"] <= PERP_LAG_MAX_H]
    print("="*80)
    print("【Track A Step-2 判定】新規上場ショート 執行可能性 GO/NO-GO（事前登録固定）")
    print("="*80)
    print(f"Spot新規上場 {n_total} / perpあり {len(have)} / E4時点(≤+{PERP_LAG_MAX_H:.0f}h)で取引可能 {len(eligible)}")
    if len(have) > 0:
        print(f"perp上場ラグ: 中央{have['lag_h'].median():+.1f}h  "
              f"同時以前(≤0h) {(have['lag_h']<=0).sum()}銘柄 / +24h以内 {(have['lag_h']<=24).sum()}銘柄")

    # funding
    fund = pd.read_csv(FUND_CSV)
    fund["t"] = pd.to_datetime(fund["funding_time_utc"])
    fund["rate"] = pd.to_numeric(fund["funding_rate"], errors="coerce")
    fund_by = {s: g.sort_values("t") for s, g in fund.groupby("perp_symbol")}
    print(f"funding履歴: {len(fund)}行 / {len(fund_by)}perp  "
          f"全体平均rate {fund['rate'].mean()*100:+.4f}%/8h  負の割合 {(fund['rate']<0).mean()*100:.0f}%")

    # klines
    kdf = pd.read_csv(KLINES_CSV)
    kdf["t"] = pd.to_datetime(kdf["open_time_utc"])
    kl = {s: g.sort_values("t").reset_index(drop=True) for s, g in kdf.groupby("symbol")}
    perp_map = dict(zip(perp["symbol"], perp["perp_symbol"]))
    elig_set = set(eligible["symbol"])

    def short_trade(sym, H, use_stop=False):
        g = kl.get(sym)
        if g is None or E_IDX >= len(g):
            return None
        p_entry = g["close"].iloc[E_IDX]
        t_entry = g["t"].iloc[E_IDX] + pd.Timedelta(hours=1)   # close時刻
        hor = E_IDX + H
        if hor >= len(g) or not np.isfinite(p_entry) or p_entry <= 0:
            return None
        win = g.iloc[E_IDX+1:hor+1]
        mae = win["high"].max() / p_entry - 1.0
        t_exit = g["t"].iloc[hor] + pd.Timedelta(hours=1)
        p_exit = g["close"].iloc[hor]
        if not np.isfinite(p_exit) or p_exit <= 0:
            return None
        if use_stop and mae >= STOP_REF:
            price_pnl = -STOP_REF
        else:
            price_pnl = -(p_exit / p_entry - 1.0)
        # funding（entry〜exitの8hポイント合算・ショートは+rate受取り）
        fsum = 0.0
        pg = fund_by.get(perp_map.get(sym, ""), None)
        if pg is not None:
            m = (pg["t"] > t_entry) & (pg["t"] <= t_exit)
            fsum = pg.loc[m, "rate"].sum()
        return price_pnl - COST_RT + fsum, fsum, mae

    print("\n" + "─"*80)
    print(f"■ funding調整後ショート（E4・コスト{COST_RT*100:.1f}%・perp即時上場銘柄のみ n={len(elig_set)}）")
    go_any = False
    for hl, H in HORIZONS.items():
        res = [(s, *r) for s in elig_set if (r := short_trade(s, H)) is not None]
        if not res:
            print(f"  {hl}: データ不足"); continue
        net = np.array([r[1] for r in res]); fs = np.array([r[2] for r in res])
        n = len(net); mean, med = net.mean(), np.median(net)
        t = t_stat(net)
        # LOO/top3: 符号を最も支える=最大利益側を除外（保守）
        order = np.argsort(-net)
        loo_ok = np.delete(net, order[0]).mean() > 0
        top3_ok = np.delete(net, order[:3]).mean() > 0
        c_n = n >= 30
        c_sign = mean > 0 and med > 0
        c_t = abs(t) >= 2.0 and mean > 0
        cell_go = c_n and c_sign and c_t and loo_ok and top3_ok
        flags = f"n{'✅' if c_n else '❌'} 符号{'✅' if c_sign else '❌'} t{'✅' if c_t else '❌'} " \
                f"LOO{'✅' if loo_ok else '❌'} top3除外{'✅' if top3_ok else '❌'}"
        in_go = hl in GO_HORIZONS
        if cell_go and in_go:
            go_any = True
        print(f"  {hl:>9}: n={n:3d} 平均{mean*100:+7.2f}% 中央{med*100:+7.2f}% t{t:+5.2f} "
              f"funding寄与 平均{fs.mean()*100:+.2f}%  [{flags}]"
              + ("  ← GO対象" if in_go else "") + ("  🟢通過" if cell_go and in_go else ""))
        # 参考: stop+50%
        res_s = [short_trade(s, H, use_stop=True) for s in elig_set]
        vs = np.array([r[0] for r in res_s if r is not None])
        print(f"        (参考 stop+{int(STOP_REF*100)}%: 平均{vs.mean()*100:+.2f}% 中央{np.median(vs)*100:+.2f}%)")

    print("\n" + "="*80)
    if len(elig_set) < 30:
        print(f"■ 判定: 🔴 NO-GO（E4時点でperp取引可能な銘柄 {len(elig_set)} < 30）")
    elif go_any:
        print("■ 判定: 🟢 GO（事前登録条件を通過 → Step-3 資金曲線の現実化へ）")
    else:
        print("■ 判定: 🔴 NO-GO（funding調整後に +24h/+7d の優位が消えた）")
    print("="*80)
    print("※GOでも即実装しない。Step-3(資金曲線現実化)→Step-4(手動小口トライアル設計)の順。")
    print("※NO-GOなら『新規上場を買わない』ルールだけ残してショート化は閉じる（事前登録どおり）。")


if __name__ == "__main__":
    main()
