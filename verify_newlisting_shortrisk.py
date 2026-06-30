#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_newlisting_shortrisk.py
==============================
【Binance Spot 新規取引開始後リターン / ショート・リスク特性化】

目的（最重要）: 「下がるか」の再確認ではなく、
  "新規上場を等金額ショートしたとき、少数の暴騰銘柄でバスケットが壊れないか" を見る。

★リスク条件は実行前に固定（後出し調整しない）。データは既存の
  newlisting_universe.csv / newlisting_klines_1h.csv のみ（新規取得なし）。

監査注記（コード冒頭に明記）:
  - 生存者バイアス: universeは status=TRADING の現存ペアのみ。上場廃止銘柄は欠落。
    → ロング回避結論は保守的に強まる / ショート成績は過小評価（保守的）。暴騰生存銘柄は含む。
  - 「上場日」= 最古klineのopen time = Binance Spot 新規取引開始イベント（新規ローンチではない）。

本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。欠損は補完しない。
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

UNIVERSE_CSV = "newlisting_universe.csv"
KLINES_CSV   = "newlisting_klines_1h.csv"

# ── 事前固定（変更禁止）──
ENTRIES  = {"E1_1hC": 0, "E4_4hC": 3}        # close index
HORIZONS = {"+24h": 24, "+7d": 168, "+30d": 720}
COST_RT  = 0.010                              # 往復1.0%（主）
STOP_LEVELS = [None, 0.30, 0.50, 1.00]        # 逆行(上昇)で手仕舞い。None=損切りなし
MAX_CONCURRENT = 10                           # 同時保有上限（資金曲線・参考）
# ─────────────────────────


def load_klines() -> dict:
    df = pd.read_csv(KLINES_CSV)
    df["t"] = pd.to_datetime(df["open_time_utc"])
    return {s: g.sort_values("t").reset_index(drop=True) for s, g in df.groupby("symbol")}


def short_trade(g: pd.DataFrame, e_idx: int, H: int, stop: float | None):
    """1銘柄のショート結果。戻り: (short_net, mae, stopped) or None。
       short_net = ショートのネット損益（+ =勝ち）。stop指定時はMAE>=stopで手仕舞い。"""
    if e_idx >= len(g):
        return None
    p_entry = g["close"].iloc[e_idx]
    hor_idx = e_idx + H
    if hor_idx >= len(g) or not np.isfinite(p_entry) or p_entry <= 0:
        return None
    # 保有中(entry直後〜horizon)の最大逆行(上昇) = ショート含み損の最大
    win = g.iloc[e_idx+1:hor_idx+1]
    if len(win) == 0:
        return None
    mae = win["high"].max() / p_entry - 1.0     # +なら上昇=ショート不利
    if stop is not None and mae >= stop:
        # 逆行stopで手仕舞い: ショートは -stop の損失 + コスト
        return -stop - COST_RT, mae, True
    p_hor = g["close"].iloc[hor_idx]
    if not np.isfinite(p_hor) or p_hor <= 0:
        return None
    raw = p_hor / p_entry - 1.0
    short_net = -raw - COST_RT                   # ショート: トークンが下げれば+
    return short_net, mae, False


def maxdd(cum: np.ndarray):
    if len(cum) == 0:
        return np.nan
    peak = np.maximum.accumulate(cum)
    return (cum - peak).min()


def main():
    if not (os.path.exists(UNIVERSE_CSV) and os.path.exists(KLINES_CSV)):
        print("❌ CSVが無い。"); sys.exit(1)
    uni = pd.read_csv(UNIVERSE_CSV)
    uni["listing"] = pd.to_datetime(uni["listing_date_utc"])
    ld = dict(zip(uni["symbol"], uni["listing"]))
    kl = load_klines()
    print("="*82)
    print("【Binance Spot 新規取引開始後リターン / ショート・リスク特性化】")
    print(f"  銘柄 {len(kl)} / コスト往復{COST_RT*100:.1f}% / 等金額1ユニット")
    print("  監査: 生存者バイアスあり(現存ペアのみ)→ショート成績は保守的・過小評価")
    print("="*82)

    for ek, eidx in ENTRIES.items():
        for hl, H in HORIZONS.items():
            print(f"\n{'─'*82}\n■ {ek} × 保有{hl}")
            # 損切りなしの基本分布
            base = []
            syms_used = []
            for s, g in kl.items():
                r = short_trade(g, eidx, H, None)
                if r is not None:
                    base.append(r); syms_used.append(s)
            net = np.array([b[0] for b in base])
            mae = np.array([b[1] for b in base])
            n = len(net)
            if n == 0:
                print("  データ不足"); continue
            winrate = (net > 0).mean()*100      # ショート勝率(=トークン下落)
            print(f"  [損切りなし] n={n}  ショート勝率{winrate:.0f}%  "
                  f"平均{net.mean()*100:+.2f}%  中央{np.median(net)*100:+.2f}%")
            # テール: 最悪(最大暴騰=ショート最大損)
            order = np.argsort(net)   # 昇順=最悪が先頭
            worst = order[:3]
            print(f"  最悪3トレード(暴騰): " +
                  " / ".join(f"{syms_used[i]} {net[i]*100:+.0f}%(MAE+{mae[i]*100:.0f}%)" for i in worst))
            total = net.sum()
            total_ex3 = np.delete(net, worst).sum()
            print(f"  合計P&L(等金額): {total*100:+.1f}%  上位3暴騰除外: {total_ex3*100:+.1f}%  "
                  f"(1トレード=資本の1/{n})")
            print(f"  MAE(保有中最大逆行・上昇): 中央+{np.median(mae)*100:.0f}%  "
                  f"75%ile+{np.percentile(mae,75)*100:.0f}%  最大+{mae.max()*100:.0f}%")
            # 損切り別
            print(f"  [損切り別] 逆行で手仕舞い → 合計P&L / 勝率 / 最悪トレード")
            for stop in STOP_LEVELS:
                rs = [short_trade(g, eidx, H, stop) for g in kl.values()]
                rs = [r for r in rs if r is not None]
                v = np.array([r[0] for r in rs])
                stops = np.array([r[2] for r in rs])
                lbl = "なし" if stop is None else f"+{int(stop*100)}%"
                print(f"     stop {lbl:>5}: 合計{v.sum()*100:+7.1f}%  勝率{(v>0).mean()*100:3.0f}%  "
                      f"最悪{v.min()*100:+6.0f}%  手仕舞い率{stops.mean()*100:3.0f}%")
            # 資金曲線(上場日順・同時保有上限なし=全部1ユニット逐次)と最大DD
            seq = sorted(zip([ld[s] for s in syms_used], net), key=lambda x: x[0])
            cum = np.cumsum([x[1] for x in seq])
            print(f"  資金曲線(上場日順・逐次): 最終{cum[-1]*100:+.1f}%  最大DD{maxdd(cum)*100:+.1f}%")

    print("\n" + "="*82)
    print("■ 読み方")
    print("="*82)
    print("""  ・「ショート勝率」が高くても、最悪3トレード(暴騰)と合計P&Lの差が大きければ右裾依存。
  ・損切りなしで合計マイナス→損切りで合計プラスに転じるなら、暴騰の手仕舞いが鍵。
  ・MAEが大きい=保有中に大きく逆行=実際は清算/追証で持ちきれない可能性。
  ・生存者バイアスで上場廃止(暴落)銘柄が欠落=ショート成績は本来もっと良い可能性(保守的)。
  → ここで右裾に殺されるなら、執行(perp/funding)を調べる前に不採用。耐えるなら執行確認へ。""")


if __name__ == "__main__":
    main()
