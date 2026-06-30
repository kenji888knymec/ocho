#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_newlisting_event_study.py
================================
【CEX新規上場後リターン検証 / 分析】事前登録 verify_newlisting_PREREG.md に従う。

★このコードは「データを見る前」に固定する。CSV投入後に窓・閾値・基準を動かさない。

入力（同ディレクトリ）:
  newlisting_universe.csv   : symbol, listing_date_utc, first_day_quote_vol_usd
  newlisting_klines_1h.csv  : symbol, open_time_utc, open, high, low, close, volume, quote_vol
  /tmp/ohlcv_long/ohlcv_2024_2026/BTC_1h.csv : 市場調整(btc_excess)用のBTC 1h

エントリー（§4）: E0=最初の1h足open（参考のみ） / E1=最初の1h足close / E4=4本目の1h足close
  採用判断は E1 / E4 のみ。
保有（§5）: +1h / +24h / +3d / +7d / +30d
コスト（§5）: 0.5% / 1.0% / 2.0%（往復控除）
市場調整: btc_excess = raw − BTC同区間リターン

合格（§6・E1/E4のセルで全て満たす）:
  ① n≥30 ② 平均と中央値が同符号 ③ |t|≥2 ④ train/test同符号
  ⑤ leave-one-token-out で符号維持 ⑥ 上位3銘柄除外で符号維持
  ⑦ btc_excess でも同符号 ⑧ コスト0.5%→1.0%で符号維持

本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。
欠損は補完しない（np.nan）。
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

UNIVERSE_CSV = "newlisting_universe.csv"
KLINES_CSV   = "newlisting_klines_1h.csv"
BTC_CSV      = "/tmp/ohlcv_long/ohlcv_2024_2026/BTC_1h.csv"

HORIZONS = {"+1h": 1, "+24h": 24, "+3d": 72, "+7d": 168, "+30d": 720}  # 時間
ENTRIES  = {"E0_open": ("open", 0), "E1_1hC": ("close", 0), "E4_4hC": ("close", 3)}
COSTS    = [0.005, 0.010, 0.020]
ADOPT_ENTRIES = ["E1_1hC", "E4_4hC"]   # 採用判断はこの2つのみ（E0は参考）


def load_btc_hourly() -> pd.Series:
    if not os.path.exists(BTC_CSV):
        print(f"⚠ {BTC_CSV} が無い。btc_excessはスキップ（rawのみ）")
        return pd.Series(dtype=float)
    df = pd.read_csv(BTC_CSV)
    t = pd.to_datetime(df["datetime_utc"]).dt.floor("h")
    s = pd.Series(df["close"].values, index=t).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def load_klines() -> dict:
    if not os.path.exists(KLINES_CSV):
        print(f"❌ {KLINES_CSV} が無い。Macで fetch_binance_newlistings.py を実行し配置してください。")
        sys.exit(1)
    df = pd.read_csv(KLINES_CSV)
    df["t"] = pd.to_datetime(df["open_time_utc"])
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("t").reset_index(drop=True)
        out[sym] = g
    return out


def t_stat(v: np.ndarray):
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2 or v.std(ddof=1) == 0:
        return np.nan
    return v.mean() / (v.std(ddof=1) / np.sqrt(n))


def returns_for_entry(g: pd.DataFrame, btc: pd.Series, entry_kind: str, entry_idx: int, H: int):
    """1銘柄の (raw, btc_excess) を返す。窓不足は (nan,nan)。"""
    col, e = entry_kind, entry_idx
    # エントリー価格・時刻
    if col == "open":
        if e >= len(g): return np.nan, np.nan
        p_entry = g["open"].iloc[e]; t_entry = g["t"].iloc[e]
        hor_idx = e + H - 1                      # close[H-1] が +Hh
    else:  # close
        if e >= len(g): return np.nan, np.nan
        p_entry = g["close"].iloc[e]; t_entry = g["t"].iloc[e]
        hor_idx = e + H                          # close[e+H] が +Hh
    if hor_idx >= len(g) or not np.isfinite(p_entry) or p_entry <= 0:
        return np.nan, np.nan
    p_hor = g["close"].iloc[hor_idx]; t_hor = g["t"].iloc[hor_idx]
    if not np.isfinite(p_hor) or p_hor <= 0:
        return np.nan, np.nan
    raw = p_hor / p_entry - 1.0
    # btc_excess（E1/E4のみ・open_time基準で同区間のBTCリターン）
    bex = np.nan
    if len(btc) > 0:
        try:
            b0 = btc.asof(t_entry); b1 = btc.asof(t_hor)
            if np.isfinite(b0) and np.isfinite(b1) and b0 > 0:
                bex = raw - (b1 / b0 - 1.0)
        except Exception:
            bex = np.nan
    return raw, bex


def agg_block(label, vals):
    v = vals[np.isfinite(vals)]
    n = len(v)
    if n == 0:
        return None
    return {"n": n, "mean": v.mean(), "med": np.median(v), "negpct": (v < 0).mean()*100,
            "t": t_stat(v)}


def check_pass(rows, listing_dates):
    """rows: list[(symbol, raw, bex)] for one (entry,horizon). 各コストで合否を返す。"""
    syms = [r[0] for r in rows]
    raw = np.array([r[1] for r in rows], float)
    bex = np.array([r[2] for r in rows], float)
    ld  = np.array([listing_dates[s] for s in syms])
    mask = np.isfinite(raw)
    raw, bex, ld, syms = raw[mask], bex[mask], ld[mask], list(np.array(syms)[mask])
    res = {}
    for cost in COSTS:
        net = raw - cost
        n = len(net)
        if n == 0:
            res[cost] = None; continue
        mean, med = net.mean(), np.median(net)
        sign = np.sign(mean)
        c1 = n >= 30
        c2 = np.sign(mean) == np.sign(med) and med != 0
        c3 = abs(t_stat(net)) >= 2.0
        # train/test split by listing date median
        order = np.argsort(ld)
        half = n // 2
        tr, te = net[order[:half]], net[order[half:]]
        c4 = (len(tr) > 0 and len(te) > 0 and np.sign(tr.mean()) == sign and np.sign(te.mean()) == sign)
        # leave-one-token-out: 符号を最も支える1件を抜いて符号維持
        if sign > 0:
            drop = np.argmax(net)
        else:
            drop = np.argmin(net)
        loo = np.delete(net, drop)
        c5 = len(loo) > 0 and np.sign(loo.mean()) == sign
        # top3（絶対値）除外で符号維持
        top3 = np.argsort(-np.abs(net))[:3]
        d3 = np.delete(net, top3)
        c6 = len(d3) > 0 and np.sign(d3.mean()) == sign
        # btc_excess 同符号（コスト控除後）
        be = bex[np.isfinite(bex)] - cost
        c7 = len(be) > 0 and np.sign(be.mean()) == sign
        res[cost] = {"n": n, "mean": mean, "med": med, "t": t_stat(net),
                     "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6, "c7": c7}
    return res


def main():
    print("="*78)
    print("【CEX新規上場後リターン検証】Binance Spot USDT（事前登録固定）")
    print("="*78)
    if not os.path.exists(UNIVERSE_CSV):
        print(f"❌ {UNIVERSE_CSV} が無い。配置してください。"); sys.exit(1)
    uni = pd.read_csv(UNIVERSE_CSV)
    uni["listing"] = pd.to_datetime(uni["listing_date_utc"])
    listing_dates = dict(zip(uni["symbol"], uni["listing"]))
    print(f"ユニバース: {len(uni)}銘柄  期間 {uni['listing'].min().date()}〜{uni['listing'].max().date()}")
    print(f"初日quote_vol: 中央${uni['first_day_quote_vol_usd'].median()/1e6:.0f}M")

    btc = load_btc_hourly()
    print(f"BTC 1h: {'OK '+str(len(btc))+'本' if len(btc)>0 else '無し(btc_excessスキップ)'}")
    kl = load_klines()
    print(f"kline銘柄: {len(kl)}")

    # ── 記述: raw平均/中央値 一覧（E0/E1/E4 × 各保有）──
    print("\n■ 記述統計（コスト控除前 raw・%）: 平均 / 中央 / 負割合 / n")
    for ek, (col, eidx) in ENTRIES.items():
        print(f"\n  [{ek}]")
        print(f"    {'保有':>6} {'平均%':>8} {'中央%':>8} {'負%':>5} {'n':>4}")
        for hl, H in HORIZONS.items():
            vals = np.array([returns_for_entry(kl[s], btc, col, eidx, H)[0] for s in kl], float)
            a = agg_block(hl, vals)
            if a:
                print(f"    {hl:>6} {a['mean']*100:8.2f} {a['med']*100:8.2f} {a['negpct']:5.0f} {a['n']:4d}")

    # ── 合格判定（採用entry E1/E4 × 各保有）──
    print("\n" + "="*78)
    print("■ 合格基準チェック（E1/E4 × 各保有・コスト別）")
    print("="*78)
    any_pass = False
    for ek in ADOPT_ENTRIES:
        col, eidx = ENTRIES[ek]
        print(f"\n[{ek}]")
        for hl, H in HORIZONS.items():
            rows = [(s, *returns_for_entry(kl[s], btc, col, eidx, H)) for s in kl]
            res = check_pass(rows, listing_dates)
            for cost in COSTS:
                r = res.get(cost)
                if r is None:
                    continue
                flags = "".join(["①" if r["c1"] else "・", "②" if r["c2"] else "・",
                                 "③" if r["c3"] else "・", "④" if r["c4"] else "・",
                                 "⑤" if r["c5"] else "・", "⑥" if r["c6"] else "・",
                                 "⑦" if r["c7"] else "・"])
                # ⑧ は cost0.5→1.0 で符号維持: ここでは各セルで参考表示
                allc = all([r["c1"],r["c2"],r["c3"],r["c4"],r["c5"],r["c6"],r["c7"]])
                mark = "  ✅候補" if allc else ""
                if cost == 0.005:
                    # ⑧チェック: cost1.0%でも同符号か
                    r10 = res.get(0.010)
                    c8 = (r10 is not None and np.sign(r10["mean"]) == np.sign(r["mean"]))
                    mark += ("  ⑧✅" if c8 else "  ⑧✗") if allc else ""
                    if allc and c8:
                        any_pass = True
                print(f"  {hl:>5} cost{cost*100:>4.1f}%: 平均{r['mean']*100:+7.2f}% 中央{r['med']*100:+7.2f}% "
                      f"t{r['t']:+5.2f} n{r['n']:3d} [{flags}]{mark}")

    print("\n" + "="*78)
    print(f"■ 総合: {'✅ 事前登録の合格セルあり（要・実行可能性の別途確認）' if any_pass else '❌ 合格セルなし'}")
    print("="*78)
    print("※統計優位が出ても、初値で買えた前提でなくE1/E4。実際に買えるか(取引所/KYC/スリッページ/")
    print("  先物有無)は統計の外。合格でも即実装しない（事前登録§8・§9）。")


if __name__ == "__main__":
    main()
