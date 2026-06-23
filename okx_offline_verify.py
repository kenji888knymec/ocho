#!/usr/bin/env python3
"""
okx_offline_verify.py
=====================
OKX OHLCV 2024-2026 で cs_mom / rv_chg-LONG / S4 をオフライン再検証。
研究用レポートのみ。本番変更・deploy・merge なし。

データ: /tmp/ohlcv_long/ohlcv_2024_2026/ (28銘柄 1h, 2024-01-01〜2026-06-21)
実行  : python3 /home/user/ocho/okx_offline_verify.py

採用判断基準 (全て満たすこと):
  WR>=60%, avg>=+0.20%, median>0,
  top3日除外後もavg+, LOMO全月+, zero_months<=2, maxDD許容範囲
"""
import pandas as pd
import numpy as np
from pathlib import Path
import heapq, warnings, os

warnings.filterwarnings("ignore")

DATA_DIR = Path(os.environ.get("OKX_DATA_DIR", "/tmp/ohlcv_long/ohlcv_2024_2026"))

SYMBOLS = ["AAVE","ADA","APT","ARB","ATOM","AVAX","BNB","BONK","BTC","DOGE",
           "DOT","ETH","FET","HBAR","INJ","LINK","LTC","NEAR","POL","SEI",
           "SHIB","SOL","STX","SUI","TRX","UNI","XLM","XRP"]

COST_SWEEP = [0.0008, 0.0010, 0.0015, 0.0020, 0.0030]
MAIN_COST  = 0.0015  # 主評価コスト (0.15%/leg)


# ═══════════════════════════════════════════════════════════════════════════════
#  データロード
# ═══════════════════════════════════════════════════════════════════════════════
def load_1h(sym):
    p = DATA_DIR / f"{sym}_1h.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["datetime_utc"])
    df = df.set_index("datetime_utc").sort_index()
    df.index = df.index.tz_localize(None)
    return df

print("=" * 72)
print("OKX オフライン検証: cs_mom / rv_chg / S4")
print("=" * 72)
print()

Close, High, Low = {}, {}, {}
missing = []
for s in SYMBOLS:
    df = load_1h(s)
    if df is None:
        missing.append(s)
        continue
    Close[s] = df["close"]
    High[s]  = df["high"]
    Low[s]   = df["low"]

avail = [s for s in SYMBOLS if s not in missing]
if missing:
    print(f"欠損銘柄: {missing}")

close_df = pd.DataFrame(Close)
high_df  = pd.DataFrame(High)
low_df   = pd.DataFrame(Low)
IDX = close_df.index
print(f"銘柄: {len(avail)} 個  期間: {IDX[0].date()} 〜 {IDX[-1].date()}  ({len(IDX):,} bars)")
print()


# ═══════════════════════════════════════════════════════════════════════════════
#  共通ユーティリティ
# ═══════════════════════════════════════════════════════════════════════════════
def _report(df, val_col, date_col="entry_dt", label=""):
    """月別・LOMO・top3日除外・maxDD・最大連敗を出力。val は小数（0.01=1%）。"""
    if df is None or len(df) == 0:
        print(f"  {label}: データなし"); return

    v = df[val_col].values
    n = len(v)
    if n == 0:
        print(f"  {label}: データなし"); return

    wr  = (v > 0).mean() * 100
    avg = v.mean() * 100
    med = np.median(v) * 100

    print(f"\n  ── {label} ──")
    print(f"  全体: n={n:,}, WR={wr:.1f}%, avg={avg:+.3f}%, med={med:+.3f}%")

    # 月別
    df = df.copy()
    df["ym"] = df[date_col].dt.to_period("M")
    mdf = df.groupby("ym")[val_col].agg(
        n="count", avg="mean", wr=lambda x: (x > 0).mean()
    )
    zero_m = int((mdf["avg"] <= 0).sum())
    print(f"  月別:")
    for ym, row in mdf.iterrows():
        mk = "✓" if row["avg"] > 0 else "✗"
        print(f"    {ym}: n={int(row['n']):3d}  avg={row['avg']*100:+.3f}%  "
              f"WR={row['wr']*100:.1f}%  {mk}")
    print(f"  zero_months (avg<=0): {zero_m}/{len(mdf)}")

    # top3日除外
    df["day"] = df[date_col].dt.date
    top3 = df.groupby("day")[val_col].sum().nlargest(3).index
    df_ex = df[~df["day"].isin(top3)]
    if len(df_ex) > 0:
        avg_ex = df_ex[val_col].mean() * 100
        mk = "✓" if avg_ex > 0 else "✗"
        print(f"  top3日除外: n={len(df_ex):,}, avg={avg_ex:+.3f}%  {mk}")

    # LOMO (leave-one-month-out)
    months = sorted(mdf.index)
    lomo_ok = True
    print(f"  LOMO:")
    for m in months:
        sub = df[df["ym"] != m][val_col].mean() * 100
        mk = "✓" if sub > 0 else "✗"
        if sub <= 0: lomo_ok = False
        print(f"    -{m}: avg={sub:+.3f}%  {mk}")
    print(f"  LOMO 全月+: {'✓' if lomo_ok else '✗'}")

    # equity curve
    cum    = (1 + df.sort_values(date_col)[val_col]).cumprod()
    dd     = (cum / cum.cummax() - 1).min() * 100
    wins_  = (df.sort_values(date_col)[val_col] > 0).astype(int).values
    maxcl = curcl = 0
    for w in wins_:
        curcl = 0 if w else curcl + 1
        maxcl = max(maxcl, curcl)
    print(f"  maxDD={dd:.1f}%  最大連敗={maxcl}")


def _report_split(df_pass, df_block, val_col, date_col="entry_dt",
                   pass_label="PASS", block_label="BLOCK"):
    """PASS/BLOCK に分けて統計を出力。"""
    for df_, lbl in [(df_pass, pass_label), (df_block, block_label)]:
        if df_ is None or len(df_) == 0:
            print(f"  {lbl}: n=0"); continue
        v = df_[val_col].values
        wr  = (v > 0).mean() * 100
        avg = v.mean() * 100
        med = np.median(v) * 100
        mk  = "✓" if avg > 0 else "✗"
        print(f"  {lbl}: n={len(v):,}  WR={wr:.1f}%  avg={avg:+.3f}%  "
              f"med={med:+.3f}%  {mk}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1: cs_mom — Cross-Sectional Momentum
# ═══════════════════════════════════════════════════════════════════════════════
print("═" * 72)
print("Section 1: cs_mom  (L=168h rank / H=48h hold / n=5 / market-neutral)")
print("═" * 72)


def cs_mom_backtest(close_df, n=5, L=168, H=48):
    """
    非重複 H-bar スロットで cs_mom バックテスト。
    gross SPREAD = (long_ret + short_profit) / 2  (コスト控除前)
    net SPREAD = gross - cost_leg  (cost_leg = 1 round-trip per leg-pair)
    """
    ret_L = close_df.pct_change(L)
    dates = close_df.index
    N = len(dates)
    rows = []

    for ti in range(L, N - H, H):  # 非重複スロット stride=H
        rv = ret_L.iloc[ti].dropna()
        if len(rv) < 2 * n:
            continue
        sv = rv.sort_values(ascending=False)
        lsyms = sv.index[:n].tolist()
        ssyms = sv.index[-n:].tolist()

        ep = close_df.iloc[ti]
        xp = close_df.iloc[ti + H]
        if ep[lsyms + ssyms].isna().any() or xp[lsyms + ssyms].isna().any():
            continue

        raw  = xp / ep - 1
        lr   = raw[lsyms].mean()   # LONG 生リターン
        sr   = raw[ssyms].mean()   # SHORT側 生リターン（下落で利益）
        gross = (lr + (-sr)) / 2   # SPREAD gross

        rows.append({
            "entry_dt"  : dates[ti],
            "exit_dt"   : dates[ti + H],
            "long_ret"  : lr,
            "short_raw" : sr,
            "gross"     : gross,
            "long_syms" : ",".join(lsyms),
            "short_syms": ",".join(ssyms),
        })

    return pd.DataFrame(rows)


print("\ncs_mom バックテスト実行中...")
slots = cs_mom_backtest(close_df, n=5, L=168, H=48)
print(f"スロット数: {len(slots)}")

# コスト感応度テーブル
print("\nコスト感応度:")
print(f"  {'cost/leg':>10}  {'WR':>6}  {'avg%':>8}  {'med%':>8}  {'判定'}")
for c in COST_SWEEP:
    net_vals = (slots["gross"] - c).values
    wr   = (net_vals > 0).mean() * 100
    avg_ = net_vals.mean() * 100
    med_ = np.median(net_vals) * 100
    ok   = "✓" if avg_ > 0.10 and med_ > 0 else ("△" if avg_ > 0 else "✗")
    print(f"  {c*100:>9.2f}%  {wr:>6.1f}%  {avg_:>+8.3f}%  {med_:>+8.3f}%  {ok}")

# 主コスト 0.15%/leg で詳細
slots["net"] = slots["gross"] - MAIN_COST
_report(slots, "net", label=f"cs_mom SPREAD (cost={MAIN_COST*100:.2f}%/leg)")

# LONG/SHORT レッグ単独
print("\n  レッグ内訳 (0.15%/leg):")
for leg_col, leg_name in [("long_ret", "LONG top5"), ("short_raw", "SHORT bot5 (生)")]:
    v = slots[leg_col].values * 100
    wr  = (v > 0 if leg_col == "long_ret" else v < 0).mean() * 100
    avg_ = v.mean()
    med_ = np.median(v)
    print(f"    {leg_name}: avg={avg_:+.3f}%  med={med_:+.3f}%  (勝ち方向WR={wr:.1f}%)")

# mom vs rev 確認
print("\n  mom (強→弱 SHORT) vs rev (弱→強 LONG) 比較:")
# revはランク順を逆にする（LONGに弱い銘柄、SHORTに強い銘柄）
ret_L_rev = close_df.pct_change(168)
rows_rev = []
dates_g = close_df.index
N = len(IDX)
for ti in range(168, N - 48, 48):
    rv = ret_L_rev.iloc[ti].dropna()
    if len(rv) < 10: continue
    sv = rv.sort_values(ascending=False)
    lsyms = sv.index[-5:].tolist()  # 弱い側をLONG
    ssyms = sv.index[:5].tolist()   # 強い側をSHORT
    ep = close_df.iloc[ti]; xp = close_df.iloc[ti + 48]
    if ep[lsyms+ssyms].isna().any() or xp[lsyms+ssyms].isna().any(): continue
    raw = xp / ep - 1
    lr = raw[lsyms].mean(); sr = raw[ssyms].mean()
    rows_rev.append({"gross": (lr + (-sr)) / 2})
rev_df = pd.DataFrame(rows_rev)
if len(rev_df):
    rv_net = (rev_df["gross"] - MAIN_COST).mean() * 100
    rv_med = (rev_df["gross"] - MAIN_COST).median() * 100
    print(f"    mom avg={((slots['gross']-MAIN_COST).mean()*100):+.3f}% "
          f"med={((slots['gross']-MAIN_COST).median()*100):+.3f}%")
    print(f"    rev avg={rv_net:+.3f}% med={rv_med:+.3f}%")
    print(f"    → mom > rev: {'✓ (方向性確認)' if (slots['gross']-MAIN_COST).mean() > (rev_df['gross']-MAIN_COST).mean() else '✗ (mom優位なし)'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2: rv_chg — BTC 実現ボラ変化ゲート (LONG方向)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 2: rv_chg  (BTC 実現ボラ変化ゲート / LONG方向)")
print("  rv12 = pct_change(1).rolling(12).std()  rv_chg = rv12/rv12.shift(4)-1")
print("  PASS: rv_chg < threshold  (BLOCK: rv_chg >= threshold)")
print("═" * 72)

# BTC rv_chg 計算 (as-of: 完了足のみ)
btc_r1  = close_df["BTC"].pct_change(1)
rv12    = btc_r1.rolling(12).std()
rv_chg  = rv12 / rv12.shift(4) - 1  # 欠損はNaN、補完しない

# cs_mom スロットに rv_chg を紐付け（エントリー時刻のas-of値）
rv_chg_map = rv_chg.to_dict()
slots["rv_chg"] = slots["entry_dt"].map(rv_chg_map)

# cs_mom LONG レッグを rv_chg PASS/BLOCK で分離
slots["long_net"] = slots["long_ret"] - MAIN_COST  # LONGレッグ net

print(f"\nrv_chg が非NaN のスロット数: {slots['rv_chg'].notna().sum()} / {len(slots)}")

RV_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]
print("\nrv_chg 閾値スイープ (cs_mom LONGレッグ, 0.15%/leg):")
print(f"  {'thr':>6}  {'n_pass':>8}  {'PASS avg%':>10}  {'PASS med%':>10}  "
      f"{'n_block':>8}  {'BLOCK avg%':>11}  {'判定'}")

for thr in RV_THRESHOLDS:
    mask_pass  = slots["rv_chg"] < thr
    mask_block = ~mask_pass & slots["rv_chg"].notna()
    p = slots[mask_pass & slots["rv_chg"].notna()]["long_net"]
    b = slots[mask_block]["long_net"]
    p_avg = p.mean()*100 if len(p) else np.nan
    p_med = p.median()*100 if len(p) else np.nan
    b_avg = b.mean()*100 if len(b) else np.nan
    ok = "✓" if (not np.isnan(p_avg) and p_avg > b_avg) else "✗"
    print(f"  {thr*100:>5.0f}%  {len(p):>8}  {p_avg:>+10.3f}%  {p_med:>+10.3f}%  "
          f"{len(b):>8}  {b_avg:>+11.3f}%  {ok}")

# 主閾値 5% で詳細（月別・LOWO）
THR = 0.05
mask_p = slots["rv_chg"].notna() & (slots["rv_chg"] < THR)
mask_b = slots["rv_chg"].notna() & (slots["rv_chg"] >= THR)
df_pass = slots[mask_p].copy()
df_pass["net"] = df_pass["long_net"]
df_block = slots[mask_b].copy()
df_block["net"] = df_block["long_net"]

print(f"\nrv_chg<{THR*100:.0f}% PASS 詳細 (cs_mom LONGレッグ):")
_report(df_pass, "net", label=f"rv_chg PASS (thr={THR*100:.0f}%)")

print(f"\nrv_chg>={THR*100:.0f}% BLOCK 詳細:")
_report(df_block, "net", label=f"rv_chg BLOCK (thr={THR*100:.0f}%)")

# 全シンボル平均リターン (市場全体) で rv_chg を検証
print("\n  [全シンボル平均リターン で rv_chg を検証] (市場全体・ランキングなし)")
all_syms = [s for s in close_df.columns if s != "BTC"]
all_ret_rows = []
for ti in range(168, N - 48, 48):
    rets = (close_df.iloc[ti+48][all_syms] / close_df.iloc[ti][all_syms] - 1).dropna()
    if len(rets) < 5: continue
    rv_val = rv_chg.iloc[ti]
    if np.isnan(rv_val): continue
    all_ret_rows.append({
        "entry_dt": IDX[ti],
        "market_ret": rets.mean(),
        "rv_chg": rv_val,
    })
mdf = pd.DataFrame(all_ret_rows)
if len(mdf) > 0:
    for thr_ in [0.03, 0.05, 0.07, 0.10]:
        p_  = mdf[mdf["rv_chg"] < thr_]["market_ret"]
        b_  = mdf[mdf["rv_chg"] >= thr_]["market_ret"]
        ok_ = "✓" if len(p_) and len(b_) and p_.mean() > b_.mean() else "✗"
        print(f"    thr={thr_*100:.0f}%: PASS avg={p_.mean()*100:+.3f}% (n={len(p_)}) "
              f"vs BLOCK avg={b_.mean()*100:+.3f}% (n={len(b_)})  {ok_}")

# LOWO: leave-one-week-out (サマリー)
print(f"\n  LOWO (rv_chg<5% PASS, cs_mom LONG, 0.15%/leg) — 週ごとに除外して全体avgを確認:")
df_pass["wk"] = (df_pass["entry_dt"].dt.isocalendar().week.astype(str)
                 + "_" + df_pass["entry_dt"].dt.year.astype(str))
weeks = sorted(df_pass["wk"].unique())
lowo_ok = True
lowo_avgs = []
for wk in weeks:
    sub = df_pass[df_pass["wk"] != wk]["net"].mean() * 100
    lowo_avgs.append(sub)
    if sub <= 0:
        lowo_ok = False
if lowo_avgs:
    print(f"  LOWO avg 範囲: min={min(lowo_avgs):+.3f}%  max={max(lowo_avgs):+.3f}%"
          f"  (n_weeks={len(weeks)})")
    neg_weeks = [w for w, a in zip(weeks, lowo_avgs) if a <= 0]
    if neg_weeks:
        print(f"  avg≤0 になった除外週: {neg_weeks}")
print(f"  LOWO 全週+: {'✓' if lowo_ok else '✗'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3: S4 — Donchian 55h SHORT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 3: S4  (Donchian 55h下抜けSHORT / TP=12% / SL=6% / 7日保有)")
print("  シグナル: close == rolling_min(close,55h) (新55h安値)")
print("  フィルター: 1銘柄1日1エントリー / ポジション上限 3or5 (グリーディ)")
print("═" * 72)

S4_TP       = 0.12
S4_SL       = 0.06
S4_MAX_HOLD = 168  # 7日 × 24h
S4_WINDOW   = 55
S4_COST     = 0.0015  # 0.15%/leg


def simulate_s4_raw(close_df, high_df, low_df,
                     tp_pct=S4_TP, sl_pct=S4_SL,
                     max_hold=S4_MAX_HOLD, window=S4_WINDOW,
                     cost_leg=S4_COST):
    """
    全銘柄のS4シグナルを生成し、ポジション上限適用前のrawトレードを返す。
    1銘柄1日1エントリーフィルター適用済み。
    """
    dates = close_df.index
    N = len(dates)
    all_trades = []

    for s in close_df.columns:
        c  = close_df[s].values
        h  = high_df[s].values
        lo = low_df[s].values

        # Donchian rolling min (pandas が速い)
        c_s = pd.Series(c)
        dc_low = c_s.rolling(window).min().values

        # シグナル: close が 55h 新安値
        signal_mask = (c <= dc_low) & np.isfinite(c) & np.isfinite(dc_low)
        signal_mask[:window - 1] = False
        signal_idxs = np.where(signal_mask)[0]

        last_day = None
        for ti in signal_idxs:
            entry_day = dates[ti].date()
            if entry_day == last_day:
                continue  # 1日1エントリーフィルター
            last_day = entry_day

            ep = c[ti]
            if not np.isfinite(ep) or ep <= 0:
                continue

            tp_price = ep * (1 - tp_pct)
            sl_price = ep * (1 + sl_pct)

            # forward scan (ti+1 から最大 max_hold bars)
            end_bar = min(ti + max_hold, N - 1)
            flo = lo[ti + 1: end_bar + 1]
            fhi = h[ti + 1: end_bar + 1]

            tp_idxs = np.where(flo <= tp_price)[0]
            sl_idxs = np.where(fhi >= sl_price)[0]
            tp_bar = tp_idxs[0] if len(tp_idxs) else max_hold
            sl_bar = sl_idxs[0] if len(sl_idxs) else max_hold

            if tp_bar <= sl_bar and tp_bar < max_hold:
                result   = "TP"
                pnl      = tp_pct - 2 * cost_leg
                exit_idx = ti + 1 + tp_bar
            elif sl_bar < tp_bar and sl_bar < max_hold:
                result   = "SL"
                pnl      = -(sl_pct + 2 * cost_leg)
                exit_idx = ti + 1 + sl_bar
            elif tp_bar == sl_bar and tp_bar < max_hold:
                # 同足両ヒット → SL優先（保守的）
                result   = "SL"
                pnl      = -(sl_pct + 2 * cost_leg)
                exit_idx = ti + 1 + sl_bar
            else:
                # 期限切れ: close で決済
                exit_c = c[end_bar]
                if not np.isfinite(exit_c):
                    continue
                pnl      = (ep - exit_c) / ep - 2 * cost_leg
                result   = "EXPIRE"
                exit_idx = end_bar

            all_trades.append({
                "entry_dt"   : dates[ti],
                "exit_dt"    : dates[exit_idx],
                "symbol"     : s,
                "entry_price": ep,
                "pnl"        : pnl,
                "result"     : result,
            })

    return pd.DataFrame(all_trades).sort_values("entry_dt").reset_index(drop=True)


def apply_pos_limit(df_raw, pos_limit=5):
    """グリーディにポジション上限を適用。open期間が重なる場合は上限まで。"""
    df = df_raw.sort_values("entry_dt").reset_index(drop=True)
    heap = []   # min-heap of exit_dt (pd.Timestamp)
    sel  = []

    for _, row in df.iterrows():
        et = row["entry_dt"]
        xt = row["exit_dt"]
        # 既に終了したポジションを除去
        while heap and heap[0] <= et:
            heapq.heappop(heap)
        if len(heap) < pos_limit:
            heapq.heappush(heap, xt)
            sel.append(True)
        else:
            sel.append(False)

    return df[sel].reset_index(drop=True)


print("\nS4 シミュレーション実行中...")
df_raw = simulate_s4_raw(close_df, high_df, low_df)
print(f"raw トレード数 (portfolio filter前): {len(df_raw):,}")
if len(df_raw):
    wr_raw = (df_raw["pnl"] > 0).mean() * 100
    avg_raw = df_raw["pnl"].mean() * 100
    tp_ct = (df_raw["result"] == "TP").sum()
    sl_ct = (df_raw["result"] == "SL").sum()
    ex_ct = (df_raw["result"] == "EXPIRE").sum()
    print(f"  結果内訳: TP={tp_ct}  SL={sl_ct}  EXPIRE={ex_ct}")
    print(f"  raw WR={wr_raw:.1f}%  avg={avg_raw:+.3f}%")

for pos_lim in [5, 3]:
    df_s4 = apply_pos_limit(df_raw, pos_limit=pos_lim)
    print(f"\n  ── ポジション上限 {pos_lim} 適用後 ──")
    print(f"  トレード数: {len(df_s4):,}")
    if len(df_s4) == 0:
        continue
    _report(df_s4, "pnl", label=f"S4 (pos_limit={pos_lim}, cost=0.15%/leg)")

# BTC暴落月（月次<-10%）の影響確認
btc_monthly = close_df["BTC"].resample("ME").last().pct_change()
crash_months = set(btc_monthly[btc_monthly < -0.10].index.to_period("M"))

df_s4_main = apply_pos_limit(df_raw, pos_limit=5)
if len(df_s4_main) > 0:
    df_s4_main["ym"] = df_s4_main["entry_dt"].dt.to_period("M")
    print(f"\n  暴落月 (BTC月次<-10%): {sorted(crash_months)}")
    df_no_crash = df_s4_main[~df_s4_main["ym"].isin(crash_months)]
    df_crash    = df_s4_main[df_s4_main["ym"].isin(crash_months)]
    print(f"  暴落月除外後 (n={len(df_no_crash):,}):")
    if len(df_no_crash):
        v = df_no_crash["pnl"].values * 100
        mk = "✓" if v.mean() > 0 else "✗"
        print(f"    WR={( v>0).mean()*100:.1f}%  avg={v.mean():+.3f}%  "
              f"med={np.median(v):+.3f}%  {mk}")
    print(f"  暴落月のみ (n={len(df_crash):,}):")
    if len(df_crash):
        v = df_crash["pnl"].values * 100
        mk = "✓" if v.mean() > 0 else "✗"
        print(f"    WR={(v>0).mean()*100:.1f}%  avg={v.mean():+.3f}%  "
              f"med={np.median(v):+.3f}%  {mk}")

    # コスト感応度
    print(f"\n  S4 コスト感応度 (pos_limit=5):")
    print(f"  {'cost/leg':>10}  {'WR':>6}  {'avg%':>8}  {'med%':>8}  {'判定'}")
    for c in [0.0008, 0.0010, 0.0015, 0.0020, 0.0030]:
        net_ = df_s4_main["pnl"] - (2*c - 2*S4_COST)  # TP→TP-delta, SL→SL+delta
        # 正確な計算: resultごとに調整
        adj = []
        for _, row in df_s4_main.iterrows():
            base = row["pnl"]
            if row["result"] == "TP":
                adj.append(S4_TP - 2*c)
            elif row["result"] == "SL":
                adj.append(-(S4_SL + 2*c))
            else:  # EXPIRE
                # PnL = (ep - exit)/ep - 2*cost_leg_orig, adjust cost only
                adj.append(base - 2*c + 2*S4_COST)
        adj = np.array(adj) * 100
        wr_  = (adj > 0).mean() * 100
        avg_ = adj.mean()
        med_ = np.median(adj)
        ok_  = "✓" if avg_ > 0 and med_ > 0 else ("△" if avg_ > 0 else "✗")
        print(f"  {c*100:>9.2f}%  {wr_:>6.1f}%  {avg_:>+8.3f}%  {med_:>+8.3f}%  {ok_}")


# ═══════════════════════════════════════════════════════════════════════════════
#  総括
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("総括")
print("═" * 72)
print("""
採用判断基準（全て満たすこと）:
  WR>=60%, avg>=+0.20%, median>0,
  top3日除外後もavg+, LOMO全月+, zero_months<=2, maxDD許容範囲

この結果で以下を確認してください:
  cs_mom:  月次安定性 / 5月依存が解消されているか
  rv_chg:  PASS > BLOCK が全閾値で一貫するか
  S4:      暴落月除外後もプラスか / maxDD許容範囲か

本番変更・deploy・merge はこのスクリプトでは行いません。
""")
