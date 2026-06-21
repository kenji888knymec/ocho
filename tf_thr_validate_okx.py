"""
tf_thr_validate_okx.py
=======================
OKX swap OHLCVデータを使った、SHORT ML モデルの train-fold threshold検証。
tpsl_ml.py と完全に同一のロジック。データソースのみ /tmp/ohlcv_okx/ に変更。

前提: fetch_okx_ohlcv.py を実行して /tmp/ohlcv_okx/{SYM}_1h.csv が揃っていること。
（build_v2_okx.py は不要 — このスクリプトが独立してOHLCVを読み込む）

確認する6基準（c=0.08/leg基準）:
  1. WR ≥ 60%
  2. avg PnL_Net ≥ +0.15%
  3. median PnL_Net > 0
  4. top3日除外後も avg プラス
  5. leave-one-week-out 全週プラス
  6. active_months ≥ 6, zero_months ≤ 2

パラメータ: TRAIN_M=12, TEST_M=2, PCT=98, no-overlap
TP/SL検証: (1.2%/1.0%) 主力 + (1.0%/1.0%) 比較
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("/tmp/ohlcv_okx")
SYMBOLS = ["AAVE","ADA","APT","ARB","ATOM","AVAX","BNB","BONK","BTC","DOGE",
           "DOT","ETH","FET","HBAR","INJ","LINK","LTC","NEAR","POL","SEI",
           "SHIB","SOL","STX","SUI","TRX","UNI","XLM","XRP"]
HORIZON = 48
CLEG_LIST = [0.00055, 0.0008, 0.0010, 0.0015]
TRAIN_M, TEST_M, PCT = 12, 2, 98
NONFEAT = ["label", "pnl", "symbol", "dt", "month", "win", "lose", "bm", "be_ret"]
COMBOS = [(0.012, 0.010), (0.010, 0.010), (0.015, 0.012), (0.010, 0.012)]


# ---------- データ確認 ----------
missing = [s for s in SYMBOLS if not (DATA_DIR / f"{s}_1h.csv").exists()]
if missing:
    print(f"WARNING: 以下の銘柄のCSVが見つかりません: {missing}")
    print(f"fetch_okx_ohlcv.py を実行してください。")
    SYMBOLS = [s for s in SYMBOLS if s not in missing]
    if not SYMBOLS:
        raise SystemExit("OHLCV CSVが1つもありません")

# ---------- OHLCV ロード ----------
print(f"OKX OHLCV ロード中 ({len(SYMBOLS)} 銘柄)...")
O, H, L, C, V = {}, {}, {}, {}, {}
for s in SYMBOLS:
    df = pd.read_csv(DATA_DIR / f"{s}_1h.csv", parse_dates=["datetime_utc"])
    df = df.set_index("datetime_utc").sort_index()
    df.index = df.index.tz_localize(None)
    O[s], H[s], L[s], C[s], V[s] = df["open"], df["high"], df["low"], df["close"], df["volume"]

close_df = pd.DataFrame(C)
high_df  = pd.DataFrame(H)
low_df   = pd.DataFrame(L)
vol_df   = pd.DataFrame(V)
idx = close_df.index
print(f"  期間: {idx[0]} 〜 {idx[-1]}  ({len(idx)} 時間足)")


# ---------- 共通特徴量 ----------
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

btc = close_df["BTC"]
btc_r1, btc_r4   = btc.pct_change(1), btc.pct_change(4)
btc_r24, btc_r72 = btc.pct_change(24), btc.pct_change(72)
btc_vol  = btc_r1.rolling(24).std()
btc_rsi  = rsi(btc, 14)
btc_rv12 = btc_r1.rolling(12).std()
btc_rvchg = btc_rv12 / btc_rv12.shift(4) - 1
btc_ma50  = btc.rolling(50).mean()
btc_ma200 = btc.rolling(200).mean()
btc_trend   = btc / btc_ma50  - 1
btc_trend_l = btc / btc_ma200 - 1
btc_dist_hi = btc / btc.rolling(168).max() - 1
ret24_all  = close_df.pct_change(24)
ret72_all  = close_df.pct_change(72)
ret168_all = close_df.pct_change(168)
cs24  = ret24_all.rank(axis=1,  pct=True)
cs72  = ret72_all.rank(axis=1,  pct=True)
cs168 = ret168_all.rank(axis=1, pct=True)


def build_features(s):
    c = close_df[s]; h = high_df[s]; l = low_df[s]; v = vol_df[s]
    r1 = c.pct_change(1)
    atr = (h - l).rolling(24).mean() / c
    f = pd.DataFrame(index=idx)
    f["ret_1h"]   = c.pct_change(1);  f["ret_4h"]  = c.pct_change(4)
    f["ret_24h"]  = c.pct_change(24); f["ret_72h"]  = c.pct_change(72)
    f["ret_168h"] = c.pct_change(168);f["ret_336h"] = c.pct_change(336)
    f["rsi_14"]   = rsi(c, 14);       f["rsi_4"]    = rsi(c, 4)
    f["vol_24h"]  = r1.rolling(24).std()
    f["vol_chg"]  = r1.rolling(12).std() / r1.rolling(12).std().shift(4) - 1
    f["volvol"]   = r1.rolling(24).std().rolling(48).std()
    f["dist_hi_24"]  = c / h.rolling(24).max()  - 1
    f["dist_lo_24"]  = c / l.rolling(24).min()  - 1
    f["dist_hi_168"] = c / h.rolling(168).max() - 1
    f["dist_lo_168"] = c / l.rolling(168).min() - 1
    f["atr"]         = atr
    f["dist_hi_atr"] = (c / h.rolling(24).max() - 1) / atr
    f["vol_z"]    = (v - v.rolling(48).mean()) / v.rolling(48).std()
    f["vol_trend"]= v.rolling(24).mean() / v.rolling(168).mean() - 1
    f["rel_str_24"]  = c.pct_change(24)  - btc_r24
    f["rel_str_72"]  = c.pct_change(72)  - btc_r72
    f["cs_rank_24"]  = cs24[s];  f["cs_rank_72"]  = cs72[s];  f["cs_rank_168"] = cs168[s]
    f["btc_ret1"]  = btc_r1;  f["btc_ret4"]  = btc_r4
    f["btc_ret24"] = btc_r24; f["btc_ret72"] = btc_r72
    f["btc_vol"]   = btc_vol;    f["btc_rsi"]   = btc_rsi
    f["btc_rvchg"] = btc_rvchg;  f["btc_trend"]   = btc_trend
    f["btc_trend_l"] = btc_trend_l; f["btc_dist_hi"] = btc_dist_hi
    f["hour"]   = idx.hour
    f["dow"]    = idx.dayofweek
    f["sym_id"] = SYMBOLS.index(s)
    return f

print("特徴量キャッシュ作成中...")
FEAT_CACHE = {s: build_features(s) for s in SYMBOLS}


# ---------- TP/SL別ラベル計算 ----------
def compute_labels(tp, sl):
    """SHORT方向TP/SL先着ラベル。戻り値: symbol -> DataFrame(label, win, lose, bm)"""
    out = {}
    n = len(idx)
    for s in SYMBOLS:
        ep = close_df[s].values
        hi = high_df[s].values
        lo = low_df[s].values
        ftp = np.full(n, HORIZON + 1, float)
        fsl = np.full(n, HORIZON + 1, float)
        tp_p = ep * (1 - tp)  # SHORT TP: 下落ターゲット
        sl_p = ep * (1 + sl)  # SHORT SL: 上昇ロスカット
        for j in range(1, HORIZON + 1):
            hj = np.roll(hi, -j); lj = np.roll(lo, -j)
            hj[-j:] = np.nan;    lj[-j:] = np.nan
            ftp[(lj <= tp_p) & (ftp > HORIZON)] = j  # LOW足がTP下回る
            fsl[(hj >= sl_p) & (fsl > HORIZON)] = j  # HIGH足がSL上回る
        lab = np.full(n, np.nan)
        win  = ftp < fsl
        lose = fsl <= ftp
        bm   = (ftp > HORIZON) & (fsl > HORIZON)
        lab[win]  = 1
        lab[lose] = 0
        lab[bm]   = 0
        out[s] = pd.DataFrame({"label": lab, "win": win, "lose": lose, "bm": bm}, index=idx)
    return out


def make_dataset(tp, sl):
    labs = compute_labels(tp, sl)
    rows = []
    for s in SYMBOLS:
        f = FEAT_CACHE[s].copy()
        d = labs[s]
        f["label"] = d["label"]; f["win"] = d["win"]
        f["lose"]  = d["lose"];  f["bm"]  = d["bm"]
        f["symbol"] = s; f["dt"] = idx
        rows.append(f)
    full = pd.concat(rows, ignore_index=True)
    full = full[full["dt"].dt.hour % 2 == 0]  # 2h間引き
    full = full.dropna(subset=["label", "ret_336h", "rsi_14", "vol_z", "btc_rvchg", "btc_trend_l"])
    return full


# ---------- 選抜（train-fold閾値 + no-overlap） ----------
def gen_select(df, tp, sl):
    df = df.copy()
    df["month"] = df["dt"].dt.to_period("M")
    FEAT = [c for c in df.columns if c not in NONFEAT]
    months = sorted(df["month"].unique())
    sel_rows = []
    fold_ct  = 0
    i = TRAIN_M
    while i + TEST_M <= len(months):
        tr = df[df["month"].isin(months[i - TRAIN_M:i])]
        te = df[df["month"].isin(months[i:i + TEST_M])]
        if len(tr) >= 3000 and len(te) >= 200:
            mdl = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.02, max_depth=6, num_leaves=48,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=120,
                reg_lambda=2.0, verbose=-1)
            mdl.fit(tr[FEAT], tr["label"])
            thr = np.percentile(mdl.predict_proba(tr[FEAT])[:, 1], PCT)
            te = te.copy()
            te["prob"] = mdl.predict_proba(te[FEAT])[:, 1]
            passed = te[te["prob"] >= thr]
            sel_rows.append(passed.copy())
            fold_ct += 1
            print(f"  fold{fold_ct} [{months[i-TRAIN_M]}〜{months[i+TEST_M-1]}]: "
                  f"train={len(tr)} test={len(te)} thr={thr:.4f} pass={len(passed)}")
        i += TEST_M

    sel = pd.concat(sel_rows, ignore_index=True) if sel_rows else pd.DataFrame()
    if len(sel) == 0:
        return sel

    # no-overlap: 同symbol 48h以内の重複除去（dt昇順で先着優先）
    sel = sel.sort_values("dt").reset_index(drop=True)
    keep = []; last_exit = {}
    for ix, r in sel.iterrows():
        s, t = r["symbol"], r["dt"]
        if s in last_exit and t < last_exit[s]:
            continue
        keep.append(ix)
        last_exit[s] = t + pd.Timedelta(hours=HORIZON)
    return sel.loc[keep].copy()


# ---------- 頑健性レポート ----------
def robustness(sel, tp, sl, tag):
    print(f"\n{'='*72}")
    print(f"=== OKX | {tag} | TP={tp*100:.1f}%/SL={sl*100:.1f}% | no-overlap n={len(sel)} ===")
    print(f"{'='*72}")
    if len(sel) < 10:
        print("  選抜数不足"); return False

    sel = sel.copy()
    sel["month"] = sel["dt"].dt.to_period("M")
    sel["week"]  = sel["dt"].dt.to_period("W")
    sel["date"]  = sel["dt"].dt.date

    wr = (sel["label"] == 1).mean()
    n_months = sel["month"].nunique()
    print(f"  n={len(sel)}, WR={wr:.1%}, active_months={n_months}")

    ref_avg = ref_med = ref_lowo_ok = ref_extop = None
    for cleg in CLEG_LIST:
        p = np.where(sel["label"] == 1, tp * 100, -sl * 100) - 2 * cleg * 100
        sel2 = sel.assign(_p=p)

        pmon  = sel2.groupby("month")["_p"].mean()
        pm    = (pmon > 0).sum()

        lowo  = []
        for w in sel["week"].unique():
            rest = sel[sel["week"] != w]
            pr = np.where(rest["label"] == 1, tp * 100, -sl * 100) - 2 * cleg * 100
            lowo.append(pr.mean() if len(pr) else np.nan)
        lowo = [x for x in lowo if not np.isnan(x)]
        lowo_ok = all(x > 0 for x in lowo)

        dpnl = sel2.groupby("date")["_p"].sum().sort_values(ascending=False)
        top3  = dpnl.head(3).index
        extop = sel2[~sel2["date"].isin(top3)]["_p"].mean()

        flag = "✅" if lowo_ok else "❌"
        print(f"  c={cleg*100:.3f}%/leg: avg={p.mean():+.3f}% med={np.median(p):+.3f}% "
              f"+月={pm}/{n_months} "
              f"LOWO{flag}[{min(lowo):+.3f}〜{max(lowo):+.3f}] "
              f"top3日除外avg={extop:+.3f}%")

        if cleg == 0.0008:
            ref_avg     = p.mean()
            ref_med     = np.median(p)
            ref_lowo_ok = lowo_ok
            ref_extop   = extop

    # 資金曲線
    p08  = np.where(sel["label"] == 1, tp * 100, -sl * 100) - 2 * 0.0008 * 100
    eq   = np.cumsum(p08)
    dd   = eq - np.maximum.accumulate(eq)
    loss = (p08 < 0).astype(int); mc = cur = 0
    for x in loss:
        cur = cur + 1 if x else 0; mc = max(mc, cur)
    print(f"  資金曲線(c=0.08): 累積={eq[-1]:+.1f}% maxDD={dd.min():+.1f}% 最大連敗={mc}")

    # 6基準判定
    c1 = wr >= 0.60
    c2 = ref_avg  >= 0.15
    c3 = ref_med  >  0
    c4 = ref_extop > 0
    c5 = ref_lowo_ok
    zero_m = n_months - (pmon > 0).sum()
    c6 = (n_months >= 6) and (zero_m <= 2)
    all_ok = all([c1, c2, c3, c4, c5, c6])

    print(f"\n  【6基準 (c=0.08/leg)】")
    print(f"  1. WR≥60%:      {'✅' if c1 else '❌'} ({wr:.1%})")
    print(f"  2. avg≥+0.15%:  {'✅' if c2 else '❌'} ({ref_avg:+.3f}%)")
    print(f"  3. median>0:    {'✅' if c3 else '❌'} ({ref_med:+.3f}%)")
    print(f"  4. top3日除外+: {'✅' if c4 else '❌'} ({ref_extop:+.3f}%)")
    print(f"  5. LOWO全+:     {'✅' if c5 else '❌'}")
    print(f"  6. active_m≥6: {'✅' if c6 else '❌'} ({n_months}ヶ月, zero={zero_m})")
    verdict = "✅ GO" if all_ok else "❌ 未達"
    print(f"  → {verdict}")
    return all_ok


def main():
    print(f"\nOKX SHORT ML 検証")
    print(f"設定: TRAIN_M={TRAIN_M}, TEST_M={TEST_M}, PCT={PCT}th, no-overlap, SHORT")
    print(f"データ: {DATA_DIR}\n")

    passed = []
    for tp, sl in COMBOS:
        print(f"\n{'─'*50}")
        print(f"TP={tp*100:.1f}%/SL={sl*100:.1f}%: ラベル再計算中...")
        full = make_dataset(tp, sl)
        base_wr = (full["label"] == 1).mean()
        print(f"  母集団: n={len(full)}, baseline WR={base_wr:.1%}")
        sel = gen_select(full, tp, sl)
        if len(sel) == 0:
            print("  → 選抜ゼロ"); continue
        ok = robustness(sel, tp, sl, f"{TRAIN_M}M→{TEST_M}M pct{PCT}")
        if ok:
            passed.append((tp, sl))

    print(f"\n{'='*72}")
    print("=== OKX検証 最終結論 ===")
    if passed:
        print(f"✅ 合格パラメータ: {[(f'TP={t*100:.1f}%/SL={s*100:.1f}%') for t,s in passed]}")
        print("SHORT_ML_SHADOW 実装GO（上記パラメータを使用すること）")
    else:
        print("❌ 全パラメータ未達 → SHORT_ML_SHADOW 実装しない")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
