"""
build_v2_okx.py
================
OKX swap OHLCVからdataset2_short_okx.pklを生成する。
build_v2.py と同一ロジック（Bybit → OKX 差し替えのみ）。

前提: fetch_okx_ohlcv.py を実行して /tmp/ohlcv_okx/ にCSVが揃っていること。

注意:
- BONK → 1000BONK/USDT:USDT (price = 1000 BONK/USDT) → 価格スケール1000x
  ただし TP/SL はパーセンテージ判定のため、スケールの違いは影響しない。
- SHIB → 1000SHIB/USDT:USDT → 同上
- 特徴量: tpsl_ml.py と同一37特徴量
- TP=SL=1.0% (ラベル用。tpsl_ml.pyがTP/SLを別途試す)
"""
import pandas as pd, numpy as np
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore")

DATA_DIR = Path("/tmp/ohlcv_okx")  # ← OKXデータディレクトリ
SYMBOLS = ["AAVE","ADA","APT","ARB","ATOM","AVAX","BNB","BONK","BTC","DOGE",
           "DOT","ETH","FET","HBAR","INJ","LINK","LTC","NEAR","POL","SEI",
           "SHIB","SOL","STX","SUI","TRX","UNI","XLM","XRP"]
TP, SL, HORIZON, COST = 0.010, 0.010, 48, 0.00055
t0=time.time()

# ---------- OHLCV ロード ----------
O,H,L,C,V = {},{},{},{},{}
missing = []
for s in SYMBOLS:
    fpath = DATA_DIR/f"{s}_1h.csv"
    if not fpath.exists():
        print(f"  WARNING: {fpath} が存在しない → スキップ")
        missing.append(s)
        continue
    df = pd.read_csv(fpath, parse_dates=["datetime_utc"])
    df = df.set_index("datetime_utc").sort_index(); df.index=df.index.tz_localize(None)
    O[s],H[s],L[s],C[s],V[s]=df["open"],df["high"],df["low"],df["close"],df["volume"]

if missing:
    print(f"欠損銘柄: {missing} → 除外してビルド続行")
    SYMBOLS = [s for s in SYMBOLS if s not in missing]

close_df=pd.DataFrame(C); high_df=pd.DataFrame(H); low_df=pd.DataFrame(L); vol_df=pd.DataFrame(V)
idx=close_df.index

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); return 100-100/(1+up/dn)

btc=close_df["BTC"]
btc_r1,btc_r4,btc_r24,btc_r72=btc.pct_change(1),btc.pct_change(4),btc.pct_change(24),btc.pct_change(72)
btc_vol=btc_r1.rolling(24).std(); btc_rsi=rsi(btc,14)
btc_rv12=btc_r1.rolling(12).std(); btc_rvchg=btc_rv12/btc_rv12.shift(4)-1
btc_ma50=btc.rolling(50).mean(); btc_ma200=btc.rolling(200).mean()
btc_trend=(btc/btc_ma50-1); btc_trend_l=(btc/btc_ma200-1)
btc_dist_hi=btc/btc.rolling(168).max()-1
ret24_all=close_df.pct_change(24); ret72_all=close_df.pct_change(72); ret168_all=close_df.pct_change(168)
cs24=ret24_all.rank(axis=1,pct=True); cs72=ret72_all.rank(axis=1,pct=True); cs168=ret168_all.rank(axis=1,pct=True)

ep=close_df.values; hi=high_df.values; lo=low_df.values; sym_map={s:i for i,s in enumerate(close_df.columns)}
n_t=len(idx)

rows=[]
for si,s in enumerate(SYMBOLS):
    print(f"[{si+1}/{len(SYMBOLS)}] {s}")
    c=close_df[s]; h=high_df[s]; l=low_df[s]; v=vol_df[s]; r1=c.pct_change(1)
    atr=(h-l).rolling(24).mean()/c

    # 特徴量
    f=pd.DataFrame(index=idx)
    f["ret_1h"]=c.pct_change(1); f["ret_4h"]=c.pct_change(4); f["ret_24h"]=c.pct_change(24)
    f["ret_72h"]=c.pct_change(72); f["ret_168h"]=c.pct_change(168); f["ret_336h"]=c.pct_change(336)
    f["rsi_14"]=rsi(c,14); f["rsi_4"]=rsi(c,4)
    f["vol_24h"]=r1.rolling(24).std(); f["vol_chg"]=r1.rolling(12).std()/r1.rolling(12).std().shift(4)-1
    f["volvol"]=r1.rolling(24).std().rolling(48).std()
    f["dist_hi_24"]=c/h.rolling(24).max()-1; f["dist_lo_24"]=c/l.rolling(24).min()-1
    f["dist_hi_168"]=c/h.rolling(168).max()-1; f["dist_lo_168"]=c/l.rolling(168).min()-1
    f["atr"]=atr; f["dist_hi_atr"]=(c/h.rolling(24).max()-1)/atr
    f["vol_z"]=(v-v.rolling(48).mean())/v.rolling(48).std()
    f["vol_trend"]=v.rolling(24).mean()/v.rolling(168).mean()-1
    f["rel_str_24"]=c.pct_change(24)-btc_r24; f["rel_str_72"]=c.pct_change(72)-btc_r72
    f["cs_rank_24"]=cs24[s]; f["cs_rank_72"]=cs72[s]; f["cs_rank_168"]=cs168[s]
    f["btc_ret1"]=btc_r1; f["btc_ret4"]=btc_r4; f["btc_ret24"]=btc_r24; f["btc_ret72"]=btc_r72
    f["btc_vol"]=btc_vol; f["btc_rsi"]=btc_rsi; f["btc_rvchg"]=btc_rvchg
    f["btc_trend"]=btc_trend; f["btc_trend_l"]=btc_trend_l; f["btc_dist_hi"]=btc_dist_hi
    f["hour"]=idx.hour; f["dow"]=idx.dayofweek; f["sym_id"]=SYMBOLS.index(s)

    # SHORT ラベル（TP/SL先着）
    ep_s=c.values; hi_s=h.values; lo_s=l.values
    ftp=np.full(n_t,HORIZON+1,float); fsl=np.full(n_t,HORIZON+1,float)
    tp_price=ep_s*(1-TP); sl_price=ep_s*(1+SL)
    for j in range(1,HORIZON+1):
        hj=np.roll(hi_s,-j); lj=np.roll(lo_s,-j)
        hj[-j:]=np.nan; lj[-j:]=np.nan
        th=lj<=tp_price; sh=hj>=sl_price
        ftp[th&(ftp>HORIZON)]=j; fsl[sh&(fsl>HORIZON)]=j
    lab=np.full(n_t,np.nan)
    lab[ftp<fsl]=1; lab[fsl<=ftp]=0; lab[(ftp>HORIZON)&(fsl>HORIZON)]=0

    # PnL（後段でcost別に再計算するためここは定数コスト）
    pnl=np.where(lab==1, TP*100, -SL*100) - 2*COST*100

    f["label"]=lab; f["pnl"]=pnl; f["symbol"]=s; f["dt"]=idx
    # 2h毎サンプリング
    f=f[f["dt"].dt.hour%2==0]
    rows.append(f)

ds=pd.concat(rows,ignore_index=True)
ds=ds.dropna(subset=["label","ret_336h","rsi_14","vol_z","btc_rvchg","btc_trend_l"])
print(f"\ndataset: {len(ds)} rows, label dist: {ds['label'].value_counts().to_dict()}")
print(f"date range: {ds['dt'].min()} - {ds['dt'].max()}")
out_path=Path("/tmp/dataset2_short_okx.pkl")
ds.to_pickle(out_path)
print(f"保存: {out_path}  ({time.time()-t0:.1f}秒)")
print(f"\n次のステップ: python3 /tmp/tf_thr_validate_okx.py")
