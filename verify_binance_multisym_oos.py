#!/usr/bin/env python3
"""
verify_binance_multisym_oos.py  【Binance metrics 横展開OOS検証（多銘柄）】
==========================================================================
目的:
  verify_binance_metrics_oos.py（BTCUSDT版）で見つかった信号
    lsr_all 高 → 8h/24h後 下落
    lsr_top 高 → 8h後   下落
    taker   高 → 24h後  上昇
  が、BTC固有の偶然ではなく 他の主要銘柄でも同方向に出るか を確認する。

  ★これは「横展開（breadth）」の検証であり、深掘り（depth）ではない。
   閾値探索・条件追加は一切しない。BTC版と完全に同じ設計を銘柄だけ増やす。

検証設計（BTC版と完全に同一・後から条件追加なし）:
  特徴量 5種 (as-of: 完了データのみ):
    lsr_all  = count_long_short_ratio          （全アカウントLSR）
    lsr_top  = sum_toptrader_long_short_ratio   （大口ポジションLSR）
    oi_chg   = sum_open_interest の1h前比変化率
    taker    = sum_taker_long_short_vol_ratio   （taker買い圧）
    funding  = fundingRate（直近settlement値・as-of）
  目的変数: 8h先・24h先の価格リターン
  分位: 月内3分位（low/mid/high）固定
  期間: BTC版と同じ月リスト（既定=2021-01〜2026-05 全月。下の USE_FULL_MONTHS で切替）

対象銘柄（固定）: BTCUSDT / ETHUSDT / SOLUSDT / DOGEUSDT / BNBUSDT
  ※新しめの銘柄は古い月の metrics が無い場合がある。その月は自動SKIP（補完しない）。

as-of制約: 特徴量は完了時刻のデータのみ。未来情報不使用。
欠損制約: np.nan のまま（0補完・平均補完 絶対禁止）。

★実行判断: まず BTCUSDT全月（verify_binance_metrics_oos.py）の結果を見て、
  lsr_all / lsr_top / taker_24h の方向性が維持されていた場合のみ本スクリプトを実行する。
  BTC全月でシグナルが崩れたら、本スクリプトは実行しない。

実行: python3 verify_binance_multisym_oos.py
依存: pandas numpy   （pip install pandas numpy）
出力: binance_multisym_out/ に
  summary_by_symbol.csv      — 特徴量×銘柄×全期間・ret の集計
  detail_monthly.csv         — 特徴量×銘柄×月別の effect
  consistency_by_symbol.csv  — 銘柄別 一貫性サマリ
  consistency_pool.csv       — 全銘柄プール 一貫性サマリ
"""
import io
import ssl
import urllib.request
import zipfile
from calendar import monthrange
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ─── 設定 ─────────────────────────────────────────────────────────
BASE_DAILY   = "https://data.binance.vision/data/futures/um/daily"
BASE_MONTHLY = "https://data.binance.vision/data/futures/um/monthly"
SSL_CTX = ssl.create_default_context()
HDRS    = {"User-Agent": "Mozilla/5.0"}
OUT_DIR = Path("binance_multisym_out")

# 対象銘柄（固定・BTCを先頭に置きBTC版との突き合わせを容易にする）
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT"]

# 月リスト: BTC版（verify_binance_metrics_oos.py）と完全に同じ設計。
#   USE_FULL_MONTHS = True  → 2021-01〜2026-05 全月（65ヶ月。BTC全月版と同一）
#   USE_FULL_MONTHS = False → 代表月（年2回・1月/7月。軽量・初回の横断確認向き）
# どちらも「設計」ではなく「月数」の違いのみ。閾値・特徴量・分位・評価は不変。
USE_FULL_MONTHS = True

_START = (2021, 1)
_END   = (2026, 5)
_FULL_MONTHS = [
    (y, m)
    for y in range(_START[0], _END[0] + 1)
    for m in range(1, 13)
    if _START <= (y, m) <= _END
]
_REP_MONTHS = [
    (2021, 1), (2021, 7),
    (2022, 1), (2022, 7),
    (2023, 1), (2023, 7),
    (2024, 1), (2024, 7),
    (2025, 1), (2025, 7),
    (2026, 1),
]
MONTHS = _FULL_MONTHS if USE_FULL_MONTHS else _REP_MONTHS

# 特徴量定義（BTC版と完全に同一・固定）
FEATURES = [
    ("lsr_all", "count_long_short_ratio",          "高LSR → 逆張りでDOWN期待"),
    ("lsr_top", "sum_toptrader_long_short_ratio",   "大口高LSR → 逆張りでDOWN期待"),
    ("oi_chg",  "oi_chg_rate",                     "OI急増 → 方向は未知・確認"),
    ("taker",   "sum_taker_long_short_vol_ratio",   "taker買い圧高 → UP/DOWN未知・確認"),
    ("funding", "fundingRate",                      "高Funding → 逆張りでDOWN期待"),
]

# 横展開で特に追跡する 3信号（BTCで一貫性が出たもの）
FOCUS = [("lsr_all", "ret_8h"), ("lsr_all", "ret_24h"),
         ("lsr_top", "ret_8h"), ("taker", "ret_24h")]

KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol", "ignore",
]

METRICS_NUM_COLS = [
    "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]


# ─── ダウンロード共通 ─────────────────────────────────────────────
def _fetch_zip_csv(url: str) -> pd.DataFrame | None:
    req = urllib.request.Request(url, headers=HDRS)
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  [HTTP {e.code}] {url}")
        return None
    except Exception as e:
        print(f"  [ERR] {url}: {e}")
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, header=0)


def fetch_metrics_day(sym: str, d: date) -> pd.DataFrame | None:
    return _fetch_zip_csv(f"{BASE_DAILY}/metrics/{sym}/{sym}-metrics-{d}.zip")


def fetch_klines_day(sym: str, d: date) -> pd.DataFrame | None:
    df = _fetch_zip_csv(f"{BASE_DAILY}/klines/{sym}/1h/{sym}-1h-{d}.zip")
    if df is not None:
        df.columns = KLINES_COLS
    return df


def fetch_funding_month(sym: str, y: int, m: int) -> pd.DataFrame | None:
    return _fetch_zip_csv(f"{BASE_MONTHLY}/fundingRate/{sym}/{sym}-fundingRate-{y}-{m:02d}.zip")


def _to_utc_ns(series: pd.Series) -> pd.Series:
    raw = series.iloc[0]
    if isinstance(raw, str):
        return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")
    unit = "ms" if raw > 1e12 else "s"
    return pd.to_datetime(series, unit=unit, utc=True).astype("datetime64[ns, UTC]")


# ─── 月単位データ構築（BTC版 load_one_month と完全に同一ロジック・銘柄引数のみ追加） ──
def load_one_month(sym: str, y: int, m: int) -> pd.DataFrame | None:
    num_days = monthrange(y, m)[1]
    month_days = [date(y, m, d + 1) for d in range(num_days)]
    nm = m % 12 + 1
    ny = y + (1 if m == 12 else 0)
    extra_day = date(ny, nm, 1)

    # 1. metrics (当月のみ)
    m_frames = [fetch_metrics_day(sym, d) for d in month_days]
    m_frames = [f for f in m_frames if f is not None]
    if not m_frames:
        return None
    metrics = pd.concat(m_frames, ignore_index=True)

    # 2. klines 1h (当月 + 翌月1日: forward return 計算用)
    k_frames = [fetch_klines_day(sym, d) for d in month_days + [extra_day]]
    k_frames = [f for f in k_frames if f is not None]
    if not k_frames:
        return None
    klines = pd.concat(k_frames, ignore_index=True)
    klines["_utc"] = _to_utc_ns(klines["open_time"])
    klines["close"] = klines["close"].astype(float)
    klines = klines.sort_values("_utc").drop_duplicates("_utc").reset_index(drop=True)

    # forward return (shift): ラベルとして使用。as-of制約は特徴量側で担保。
    klines["ret_8h"]  = klines["close"].shift(-8)  / klines["close"] - 1
    klines["ret_24h"] = klines["close"].shift(-24) / klines["close"] - 1

    # 3. metrics → 1h リサンプル（各bucketの最終値 = as-of）
    metrics["_utc_raw"] = _to_utc_ns(metrics["create_time"])
    metrics["_utc"] = metrics["_utc_raw"].dt.floor("1h")
    avail_cols = [c for c in METRICS_NUM_COLS if c in metrics.columns]
    metrics_1h = (
        metrics.sort_values("_utc_raw")
               .groupby("_utc")[avail_cols]
               .last()
               .reset_index()
    )
    metrics_1h = metrics_1h.sort_values("_utc").reset_index(drop=True)

    # OI変化率（前1hとの比）
    if "sum_open_interest" in metrics_1h.columns:
        metrics_1h["oi_chg_rate"] = metrics_1h["sum_open_interest"].pct_change(1)

    # 4. metrics × klines 結合（_utcで精度統一済み）
    merged = pd.merge_asof(
        metrics_1h.sort_values("_utc"),
        klines[["_utc", "close", "ret_8h", "ret_24h"]].sort_values("_utc"),
        on="_utc", direction="backward"
    )

    # 5. Funding（月次・as-of: 直近settlementの値を使用）
    fund = fetch_funding_month(sym, y, m)
    if fund is not None:
        ts_col = next((c for c in fund.columns if "time" in c.lower()), None)
        if ts_col:
            fund["_utc_f"] = _to_utc_ns(fund[ts_col])
            fr_col = next((c for c in fund.columns if "rate" in c.lower()), None)
            if fr_col:
                fund["fundingRate"] = pd.to_numeric(fund[fr_col], errors="coerce")
                merged = pd.merge_asof(
                    merged.sort_values("_utc"),
                    fund[["_utc_f", "fundingRate"]].rename(columns={"_utc_f": "_utc"}).sort_values("_utc"),
                    on="_utc", direction="backward"
                )
    if "fundingRate" not in merged.columns:
        merged["fundingRate"] = np.nan

    merged["symbol"] = sym
    merged["year"]   = y
    merged["month"]  = m
    return merged


# ─── 3分位分析（BTC版 tercile_stats と完全に同一） ──────────────────
def tercile_stats(df: pd.DataFrame, feat_col: str, ret_col: str) -> pd.DataFrame | None:
    sub = df[[feat_col, ret_col]].dropna()
    if len(sub) < 9:
        return None
    q33 = sub[feat_col].quantile(1 / 3)
    q67 = sub[feat_col].quantile(2 / 3)
    rows = []
    for label, mask in [
        ("low",  sub[feat_col] <= q33),
        ("mid", (sub[feat_col] > q33) & (sub[feat_col] <= q67)),
        ("high", sub[feat_col] > q67),
    ]:
        g = sub.loc[mask, ret_col]
        rows.append({
            "quantile": label, "n": len(g),
            "avg":  round(g.mean() * 100, 4),
            "med":  round(g.median() * 100, 4),
        })
    return pd.DataFrame(rows)


def _effect(df: pd.DataFrame, fcol: str, rc: str):
    stats = tercile_stats(df, fcol, rc)
    if stats is None:
        return None, None
    hi = stats.loc[stats["quantile"] == "high", "avg"].values[0]
    lo = stats.loc[stats["quantile"] == "low",  "avg"].values[0]
    return round(hi - lo, 4), int(stats["n"].sum())


# ─── メイン ──────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 72)
    print(f"Binance metrics 横展開OOS検証  銘柄: {SYMBOLS}")
    print(f"月数: {len(MONTHS)}ヶ月  ({'全月 2021-01〜2026-05' if USE_FULL_MONTHS else '代表月(1月/7月)'})")
    print(f"特徴量: {[f[0] for f in FEATURES]}（BTC版と完全に同一・閾値探索なし）")
    print("=" * 72)

    # 銘柄ごとに全月ロード
    per_symbol = {}
    for sym in SYMBOLS:
        print(f"\n■ {sym} ロード中...")
        frames = []
        for (y, m) in MONTHS:
            frame = load_one_month(sym, y, m)
            if frame is not None:
                frames.append(frame)
                print(f"  [{sym} {y}-{m:02d}] rows={len(frame)}")
            else:
                print(f"  [{sym} {y}-{m:02d}] SKIP")
        if frames:
            per_symbol[sym] = pd.concat(frames, ignore_index=True)
            print(f"  → {sym} 合計 {len(per_symbol[sym])} 行 / {len(frames)} ヶ月")
        else:
            print(f"  → {sym} データ取得ゼロ。除外。")

    if not per_symbol:
        print("[ERROR] 全銘柄でデータ取得失敗。")
        return

    summary_rows = []   # 特徴量×銘柄×全期間
    detail_rows  = []   # 特徴量×銘柄×月別
    cons_sym_rows = []  # 銘柄別 一貫性

    for sym, df in per_symbol.items():
        for fkey, fcol, _ in FEATURES:
            if fcol not in df.columns:
                continue
            for rc in ["ret_8h", "ret_24h"]:
                eff_all, n_all = _effect(df, fcol, rc)
                if eff_all is not None:
                    summary_rows.append({
                        "symbol": sym, "feature": fkey, "ret": rc,
                        "effect_hi_lo": eff_all, "n_total": n_all,
                    })
                # 月別
                pos = neg = tot = 0
                for (yr, mo) in sorted(df.groupby(["year", "month"]).groups.keys()):
                    sub_m = df[(df["year"] == yr) & (df["month"] == mo)]
                    eff_m, n_m = _effect(sub_m, fcol, rc)
                    if eff_m is None:
                        continue
                    detail_rows.append({
                        "symbol": sym, "feature": fkey, "ret": rc,
                        "period": f"{yr}-{mo:02d}", "effect_hi_lo": eff_m, "n": n_m,
                    })
                    tot += 1
                    if eff_m > 0:
                        pos += 1
                    elif eff_m < 0:
                        neg += 1
                if tot:
                    cons_sym_rows.append({
                        "symbol": sym, "feature": fkey, "ret": rc,
                        "pos_months": pos, "neg_months": neg, "total_months": tot,
                        "pct_pos": round(pos / tot * 100, 1),
                        "effect_all": eff_all,
                    })

    detail_df  = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    cons_sym_df = pd.DataFrame(cons_sym_rows)

    # ─── 銘柄別 一貫性サマリ（FOCUS信号を強調） ──────────────────
    print(f"\n{'='*72}")
    print("銘柄別 一貫性サマリ（月別 effect_hi_lo の符号カウント）")
    print(f"{'='*72}")
    print(f"  {'symbol':<9} {'feature':<9} {'ret':<8} {'+mo':>4} {'/tot':>5} {'%pos':>6} {'effect_all':>11}")
    for sym in per_symbol:
        for fkey, _, _ in FEATURES:
            for rc in ["ret_8h", "ret_24h"]:
                r = cons_sym_df[(cons_sym_df["symbol"] == sym) &
                                (cons_sym_df["feature"] == fkey) &
                                (cons_sym_df["ret"] == rc)]
                if r.empty:
                    continue
                row = r.iloc[0]
                mark = "  ◀FOCUS" if (fkey, rc) in FOCUS else ""
                print(f"  {sym:<9} {fkey:<9} {rc:<8} "
                      f"{int(row['pos_months']):>4} {int(row['total_months']):>5} "
                      f"{row['pct_pos']:>5.0f}% {row['effect_all']:>+11.4f}{mark}")

    # ─── 全銘柄プール 一貫性サマリ ───────────────────────────────
    # プール = 全銘柄×全月の (symbol,month) effect を1サンプルとして符号集計
    print(f"\n{'='*72}")
    print("全銘柄プール 一貫性サマリ（symbol×month の effect 符号カウント）")
    print(f"{'='*72}")
    pool_rows = []
    print(f"  {'feature':<9} {'ret':<8} {'+sm':>5} {'/tot':>6} {'%pos':>6} {'mean_eff':>10}")
    for fkey, _, _ in FEATURES:
        for rc in ["ret_8h", "ret_24h"]:
            sub = detail_df[(detail_df["feature"] == fkey) & (detail_df["ret"] == rc)]
            if sub.empty:
                continue
            pos = int((sub["effect_hi_lo"] > 0).sum())
            neg = int((sub["effect_hi_lo"] < 0).sum())
            tot = len(sub)
            mean_eff = round(sub["effect_hi_lo"].mean(), 4)
            mark = "  ◀FOCUS" if (fkey, rc) in FOCUS else ""
            pool_rows.append({
                "feature": fkey, "ret": rc,
                "pos_symmonths": pos, "neg_symmonths": neg, "total_symmonths": tot,
                "pct_pos": round(pos / tot * 100, 1), "mean_effect": mean_eff,
            })
            print(f"  {fkey:<9} {rc:<8} {pos:>5} {tot:>6} "
                  f"{pos/tot*100:>5.0f}% {mean_eff:>+10.4f}{mark}")
    pool_df = pd.DataFrame(pool_rows)

    # ─── FOCUS 3信号の銘柄横断クロス表 ──────────────────────────
    print(f"\n{'='*72}")
    print("FOCUS信号 銘柄横断クロス（%pos: 月別effectが期待方向に出た割合）")
    print("  期待方向: lsr_all/lsr_top=DOWN(%posが低いほど良) / taker_24h=UP(%posが高いほど良)")
    print(f"{'='*72}")
    print(f"  {'signal':<18} " + " ".join(f"{s.replace('USDT',''):>7}" for s in per_symbol))
    for fkey, rc in FOCUS:
        cells = []
        for sym in per_symbol:
            r = cons_sym_df[(cons_sym_df["symbol"] == sym) &
                            (cons_sym_df["feature"] == fkey) &
                            (cons_sym_df["ret"] == rc)]
            cells.append(f"{r.iloc[0]['pct_pos']:>6.0f}%" if not r.empty else f"{'--':>7}")
        print(f"  {fkey+'/'+rc:<18} " + " ".join(cells))

    # ─── CSV 保存 ─────────────────────────────────────────────────
    if not summary_df.empty:
        summary_df.to_csv(OUT_DIR / "summary_by_symbol.csv", index=False)
    if not detail_df.empty:
        detail_df.to_csv(OUT_DIR / "detail_monthly.csv", index=False)
    if not cons_sym_df.empty:
        cons_sym_df.to_csv(OUT_DIR / "consistency_by_symbol.csv", index=False)
    if not pool_df.empty:
        pool_df.to_csv(OUT_DIR / "consistency_pool.csv", index=False)
    print(f"\n出力先: {OUT_DIR}/")
    print("  summary_by_symbol.csv      — 特徴量×銘柄×全期間 effect")
    print("  detail_monthly.csv         — 特徴量×銘柄×月別 effect")
    print("  consistency_by_symbol.csv  — 銘柄別 一貫性サマリ")
    print("  consistency_pool.csv       — 全銘柄プール 一貫性サマリ")
    print("=" * 72)
    print("\n判定の見方:")
    print("  lsr_all / lsr_top : 複数銘柄で %pos が低い(≤30%目安=DOWN方向)なら横展開成立")
    print("  taker_24h         : 複数銘柄で %pos が高い(≥70%目安=UP方向)なら横展開成立")
    print("  BTC以外でも同方向 → BTC固有でない / 50%前後に散る → BTCのたまたま")


if __name__ == "__main__":
    main()
