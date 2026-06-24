#!/usr/bin/env python3
"""
verify_binance_metrics_oos.py  【Binance metrics 小規模OOS検証】
================================================================
目的:
  Binance data.vision futures/um の OI / LSR / Funding / taker比率が、
  BTCUSDTの未来8h/24hリターンを3分位で安定して分離するか
  複数年・複数月のOOSで検証する。

検証設計（事前固定・後から条件追加なし）:
  特徴量 5種 (as-of: 完了データのみ):
    lsr_all  = count_long_short_ratio      （全アカウントLSR）
    lsr_top  = sum_toptrader_long_short_ratio  （大口ポジションLSR）
    oi_chg   = sum_open_interest の1h前比変化率
    taker    = sum_taker_long_short_vol_ratio  （taker買い圧）
    funding  = fundingRate（直近settlement値・as-of）
  目的変数: 8h先・24h先のBTC価格リターン
  分位: 月内3分位（low/mid/high）固定
  期間: 2021〜2026の代表月（年2回・1月と7月）= 最大11ヶ月

評価（手動運用基準は適用しない・存在確認のみ）:
  - high − low effect の符号が複数月で一貫しているか
  - 年別・月別で符号が崩れないか
  - 全期間のavg/medの方向感

as-of制約: 特徴量は完了時刻のデータのみ。未来情報不使用。
欠損制約: np.nan のまま（0補完・平均補完 絶対禁止）。

実行: python3 verify_binance_metrics_oos.py
依存: pandas numpy   （pip install pandas numpy）
出力: binance_oos_out/ に summary.csv / detail_monthly.csv / raw_merged.csv
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
SYM     = "BTCUSDT"
SSL_CTX = ssl.create_default_context()
HDRS    = {"User-Agent": "Mozilla/5.0"}
OUT_DIR = Path("binance_oos_out")

# 代表月: 年2回(1月・7月)、2021〜2026
REP_MONTHS = [
    (2021, 1), (2021, 7),
    (2022, 1), (2022, 7),
    (2023, 1), (2023, 7),
    (2024, 1), (2024, 7),
    (2025, 1), (2025, 7),
    (2026, 1),
]

# 特徴量定義（事前固定）
FEATURES = [
    ("lsr_all", "count_long_short_ratio",          "高LSR → 逆張りでDOWN期待"),
    ("lsr_top", "sum_toptrader_long_short_ratio",   "大口高LSR → 逆張りでDOWN期待"),
    ("oi_chg",  "oi_chg_rate",                     "OI急増 → 方向は未知・確認"),
    ("taker",   "sum_taker_long_short_vol_ratio",   "taker買い圧高 → UP/DOWN未知・確認"),
    ("funding", "fundingRate",                      "高Funding → 逆張りでDOWN期待"),
]

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


def fetch_metrics_day(d: date) -> pd.DataFrame | None:
    return _fetch_zip_csv(f"{BASE_DAILY}/metrics/{SYM}/{SYM}-metrics-{d}.zip")


def fetch_klines_day(d: date) -> pd.DataFrame | None:
    df = _fetch_zip_csv(f"{BASE_DAILY}/klines/{SYM}/1h/{SYM}-1h-{d}.zip")
    if df is not None:
        df.columns = KLINES_COLS
    return df


def fetch_funding_month(y: int, m: int) -> pd.DataFrame | None:
    return _fetch_zip_csv(f"{BASE_MONTHLY}/fundingRate/{SYM}/{SYM}-fundingRate-{y}-{m:02d}.zip")


def _to_utc_ns(series: pd.Series) -> pd.Series:
    raw = series.iloc[0]
    if isinstance(raw, str):
        return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")
    unit = "ms" if raw > 1e12 else "s"
    return pd.to_datetime(series, unit=unit, utc=True).astype("datetime64[ns, UTC]")


# ─── 月単位データ構築 ─────────────────────────────────────────────
def load_one_month(y: int, m: int) -> pd.DataFrame | None:
    num_days = monthrange(y, m)[1]
    month_days = [date(y, m, d + 1) for d in range(num_days)]
    nm = m % 12 + 1
    ny = y + (1 if m == 12 else 0)
    extra_day = date(ny, nm, 1)

    # 1. metrics (当月のみ)
    m_frames = [fetch_metrics_day(d) for d in month_days]
    m_frames = [f for f in m_frames if f is not None]
    if not m_frames:
        return None
    metrics = pd.concat(m_frames, ignore_index=True)
    print(f"  [{y}-{m:02d}] metrics rows={len(metrics)}", end="")

    # 2. klines 1h (当月 + 翌月1日: forward return 計算用)
    k_frames = [fetch_klines_day(d) for d in month_days + [extra_day]]
    k_frames = [f for f in k_frames if f is not None]
    if not k_frames:
        print(" | klines NG")
        return None
    klines = pd.concat(k_frames, ignore_index=True)
    klines["_utc"] = _to_utc_ns(klines["open_time"])
    klines["close"] = klines["close"].astype(float)
    klines = klines.sort_values("_utc").drop_duplicates("_utc").reset_index(drop=True)
    print(f" | klines rows={len(klines)}", end="")

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
    fund = fetch_funding_month(y, m)
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
                print(f" | funding rows={len(fund)}", end="")
    if "fundingRate" not in merged.columns:
        merged["fundingRate"] = np.nan

    merged["year"]  = y
    merged["month"] = m
    print()
    return merged


# ─── 3分位分析 ──────────────────────────────────────────────────
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


# ─── メイン ──────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print(f"Binance metrics OOS検証  対象: {SYM}  代表月: {len(REP_MONTHS)}ヶ月")
    print(f"特徴量: {[f[0] for f in FEATURES]}")
    print("=" * 70)

    all_frames = []
    for y, m in REP_MONTHS:
        print(f"\n■ {y}-{m:02d} 取得中...")
        frame = load_one_month(y, m)
        if frame is not None:
            all_frames.append(frame)
        else:
            print(f"  [{y}-{m:02d}] SKIP（取得失敗）")

    if not all_frames:
        print("[ERROR] データが1件も取得できませんでした。")
        return

    all_df = pd.concat(all_frames, ignore_index=True)
    print(f"\n合計 {len(all_df)} 行 / {len(all_frames)} ヶ月\n")

    detail_rows  = []
    summary_rows = []

    for fkey, fcol, fdesc in FEATURES:
        if fcol not in all_df.columns:
            print(f"[SKIP] {fkey}: 列 '{fcol}' なし")
            continue

        print(f"\n{'='*70}")
        print(f"特徴量: {fkey}  列: {fcol}")
        print(f"仮説: {fdesc}")

        for rc in ["ret_8h", "ret_24h"]:
            # 全期間
            stats = tercile_stats(all_df, fcol, rc)
            if stats is None:
                continue
            hi  = stats.loc[stats["quantile"] == "high", "avg"].values[0]
            lo  = stats.loc[stats["quantile"] == "low",  "avg"].values[0]
            eff = hi - lo
            print(f"\n  [{rc}] 全期間 (n={stats['n'].sum()})")
            print("  " + stats.to_string(index=False))
            print(f"  high-low effect: {eff:+.4f}%")
            summary_rows.append({
                "feature": fkey, "period": "ALL", "ret": rc,
                "avg_low":  stats.loc[stats["quantile"]=="low",  "avg"].values[0],
                "avg_mid":  stats.loc[stats["quantile"]=="mid",  "avg"].values[0],
                "avg_high": stats.loc[stats["quantile"]=="high", "avg"].values[0],
                "med_low":  stats.loc[stats["quantile"]=="low",  "med"].values[0],
                "med_mid":  stats.loc[stats["quantile"]=="mid",  "med"].values[0],
                "med_high": stats.loc[stats["quantile"]=="high", "med"].values[0],
                "effect_hi_lo": round(eff, 4),
                "n_total": int(stats["n"].sum()),
            })

        # 年別
        for yr in sorted(all_df["year"].unique()):
            sub_y = all_df[all_df["year"] == yr]
            for rc in ["ret_8h", "ret_24h"]:
                stats = tercile_stats(sub_y, fcol, rc)
                if stats is None:
                    continue
                hi  = stats.loc[stats["quantile"]=="high","avg"].values[0]
                lo  = stats.loc[stats["quantile"]=="low", "avg"].values[0]
                detail_rows.append({
                    "feature": fkey, "period": str(yr), "ret": rc,
                    "avg_low":  stats.loc[stats["quantile"]=="low",  "avg"].values[0],
                    "avg_high": stats.loc[stats["quantile"]=="high", "avg"].values[0],
                    "effect_hi_lo": round(hi - lo, 4),
                    "n": int(stats["n"].sum()),
                })

        # 月別
        for (yr, mo) in sorted(all_df.groupby(["year", "month"]).groups.keys()):
            sub_m = all_df[(all_df["year"] == yr) & (all_df["month"] == mo)]
            for rc in ["ret_8h", "ret_24h"]:
                stats = tercile_stats(sub_m, fcol, rc)
                if stats is None:
                    continue
                hi  = stats.loc[stats["quantile"]=="high","avg"].values[0]
                lo  = stats.loc[stats["quantile"]=="low", "avg"].values[0]
                detail_rows.append({
                    "feature": fkey, "period": f"{yr}-{mo:02d}", "ret": rc,
                    "avg_low":  stats.loc[stats["quantile"]=="low",  "avg"].values[0],
                    "avg_high": stats.loc[stats["quantile"]=="high", "avg"].values[0],
                    "effect_hi_lo": round(hi - lo, 4),
                    "n": int(stats["n"].sum()),
                })

    # ─── 一貫性サマリ ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("一貫性確認（月別 effect_hi_lo の符号カウント）")
    print(f"{'='*70}")
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        mon_detail = detail_df[detail_df["period"].str.match(r"^\d{4}-\d{2}$")]
        print(f"  {'feature':<10} {'ret':<8} {'+ months':>8} {'/ total':>7} {'% pos':>7} {'| - months':>10}")
        for fkey, _, _ in FEATURES:
            for rc in ["ret_8h", "ret_24h"]:
                sub = mon_detail[(mon_detail["feature"] == fkey) & (mon_detail["ret"] == rc)]
                if sub.empty:
                    continue
                pos   = (sub["effect_hi_lo"] > 0).sum()
                neg   = (sub["effect_hi_lo"] < 0).sum()
                total = len(sub)
                pct   = pos / total * 100 if total else 0
                print(f"  {fkey:<10} {rc:<8} {pos:>8} {total:>7} {pct:>6.0f}% {neg:>10}")

    # ─── CSV 保存 ─────────────────────────────────────────────────
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(OUT_DIR / "detail_monthly.csv", index=False)
    all_df.to_csv(OUT_DIR / "raw_merged.csv", index=False)
    print(f"\n出力先: {OUT_DIR}/")
    print("  summary.csv        — 特徴量×全期間・ret の集計")
    print("  detail_monthly.csv — 特徴量×月別・年別の effect")
    print("  raw_merged.csv     — 結合後の全行（確認用）")
    print("=" * 70)


if __name__ == "__main__":
    main()
