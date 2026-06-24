#!/usr/bin/env python3
"""
verify_funding_carry.py  【Funding Carry（Cash & Carry）過去検証】
==================================================================
目的:
  「現物買い + Perp売り」をただ保有し続けたら、Fundingだけで
  過去どれだけの収益/損失になったかを、Binance公開のfundingRate
  （手数料も方向予測も無し・足し算だけ）で正直に確認する。

  ★これは方向予測シグナルではない。市場ニュートラル構造収益の検証。
   勝率という概念はない。年率とマイナス期間（耐性）で評価する。
   閾値探索・条件追加は一切しない（足し算のみ＝過学習が原理的に入らない）。

ポジション前提:
  Cash & Carry = 現物 LONG + Perp SHORT（デルタ≒ゼロ）
  → Perp SHORT は Funding>0 のとき「受け取る」側。
    よって保有者のFunding収益 = +fundingRate（settlementごと）。
    Funding<0 の期間は逆に「支払う」側になる。

データ: data.binance.vision futures/um/monthly/fundingRate（無料・2020〜）
  ※ローカル実行（リモートはプロキシでdata.binance.visionがブロック）

対象銘柄（固定）: BTCUSDT / ETHUSDT / SOLUSDT / DOGEUSDT / BNBUSDT
期間: 2020-01〜2026-05（取得可能な月のみ・欠損月は自動SKIP・補完しない）

評価指標（事前固定）:
  - 累積Funding（グロス, %）
  - 年率換算（グロス, %/年）
  - 手数料 往復0.30% 控除後の純年率（全期間保有前提・1回だけ控除）
  - break-even日数（手数料0.30%を平均Fundingで回収するのに何日か）
  - Funding<0 の最長連続期間（日）と、その間の累積マイナス幅
  - カレッジ曲線（累積Funding）の最大ドローダウン
  - 月別・年別の安定性（プラス月の割合）

★これは検証のみ。取引・API接続・Bot化・本番接続は一切しない。
  main.py変更なし / deploy / merge なし / 有料データなし。

実行: python3 verify_funding_carry.py
依存: pandas numpy   （pip install pandas numpy）
出力: funding_carry_out/ に
  summary_by_symbol.csv  — 銘柄別 全指標
  monthly_funding.csv    — 銘柄×月別の平均Funding・件数
  yearly_funding.csv     — 銘柄×年別の累積・平均Funding
"""
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ─── 設定 ─────────────────────────────────────────────────────────
BASE_MONTHLY = "https://data.binance.vision/data/futures/um/monthly"
SSL_CTX = ssl.create_default_context()
HDRS    = {"User-Agent": "Mozilla/5.0"}
OUT_DIR = Path("funding_carry_out")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT"]

# 取得対象月（2020-01〜2026-05 全月。欠損は自動SKIP・補完しない）
_START = (2020, 1)
_END   = (2026, 5)
MONTHS = [
    (y, m)
    for y in range(_START[0], _END[0] + 1)
    for m in range(1, 13)
    if _START <= (y, m) <= _END
]

# 手数料（往復・保守的）: 現物 0.10%×2 + Perp 0.05%×2 = 0.30%（1サイクルに1回）
ROUND_TRIP_FEE_PCT = 0.30


# ─── ダウンロード ─────────────────────────────────────────────────
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


def fetch_funding_month(sym: str, y: int, m: int) -> pd.DataFrame | None:
    return _fetch_zip_csv(f"{BASE_MONTHLY}/fundingRate/{sym}/{sym}-fundingRate-{y}-{m:02d}.zip")


def _to_utc(series: pd.Series) -> pd.Series:
    raw = series.iloc[0]
    if isinstance(raw, str):
        return pd.to_datetime(series, utc=True)
    unit = "ms" if raw > 1e12 else "s"
    return pd.to_datetime(series, unit=unit, utc=True)


# ─── 1銘柄ぶんの fundingRate を全月ロード ─────────────────────────
def load_funding(sym: str) -> pd.DataFrame | None:
    frames = []
    for (y, m) in MONTHS:
        df = fetch_funding_month(sym, y, m)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    fund = pd.concat(frames, ignore_index=True)

    # 時刻列・レート列を robust に検出（probe と同じ方式）
    ts_col = next((c for c in fund.columns if "time" in c.lower()), None)
    fr_col = next((c for c in fund.columns if "rate" in c.lower()), None)
    if ts_col is None or fr_col is None:
        print(f"  [{sym}] 列検出失敗: {list(fund.columns)}")
        return None

    fund["_utc"]  = _to_utc(fund[ts_col])
    fund["rate"]  = pd.to_numeric(fund[fr_col], errors="coerce")
    fund = fund.dropna(subset=["_utc", "rate"])           # 欠損は補完せず除外
    fund = fund.sort_values("_utc").drop_duplicates("_utc").reset_index(drop=True)
    fund["year"]  = fund["_utc"].dt.year
    fund["month"] = fund["_utc"].dt.month
    return fund


# ─── 指標計算 ─────────────────────────────────────────────────────
def carry_metrics(sym: str, fund: pd.DataFrame) -> dict:
    rate = fund["rate"].to_numpy()                  # decimal（0.0001 = 0.01%）
    rate_pct = rate * 100.0                          # %に変換
    n = len(rate_pct)

    # 期間（年）
    span_days = (fund["_utc"].iloc[-1] - fund["_utc"].iloc[0]).total_seconds() / 86400.0
    years = span_days / 365.25 if span_days > 0 else np.nan

    # 累積・年率（グロス）
    cum_gross = float(np.nansum(rate_pct))           # %
    ann_gross = cum_gross / years if years and years > 0 else np.nan

    # 手数料控除（全期間保有前提・往復1回だけ）
    cum_net = cum_gross - ROUND_TRIP_FEE_PCT
    ann_net = cum_net / years if years and years > 0 else np.nan

    # 平均Funding（settlementあたり・日あたり）
    avg_per_settle = float(np.nanmean(rate_pct))
    settles_per_day = n / span_days if span_days > 0 else np.nan
    avg_per_day = avg_per_settle * settles_per_day if np.isfinite(settles_per_day) else np.nan
    breakeven_days = (ROUND_TRIP_FEE_PCT / avg_per_day) if avg_per_day and avg_per_day > 0 else np.nan

    # プラス settlement の割合
    pct_pos = float((rate_pct > 0).mean() * 100)

    # Funding<0 の最長連続期間 と その間の累積マイナス幅
    longest_neg_streak = 0
    cur_streak = 0
    worst_neg_sum = 0.0
    cur_sum = 0.0
    for r in rate_pct:
        if r < 0:
            cur_streak += 1
            cur_sum += r
            longest_neg_streak = max(longest_neg_streak, cur_streak)
            worst_neg_sum = min(worst_neg_sum, cur_sum)
        else:
            cur_streak = 0
            cur_sum = 0.0
    neg_streak_days = longest_neg_streak / settles_per_day if settles_per_day and settles_per_day > 0 else np.nan

    # 累積Funding曲線（carry equity）の最大ドローダウン
    cum_curve = np.nancumsum(rate_pct)
    running_max = np.maximum.accumulate(cum_curve)
    drawdown = running_max - cum_curve
    max_dd = float(np.nanmax(drawdown)) if len(drawdown) else np.nan

    return {
        "symbol": sym,
        "settlements": n,
        "years": round(years, 2) if np.isfinite(years) else None,
        "settles_per_day": round(settles_per_day, 2) if np.isfinite(settles_per_day) else None,
        "cum_gross_pct": round(cum_gross, 3),
        "ann_gross_pct": round(ann_gross, 3) if np.isfinite(ann_gross) else None,
        "ann_net_pct(fee0.30once)": round(ann_net, 3) if np.isfinite(ann_net) else None,
        "avg_per_day_pct": round(avg_per_day, 5) if np.isfinite(avg_per_day) else None,
        "breakeven_days(fee0.30)": round(breakeven_days, 1) if np.isfinite(breakeven_days) else None,
        "pct_positive_settles": round(pct_pos, 1),
        "longest_neg_streak_days": round(neg_streak_days, 1) if np.isfinite(neg_streak_days) else None,
        "worst_neg_streak_sum_pct": round(worst_neg_sum, 3),
        "carry_curve_maxDD_pct": round(max_dd, 3) if np.isfinite(max_dd) else None,
        "first": str(fund["_utc"].iloc[0].date()),
        "last":  str(fund["_utc"].iloc[-1].date()),
    }


# ─── メイン ──────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 78)
    print("Funding Carry（Cash & Carry）過去検証  ※方向予測ではない・足し算のみ")
    print(f"銘柄: {SYMBOLS}")
    print(f"期間: {MONTHS[0][0]}-{MONTHS[0][1]:02d}〜{MONTHS[-1][0]}-{MONTHS[-1][1]:02d}  手数料: 往復{ROUND_TRIP_FEE_PCT}%")
    print("=" * 78)

    summary_rows = []
    monthly_rows = []
    yearly_rows  = []

    for sym in SYMBOLS:
        print(f"\n■ {sym} fundingRate ロード中...")
        fund = load_funding(sym)
        if fund is None or fund.empty:
            print(f"  [{sym}] データなし。SKIP。")
            continue
        print(f"  settlements={len(fund)}  {fund['_utc'].iloc[0].date()}〜{fund['_utc'].iloc[-1].date()}")

        m = carry_metrics(sym, fund)
        summary_rows.append(m)

        # 月別（プラス月の割合を見るため）
        g = fund.groupby(["year", "month"])["rate"].agg(["mean", "count"]).reset_index()
        for _, row in g.iterrows():
            monthly_rows.append({
                "symbol": sym, "year": int(row["year"]), "month": int(row["month"]),
                "mean_funding_pct": round(row["mean"] * 100, 5),
                "n_settles": int(row["count"]),
            })

        # 年別
        gy = fund.groupby("year")["rate"].agg(["sum", "mean", "count"]).reset_index()
        for _, row in gy.iterrows():
            yearly_rows.append({
                "symbol": sym, "year": int(row["year"]),
                "cum_funding_pct": round(row["sum"] * 100, 3),
                "mean_funding_pct": round(row["mean"] * 100, 5),
                "n_settles": int(row["count"]),
            })

    if not summary_rows:
        print("\n[ERROR] 全銘柄でデータ取得失敗。")
        return

    summary_df = pd.DataFrame(summary_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    yearly_df  = pd.DataFrame(yearly_rows)

    # 月別プラス割合を summary に付与
    pos_share = (
        monthly_df.assign(pos=lambda d: d["mean_funding_pct"] > 0)
                  .groupby("symbol")["pos"].mean()
                  .mul(100).round(1)
    )
    summary_df["pct_positive_months"] = summary_df["symbol"].map(pos_share)

    # ─── 出力 ────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("銘柄別サマリ（Cash & Carry をただ保有し続けた場合）")
    print(f"{'='*78}")
    show_cols = [
        "symbol", "years", "settles_per_day",
        "ann_gross_pct", "ann_net_pct(fee0.30once)",
        "pct_positive_settles", "pct_positive_months",
        "longest_neg_streak_days", "worst_neg_streak_sum_pct", "carry_curve_maxDD_pct",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary_df[show_cols].to_string(index=False))

    print(f"\n{'='*78}")
    print("年別 累積Funding（%）— 安定性確認")
    print(f"{'='*78}")
    if not yearly_df.empty:
        pivot = yearly_df.pivot_table(index="year", columns="symbol",
                                      values="cum_funding_pct", aggfunc="first")
        print(pivot.round(2).to_string())

    summary_df.to_csv(OUT_DIR / "summary_by_symbol.csv", index=False)
    monthly_df.to_csv(OUT_DIR / "monthly_funding.csv", index=False)
    yearly_df.to_csv(OUT_DIR / "yearly_funding.csv", index=False)
    print(f"\n出力先: {OUT_DIR}/")
    print("  summary_by_symbol.csv  — 銘柄別 全指標")
    print("  monthly_funding.csv    — 銘柄×月別 平均Funding")
    print("  yearly_funding.csv     — 銘柄×年別 累積Funding")
    print("=" * 78)
    print("\n判定の見方（事前固定・閾値探索なし）:")
    print("  ann_net_pct        : 手数料控除後の年率。プラスでなければ話にならない")
    print("  longest_neg_streak : Fundingマイナスが何日続いたか（耐性の核心）")
    print("  worst_neg_streak_sum: その間にどれだけ払ったか（最悪期の痛み）")
    print("  carry_curve_maxDD  : 累積Funding曲線の最大落ち込み")
    print("  pct_positive_months: プラス月の割合（高いほど安定）")
    print("  ※スリッページ・清算・maker/taker差・取引所リスクは含まない（無料データの限界）")


if __name__ == "__main__":
    main()
