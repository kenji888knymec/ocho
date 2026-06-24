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
   閾値探索・条件追加はしない。ただし「過学習が無い」とは言い切らない。
   残るバイアス（必ず意識する）:
     - 取引所バイアス（Binanceのみ）
     - 銘柄選択バイアス（BTC/ETH/SOL/DOGE/BNBを選んだこと）
     - サバイバーシップ（上場後生き残った銘柄だけ）
     - 手数料仮定の甘さ/厳しさ（0.30%/0.60%の2段で見る）
     - スリッページ・証拠金管理・清算は未モデル化
   → 閾値探索型の過学習は入りにくいが、設計バイアスは残る、が正確。

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
  roll12_monthly.csv     — BTC/ETH/DOGE の Rolling 12ヶ月年率グロス
"""
from __future__ import annotations
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

# 手数料（往復・1サイクルに1回）。2段階で見る:
#   0.30% = 現物0.10%×2 + Perp0.05%×2（maker寄り・保守的下限）
#   0.60% = よりtaker寄り・スプレッド/スリッページ込みの厳しめ想定
# 結果が0.30%でだけプラスなら、0.60%で消える可能性を疑う。
ROUND_TRIP_FEE_PCT  = 0.30
ROUND_TRIP_FEE_PCT2 = 0.60

# 深掘り対象銘柄（BTC/ETH中心。DOGEは参考のみ。SOL/BNBは深掘り対象外）
DEEP_SYMS = ["BTCUSDT", "ETHUSDT"]
REF_SYMS  = ["DOGEUSDT"]

# 期間別サマリの開始年月（label, (year, month)）
SUBPERIODS = [
    ("2022-01+", (2022, 1)),
    ("2024-01+", (2024, 1)),
    ("2025-01+", (2025, 1)),
]

# 資本効率の想定倍率（想定元本 × N = 実際に必要な証拠金）
# 現物全額 + Perp証拠金 + バッファ ≒ 想定元本の1.2〜2.0倍
CAPITAL_MULTIPLIERS = [1.2, 1.5, 2.0]

# 現在地スナップショット: trailing window の日数ラベルと期間
WINDOWS = [("30日", 30), ("60日", 60), ("90日", 90), ("180日", 180), ("12ヶ月", 365)]

# ハードルレート（net年率の最低基準・事前固定・閾値探索しない）
# 根拠: 無リスク金利(~4-5%) + 取引所/清算/手間リスクプレミアム(~3-4%) = 8%
HURDLE_PCT = 8.0


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

    # 手数料控除（全期間保有前提・往復1回だけ）2段階
    cum_net  = cum_gross - ROUND_TRIP_FEE_PCT
    ann_net  = cum_net / years if years and years > 0 else np.nan
    cum_net2 = cum_gross - ROUND_TRIP_FEE_PCT2
    ann_net2 = cum_net2 / years if years and years > 0 else np.nan

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
        "ann_net_pct(fee0.30)": round(ann_net, 3) if np.isfinite(ann_net) else None,
        "ann_net_pct(fee0.60)": round(ann_net2, 3) if np.isfinite(ann_net2) else None,
        "avg_per_day_pct": round(avg_per_day, 5) if np.isfinite(avg_per_day) else None,
        "breakeven_days(fee0.30)": round(breakeven_days, 1) if np.isfinite(breakeven_days) else None,
        "pct_positive_settles": round(pct_pos, 1),
        "longest_neg_streak_days": round(neg_streak_days, 1) if np.isfinite(neg_streak_days) else None,
        "worst_neg_streak_sum_pct": round(worst_neg_sum, 3),
        "carry_curve_maxDD_pct": round(max_dd, 3) if np.isfinite(max_dd) else None,
        "first": str(fund["_utc"].iloc[0].date()),
        "last":  str(fund["_utc"].iloc[-1].date()),
    }


def carry_metrics_subperiod(sym: str, fund: pd.DataFrame, start_ym: tuple) -> dict | None:
    """特定年月以降のサブ期間のみで carry_metrics を計算する。データが少ない場合は None。"""
    year0, month0 = start_ym
    sub = fund[
        (fund["year"] > year0) | ((fund["year"] == year0) & (fund["month"] >= month0))
    ].copy().reset_index(drop=True)
    if len(sub) < 20:
        return None
    return carry_metrics(sym, sub)


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
    fund_by_sym  = {}   # 深掘り分析用に fund DataFrame を保持

    for sym in SYMBOLS:
        print(f"\n■ {sym} fundingRate ロード中...")
        fund = load_funding(sym)
        if fund is None or fund.empty:
            print(f"  [{sym}] データなし。SKIP。")
            continue
        fund_by_sym[sym] = fund
        print(f"  settlements={len(fund)}  {fund['_utc'].iloc[0].date()}〜{fund['_utc'].iloc[-1].date()}")

        m = carry_metrics(sym, fund)
        summary_rows.append(m)

        # 月別（プラス月の割合・Rolling 12ヶ月計算のため sum も追加）
        g = fund.groupby(["year", "month"])["rate"].agg(["sum", "mean", "count"]).reset_index()
        for _, row in g.iterrows():
            monthly_rows.append({
                "symbol": sym, "year": int(row["year"]), "month": int(row["month"]),
                "sum_funding_pct":  round(row["sum"]  * 100, 5),
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
        "ann_gross_pct", "ann_net_pct(fee0.30)", "ann_net_pct(fee0.60)",
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

    # ─── BTC/ETH 深掘り分析 ──────────────────────────────────────
    deep_syms_available = [s for s in DEEP_SYMS + REF_SYMS if s in fund_by_sym]
    if deep_syms_available:
        print(f"\n{'='*78}")
        print("【深掘り分析】BTC/ETH 中心（DOGE=参考。SOL/BNBは対象外）")
        print(f"{'='*78}")

        # ── 1. 期間別サマリ ──────────────────────────────────────
        print("\n▶ 1. 期間別サマリ（全期間 / 2022-01以降 / 2024-01以降 / 2025-01以降）")
        for sym in deep_syms_available:
            fund = fund_by_sym[sym]
            base = summary_df[summary_df["symbol"] == sym].iloc[0]
            tag  = "深掘り" if sym in DEEP_SYMS else "参考"
            print(f"\n  {sym}（{tag}）")
            sp_rows = []

            # 全期間行
            ps_val = pos_share.get(sym)
            pos_m_full = float(ps_val) if ps_val is not None and pd.notna(ps_val) else None
            sp_rows.append({
                "期間":             "全期間",
                "years":            base["years"],
                "ann_gross_%":      base["ann_gross_pct"],
                "ann_net(0.30)%":   base["ann_net_pct(fee0.30)"],
                "ann_net(0.60)%":   base["ann_net_pct(fee0.60)"],
                "pos_months%":      pos_m_full,
                "longest_neg_d":    base["longest_neg_streak_days"],
                "worst_streak%":    base["worst_neg_streak_sum_pct"],
                "maxDD%":           base["carry_curve_maxDD_pct"],
            })

            # サブ期間行
            for label, start_ym in SUBPERIODS:
                ms = carry_metrics_subperiod(sym, fund, start_ym)
                sub_m = monthly_df[
                    (monthly_df["symbol"] == sym) &
                    ((monthly_df["year"] > start_ym[0]) |
                     ((monthly_df["year"] == start_ym[0]) &
                      (monthly_df["month"] >= start_ym[1])))
                ]
                pos_m = round(float((sub_m["sum_funding_pct"] > 0).mean() * 100), 1) \
                        if len(sub_m) > 0 else None
                if ms is None:
                    sp_rows.append({
                        "期間": label, "years": None, "ann_gross_%": None,
                        "ann_net(0.30)%": None, "ann_net(0.60)%": None,
                        "pos_months%": pos_m, "longest_neg_d": None,
                        "worst_streak%": None, "maxDD%": None,
                    })
                else:
                    sp_rows.append({
                        "期間":           label,
                        "years":          ms["years"],
                        "ann_gross_%":    ms["ann_gross_pct"],
                        "ann_net(0.30)%": ms["ann_net_pct(fee0.30)"],
                        "ann_net(0.60)%": ms["ann_net_pct(fee0.60)"],
                        "pos_months%":    pos_m,
                        "longest_neg_d":  ms["longest_neg_streak_days"],
                        "worst_streak%":  ms["worst_neg_streak_sum_pct"],
                        "maxDD%":         ms["carry_curve_maxDD_pct"],
                    })

            sp_df = pd.DataFrame(sp_rows)
            with pd.option_context("display.width", 200, "display.max_columns", None):
                print(sp_df.to_string(index=False))

        # ── 2. 最悪月・連続マイナス月 ────────────────────────────
        print(f"\n{'─'*60}")
        print("▶ 2. 最悪月・連続マイナス月（月別累積Funding基準）")
        for sym in deep_syms_available:
            mdf = monthly_df[monthly_df["symbol"] == sym].sort_values(["year", "month"]).copy()
            if mdf.empty:
                continue
            worst_row = mdf.loc[mdf["sum_funding_pct"].idxmin()]
            # 連続マイナス月の最長ストリーク計算
            neg_cur, neg_max = 0, 0
            neg_start_cur, neg_worst_start = None, None
            for _, row in mdf.iterrows():
                if row["sum_funding_pct"] < 0:
                    if neg_cur == 0:
                        neg_start_cur = (int(row["year"]), int(row["month"]))
                    neg_cur += 1
                    if neg_cur > neg_max:
                        neg_max = neg_cur
                        neg_worst_start = neg_start_cur
                else:
                    neg_cur = 0
            tag = "深掘り" if sym in DEEP_SYMS else "参考"
            print(f"\n  {sym}（{tag}）")
            print(f"    最悪月: {int(worst_row['year'])}-{int(worst_row['month']):02d}  "
                  f"月累積Funding={worst_row['sum_funding_pct']:.3f}%")
            wstart = (f"({neg_worst_start[0]}-{neg_worst_start[1]:02d}〜)"
                      if neg_worst_start else "")
            print(f"    連続マイナス月最長: {neg_max}ヶ月 {wstart}")

        # ── 3. Rolling 12ヶ月 年率グロス ──────────────────────────
        print(f"\n{'─'*60}")
        print("▶ 3. Rolling 12ヶ月 年率グロス（%）— 手数料未控除")
        print("   （直近12ヶ月の累積Fundingを年率相当とみなす）")
        roll12_out = []
        for sym in deep_syms_available:
            mdf = monthly_df[monthly_df["symbol"] == sym].sort_values(["year", "month"]).copy()
            mdf["roll12"] = mdf["sum_funding_pct"].rolling(12).sum()
            valid = mdf.dropna(subset=["roll12"])
            for _, row in valid.iterrows():
                roll12_out.append({
                    "symbol":              sym,
                    "year_month":          f"{int(row['year'])}-{int(row['month']):02d}",
                    "roll12_gross_ann_%":  round(float(row["roll12"]), 3),
                })
            r = valid["roll12"]
            tag = "深掘り" if sym in DEEP_SYMS else "参考"
            print(f"\n  {sym}（{tag}）: "
                  f"min={r.min():.2f}%  max={r.max():.2f}%  "
                  f"median={r.median():.2f}%  "
                  f"直近12ヶ月={r.iloc[-1]:.2f}%  "
                  f"マイナス年率={int((r < 0).sum())}ヶ月/{len(r)}ヶ月中")
        if roll12_out:
            roll12_df_out = pd.DataFrame(roll12_out)
            roll12_df_out.to_csv(OUT_DIR / "roll12_monthly.csv", index=False)
            print(f"\n  (詳細: {OUT_DIR}/roll12_monthly.csv)")

        # ── 4. 資金効率 ROE ──────────────────────────────────────
        print(f"\n{'─'*60}")
        print("▶ 4. 資金効率 ROE（実必要資金 = 想定元本 × 倍率）")
        print("   ×1.2: 効率重視（レバ活用）  ×1.5: 標準  ×2.0: 低レバ・余裕重視")
        print("   ROE = 純年率Funding収益 ÷ 実必要資本倍率")
        roe_rows = []
        for sym in deep_syms_available:
            base = summary_df[summary_df["symbol"] == sym].iloc[0]
            ann_030 = base["ann_net_pct(fee0.30)"]
            ann_060 = base["ann_net_pct(fee0.60)"]
            tag = "深掘り" if sym in DEEP_SYMS else "参考"
            for mult in CAPITAL_MULTIPLIERS:
                roe_030 = round(ann_030 / mult, 2) if ann_030 is not None else None
                roe_060 = round(ann_060 / mult, 2) if ann_060 is not None else None
                roe_rows.append({
                    "symbol":           sym,
                    "tag":              tag,
                    "資本倍率":          f"×{mult:.1f}",
                    "ann_net(fee0.30)%": ann_030,
                    "ann_net(fee0.60)%": ann_060,
                    "ROE(fee0.30)%":    roe_030,
                    "ROE(fee0.60)%":    roe_060,
                })
        if roe_rows:
            roe_df_out = pd.DataFrame(roe_rows)
            with pd.option_context("display.width", 200, "display.max_columns", None):
                print(roe_df_out.to_string(index=False))

        # ── 5. 現在地スナップショット（Trailing Window） ─────────────────
        print(f"\n{'─'*60}")
        print("▶ 5. 現在地スナップショット（Trailing Window）  BTC/ETHのみ")
        print("   直近N日の累積Fundingを年率換算した「今この瞬間の利回り」")
        print(f"   ハードル: net≥{HURDLE_PCT:.0f}%（清算/取引所/手間リスクを背負う最低ライン・固定）")
        snapshot_rows = []
        for sym in [s for s in DEEP_SYMS if s in fund_by_sym]:
            fund_s   = fund_by_sym[sym]
            last_utc = fund_s["_utc"].iloc[-1]
            print(f"\n  {sym}  (データ末尾: {last_utc.date()})")
            s_rows = []
            for label, days in WINDOWS:
                cutoff = last_utc - pd.Timedelta(days=days)
                sub = fund_s[fund_s["_utc"] >= cutoff]
                if len(sub) < 2:
                    continue
                actual_d = (sub["_utc"].iloc[-1] - sub["_utc"].iloc[0]).total_seconds() / 86400
                if actual_d < 1:
                    continue
                gross_ann = float(sub["rate"].sum()) * 100 * (365.0 / actual_d)
                net_030   = gross_ann - ROUND_TRIP_FEE_PCT
                net_060   = gross_ann - ROUND_TRIP_FEE_PCT2
                row = {
                    "window":      label,
                    "actual_days": round(actual_d, 1),
                    "gross_ann_%": round(gross_ann, 2),
                    "net_030_%":   round(net_030,   2),
                    "net_060_%":   round(net_060,   2),
                }
                for mult in CAPITAL_MULTIPLIERS:
                    row[f"ROE×{mult:.1f}(030)%"] = round(net_030 / mult, 2) if np.isfinite(net_030) else None
                    row[f"ROE×{mult:.1f}(060)%"] = round(net_060 / mult, 2) if np.isfinite(net_060) else None
                row["pass_030(≥8%)"] = "YES" if (np.isfinite(net_030) and net_030 >= HURDLE_PCT) else "NO"
                row["pass_060(≥8%)"] = "YES" if (np.isfinite(net_060) and net_060 >= HURDLE_PCT) else "NO"
                s_rows.append(row)
                snapshot_rows.append({"symbol": sym, **row})
            if s_rows:
                with pd.option_context("display.width", 200, "display.max_columns", None):
                    print(pd.DataFrame(s_rows).to_string(index=False))
        if snapshot_rows:
            snap_df_out = pd.DataFrame(snapshot_rows)
            snap_df_out.to_csv(OUT_DIR / "current_yield_snapshot.csv", index=False)
            print(f"\n  (詳細: {OUT_DIR}/current_yield_snapshot.csv)")
        print(f"  YES = 検討開始ライン（即運用GOではない。取引所/税務/清算/実手数料を調べる段階）")
        print(f"  NO  = 待機（米国債等の代替に見劣り。利回りが太るまで張らない）")

    # ─── CSV 保存 ──────────────────────────────────────────────────
    summary_df.to_csv(OUT_DIR / "summary_by_symbol.csv", index=False)
    monthly_df.to_csv(OUT_DIR / "monthly_funding.csv", index=False)
    yearly_df.to_csv(OUT_DIR / "yearly_funding.csv", index=False)
    print(f"\n出力先: {OUT_DIR}/")
    print("  summary_by_symbol.csv  — 銘柄別 全指標")
    print("  monthly_funding.csv    — 銘柄×月別 平均・合計Funding")
    print("  yearly_funding.csv     — 銘柄×年別 累積Funding")
    print("  roll12_monthly.csv     — BTC/ETH/DOGE Rolling 12ヶ月年率グロス")
    print("  current_yield_snapshot.csv — BTC/ETH Trailing窓別 現在地スナップショット")
    print("=" * 78)
    print("\n判定の見方（事前固定・閾値探索なし・ただし設計バイアスは残る）:")
    print("  ann_net(fee0.30)   : 控除後年率（下限コスト）。プラスでなければ話にならない")
    print("  ann_net(fee0.60)   : 厳しめコスト。0.30%でだけプラスなら実運用で消える疑い")
    print("  longest_neg_streak : Fundingマイナスが何日続いたか（耐性の核心）")
    print("  worst_neg_streak_sum: その間にどれだけ払ったか（最悪期の痛み）")
    print("  carry_curve_maxDD  : 累積Funding曲線の最大落ち込み")
    print("  pct_positive_months: プラス月の割合（高いほど安定）")
    print("\n  GO候補の目安（厳しめに見る）:")
    print("    - BTC/ETH で net年率(0.60%控除後) が明確にプラス")
    print("    - 2022年型の弱気相場でも完全崩壊していない")
    print("    - マイナスFunding連続期間・worst streak が資金的/心理的に耐えられる")
    print("  STOPの目安:")
    print("    - BTCだけ / DOGE・BNBだけ妙に高い / net年率2〜3%程度 / 0.60%で消える")
    print("    - マイナスFundingが数ヶ月続く / 最大DDが大きい")
    print("\n  残るバイアス: Binanceのみ / 銘柄選択 / サバイバーシップ /")
    print("    スリッページ・清算・証拠金管理・取引所リスクは未モデル化（無料データの限界）")


if __name__ == "__main__":
    main()
