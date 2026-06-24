#!/usr/bin/env python3
"""
probe_binance_public_data.py  【Binance公開データ形式チェック】
=================================================================
目的:
  data.binance.vision の futures/um/daily/metrics が、
  実際に検証に使える形式か確認する。

確認すること:
  1. metrics の列名・時刻形式・粒度・欠損
  2. OI / Long/Short Ratio / taker buy/sell比率が入っているか
  3. klines(1h) / fundingRate と同期間で結合できるか

条件:
  - BTCUSDT のみ、2024-01-01 〜 2024-01-03 (3日間)
  - 有料データなし / APIキー不要
  - main.py変更なし / deploy不要 / 本番非接続
  - 欠損は np.nan 扱い（0補完・推測補完 厳禁）

実行: python3 probe_binance_public_data.py
依存: 標準ライブラリ + pandas（pip install pandas）
"""
import io
import ssl
import urllib.request
import zipfile
from datetime import date, timedelta

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily"
SYM  = "BTCUSDT"
SSL_CTX = ssl.create_default_context()
HDRS = {"User-Agent": "Mozilla/5.0"}

DATES = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


# ─── ダウンロード共通 ─────────────────────────────────────────────────
def fetch_zip_csv(url: str) -> pd.DataFrame | None:
    req = urllib.request.Request(url, headers=HDRS)
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        print(f"    [HTTP {e.code}] {url}")
        return None
    except Exception as e:
        print(f"    [ERR {type(e).__name__}] {url}")
        return None

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return pd.read_csv(f, header=0)


# ─── 1. metrics ──────────────────────────────────────────────────────
def fetch_metrics(d: date) -> pd.DataFrame | None:
    url = f"{BASE}/metrics/{SYM}/{SYM}-metrics-{d}.zip"
    return fetch_zip_csv(url)


# ─── 2. klines (1h) ──────────────────────────────────────────────────
KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol", "ignore",
]

def fetch_klines(d: date, interval: str = "1h") -> pd.DataFrame | None:
    url = f"{BASE}/klines/{SYM}/{interval}/{SYM}-{interval}-{d}.zip"
    df = fetch_zip_csv(url)
    if df is not None:
        df.columns = KLINES_COLS
    return df


# ─── 3. fundingRate（月次のみ。daily には存在しない） ────────────────
FUNDING_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"

def fetch_funding_month(y: int, m: int) -> pd.DataFrame | None:
    url = f"{FUNDING_BASE}/{SYM}/{SYM}-fundingRate-{y}-{m:02d}.zip"
    return fetch_zip_csv(url)


# ─── ユーティリティ ───────────────────────────────────────────────────
def ts_to_jst(ms_series: pd.Series) -> pd.Series:
    return pd.to_datetime(ms_series, unit="ms", utc=True).dt.tz_convert("Asia/Tokyo")

def show_head(name: str, df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"{name}  rows={len(df)}  cols={len(df.columns)}")
    print(f"  columns: {list(df.columns)}")
    na = df.isna().sum()
    na_cols = na[na > 0]
    print(f"  nulls : {dict(na_cols) if len(na_cols) else '(なし)'}")
    print(df.head(3).to_string())


def main():
    print("=" * 60)
    print("Binance data.vision 形式チェック（3日間・BTCUSDT）")
    print(f"対象: {DATES[0]} 〜 {DATES[-1]}")
    print("=" * 60)

    # ── metrics ──────────────────────────────────────────────
    print("\n■ 1. metrics（OI / LSR / taker比率）")
    mlist = []
    for d in DATES:
        print(f"  fetch metrics {d} ...", end=" ")
        df = fetch_metrics(d)
        if df is not None:
            print(f"OK rows={len(df)}")
            mlist.append(df)
        else:
            print("NG")

    if mlist:
        metrics = pd.concat(mlist, ignore_index=True)
        show_head("metrics (3日合計)", metrics)

        # 時刻列を探す（create_time は "2024-01-01 00:05:00" の文字列日時）
        ts_col = next((c for c in metrics.columns if "time" in c.lower() or "timestamp" in c.lower()), None)
        if ts_col:
            raw = metrics[ts_col].iloc[0]
            if isinstance(raw, str):
                # 文字列日時: Binance metrics の create_time 形式
                metrics["_jst"] = pd.to_datetime(metrics[ts_col], utc=True).dt.tz_convert("Asia/Tokyo")
                unit_desc = "string-datetime"
            else:
                unit = "ms" if raw > 1e12 else "s"
                metrics["_jst"] = pd.to_datetime(metrics[ts_col], unit=unit, utc=True).dt.tz_convert("Asia/Tokyo")
                unit_desc = f"numeric-{unit}"
            diffs = metrics["_jst"].diff().dropna()
            most_common = diffs.value_counts().index[0] if len(diffs) else None
            print(f"\n  時刻列: {ts_col}  形式: {unit_desc}")
            print(f"  最頻インターバル: {most_common}  （≒粒度）")
            print(f"  先頭3行 JST: {metrics['_jst'].head(3).tolist()}")

        # OI / LSR 関連列を表示
        oi_cols  = [c for c in metrics.columns if "open_interest" in c.lower() or "sum_open_interest" in c.lower()]
        lsr_cols = [c for c in metrics.columns if "long" in c.lower() or "short" in c.lower() or "ratio" in c.lower()]
        tkr_cols = [c for c in metrics.columns if "taker" in c.lower() or "buy" in c.lower() or "sell" in c.lower()]
        print(f"\n  OI系列  : {oi_cols}")
        print(f"  LSR系列 : {lsr_cols}")
        print(f"  taker系列: {tkr_cols}")
        if oi_cols:
            print(f"\n  OI先頭値サンプル: {metrics[oi_cols[0]].head(3).tolist()}")
        if lsr_cols:
            print(f"  LSR先頭値サンプル: {metrics[lsr_cols[0]].head(3).tolist()}")
    else:
        print("  metrics 取得失敗。URLパスを要確認。")
        metrics = None

    # ── klines (1h) ──────────────────────────────────────────
    print("\n■ 2. klines (1h)")
    klist = []
    for d in DATES:
        print(f"  fetch klines {d} ...", end=" ")
        df = fetch_klines(d)
        if df is not None:
            print(f"OK rows={len(df)}")
            klist.append(df)
        else:
            print("NG")

    if klist:
        klines = pd.concat(klist, ignore_index=True)
        klines["_jst"] = ts_to_jst(klines["open_time"])
        show_head("klines 1h (3日合計)", klines)
    else:
        klines = None

    # ── fundingRate（月次ファイルから対象月を取得し3日分に絞る） ──────
    print("\n■ 3. fundingRate（月次ファイル）")
    fy, fm = DATES[0].year, DATES[0].month
    print(f"  fetch fundingRate {fy}-{fm:02d} (月次) ...", end=" ")
    funding = fetch_funding_month(fy, fm)
    if funding is not None:
        print(f"OK rows={len(funding)}（月全体）")
        show_head("fundingRate (月次・先頭)", funding)
        ts_col = next((c for c in funding.columns if "time" in c.lower()), None)
        if ts_col:
            raw = funding[ts_col].iloc[0]
            if isinstance(raw, str):
                funding["_jst"] = pd.to_datetime(funding[ts_col], utc=True).dt.tz_convert("Asia/Tokyo")
            else:
                unit = "ms" if raw > 1e12 else "s"
                funding["_jst"] = pd.to_datetime(funding[ts_col], unit=unit, utc=True).dt.tz_convert("Asia/Tokyo")
            interval = funding["funding_interval_hours"].iloc[0] if "funding_interval_hours" in funding.columns else "?"
            print(f"  Funding先頭3件(JST): {funding['_jst'].head(3).tolist()}")
            print(f"  Funding間隔(h): {interval}")
    else:
        funding = None

    # ── 結合テスト（dtype を UTC ns に統一してから merge_asof） ──────────
    print("\n■ 4. metrics × klines 結合テスト（merge_asof）")
    if metrics is not None and klines is not None and "_jst" in metrics.columns and "_jst" in klines.columns:
        m2 = metrics.copy()
        k2 = klines.copy()
        # 精度(us/ms)とtzのゆらぎを吸収: 一旦UTC・ns精度に正規化
        m2["_utc"] = pd.to_datetime(m2["_jst"], utc=True).astype("datetime64[ns, UTC]")
        k2["_utc"] = pd.to_datetime(k2["_jst"], utc=True).astype("datetime64[ns, UTC]")
        m2 = m2.sort_values("_utc")
        k2 = k2.sort_values("_utc")
        merged = pd.merge_asof(
            m2, k2[["_utc", "close", "volume"]].rename(columns={"close": "k_close", "volume": "k_vol"}),
            on="_utc", direction="backward"
        )
        print(f"  メトリクス行数: {len(m2)}  klines行数: {len(k2)}  結合後: {len(merged)}")
        na_close = merged["k_close"].isna().sum()
        print(f"  k_close 欠損: {na_close}行 ({na_close/len(merged)*100:.1f}%)")
        print(merged[["_utc", "k_close"]].head(3).to_string())
    else:
        print("  metrics または klines が未取得のため結合スキップ。")

    # ── 総合判定 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("総合判定:")
    ok_m   = metrics is not None and len(metrics) > 0
    ok_k   = klines  is not None and len(klines)  > 0
    ok_f   = funding is not None and len(funding)  > 0
    ok_oi  = ok_m and any("open_interest" in c.lower() or "sum_open_interest" in c.lower() for c in metrics.columns)
    ok_lsr = ok_m and any("long" in c.lower() or "ratio" in c.lower() for c in metrics.columns)
    ok_tkr = ok_m and any("taker" in c.lower() for c in metrics.columns)
    ok_jn  = ok_m and ok_k and metrics is not None and "k_close" in (merged.columns if 'merged' in dir() else [])

    print(f"  metrics取得        : {'✅' if ok_m else '❌'}")
    print(f"  klines取得         : {'✅' if ok_k else '❌'}")
    print(f"  fundingRate取得    : {'✅' if ok_f else '❌'}")
    print(f"  OI列あり           : {'✅' if ok_oi else '❌'}")
    print(f"  LSR列あり          : {'✅' if ok_lsr else '❌'}")
    print(f"  taker比率列あり    : {'✅' if ok_tkr else '❌'}")
    print(f"  klines結合可能     : {'✅' if ok_jn else '❌'}")

    all_ok = all([ok_m, ok_k, ok_f, ok_oi, ok_lsr, ok_jn])
    print(f"\n  → {'✅ 形式OK。次フェーズ（数年分OOS検証）に進める' if all_ok else '❌ 要調査。URLパス or 列名を再確認'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
