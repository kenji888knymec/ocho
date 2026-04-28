"""
shadow_short_a_analysis.py — SHORT_A シャドウ集計
short_recent_pause がなければ、新SHORT_A条件は何件マッチしていたかを分析する。

SHORT_A条件:
  side=SHORT;mode=RANGE;rsi_min=50;rsi_max=55;p2_min=0;volratio_max=1.2;ai_max=0.6

実行: python shadow_short_a_analysis.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
"""

import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import google.auth
import httplib2
from googleapiclient.discovery import build
from google_auth_httplib2 import AuthorizedHttp

warnings.filterwarnings("ignore")

SPREADSHEET_ID = os.environ.get(
    "SPREADSHEET_ID",
    "1XwWkzijIwRlafg2zDgPHQ4tgjYModapFI3T_wbYS9_8",
)
V2_SHADOW_SHEET = os.environ.get("V2_SHADOW_SHEET", "v2_shadow_ai")
HEADER_COL_END = "AZ"
JST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 30


# ==========================================
# Google Sheets 取得
# ==========================================

def _sheets_service(creds):
    http = httplib2.Http(timeout=120)
    authed_http = AuthorizedHttp(creds, http=http)
    return build("sheets", "v4", http=authed_http, cache_discovery=False)


def fetch_done() -> pd.DataFrame:
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds, _ = google.auth.default(scopes=scopes)
    svc = _sheets_service(creds)

    hdr_res = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A1:{HEADER_COL_END}1",
    ).execute()
    headers = [str(h).strip() for h in (hdr_res.get("values", [[]]) or [[]])[0]]
    if not headers:
        raise RuntimeError("ヘッダーが空です")

    data_res = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A2:{HEADER_COL_END}",
    ).execute()
    rows = data_res.get("values", []) or []
    if not rows:
        raise RuntimeError("データ行が空です")

    n_cols = len(headers)
    fixed = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
    df_all = pd.DataFrame(fixed, columns=headers)
    print(f"[INFO] 総行数: {len(df_all)}", flush=True)

    status = df_all["EvalStatus"].astype(str).str.strip().str.upper()
    df_done = df_all[status == "DONE"].copy()
    print(f"[INFO] DONE件数: {len(df_done)}", flush=True)
    return df_done


# ==========================================
# 前処理
# ==========================================

def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def _to_win(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = []
    for v in s:
        if v in {"win", "w", "1", "true", "yes"}:
            out.append(1.0)
        elif v in {"lose", "l", "0", "false", "no"}:
            out.append(0.0)
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 日付
    if "Datetime_JST" not in df.columns:
        raise RuntimeError("Datetime_JST 列が見つかりません")
    df["_dt"] = pd.to_datetime(df["Datetime_JST"], errors="coerce", utc=False)
    df["_date"] = df["_dt"].dt.date

    # 数値列
    for col in ("RSI", "P2_FundingScore", "VolRatio", "AI_Prob_Win",
                "BTC_Mode_RANGE", "PnL_Pct"):
        if col in df.columns:
            df[f"_n_{col}"] = _to_num(df[col])
        else:
            df[f"_n_{col}"] = np.nan

    # BTC_Mode (RANGE判定用)
    # main.py line 1548: mode=RANGE は BTC_Mode_Compat == "RANGE" に対応
    is_range = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    for cand in ("BTC_Mode_Compat", "BTC_Mode_RANGE", "BTC_Mode", "BTC_HTF_Mode", "BTCMode"):
        if cand not in df.columns:
            continue
        col_vals = df[cand].astype(str).str.strip()
        non_empty = col_vals.replace("nan", "") != ""
        if not non_empty.any():
            continue
        if cand == "BTC_Mode_RANGE":
            # 数値フラグ: 1=RANGE
            numeric = _to_num(df[cand])
            is_range = (numeric == 1).where(numeric.notna(), np.nan)
        else:
            # 文字列: "RANGE" かどうか
            upper = col_vals.str.upper()
            is_range = upper.where(non_empty, np.nan).apply(
                lambda v: True if v == "RANGE" else (False if pd.notna(v) and v != "NAN" else np.nan)
            )
        break
    df["_is_range"] = is_range

    # WinLose
    if "WinLose" in df.columns:
        df["_win"] = _to_win(df["WinLose"])
    else:
        df["_win"] = np.nan

    df["_pnl"] = df["_n_PnL_Pct"]

    # Direction
    df["_dir"] = df["Direction"].astype(str).str.strip().str.upper() if "Direction" in df.columns else ""

    return df


# ==========================================
# SHORT_A 条件フィルタ
# ==========================================

def apply_short_a(df: pd.DataFrame, use_ai_max: bool = True) -> pd.DataFrame:
    """
    SHORT_A条件でフィルタ。
    use_ai_max=False にすると ai_max=0.6 を外した比較ができる。
    """
    mask = pd.Series([True] * len(df), index=df.index)

    # side=SHORT
    mask &= df["_dir"] == "SHORT"

    # mode=RANGE → _is_range == True
    rng = df["_is_range"]
    has_range = rng.notna() & (rng != "")
    mask &= has_range & (rng == True)

    # rsi_min=50, rsi_max=55 → 50 <= RSI < 55
    rsi = df["_n_RSI"]
    has_rsi = rsi.notna()
    mask &= has_rsi & (rsi >= 50) & (rsi < 55)

    # p2_min=0 → P2_FundingScore >= 0
    p2 = df["_n_P2_FundingScore"]
    has_p2 = p2.notna()
    mask &= has_p2 & (p2 >= 0)

    # volratio_max=1.2 → VolRatio < 1.2
    vr = df["_n_VolRatio"]
    has_vr = vr.notna()
    mask &= has_vr & (vr < 1.2)

    # ai_max=0.6 → AI_Prob_Win < 0.6
    if use_ai_max:
        ai = df["_n_AI_Prob_Win"]
        has_ai = ai.notna()
        mask &= has_ai & (ai < 0.6)

    return df[mask].copy()


# ==========================================
# 集計
# ==========================================

def _stats(sub: pd.DataFrame) -> dict:
    win = sub["_win"].dropna()
    pnl = sub["_pnl"].dropna()
    n = len(win)
    return {
        "n": n,
        "wr": float(win.mean()) if n > 0 else np.nan,
        "pnl_mean": float(pnl.mean()) if len(pnl) > 0 else np.nan,
        "pnl_med": float(pnl.median()) if len(pnl) > 0 else np.nan,
    }


def _fmt(s: dict) -> str:
    n = s["n"]
    if n == 0:
        return f"  n={n:4d}  (データなし)"
    wr = f"{s['wr']*100:.1f}%" if np.isfinite(s["wr"]) else "  N/A "
    pm = f"{s['pnl_mean']:+.3f}" if np.isfinite(s["pnl_mean"]) else "  N/A "
    md = f"{s['pnl_med']:+.3f}" if np.isfinite(s["pnl_med"]) else "  N/A "
    return f"  n={n:4d}  wr={wr}  avgPnL={pm}  medPnL={md}"


def report_section(label: str, sub: pd.DataFrame, all_days: set):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    s = _stats(sub)
    print(f"\n【全体】{_fmt(s)}")

    # 発火日数
    match_days = set(sub["_date"].dropna())
    active = len(match_days)
    zero = len(all_days) - active
    print(f"  発火日数: {active}日 / {len(all_days)}日  (ゼロ日: {zero}日)")

    # AI帯別
    if sub["_n_AI_Prob_Win"].notna().any():
        print("\n【AI_Prob_Win 帯別】")
        bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
        labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        sub = sub.copy()
        sub["_ai_bin"] = pd.cut(sub["_n_AI_Prob_Win"], bins=bins, labels=labels)
        for band in labels:
            g = sub[sub["_ai_bin"] == band]
            print(f"  {band}: {_fmt(_stats(g))}")

    # 日別上位10日
    if len(sub) > 0:
        day_counts = sub.groupby("_date").size().sort_values(ascending=False).head(10)
        if len(day_counts) > 0:
            print(f"\n【日別件数 (上位10日)】")
            for d, cnt in day_counts.items():
                g = sub[sub["_date"] == d]
                st = _stats(g)
                wr_s = f"{st['wr']*100:.0f}%" if np.isfinite(st["wr"]) else "N/A"
                pm_s = f"{st['pnl_mean']:+.3f}" if np.isfinite(st["pnl_mean"]) else "N/A"
                print(f"  {d}  n={cnt:3d}  wr={wr_s}  avg={pm_s}")


# ==========================================
# メイン
# ==========================================

def main():
    print("=== SHORT_A シャドウ集計 ===")
    print(f"対象: 直近{LOOKBACK_DAYS}日 EvalStatus=DONE")
    print(f"SHORT_A条件: side=SHORT mode=RANGE rsi[50,55) p2>=0 vol<1.2 ai<0.6\n")

    df_raw = fetch_done()
    df = preprocess(df_raw)

    # 列存在チェック
    checked = ["RSI", "P2_FundingScore", "VolRatio", "AI_Prob_Win",
               "BTC_Mode_Compat", "BTC_Mode_RANGE", "BTC_Mode", "BTC_HTF_Mode",
               "PnL_Pct", "WinLose", "Direction", "Datetime_JST"]
    missing = [c for c in checked if c not in df_raw.columns]
    present = [c for c in checked if c in df_raw.columns]
    print(f"[INFO] 使用列: {present}")
    if missing:
        print(f"[WARN] 欠損列: {missing}")

    # BTC_Mode列候補をスプシ全体から探す
    btc_cols = [c for c in df_raw.columns if "btc" in c.lower()]
    print(f"[INFO] BTC mode系の列: {btc_cols}")

    # _is_range の判定状況を確認
    n_range_true = int((df["_is_range"] == True).sum())
    n_range_false = int((df["_is_range"] == False).sum())
    n_range_nan = int(df["_is_range"].isna().sum())
    print(f"[INFO] _is_range: True={n_range_true}, False={n_range_false}, NaN={n_range_nan}")

    # 日付フィルタ
    now_jst = datetime.now(JST)
    cutoff = (now_jst - timedelta(days=LOOKBACK_DAYS)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cutoff)].copy()
    all_days = set(pd.date_range(cutoff, now_jst.date()).date)
    print(f"[INFO] 直近{LOOKBACK_DAYS}日 DONE: {len(df_recent)}件  対象日数: {len(all_days)}日\n")

    # SHORT全体（ベースライン）
    df_short = df_recent[df_recent["_dir"] == "SHORT"].copy()
    print(f"[INFO] SHORT DONE (直近{LOOKBACK_DAYS}日): {len(df_short)}件")

    # SHORT_A（ai_max適用あり）
    df_a = apply_short_a(df_recent, use_ai_max=True)
    report_section(f"SHORT_A マッチ（ai_max=0.6あり）", df_a, all_days)

    # SHORT_A（ai_max除外：比較用）
    df_a_noai = apply_short_a(df_recent, use_ai_max=False)
    report_section(f"SHORT_A マッチ（ai_max=0.6なし・比較用）", df_a_noai, all_days)

    # SHORT_A内のAI帯別詳細（0.2刻み）
    print(f"\n{'='*60}")
    print("  ai_max除外版でのAI帯別詳細（ai_max閾値検討用）")
    print(f"{'='*60}")
    bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df_a_noai2 = df_a_noai.copy()
    df_a_noai2["_ai_bin"] = pd.cut(df_a_noai2["_n_AI_Prob_Win"], bins=bins, labels=labels)
    for band in labels:
        g = df_a_noai2[df_a_noai2["_ai_bin"] == band]
        print(f"  {band}: {_fmt(_stats(g))}")

    # ai_max=0.4 の場合
    df_a_04 = df_a_noai[df_a_noai["_n_AI_Prob_Win"].notna() & (df_a_noai["_n_AI_Prob_Win"] < 0.4)].copy()
    print(f"\n  ai_max=0.4 に絞った場合: {_fmt(_stats(df_a_04))}")
    days_04 = set(df_a_04["_date"].dropna())
    print(f"  発火日数: {len(days_04)}日 / {len(all_days)}日")

    # ai_max=0.2 の場合
    df_a_02 = df_a_noai[df_a_noai["_n_AI_Prob_Win"].notna() & (df_a_noai["_n_AI_Prob_Win"] < 0.2)].copy()
    print(f"\n  ai_max=0.2 に絞った場合: {_fmt(_stats(df_a_02))}")
    days_02 = set(df_a_02["_date"].dropna())
    print(f"  発火日数: {len(days_02)}日 / {len(all_days)}日")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
