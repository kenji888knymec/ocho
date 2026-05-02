"""
long_candidate_map.py — LONG候補探索マップ

LONG × BTC_Mode=UP の全体成績をRSI帯・AI帯・VolRatio帯・Hour帯・P1帯で集計し、
勝率が高いセグメントを候補として抽出する。

実行: python long_candidate_map.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
オプション環境変数:
  LOOKBACK_DAYS    : 集計対象日数 (default: 30)
  BTC_MODE_FILTER  : UP/DOWN/RANGE/ALL (default: UP)
  RESEARCH_MIN_N   : 研究候補最低件数 (default: 5)
  RESEARCH_MIN_WR  : 研究候補最低勝率 (default: 0.55)
"""

import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

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
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "30"))
BTC_MODE_FILTER = os.environ.get("BTC_MODE_FILTER", "UP").upper()
RESEARCH_MIN_N = int(os.environ.get("RESEARCH_MIN_N", "5"))
RESEARCH_MIN_WR = float(os.environ.get("RESEARCH_MIN_WR", "0.55"))


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

    present = [c for c in df_all.columns if c.strip()]
    print(f"[INFO] 取得列({len(present)}): {present}", flush=True)
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


def _get_col(df: pd.DataFrame, *candidates) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan] * len(df), index=df.index)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "Datetime_JST" not in df.columns:
        raise RuntimeError("Datetime_JST 列が見つかりません")
    df["_dt"] = pd.to_datetime(df["Datetime_JST"], errors="coerce", utc=False)
    df["_date"] = df["_dt"].dt.date
    df["_hour"] = df["_dt"].dt.hour

    # Direction
    df["_dir"] = _get_col(df, "Direction").astype(str).str.strip().str.upper()

    # BTC_Mode_Compat (UP/DOWN/RANGE)
    btc_mode_raw = _get_col(df, "BTC_Mode_Compat", "BTC_Mode", "BTC_HTF_Mode", "BTCMode")
    df["_btc_mode"] = btc_mode_raw.astype(str).str.strip().str.upper()

    # WinLose / PnL
    df["_win"] = _to_win(_get_col(df, "WinLose"))
    df["_pnl"] = _to_num(_get_col(df, "PnL_Pct"))

    # 数値特徴量
    df["_rsi"]  = _to_num(_get_col(df, "RSI"))
    df["_ai"]   = _to_num(_get_col(df, "AI_Prob_Win"))
    df["_volr"] = _to_num(_get_col(df, "VolRatio"))
    df["_p1"]   = _to_num(_get_col(df, "P1_TrendScore"))
    df["_p2"]   = _to_num(_get_col(df, "P2_FundingScore"))
    df["_p3"]   = _to_num(_get_col(df, "P3_VolumeScore"))

    # notify_sent (D判定用)
    ns_raw = _get_col(df, "notify_sent", "_notify_sent")
    df["_notify"] = ns_raw.astype(str).str.strip() == "1"

    return df


# ==========================================
# 集計ヘルパー
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


def _fmt(s: dict, prefix: str = "") -> str:
    n = s["n"]
    if n == 0:
        return f"{prefix}n={n:4d}  (データなし)"
    wr = f"{s['wr']*100:.1f}%" if np.isfinite(s.get("wr", float("nan"))) else "  N/A"
    pm = f"{s['pnl_mean']:+.3f}" if np.isfinite(s.get("pnl_mean", float("nan"))) else "  N/A"
    md = f"{s['pnl_med']:+.3f}" if np.isfinite(s.get("pnl_med", float("nan"))) else "  N/A"
    return f"{prefix}n={n:4d}  wr={wr}  avg={pm}  med={md}"


def _is_candidate(s: dict, min_n: int, min_wr: float) -> bool:
    if s["n"] < min_n:
        return False
    if not np.isfinite(s.get("wr", float("nan"))):
        return False
    if s["wr"] < min_wr:
        return False
    if not np.isfinite(s.get("pnl_mean", float("nan"))) or s["pnl_mean"] <= 0:
        return False
    if not np.isfinite(s.get("pnl_med", float("nan"))) or s["pnl_med"] <= 0:
        return False
    return True


# ==========================================
# 各セクション
# ==========================================

def section_overall(df: pd.DataFrame, all_days: set) -> None:
    print(f"\n{'='*70}")
    print("  Section 1: LONG全体成績")
    print(f"{'='*70}")

    s = _stats(df)
    print(f"\n【全体DONE】 {_fmt(s)}")

    active = len(set(df["_date"].dropna()))
    print(f"  発火日数: {active}日 / {len(all_days)}日  (ゼロ日: {len(all_days)-active}日)")

    # D=実通知のみ
    live = df[df["_notify"]]
    s_live = _stats(live)
    print(f"\n【D=実通知のみ】 {_fmt(s_live)}")


def section_rsi(df: pd.DataFrame) -> list:
    print(f"\n{'='*70}")
    print("  Section 2: RSI帯別成績")
    print(f"{'='*70}")

    bands = [
        ("< 40",   df["_rsi"] < 40),
        ("40-50",  (df["_rsi"] >= 40) & (df["_rsi"] < 50)),
        ("50-60",  (df["_rsi"] >= 50) & (df["_rsi"] < 60)),
        ("60-70",  (df["_rsi"] >= 60) & (df["_rsi"] < 70)),
        ("70-80",  (df["_rsi"] >= 70) & (df["_rsi"] < 80)),
        ("80+",    df["_rsi"] >= 80),
        ("N/A",    df["_rsi"].isna()),
    ]

    candidates = []
    for label, mask in bands:
        g = df[mask & df["_rsi"].notna()] if label != "N/A" else df[mask]
        s = _stats(g)
        tag = " ★研究候補" if _is_candidate(s, RESEARCH_MIN_N, RESEARCH_MIN_WR) else ""
        print(f"  RSI {label:8s}: {_fmt(s)}{tag}")
        if tag:
            candidates.append(("RSI", label, s))
    return candidates


def section_ai(df: pd.DataFrame) -> list:
    print(f"\n{'='*70}")
    print("  Section 3: AI_Prob_Win帯別成績")
    print(f"{'='*70}")

    na_mask = df["_ai"].isna()
    s_na = _stats(df[na_mask])
    print(f"  AI blank : {_fmt(s_na)}")

    bins = [
        ("0.0-0.2", (df["_ai"] >= 0.0) & (df["_ai"] < 0.2)),
        ("0.2-0.4", (df["_ai"] >= 0.2) & (df["_ai"] < 0.4)),
        ("0.4-0.6", (df["_ai"] >= 0.4) & (df["_ai"] < 0.6)),
        ("0.6-0.8", (df["_ai"] >= 0.6) & (df["_ai"] < 0.8)),
        ("0.8+",    df["_ai"] >= 0.8),
    ]

    candidates = []
    for label, mask in bins:
        g = df[mask]
        s = _stats(g)
        tag = " ★研究候補" if _is_candidate(s, RESEARCH_MIN_N, RESEARCH_MIN_WR) else ""
        print(f"  AI {label:8s}: {_fmt(s)}{tag}")
        if tag:
            candidates.append(("AI", label, s))
    return candidates


def section_volratio(df: pd.DataFrame) -> list:
    print(f"\n{'='*70}")
    print("  Section 4: VolRatio帯別成績")
    print(f"{'='*70}")

    bands = [
        ("< 1.0",  df["_volr"] < 1.0),
        ("1.0-1.5", (df["_volr"] >= 1.0) & (df["_volr"] < 1.5)),
        ("1.5-2.0", (df["_volr"] >= 1.5) & (df["_volr"] < 2.0)),
        ("2.0-3.0", (df["_volr"] >= 2.0) & (df["_volr"] < 3.0)),
        ("3.0+",    df["_volr"] >= 3.0),
        ("N/A",     df["_volr"].isna()),
    ]

    candidates = []
    for label, mask in bands:
        g = df[mask & df["_volr"].notna()] if label != "N/A" else df[mask]
        s = _stats(g)
        tag = " ★研究候補" if _is_candidate(s, RESEARCH_MIN_N, RESEARCH_MIN_WR) else ""
        print(f"  VolR {label:8s}: {_fmt(s)}{tag}")
        if tag:
            candidates.append(("VolRatio", label, s))
    return candidates


def section_p1(df: pd.DataFrame) -> list:
    print(f"\n{'='*70}")
    print("  Section 5: P1_TrendScore帯別成績")
    print(f"{'='*70}")

    bands = [
        ("< 0",    df["_p1"] < 0),
        ("0-1",    (df["_p1"] >= 0) & (df["_p1"] < 1)),
        ("1-2",    (df["_p1"] >= 1) & (df["_p1"] < 2)),
        ("2-3",    (df["_p1"] >= 2) & (df["_p1"] < 3)),
        ("3+",     df["_p1"] >= 3),
        ("N/A",    df["_p1"].isna()),
    ]

    candidates = []
    for label, mask in bands:
        g = df[mask & df["_p1"].notna()] if label != "N/A" else df[mask]
        s = _stats(g)
        tag = " ★研究候補" if _is_candidate(s, RESEARCH_MIN_N, RESEARCH_MIN_WR) else ""
        print(f"  P1 {label:6s}: {_fmt(s)}{tag}")
        if tag:
            candidates.append(("P1", label, s))
    return candidates


def section_hour(df: pd.DataFrame) -> list:
    print(f"\n{'='*70}")
    print("  Section 6: Hour_JST帯別成績")
    print(f"{'='*70}")

    slot_bands = [
        ("0-5",   (df["_hour"] >= 0)  & (df["_hour"] <= 5)),
        ("6-11",  (df["_hour"] >= 6)  & (df["_hour"] <= 11)),
        ("12-17", (df["_hour"] >= 12) & (df["_hour"] <= 17)),
        ("18-23", (df["_hour"] >= 18) & (df["_hour"] <= 23)),
    ]

    candidates = []
    print("  [時間帯スロット]")
    for label, mask in slot_bands:
        g = df[mask]
        s = _stats(g)
        tag = " ★研究候補" if _is_candidate(s, RESEARCH_MIN_N, RESEARCH_MIN_WR) else ""
        print(f"  Hour {label:6s}: {_fmt(s)}{tag}")
        if tag:
            candidates.append(("Hour", label, s))

    print("\n  [時間別（0〜23）]")
    for h in range(24):
        g = df[df["_hour"] == h]
        s = _stats(g)
        if s["n"] == 0:
            continue
        tag = " ★" if _is_candidate(s, max(3, RESEARCH_MIN_N // 2), RESEARCH_MIN_WR) else ""
        print(f"  Hour {h:02d}h: {_fmt(s)}{tag}")
    return candidates


# ==========================================
# 2変数組み合わせ
# ==========================================

def _rsi_label(v) -> str:
    if pd.isna(v):
        return "N/A"
    if v < 40:  return "<40"
    if v < 50:  return "40-50"
    if v < 60:  return "50-60"
    if v < 70:  return "60-70"
    if v < 80:  return "70-80"
    return "80+"


def _ai_label(v) -> str:
    if pd.isna(v):
        return "N/A"
    if v < 0.2: return "0.0-0.2"
    if v < 0.4: return "0.2-0.4"
    if v < 0.6: return "0.4-0.6"
    if v < 0.8: return "0.6-0.8"
    return "0.8+"


def _volr_label(v) -> str:
    if pd.isna(v):
        return "N/A"
    if v < 1.0: return "<1.0"
    if v < 1.5: return "1.0-1.5"
    if v < 2.0: return "1.5-2.0"
    if v < 3.0: return "2.0-3.0"
    return "3.0+"


def _p1_label(v) -> str:
    if pd.isna(v):
        return "N/A"
    if v < 0:   return "<0"
    if v < 1:   return "0-1"
    if v < 2:   return "1-2"
    if v < 3:   return "2-3"
    return "3+"


def _hour_label(v) -> str:
    if pd.isna(v):
        return "N/A"
    h = int(v)
    if h <= 5:   return "0-5"
    if h <= 11:  return "6-11"
    if h <= 17:  return "12-17"
    return "18-23"


def section_cross(df: pd.DataFrame, label_a_fn, col_a: str, label_b_fn, col_b: str,
                  title: str, min_n_cross: int = 3) -> list:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    df2 = df.copy()
    df2["_la"] = df2[col_a].apply(label_a_fn)
    df2["_lb"] = df2[col_b].apply(label_b_fn)

    candidates = []
    pivot_rows = sorted(df2["_la"].unique())
    pivot_cols = sorted(df2["_lb"].unique())

    for ra in pivot_rows:
        if ra == "N/A":
            continue
        for rb in pivot_cols:
            if rb == "N/A":
                continue
            g = df2[(df2["_la"] == ra) & (df2["_lb"] == rb)]
            s = _stats(g)
            if s["n"] < min_n_cross:
                continue
            tag = " ★研究候補" if _is_candidate(s, RESEARCH_MIN_N, RESEARCH_MIN_WR) else ""
            if tag or s["n"] >= 5:
                print(f"  {ra:8s} x {rb:8s}: {_fmt(s)}{tag}")
            if tag:
                candidates.append((title, f"{ra}×{rb}", s))
    return candidates


# ==========================================
# 候補サマリ
# ==========================================

def section_candidates(all_candidates: list, df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  Section 8: 研究候補サマリ")
    print(f"{'='*70}")
    print(f"  基準: DONE >= {RESEARCH_MIN_N}件、勝率 >= {RESEARCH_MIN_WR*100:.0f}%、avgPnL > 0、medPnL > 0\n")

    if not all_candidates:
        print("  研究候補なし — 現状ブロックが妥当")
        return

    print(f"  候補数: {len(all_candidates)}件\n")
    for dim, label, s in all_candidates:
        print(f"  [{dim}] {label}")
        print(f"    {_fmt(s)}")

    # 最有望候補を n×wr でソート
    scored = []
    for dim, label, s in all_candidates:
        if np.isfinite(s.get("wr", float("nan"))):
            scored.append((dim, label, s, s["n"] * s["wr"]))
    scored.sort(key=lambda x: -x[3])

    if scored:
        print(f"\n  【最有望候補 top3（n×勝率スコア順）】")
        for dim, label, s, sc in scored[:3]:
            print(f"  [{dim}] {label}: {_fmt(s)}  (スコア={sc:.1f})")


# ==========================================
# メイン
# ==========================================

def main():
    print("=== LONG候補探索マップ ===")
    print(f"対象: 直近{LOOKBACK_DAYS}日 / EvalStatus=DONE / Direction=LONG"
          + (f" / BTC_Mode={BTC_MODE_FILTER}" if BTC_MODE_FILTER != "ALL" else " / BTC_Mode=ALL"))
    print(f"研究候補基準: DONE>={RESEARCH_MIN_N}件, 勝率>={RESEARCH_MIN_WR*100:.0f}%, avgPnL>0, medPnL>0\n")

    df_raw = fetch_done()
    df = preprocess(df_raw)

    # 日付フィルタ
    now_jst = datetime.now(JST)
    cutoff = (now_jst - timedelta(days=LOOKBACK_DAYS)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cutoff)].copy()
    all_days = set(pd.date_range(cutoff, now_jst.date()).date)
    print(f"[INFO] 直近{LOOKBACK_DAYS}日 全DONE: {len(df_recent)}件", flush=True)

    # Direction=LONG フィルタ
    df_long = df_recent[df_recent["_dir"] == "LONG"].copy()
    print(f"[INFO] LONG DONE: {len(df_long)}件", flush=True)

    # BTC_Mode フィルタ
    if BTC_MODE_FILTER != "ALL":
        df_base = df_long[df_long["_btc_mode"] == BTC_MODE_FILTER].copy()
        print(f"[INFO] LONG×{BTC_MODE_FILTER} DONE: {len(df_base)}件", flush=True)
    else:
        df_base = df_long.copy()

    if len(df_base) == 0:
        print("\n[WARN] 対象データが0件です。フィルター条件を確認してください。")
        return

    all_candidates = []

    section_overall(df_base, all_days)
    all_candidates += section_rsi(df_base)
    all_candidates += section_ai(df_base)
    all_candidates += section_volratio(df_base)
    all_candidates += section_p1(df_base)
    all_candidates += section_hour(df_base)

    # 2変数クロス集計
    print(f"\n{'='*70}")
    print("  Section 7: 2変数クロス集計")
    print(f"{'='*70}")

    df_base["_rsi_l"]  = df_base["_rsi"].apply(_rsi_label)
    df_base["_ai_l"]   = df_base["_ai"].apply(_ai_label)
    df_base["_volr_l"] = df_base["_volr"].apply(_volr_label)
    df_base["_p1_l"]   = df_base["_p1"].apply(_p1_label)
    df_base["_hour_l"] = df_base["_hour"].apply(_hour_label)

    all_candidates += section_cross(
        df_base, _rsi_label, "_rsi", _ai_label, "_ai",
        "Section 7a: RSI帯 × AI帯"
    )
    all_candidates += section_cross(
        df_base, _rsi_label, "_rsi", _volr_label, "_volr",
        "Section 7b: RSI帯 × VolRatio帯"
    )
    all_candidates += section_cross(
        df_base, _p1_label, "_p1", _rsi_label, "_rsi",
        "Section 7c: P1帯 × RSI帯"
    )
    all_candidates += section_cross(
        df_base, _hour_label, "_hour", _rsi_label, "_rsi",
        "Section 7d: Hour帯 × RSI帯"
    )

    section_candidates(all_candidates, df_base)

    print(f"\n=== 完了 ===")


if __name__ == "__main__":
    main()
