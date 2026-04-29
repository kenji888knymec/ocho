"""
shadow_4lane_analysis.py — 4レーン別シャドウ集計
AI_Model_Version から4レーンを判定して、各レーンの成績を分析する。

4レーン判定方法（approximate）:
  v2_long_20260415_081606  → long_trend
  v2_long_20260414_153708  → long_recovery
  v2_short_20260416_082845 → short_trend
  SHORT共通 or 未判定SHORT  → short_retrace（approximate）
  その他                   → unknown

実行: python shadow_4lane_analysis.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
"""

import os
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

# モデルバージョン → レーン対応
# 環境変数で上書き可能（新モデルが出たとき用）
LONG_TREND_MODEL    = os.environ.get("LONG_TREND_MODEL_VERSION",    "v2_long_20260415_081606")
LONG_RECOVERY_MODEL = os.environ.get("LONG_RECOVERY_MODEL_VERSION", "v2_long_20260414_153708")
SHORT_TREND_MODEL   = os.environ.get("SHORT_TREND_MODEL_VERSION",   "v2_short_20260416_082845")
LONG_COMMON_MODEL   = os.environ.get("LONG_MODEL_VERSION",          "v2_long_20260413_212814")
SHORT_COMMON_MODEL  = os.environ.get("SHORT_MODEL_VERSION",         "v2_short_20260413_084558")


# ==========================================
# Google Sheets 取得
# ==========================================

def _sheets_service(creds):
    http = httplib2.Http(timeout=120)
    from google_auth_httplib2 import AuthorizedHttp
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


def classify_lane(df: pd.DataFrame) -> pd.Series:
    """
    AI_Model_Version からレーンを判定（approximate）。
    既存の lane 列があればそちらを優先。
    """
    # 既存 lane 列優先
    for col in ("lane", "Lane", "signal_lane"):
        if col in df.columns:
            vals = df[col].astype(str).str.strip().str.lower()
            non_empty = ~vals.isin(["", "nan", "none"])
            if non_empty.sum() > 0:
                print(f"[INFO] lane列 '{col}' を使用（{non_empty.sum()}件）")
                return vals.where(non_empty, "unknown")

    # AI_Model_Version から判定
    ver_col = None
    for c in ("AI_Model_Version", "ai_model_version", "AI_Model_Ver"):
        if c in df.columns:
            ver_col = c
            break

    if ver_col is None:
        print("[WARN] AI_Model_Version 列が見つかりません。全行 unknown になります")
        return pd.Series(["unknown"] * len(df), index=df.index)

    ver = df[ver_col].astype(str).str.strip()
    dir_col = df["Direction"].astype(str).str.strip().str.upper() if "Direction" in df.columns else pd.Series([""] * len(df), index=df.index)

    lanes = []
    for i in range(len(df)):
        v = ver.iloc[i]
        d = dir_col.iloc[i]
        if v == LONG_TREND_MODEL:
            lanes.append("long_trend")
        elif v == LONG_RECOVERY_MODEL:
            lanes.append("long_recovery")
        elif v == SHORT_TREND_MODEL:
            lanes.append("short_trend")
        elif v == SHORT_COMMON_MODEL and d == "SHORT":
            # SHORT共通モデル = short_retrace相当（専用モデルなし）
            lanes.append("short_retrace")
        elif v == LONG_COMMON_MODEL and d == "LONG":
            lanes.append("long_trend")   # LONG共通は long_trend 寄り
        elif d == "SHORT":
            lanes.append("short_unknown")
        elif d == "LONG":
            lanes.append("long_unknown")
        else:
            lanes.append("unknown")

    return pd.Series(lanes, index=df.index)


def is_live_notify(df: pd.DataFrame) -> pd.Series:
    for col in ("notify_sent", "_notify_sent"):
        if col in df.columns:
            return df[col].astype(str).str.strip() == "1"
    if "AI_Note" in df.columns:
        note = df["AI_Note"].astype(str).str.lower()
        return note.str.contains("notify_sent=1", na=False)
    return pd.Series([False] * len(df), index=df.index)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "Datetime_JST" not in df.columns:
        raise RuntimeError("Datetime_JST 列が見つかりません")
    df["_dt"] = pd.to_datetime(df["Datetime_JST"], errors="coerce", utc=False)
    df["_date"] = df["_dt"].dt.date

    df["_win"] = _to_win(df["WinLose"]) if "WinLose" in df.columns else np.nan
    df["_pnl"] = _to_num(df["PnL_Pct"]) if "PnL_Pct" in df.columns else np.nan
    df["_ai"]  = _to_num(df["AI_Prob_Win"]) if "AI_Prob_Win" in df.columns else np.nan
    df["_dir"] = df["Direction"].astype(str).str.strip().str.upper() if "Direction" in df.columns else ""
    df["_lane"] = classify_lane(df)
    df["_live"] = is_live_notify(df)

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


def _fmt(s: dict) -> str:
    n = s["n"]
    if n == 0:
        return f"n={n:4d}  (データなし)"
    wr = f"{s['wr']*100:.1f}%" if np.isfinite(s["wr"]) else "  N/A"
    pm = f"{s['pnl_mean']:+.3f}" if np.isfinite(s["pnl_mean"]) else "  N/A"
    md = f"{s['pnl_med']:+.3f}" if np.isfinite(s["pnl_med"]) else "  N/A"
    return f"n={n:4d}  wr={wr}  avg={pm}  med={md}"


def report_lane(label: str, sub: pd.DataFrame, all_days: set):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    # 全体
    s_all = _stats(sub)
    print(f"\n【全体DONE】  {_fmt(s_all)}")

    # 発火日数
    match_days = set(sub["_date"].dropna())
    print(f"  発火日数: {len(match_days)}日 / {len(all_days)}日  "
          f"(ゼロ日: {len(all_days)-len(match_days)}日)")

    # D=実通知のみ
    live = sub[sub["_live"]]
    s_live = _stats(live)
    print(f"\n【D=実通知のみ】  {_fmt(s_live)}")

    # AI帯別
    if sub["_ai"].notna().any():
        print("\n【AI_Prob_Win 帯別（全体）】")
        bins   = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
        labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        sub2 = sub.copy()
        sub2["_ai_bin"] = pd.cut(sub2["_ai"], bins=bins, labels=labels)
        for band in labels:
            g = sub2[sub2["_ai_bin"] == band]
            print(f"  {band}: {_fmt(_stats(g))}")

    # 日別上位5日
    if len(sub) > 0:
        day_counts = sub.groupby("_date").size().sort_values(ascending=False).head(5)
        if len(day_counts) > 0:
            print("\n【日別件数 上位5日】")
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
    print("=== 4レーン シャドウ集計 ===")
    print(f"対象: 直近{LOOKBACK_DAYS}日 EvalStatus=DONE\n")
    print(f"モデル→レーン対応:")
    print(f"  {LONG_TREND_MODEL}    → long_trend")
    print(f"  {LONG_RECOVERY_MODEL} → long_recovery")
    print(f"  {SHORT_TREND_MODEL}  → short_trend")
    print(f"  {SHORT_COMMON_MODEL} → short_retrace（専用モデルなし・approximate）\n")

    df_raw = fetch_done()
    df = preprocess(df_raw)

    # 使用列確認
    key_cols = ["AI_Model_Version", "AI_Prob_Win", "WinLose", "PnL_Pct",
                "Direction", "Datetime_JST", "notify_sent"]
    present = [c for c in key_cols if c in df_raw.columns]
    missing = [c for c in key_cols if c not in df_raw.columns]
    print(f"[INFO] 使用列: {present}")
    if missing:
        print(f"[WARN] 欠損列: {missing}")

    # レーン分布確認
    lane_counts = df["_lane"].value_counts()
    print(f"\n[INFO] レーン分布（全DONE）:")
    for lane, cnt in lane_counts.items():
        print(f"  {lane}: {cnt}件")

    # 直近30日フィルタ
    now_jst = datetime.now(JST)
    cutoff = (now_jst - timedelta(days=LOOKBACK_DAYS)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cutoff)].copy()
    all_days = set(pd.date_range(cutoff, now_jst.date()).date)
    print(f"\n[INFO] 直近{LOOKBACK_DAYS}日 DONE: {len(df_recent)}件")

    # レーン分布（直近30日）
    print(f"\n[INFO] レーン分布（直近30日）:")
    for lane, cnt in df_recent["_lane"].value_counts().items():
        print(f"  {lane}: {cnt}件")

    # 4レーン別レポート
    lanes = ["long_trend", "long_recovery", "short_trend", "short_retrace"]
    for lane in lanes:
        sub = df_recent[df_recent["_lane"] == lane]
        report_lane(f"{lane}  （直近{LOOKBACK_DAYS}日）", sub, all_days)

    # unknown系もまとめて表示
    unk = df_recent[~df_recent["_lane"].isin(lanes)]
    if len(unk) > 0:
        print(f"\n[INFO] 未分類（unknown/short_unknown/long_unknown）: {len(unk)}件")
        unk_lane_counts = unk["_lane"].value_counts()
        for l, c in unk_lane_counts.items():
            g = unk[unk["_lane"] == l]
            print(f"  {l}: {_fmt(_stats(g))}")

    # 全体まとめ
    print(f"\n{'='*65}")
    print("  全体まとめ（直近30日）")
    print(f"{'='*65}")
    for lane in lanes:
        sub = df_recent[df_recent["_lane"] == lane]
        s = _stats(sub)
        live_n = int(sub["_live"].sum())
        print(f"  {lane:20s} {_fmt(s)}  D={live_n}件")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
