"""
edge_route_cross_analysis.py — route × Direction × AI帯 × WinLose クロス集計
実行: python edge_route_cross_analysis.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
"""

import os
import sys
import warnings
from datetime import timedelta, timezone
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
MIN_SAMPLES = 15
JST = timezone(timedelta(hours=9))


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

def _to_win(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    win_tokens  = {"win", "w", "1", "true", "yes"}
    lose_tokens = {"lose", "l", "0", "false", "no"}
    out = []
    for v in s:
        if v in win_tokens:
            out.append(1.0)
        elif v in lose_tokens:
            out.append(0.0)
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def classify_route(df: pd.DataFrame) -> pd.Series:
    """
    route分類。優先順位:
      1. AI_Note に feature_profile_only_no_match → no_match
      2. AI_Model_Type が SHORT/LONG_FEATURE_BYPASS → feature_route
      3. AI_Model_Type が SHORT_AI / SHORT_AI_FALLBACK → AI_route_SHORT
      4. AI_Model_Type が LONG_AI / LONG_AI_FALLBACK → AI_route_LONG
      5. その他 → rule_rank / common / other
    no_match を先に判定する理由: no_match 行でも AI_Model_Type=SHORT_AI/LONG_AI の
    場合があり、後でチェックすると AI_route に混入する。
    """
    mt = df["AI_Model_Type"].astype(str).str.strip().str.upper() if "AI_Model_Type" in df.columns else pd.Series([""] * len(df), index=df.index)
    ab = df["AI_Band"].astype(str).str.strip().str.upper() if "AI_Band" in df.columns else pd.Series([""] * len(df), index=df.index)
    note = df["AI_Note"].astype(str).str.lower() if "AI_Note" in df.columns else pd.Series([""] * len(df), index=df.index)

    routes = []
    for i in range(len(df)):
        m = mt.iloc[i]
        a = ab.iloc[i]
        n = note.iloc[i]

        # 1. no_match を最優先で判定（AI_Model_Type より先）
        if "feature_profile_only_no_match" in n:
            routes.append("no_match")
        # 2. feature_route
        elif m in ("SHORT_FEATURE_BYPASS", "LONG_FEATURE_BYPASS") or a == "FEATURE_PROFILE_BYPASS":
            routes.append("feature_route")
        # 3. AI_route SHORT
        elif m in ("SHORT_AI", "SHORT_AI_FALLBACK"):
            routes.append("AI_route_SHORT")
        # 4. AI_route LONG
        elif m in ("LONG_AI", "LONG_AI_FALLBACK"):
            routes.append("AI_route_LONG")
        # 5. その他
        elif m in ("SHORT_RULE_RANK", "LONG_RULE_RANK"):
            routes.append("rule_rank")
        elif m == "COMMON":
            routes.append("common")
        else:
            routes.append("other")

    return pd.Series(routes, index=df.index)


def is_live_notify(df: pd.DataFrame) -> pd.Series:
    """
    D=実通知判定。
    優先: notify_sent 列 または _notify_sent 列（値 "1"）。
    補助: AI_Note 内の "notify_sent=1"（列がない場合のフォールバック）。
    """
    # notify_sent 列優先
    for col in ("notify_sent", "_notify_sent"):
        if col in df.columns:
            return df[col].astype(str).str.strip() == "1"
    # フォールバック: AI_Note 文字列
    if "AI_Note" in df.columns:
        note = df["AI_Note"].astype(str).str.lower()
        return note.str.contains("notify_sent=1", na=False)
    return pd.Series([False] * len(df), index=df.index)


def ai_prob_bin(series: pd.Series) -> pd.Series:
    """AI_Prob_Win を0.2刻みのbinに変換。欠損/非数値はNaN。"""
    numeric = _to_num(series)
    bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    return pd.cut(numeric, bins=bins, labels=labels)


# ==========================================
# 集計ヘルパー
# ==========================================

def _stats(group: pd.DataFrame) -> dict:
    win = group["_win"].dropna()
    pnl = group["_pnl"].dropna()
    n = int(win.notna().sum())
    if n == 0:
        return {"n": 0, "wr": np.nan, "pnl_mean": np.nan, "pnl_med": np.nan}
    return {
        "n": n,
        "wr": float(win.mean() * 100),
        "pnl_mean": float(pnl.mean()) if len(pnl) > 0 else np.nan,
        "pnl_med": float(pnl.median()) if len(pnl) > 0 else np.nan,
    }


def _fmt(s: dict) -> str:
    n = s["n"]
    if n == 0:
        return f"  n={n:4d} ---"
    wr = s["wr"]
    pm = s["pnl_mean"]
    wr_s  = f"{wr:5.1f}%" if not np.isnan(wr) else "  N/A "
    pm_s  = f"{pm:+.3f}" if not np.isnan(pm) else "  N/A"
    flag = ""
    if not np.isnan(wr):
        if wr >= 58:
            flag = " ✅"
        elif wr <= 42:
            flag = " ❌"
    return f"  n={n:4d}  wr={wr_s}  PnL={pm_s}{flag}"


def print_table(title: str, df: pd.DataFrame, group_cols, min_n: int = MIN_SAMPLES):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    grp = df.groupby(group_cols, observed=True)
    rows = []
    for keys, g in grp:
        s = _stats(g)
        if s["n"] >= min_n:
            rows.append((keys, s))
    if not rows:
        print("  → 件数不足（判定不能）")
        return
    for keys, s in rows:
        label = " | ".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
        print(f"  [{label}]{_fmt(s)}")


# ==========================================
# 分析本体
# ==========================================

def analyze_route_direction(df: pd.DataFrame):
    print_table(
        "1. route × Direction（DONE全体）",
        df,
        ["_route", "Direction"],
    )


def analyze_route_direction_ai_bin(df: pd.DataFrame):
    print_table(
        "2. route × Direction × AI_Prob_Win帯（DONE全体）",
        df,
        ["_route", "Direction", "_ai_bin"],
    )


def analyze_feature_route_ai_bin(df: pd.DataFrame):
    sub = df[df["_route"] == "feature_route"].copy()
    print(f"\n{'='*60}")
    print("  3. feature_route 内 AI帯別")
    print(f"{'='*60}")
    print(f"  feature_route DONE件数: {len(sub)}")
    print_table(
        "  3a. feature_route × Direction × AI帯",
        sub,
        ["Direction", "_ai_bin"],
        min_n=10,
    )


def analyze_ai_route_ai_bin(df: pd.DataFrame):
    sub = df[df["_route"].isin(["AI_route_SHORT", "AI_route_LONG"])].copy()
    print(f"\n{'='*60}")
    print("  4. AI_route 内 AI帯別")
    print(f"{'='*60}")
    print(f"  AI_route DONE件数: {len(sub)}")
    print_table(
        "  4a. AI_route × Direction × AI帯",
        sub,
        ["_route", "_ai_bin"],
        min_n=10,
    )


def analyze_no_match(df: pd.DataFrame):
    sub = df[df["_route"] == "no_match"].copy()
    print(f"\n{'='*60}")
    print("  5. no_match 内（selected=true vs false）")
    print(f"{'='*60}")
    print(f"  no_match DONE件数: {len(sub)}")

    if "AI_Note" not in sub.columns:
        print("  → AI_Note列なし")
        return

    note = sub["AI_Note"].astype(str).str.lower()
    sub = sub.copy()
    sub["_selected"] = "unknown"
    sub.loc[note.str.contains("selected=true|selected=1", regex=True), "_selected"] = "selected_true"
    sub.loc[note.str.contains("selected=false|selected=0", regex=True), "_selected"] = "selected_false"

    print_table(
        "  5a. no_match × Direction × selected",
        sub,
        ["Direction", "_selected"],
        min_n=10,
    )


def analyze_live_notify(df: pd.DataFrame):
    sub = df[df["_is_live"]].copy()
    print(f"\n{'='*60}")
    print("  6. D=実通知(notify_sent=1)のみ")
    print(f"{'='*60}")
    print(f"  実通知 DONE件数: {len(sub)}")
    print_table(
        "  6a. 実通知 × route × Direction",
        sub,
        ["_route", "Direction"],
        min_n=5,
    )
    print_table(
        "  6b. 実通知 × route × AI帯",
        sub,
        ["_route", "_ai_bin"],
        min_n=5,
    )


def analyze_high_ai_danger(df: pd.DataFrame):
    """高AI(≥0.8)が feature_route と AI_route で危険かどうか確認"""
    print(f"\n{'='*60}")
    print("  7. 高AI帯(0.8-1.0)危険確認")
    print(f"{'='*60}")

    high_ai = df[df["_ai_bin"].astype(str) == "0.8-1.0"].copy()
    print(f"  AI_Prob_Win 0.8以上 DONE件数: {len(high_ai)}")

    if len(high_ai) < MIN_SAMPLES:
        print("  → サンプル不足")
        return

    print_table(
        "  7a. 高AI × route × Direction",
        high_ai,
        ["_route", "Direction"],
        min_n=5,
    )

    # feature_route 内で高AIが危険かどうか
    feat_high = high_ai[high_ai["_route"] == "feature_route"]
    feat_all  = df[df["_route"] == "feature_route"]
    if len(feat_all) >= MIN_SAMPLES:
        s_high = _stats(feat_high)
        s_all  = _stats(feat_all)
        print(f"\n  feature_route 全体:  {_fmt(s_all)}")
        print(f"  feature_route 高AI:  {_fmt(s_high)}")
        if not np.isnan(s_high["wr"]) and not np.isnan(s_all["wr"]):
            diff = s_high["wr"] - s_all["wr"]
            flag = " → 高AIは危険" if diff < -5 else (" → 高AIでも変わらない" if abs(diff) < 5 else " → 高AIの方が良い")
            print(f"  差(高AI - 全体): {diff:+.1f}%{flag}")


# ==========================================
# 総合サマリー
# ==========================================

def summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  総合サマリー")
    print(f"{'='*60}")

    route_counts = df["_route"].value_counts()
    print("\n  route別件数:")
    for route, cnt in route_counts.items():
        print(f"    {route}: {cnt}件")

    live_cnt = int(df["_is_live"].sum())
    print(f"\n  実通知(D): {live_cnt}件 / DONE全体: {len(df)}件")

    # 各routeの勝率サマリー
    print("\n  route別勝率サマリー（n≥15のみ）:")
    for route in ["feature_route", "AI_route_SHORT", "AI_route_LONG", "no_match", "rule_rank"]:
        sub = df[df["_route"] == route]
        s = _stats(sub)
        if s["n"] >= MIN_SAMPLES:
            label = f"  {route:30s}"
            print(f"{label}{_fmt(s)}")

    # 判断テーブル
    print(f"\n  判断テーブル:")
    feat = df[df["_route"] == "feature_route"]
    ai_short = df[df["_route"] == "AI_route_SHORT"]
    ai_long = df[df["_route"] == "AI_route_LONG"]

    for label, sub in [("feature_route", feat), ("AI_route_SHORT", ai_short), ("AI_route_LONG", ai_long)]:
        s = _stats(sub)
        if s["n"] < MIN_SAMPLES:
            print(f"  {label}: サンプル不足（n={s['n']}）→ 判断保留")
            continue
        wr = s["wr"]
        if np.isnan(wr):
            continue
        if wr >= 55:
            action = "✅ 継続"
        elif wr >= 48:
            action = "△ 観察継続"
        else:
            action = "❌ shadow only 降格候補"
        print(f"  {label:30s} wr={wr:.1f}% → {action}")


# ==========================================
# メイン
# ==========================================

def main():
    print("=" * 60)
    print("edge_route_cross_analysis — route×Direction×AI帯×WinLose")
    print("=" * 60)

    try:
        df = fetch_done()
    except Exception as e:
        print(f"[FATAL] データ取得失敗: {e}")
        return 1

    if len(df) < MIN_SAMPLES:
        print(f"[ERROR] DONE件数 {len(df)} < {MIN_SAMPLES}")
        return 1

    # 前処理
    df["_win"] = _to_win(df["WinLose"]) if "WinLose" in df.columns else np.nan
    df["_pnl"] = _to_num(df["PnL_Pct"]) if "PnL_Pct" in df.columns else np.nan
    df["_route"] = classify_route(df)
    df["_is_live"] = is_live_notify(df)
    df["_ai_bin"] = ai_prob_bin(df["AI_Prob_Win"]) if "AI_Prob_Win" in df.columns else np.nan

    # Direction 正規化
    if "Direction" in df.columns:
        df["Direction"] = df["Direction"].astype(str).str.strip().str.upper()

    print(f"\n使用列: {[c for c in ['Direction','AI_Model_Type','AI_Band','AI_Note','AI_Prob_Win','WinLose','PnL_Pct','EvalStatus'] if c in df.columns]}")
    print(f"DONE: {len(df)}件 / 実通知: {int(df['_is_live'].sum())}件")

    # 分析実行
    analyze_route_direction(df)
    analyze_route_direction_ai_bin(df)
    analyze_feature_route_ai_bin(df)
    analyze_ai_route_ai_bin(df)
    analyze_no_match(df)
    analyze_live_notify(df)
    analyze_high_ai_danger(df)
    summary(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
