"""
shadow_unknown_versions.py — 未分類行のAI_Model_Version調査
v2_shadow_aiのDONE行のうち、4レーンに分類されなかった行の
AI_Model_Versionを全て列挙し、正体を特定する。

実行: python shadow_unknown_versions.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
"""

import os
import warnings
from datetime import datetime, timedelta, timezone
from collections import Counter

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

# 既知の4レーンモデル
KNOWN_MODELS = {
    os.environ.get("LONG_TREND_MODEL_VERSION",    "v2_long_20260415_081606"):  "long_trend",
    os.environ.get("LONG_RECOVERY_MODEL_VERSION", "v2_long_20260414_153708"):  "long_recovery",
    os.environ.get("SHORT_TREND_MODEL_VERSION",   "v2_short_20260416_082845"): "short_trend",
    os.environ.get("SHORT_MODEL_VERSION",         "v2_short_20260413_084558"): "short_retrace(approx)",
    os.environ.get("LONG_MODEL_VERSION",          "v2_long_20260413_212814"):  "long_trend(common)",
}


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


def main():
    print("=== 未分類行 AI_Model_Version 調査 ===\n")

    df = fetch_done()

    # AI_Model_Version 列の特定
    ver_col = None
    for c in ("AI_Model_Version", "ai_model_version", "AI_Model_Ver"):
        if c in df.columns:
            ver_col = c
            break

    if ver_col is None:
        print("[ERROR] AI_Model_Version 列が見つかりません")
        print(f"全列: {list(df.columns)}")
        return

    print(f"[INFO] バージョン列: {ver_col}")

    # Direction
    dir_col = df["Direction"].astype(str).str.strip().str.upper() if "Direction" in df.columns else pd.Series([""] * len(df), index=df.index)

    # WinLose
    win_col = _to_win(df["WinLose"]) if "WinLose" in df.columns else pd.Series([np.nan] * len(df), index=df.index)

    # 全バージョン一覧
    ver = df[ver_col].astype(str).str.strip()
    ver_counts = ver.value_counts()

    print(f"\n{'='*70}")
    print(f"  全AI_Model_Version 一覧（DONE全件）")
    print(f"{'='*70}")
    print(f"  ユニーク数: {len(ver_counts)}種類")
    print()

    for v, cnt in ver_counts.items():
        label = KNOWN_MODELS.get(v, "★未分類★")
        mask = ver == v
        dirs = dir_col[mask].value_counts()
        dirs_str = "  ".join([f"{d}:{c}" for d, c in dirs.items()])

        # WR
        wins = win_col[mask].dropna()
        wr_str = f"WR={wins.mean()*100:.1f}%({len(wins)}件)" if len(wins) > 0 else "WR=N/A"

        print(f"  {v}")
        print(f"    件数: {cnt}  レーン: {label}  方向: {dirs_str}  {wr_str}")

    # 未分類のみ抽出
    unknown_mask = ~ver.isin(KNOWN_MODELS.keys())
    df_unk = df[unknown_mask].copy()
    print(f"\n{'='*70}")
    print(f"  未分類行の詳細（{len(df_unk)}件）")
    print(f"{'='*70}")

    unk_ver = df_unk[ver_col].astype(str).str.strip()
    unk_ver_counts = unk_ver.value_counts()
    print(f"  ユニーク数: {len(unk_ver_counts)}種類\n")

    for v, cnt in unk_ver_counts.items():
        mask = unk_ver == v
        dirs = dir_col[df_unk.index][mask].value_counts()
        dirs_str = "  ".join([f"{d}:{c}" for d, c in dirs.items()])

        wins = win_col[df_unk.index][mask].dropna()
        wr_str = f"WR={wins.mean()*100:.1f}%({len(wins)}件)" if len(wins) > 0 else "WR=N/A"

        # 日付範囲
        if "Datetime_JST" in df_unk.columns:
            dt = pd.to_datetime(df_unk["Datetime_JST"][mask.values], errors="coerce")
            dt_valid = dt.dropna()
            if len(dt_valid) > 0:
                date_range = f"{dt_valid.min().date()} ~ {dt_valid.max().date()}"
            else:
                date_range = "日付不明"
        else:
            date_range = "日付列なし"

        print(f"  [{cnt:6d}件] {v if v else '(空白)'}")
        print(f"           方向: {dirs_str if dirs_str else '不明'}  {wr_str}  期間: {date_range}")

    # 空白・null のみの行
    empty_mask = ver.isin(["", "nan", "None", "none"])
    n_empty = empty_mask.sum()
    if n_empty > 0:
        print(f"\n[INFO] AI_Model_Version が空白/null の行: {n_empty}件")

    # サマリー
    print(f"\n{'='*70}")
    print(f"  サマリー")
    print(f"{'='*70}")
    print(f"  DONE総件数: {len(df)}")
    print(f"  既知モデル分類済み: {(~unknown_mask).sum()}件  ({(~unknown_mask).sum()/len(df)*100:.1f}%)")
    print(f"  未分類: {len(df_unk)}件  ({len(df_unk)/len(df)*100:.1f}%)")
    print(f"  空白バージョン: {n_empty}件")

    # 未分類の中でDIRECTION別
    if len(df_unk) > 0:
        unk_dirs = dir_col[df_unk.index].value_counts()
        print(f"\n  未分類のDirection分布:")
        for d, c in unk_dirs.items():
            print(f"    {d}: {c}件")

        # 未分類の日付分布（月別）
        if "Datetime_JST" in df_unk.columns:
            dt_unk = pd.to_datetime(df_unk["Datetime_JST"], errors="coerce")
            dt_unk_valid = dt_unk.dropna()
            if len(dt_unk_valid) > 0:
                month_counts = dt_unk_valid.dt.to_period("M").value_counts().sort_index()
                print(f"\n  未分類の月別分布:")
                for m, c in month_counts.items():
                    print(f"    {m}: {c}件")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
