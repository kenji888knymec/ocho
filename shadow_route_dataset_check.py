"""
shadow_route_dataset_check.py — 4レーン学習前の最終データセット整理

目的:
  1. feature_profile を profile別に分解
     short_a / short_b / short_c / long_d / long_e / long_f
  2. 4レーン既知モデルの全期間件数 vs 直近30日件数
     long_trend / long_recovery / short_trend / short_retrace
  3. legacy系（LONG_QS_v2 / QS_v1 / 空白version / 古いv2）の件数と期間範囲

実行: python shadow_route_dataset_check.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
"""

import os
import re
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

# 既知の4レーンモデル
LONG_TREND_MODEL    = os.environ.get("LONG_TREND_MODEL_VERSION",    "v2_long_20260415_081606")
LONG_RECOVERY_MODEL = os.environ.get("LONG_RECOVERY_MODEL_VERSION", "v2_long_20260414_153708")
SHORT_TREND_MODEL   = os.environ.get("SHORT_TREND_MODEL_VERSION",   "v2_short_20260416_082845")
LONG_COMMON_MODEL   = os.environ.get("LONG_MODEL_VERSION",          "v2_long_20260413_212814")
SHORT_COMMON_MODEL  = os.environ.get("SHORT_MODEL_VERSION",         "v2_short_20260413_084558")

KNOWN_4LANE = {
    LONG_TREND_MODEL:    "long_trend",
    LONG_RECOVERY_MODEL: "long_recovery",
    SHORT_TREND_MODEL:   "short_trend",
    SHORT_COMMON_MODEL:  "short_retrace",
    LONG_COMMON_MODEL:   "long_trend_common",
}

# 検出するprofile
FEATURE_PROFILES = ["short_a", "short_b", "short_c", "long_d", "long_e", "long_f"]


# ==========================================
# Google Sheets
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


def _find_col(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "Datetime_JST" not in df.columns:
        raise RuntimeError("Datetime_JST 列が見つかりません")
    df["_dt"] = pd.to_datetime(df["Datetime_JST"], errors="coerce", utc=False)
    df["_date"] = df["_dt"].dt.date

    df["_win"] = _to_win(df["WinLose"]) if "WinLose" in df.columns else np.nan
    df["_pnl"] = _to_num(df["PnL_Pct"]) if "PnL_Pct" in df.columns else np.nan
    df["_dir"] = df["Direction"].astype(str).str.strip().str.upper() if "Direction" in df.columns else ""

    ver_col = _find_col(df, ["AI_Model_Version", "ai_model_version", "AI_Model_Ver"])
    df["_ver"] = df[ver_col].astype(str).str.strip() if ver_col else ""

    note_col = _find_col(df, ["AI_Note", "ai_note", "Note"])
    df["_note"] = df[note_col].astype(str) if note_col else ""

    # action / route系
    act_col = _find_col(df, ["Action", "action", "EntryRoute", "entry_route", "Route"])
    df["_action"] = df[act_col].astype(str) if act_col else ""

    # selected / rank系
    sel_col = _find_col(df, ["selected", "Selected", "is_selected"])
    df["_selected"] = df[sel_col].astype(str).str.lower() if sel_col else ""

    rank_col = _find_col(df, ["rank_exceeded", "RankExceeded", "rank"])
    df["_rank"] = df[rank_col].astype(str).str.lower() if rank_col else ""

    # notify_sent系
    notify_col = _find_col(df, ["notify_sent", "_notify_sent", "NotifySent"])
    if notify_col:
        df["_notify"] = df[notify_col].astype(str).str.strip() == "1"
    else:
        # AI_Note からフォールバック推定
        note_lower = df["_note"].str.lower()
        df["_notify"] = note_lower.str.contains("notify_sent=1", na=False)

    return df


# ==========================================
# レーン / profile 判定
# ==========================================

def classify_4lane(row: pd.Series) -> str:
    v = row["_ver"]
    d = row["_dir"]
    if v == LONG_TREND_MODEL:
        return "long_trend"
    if v == LONG_RECOVERY_MODEL:
        return "long_recovery"
    if v == SHORT_TREND_MODEL:
        return "short_trend"
    if v == SHORT_COMMON_MODEL and d == "SHORT":
        return "short_retrace"
    if v == LONG_COMMON_MODEL and d == "LONG":
        return "long_trend"
    return None


def classify_legacy(row: pd.Series) -> str:
    v = row["_ver"]
    if v == "LONG_QS_v2":
        return "legacy_long_qs"
    if v == "QS_v1":
        return "legacy_short_qs"
    if v in ("", "nan", "None", "none"):
        return "legacy_blank"
    if v.startswith("v2_long_20260325") or v.startswith("v2_long_2026032") or v.startswith("v2_long_2026031"):
        return "legacy_old_long"
    if v.startswith("v2_short_20260326") or v.startswith("v2_short_20260407") or v.startswith("v2_short_2026032"):
        return "legacy_old_short"
    return None


def classify_feature_profile(row: pd.Series) -> str:
    """
    AI_Note または Action から profile を抽出。
    feature_profile行のみ対象。
    """
    note = row["_note"].lower()
    action = row["_action"].lower()
    text = f"{note} {action}"

    for prof in FEATURE_PROFILES:
        # short_a / long_e のようなトークンが入っているか
        if re.search(rf"\b{prof}\b", text):
            return prof
    return None


def is_feature_profile_row(row: pd.Series) -> bool:
    action = row["_action"].upper()
    note = row["_note"].lower()
    return ("FEATURE_BYPASS" in action
            or "FEATURE_PROFILE" in action
            or "feature_bypass" in note
            or "feature_profile" in note)


# ==========================================
# 集計
# ==========================================

def _stats(sub: pd.DataFrame, all_days: set = None) -> dict:
    win = sub["_win"].dropna()
    pnl = sub["_pnl"].dropna()
    n = len(win)
    out = {
        "rows": len(sub),
        "n_win": n,
        "wr": float(win.mean()) if n > 0 else np.nan,
        "pnl_mean": float(pnl.mean()) if len(pnl) > 0 else np.nan,
        "pnl_med":  float(pnl.median()) if len(pnl) > 0 else np.nan,
        "notify1": int(sub["_notify"].sum()),
    }
    dates = set(sub["_date"].dropna())
    out["active_days"] = len(dates)
    if all_days is not None:
        out["zero_days"] = len(all_days) - len(dates)
    if len(sub) > 0 and sub["_dt"].notna().any():
        out["dt_min"] = sub["_dt"].min()
        out["dt_max"] = sub["_dt"].max()
    else:
        out["dt_min"] = None
        out["dt_max"] = None
    return out


def _fmt_stats(s: dict, with_zero=True) -> str:
    rows = s["rows"]
    if rows == 0:
        return f"rows={rows:6d}  (データなし)"
    wr = f"{s['wr']*100:.1f}%" if np.isfinite(s["wr"]) else " N/A "
    pm = f"{s['pnl_mean']:+.3f}" if np.isfinite(s["pnl_mean"]) else " N/A "
    md = f"{s['pnl_med']:+.3f}" if np.isfinite(s["pnl_med"]) else " N/A "
    out = f"rows={rows:6d}  n={s['n_win']:5d}  wr={wr}  avg={pm}  med={md}  notify1={s['notify1']:4d}"
    if with_zero and "zero_days" in s:
        out += f"  active={s['active_days']}日 zero={s['zero_days']}日"
    else:
        out += f"  active={s['active_days']}日"
    return out


def _fmt_period(s: dict) -> str:
    if s["dt_min"] is None or s["dt_max"] is None:
        return "期間不明"
    return f"{s['dt_min'].date()} ~ {s['dt_max'].date()}"


# ==========================================
# レポート
# ==========================================

def report_feature_profiles(df: pd.DataFrame, all_days: set):
    print(f"\n{'='*75}")
    print("  ① feature_profile別 分解（直近30日 DONE）")
    print(f"{'='*75}")

    fp_mask = df.apply(is_feature_profile_row, axis=1)
    df_fp = df[fp_mask].copy()
    print(f"\n  feature_profile行 総数: {len(df_fp)}件")

    if len(df_fp) == 0:
        print("  (feature_profile行が見つかりませんでした)")
        print("  ※ Action/AI_Note内に FEATURE_BYPASS/feature_profile が含まれる行を検出")
        return

    df_fp["_profile"] = df_fp.apply(classify_feature_profile, axis=1)

    # profile別
    for prof in FEATURE_PROFILES:
        sub = df_fp[df_fp["_profile"] == prof]
        s = _stats(sub, all_days)
        if sub.empty:
            print(f"\n  [{prof:>8s}] {_fmt_stats(s)}")
            continue
        print(f"\n  [{prof:>8s}] {_fmt_stats(s)}")
        print(f"           期間: {_fmt_period(s)}")

        # selected / rank_exceeded 内訳
        sel_counts = sub["_selected"].value_counts().head(5)
        if len(sel_counts) > 0 and sel_counts.iloc[0] != "":
            sel_str = ", ".join([f"{k or '空'}={v}" for k, v in sel_counts.items()])
            print(f"           selected: {sel_str}")
        rank_counts = sub["_rank"].value_counts().head(5)
        if len(rank_counts) > 0 and rank_counts.iloc[0] != "":
            rk_str = ", ".join([f"{k or '空'}={v}" for k, v in rank_counts.items()])
            print(f"           rank: {rk_str}")

    # profile未判定
    unmatched = df_fp[df_fp["_profile"].isna()]
    if len(unmatched) > 0:
        s = _stats(unmatched, all_days)
        print(f"\n  [profile不明] {_fmt_stats(s)}")
        # サンプルAI_Note
        sample_notes = unmatched["_note"].dropna().head(3).tolist()
        for note in sample_notes:
            print(f"           note例: {note[:120]}")


def report_4lane_known(df_full: pd.DataFrame, df_recent: pd.DataFrame, all_days_recent: set):
    print(f"\n{'='*75}")
    print("  ② 4レーン既知モデル 全期間 vs 直近30日")
    print(f"{'='*75}")

    for ver, lane in KNOWN_4LANE.items():
        sub_full = df_full[df_full["_ver"] == ver]
        # short_retrace は SHORT方向のみ、long_trend_common は LONG方向のみ
        if lane == "short_retrace":
            sub_full = sub_full[sub_full["_dir"] == "SHORT"]
        if lane == "long_trend_common":
            sub_full = sub_full[sub_full["_dir"] == "LONG"]

        sub_recent = df_recent[df_recent["_ver"] == ver]
        if lane == "short_retrace":
            sub_recent = sub_recent[sub_recent["_dir"] == "SHORT"]
        if lane == "long_trend_common":
            sub_recent = sub_recent[sub_recent["_dir"] == "LONG"]

        s_full = _stats(sub_full)
        s_recent = _stats(sub_recent, all_days_recent)

        print(f"\n  [{lane:>20s}]  version: {ver}")
        print(f"    全期間  : {_fmt_stats(s_full, with_zero=False)}")
        print(f"             期間: {_fmt_period(s_full)}")
        print(f"    直近30日: {_fmt_stats(s_recent)}")


def report_legacy(df_full: pd.DataFrame):
    print(f"\n{'='*75}")
    print("  ③ legacy系データ（4レーン学習から除外推奨）全期間")
    print(f"{'='*75}")

    df = df_full.copy()
    df["_legacy"] = df.apply(classify_legacy, axis=1)
    legacy_counts = df["_legacy"].value_counts(dropna=True)

    for legacy_label, _ in legacy_counts.items():
        sub = df[df["_legacy"] == legacy_label]
        s = _stats(sub)
        print(f"\n  [{legacy_label:>20s}] {_fmt_stats(s, with_zero=False)}")
        print(f"                    期間: {_fmt_period(s)}")
        # 中のversion内訳
        ver_counts = sub["_ver"].value_counts().head(5)
        if len(ver_counts) > 1 or (len(ver_counts) == 1 and ver_counts.index[0] != ""):
            print(f"                    主version:")
            for v, c in ver_counts.items():
                print(f"                      {v if v else '(空白)'}: {c}件")


def report_summary(df_full: pd.DataFrame, df_recent: pd.DataFrame):
    print(f"\n{'='*75}")
    print("  ④ サマリー（4レーン学習データセット候補）")
    print(f"{'='*75}")

    # 4レーン既知に含まれる行（全期間）
    known_versions = set(KNOWN_4LANE.keys())
    known_mask_full = df_full["_ver"].isin(known_versions)
    known_full = df_full[known_mask_full]

    # legacy
    df_full2 = df_full.copy()
    df_full2["_legacy"] = df_full2.apply(classify_legacy, axis=1)
    legacy_full = df_full2[df_full2["_legacy"].notna()]

    # feature_profile
    fp_mask = df_full.apply(is_feature_profile_row, axis=1)
    fp_full = df_full[fp_mask]

    # その他
    other = df_full[~known_mask_full & ~df_full2["_legacy"].notna() & ~fp_mask]

    total = len(df_full)
    print(f"\n  全期間DONE: {total:,}件")
    print(f"    4レーン既知モデル   : {len(known_full):6,}件 ({len(known_full)/total*100:5.1f}%) ★学習主データ候補")
    print(f"    feature_profile    : {len(fp_full):6,}件 ({len(fp_full)/total*100:5.1f}%) ★profile別評価")
    print(f"    legacy系          : {len(legacy_full):6,}件 ({len(legacy_full)/total*100:5.1f}%) ★除外推奨")
    print(f"    その他            : {len(other):6,}件 ({len(other)/total*100:5.1f}%) ※要確認")

    if len(other) > 0:
        print(f"\n  ※その他のversion上位5:")
        for v, c in other["_ver"].value_counts().head(5).items():
            print(f"      {v if v else '(空白)'}: {c}件")


# ==========================================
# メイン
# ==========================================

def main():
    print("=== shadow_route_dataset_check.py ===")
    print("目的: 4レーン学習前の最終データセット整理")
    print("  ① feature_profile を profile別に分解（直近30日）")
    print("  ② 4レーン既知モデルの全期間 vs 直近30日")
    print("  ③ legacy系の件数と期間範囲（全期間）\n")

    df_raw = fetch_done()
    df = preprocess(df_raw)

    # 列状況
    key_cols = ["AI_Model_Version", "AI_Note", "WinLose", "PnL_Pct",
                "Direction", "Datetime_JST", "Action", "selected",
                "rank_exceeded", "notify_sent"]
    present = [c for c in key_cols if c in df_raw.columns]
    missing = [c for c in key_cols if c not in df_raw.columns]
    print(f"\n[INFO] 使用列: {present}")
    if missing:
        print(f"[WARN] 欠損列: {missing}")

    # 直近30日データ
    now_jst = datetime.now(JST)
    cutoff = (now_jst - timedelta(days=LOOKBACK_DAYS)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cutoff)].copy()
    all_days_recent = set(pd.date_range(cutoff, now_jst.date()).date)
    print(f"[INFO] 直近{LOOKBACK_DAYS}日 DONE: {len(df_recent):,}件 / 対象日数: {len(all_days_recent)}日")
    print(f"[INFO] 全期間 DONE: {len(df):,}件")

    # ① feature_profile別（直近30日）
    report_feature_profiles(df_recent, all_days_recent)

    # ② 4レーン既知モデル（全期間 vs 30日）
    report_4lane_known(df, df_recent, all_days_recent)

    # ③ legacy系（全期間）
    report_legacy(df)

    # ④ サマリー
    report_summary(df, df_recent)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
