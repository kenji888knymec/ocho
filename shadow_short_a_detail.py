"""
shadow_short_a_detail.py — short_a 260件の詳細調査

目的:
  AI_Note で feature_bypass:short_a と判定された行260件について、
  実際の発火条件（RSI, mode, AI_Prob_Win, VolRatio, p2 等）を
  実データから確認し、現行の本番条件と shadow分析条件のどちらに
  合致しているかを特定する。

仮説:
  現行本番: side=SHORT;mode=RANGE;rsi_min=35;rsi_max=45;ltf_aligned=0
  shadow版: side=SHORT;mode=RANGE;rsi_min=50;rsi_max=55;p2_min=0;volratio_max=1.2;ai_max=0.6

  この2条件は重ならないため、260件のRSI分布等で識別可能。

実行: python shadow_short_a_detail.py
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

    status = df_all["EvalStatus"].astype(str).str.strip().str.upper()
    df_done = df_all[status == "DONE"].copy()
    print(f"[INFO] DONE件数: {len(df_done)}", flush=True)
    return df_done


# ==========================================
# helpers
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


def _bin_count(series: pd.Series, bins: list, labels: list) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "  (データなし)"
    cuts = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    counts = cuts.value_counts().sort_index()
    out = []
    for lbl, cnt in counts.items():
        pct = cnt / len(s) * 100
        out.append(f"    {lbl}: {cnt:4d}件 ({pct:5.1f}%)")
    return "\n".join(out)


# ==========================================
# main
# ==========================================

def main():
    print("=== short_a 260件 詳細調査 ===\n")
    print("【仮説】")
    print("  本番版: rsi_min=35, rsi_max=45, ltf_aligned=0")
    print("  shadow版: rsi_min=50, rsi_max=55, p2_min=0, volratio_max=1.2, ai_max=0.6")
    print("  → RSI分布で識別可能\n")

    df_raw = fetch_done()

    # 利用可能列の確認
    print(f"[INFO] 全カラム数: {len(df_raw.columns)}")
    cols_lower = {c.lower(): c for c in df_raw.columns}

    def find(*names):
        for n in names:
            if n in df_raw.columns:
                return n
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    note_col = find("AI_Note", "ai_note", "Note")
    if not note_col:
        raise RuntimeError("AI_Note 列が見つかりません")

    # short_a行の抽出（AI_Noteで判定）
    note = df_raw[note_col].astype(str)
    note_lower = note.str.lower()

    # \bshort_a\b で正確に判定
    short_a_mask = note_lower.str.contains(r"\bshort_a\b", regex=True, na=False)
    df = df_raw[short_a_mask].copy()
    print(f"[INFO] short_a 該当行: {len(df)}件\n")

    if len(df) == 0:
        print("short_a行が見つかりませんでした")
        return

    # 各種列を取得
    df["_note"] = df[note_col].astype(str)
    df["_note_lower"] = df["_note"].str.lower()
    df["_dir"] = df["Direction"].astype(str).str.upper() if "Direction" in df.columns else ""

    rsi_col   = find("RSI", "rsi")
    vol_col   = find("VolRatio", "volratio", "vol_ratio")
    p2_col    = find("P2_FundingScore", "p2", "P2", "P2_Score")
    ai_col    = find("AI_Prob_Win", "ai_prob_win", "AI_Prob")
    btc_col   = find("BTC_Mode_Compat", "BTC_Mode", "BTC_HTF_Mode")
    pnl_col   = find("PnL_Pct", "pnl_pct", "PnL")
    win_col   = find("WinLose", "winlose", "win")
    dt_col    = find("Datetime_JST", "datetime_jst", "Datetime")
    ver_col   = find("AI_Model_Version", "ai_model_version")
    ltf_col   = find("LTF_Aligned", "ltf_aligned", "LTF")

    df["_rsi"]   = _to_num(df[rsi_col])   if rsi_col   else np.nan
    df["_vol"]   = _to_num(df[vol_col])   if vol_col   else np.nan
    df["_p2"]    = _to_num(df[p2_col])    if p2_col    else np.nan
    df["_ai"]    = _to_num(df[ai_col])    if ai_col    else np.nan
    df["_pnl"]   = _to_num(df[pnl_col])   if pnl_col   else np.nan
    df["_win"]   = _to_win(df[win_col])   if win_col   else np.nan
    df["_btc"]   = df[btc_col].astype(str).str.upper() if btc_col else ""
    df["_dt"]    = pd.to_datetime(df[dt_col], errors="coerce") if dt_col else pd.NaT
    df["_date"]  = df["_dt"].dt.date if dt_col else None
    df["_ver"]   = df[ver_col].astype(str) if ver_col else ""
    df["_ltf"]   = df[ltf_col].astype(str) if ltf_col else ""

    # AI_Note からの抽出（notify_sent / selected / rank_exceeded / feature_probe_only / feature_bypass profile）
    df["_notify_sent"]        = df["_note_lower"].str.contains(r"notify_sent\s*=\s*1", regex=True, na=False)
    df["_selected_true"]      = df["_note_lower"].str.contains(r"selected\s*=\s*true",  regex=True, na=False)
    df["_selected_false"]     = df["_note_lower"].str.contains(r"selected\s*=\s*false", regex=True, na=False)
    df["_rank_exceeded"]      = df["_note_lower"].str.contains(r"rank_exceeded\s*=\s*true|rank_exceeded\s*=\s*1", regex=True, na=False)
    df["_feature_probe_only"] = df["_note_lower"].str.contains(r"feature_probe_only\s*=\s*true|feature_probe_only\s*=\s*1", regex=True, na=False)
    df["_feature_bypass"]     = df["_note_lower"].str.contains(r"feature_bypass", regex=True, na=False)
    df["_rejected"]           = df["_note_lower"].str.contains(r"rejected", regex=True, na=False)

    # 列存在報告
    available = []
    for nm, col in [("RSI", rsi_col), ("VolRatio", vol_col), ("P2", p2_col), ("AI", ai_col),
                    ("BTC_Mode", btc_col), ("PnL", pnl_col), ("Win", win_col), ("Date", dt_col),
                    ("Version", ver_col), ("LTF", ltf_col)]:
        if col:
            available.append(f"{nm}={col}")
    print(f"[INFO] 列マッピング: {', '.join(available)}\n")

    # ==========================================
    # ① 日付分布
    # ==========================================
    print("="*70)
    print("  ① 日付分布")
    print("="*70)
    if df["_date"].notna().any():
        date_counts = df["_date"].value_counts().sort_index()
        print(f"  期間: {df['_dt'].min()} ~ {df['_dt'].max()}")
        print(f"  ユニーク日数: {df['_date'].nunique()}")
        for d, cnt in date_counts.items():
            sub = df[df["_date"] == d]
            wins = sub["_win"].dropna()
            wr = f"{wins.mean()*100:.1f}%" if len(wins) > 0 else "N/A"
            pnl = sub["_pnl"].dropna()
            avg = f"{pnl.mean():+.3f}" if len(pnl) > 0 else "N/A"
            print(f"  {d}  n={cnt:4d}  WR={wr:>6s}  avg={avg}")

    # ==========================================
    # ② RSI分布（最重要：仮説検証）
    # ==========================================
    print("\n" + "="*70)
    print("  ② RSI分布（仮説検証の核）")
    print("="*70)
    if df["_rsi"].notna().any():
        rsi_valid = df["_rsi"].dropna()
        print(f"  RSI有効件数: {len(rsi_valid)} / {len(df)}")
        print(f"  最小: {rsi_valid.min():.2f}  最大: {rsi_valid.max():.2f}")
        print(f"  平均: {rsi_valid.mean():.2f}  中央値: {rsi_valid.median():.2f}")
        print()
        # 仮説検証用の細分割
        bins   = [0, 30, 35, 40, 45, 50, 55, 60, 65, 100]
        labels = ["<30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", ">=65"]
        print("  【RSI帯別件数】")
        print(_bin_count(df["_rsi"], bins, labels))
        print()
        # 仮説帯
        in_new   = ((df["_rsi"] >= 35) & (df["_rsi"] < 45)).sum()
        in_old   = ((df["_rsi"] >= 50) & (df["_rsi"] < 55)).sum()
        outside  = len(rsi_valid) - in_new - in_old
        print(f"  → 本番版範囲 [35-45): {in_new}件")
        print(f"  → shadow版範囲 [50-55): {in_old}件")
        print(f"  → どちらでもない範囲: {outside}件")
        if in_new > in_old:
            print(f"  ★ 結論: 本番版条件 [35-45) で発火している可能性が高い")
        elif in_old > in_new:
            print(f"  ★ 結論: shadow版条件 [50-55) で発火している可能性が高い")
        else:
            print(f"  ★ 結論: 不明（混在または別条件の可能性）")
    else:
        print("  RSI列が無効（データなし）")

    # ==========================================
    # ③ BTC Mode 分布
    # ==========================================
    print("\n" + "="*70)
    print("  ③ BTC Mode 分布")
    print("="*70)
    if df["_btc"].astype(str).ne("").any():
        mode_counts = df["_btc"].value_counts()
        for m, c in mode_counts.head(10).items():
            print(f"  {m}: {c}件 ({c/len(df)*100:.1f}%)")
    else:
        print("  BTC_Mode列が空")

    # ==========================================
    # ④ AI_Prob_Win 分布
    # ==========================================
    print("\n" + "="*70)
    print("  ④ AI_Prob_Win 分布")
    print("="*70)
    if df["_ai"].notna().any():
        ai_valid = df["_ai"].dropna()
        print(f"  AI有効件数: {len(ai_valid)} / {len(df)}")
        print(f"  平均: {ai_valid.mean():.3f}  中央値: {ai_valid.median():.3f}")
        bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
        labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        print(_bin_count(df["_ai"], bins, labels))
        n_lt06 = (df["_ai"] < 0.6).sum()
        n_ge06 = (df["_ai"] >= 0.6).sum()
        print(f"  → ai<0.6: {n_lt06}件  ai>=0.6: {n_ge06}件")
        if n_ge06 > 10:
            print(f"  ★ ai_max=0.6 制約は適用されていない可能性")
    else:
        print("  AI_Prob_Win列が無効")

    # ==========================================
    # ⑤ VolRatio 分布
    # ==========================================
    print("\n" + "="*70)
    print("  ⑤ VolRatio 分布")
    print("="*70)
    if df["_vol"].notna().any():
        vol_valid = df["_vol"].dropna()
        print(f"  Vol有効件数: {len(vol_valid)} / {len(df)}")
        print(f"  平均: {vol_valid.mean():.3f}  中央値: {vol_valid.median():.3f}")
        bins = [0, 0.5, 1.0, 1.2, 1.5, 2.0, 100]
        labels = ["<0.5", "0.5-1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", ">=2.0"]
        print(_bin_count(df["_vol"], bins, labels))
        n_lt12 = (df["_vol"] < 1.2).sum()
        n_ge12 = (df["_vol"] >= 1.2).sum()
        print(f"  → vol<1.2: {n_lt12}件  vol>=1.2: {n_ge12}件")
        if n_ge12 > 10:
            print(f"  ★ volratio_max=1.2 制約は適用されていない可能性")

    # ==========================================
    # ⑥ P2 分布
    # ==========================================
    print("\n" + "="*70)
    print("  ⑥ P2_FundingScore 分布")
    print("="*70)
    if df["_p2"].notna().any():
        p2_valid = df["_p2"].dropna()
        print(f"  P2有効件数: {len(p2_valid)} / {len(df)}")
        print(f"  平均: {p2_valid.mean():.3f}  中央値: {p2_valid.median():.3f}")
        n_neg = (df["_p2"] < 0).sum()
        n_pos = (df["_p2"] >= 0).sum()
        print(f"  → p2<0: {n_neg}件  p2>=0: {n_pos}件")

    # ==========================================
    # ⑦ AI_Note フラグ分布
    # ==========================================
    print("\n" + "="*70)
    print("  ⑦ AI_Note フラグ分布")
    print("="*70)
    flags = [
        ("notify_sent=1",        df["_notify_sent"]),
        ("selected=true",        df["_selected_true"]),
        ("selected=false",       df["_selected_false"]),
        ("rank_exceeded",        df["_rank_exceeded"]),
        ("feature_probe_only",   df["_feature_probe_only"]),
        ("feature_bypass",       df["_feature_bypass"]),
        ("rejected",             df["_rejected"]),
    ]
    for name, flag in flags:
        n_true = int(flag.sum())
        wins = df.loc[flag, "_win"].dropna()
        wr = f"{wins.mean()*100:.1f}%" if len(wins) > 0 else "N/A"
        pnl = df.loc[flag, "_pnl"].dropna()
        avg = f"{pnl.mean():+.3f}" if len(pnl) > 0 else "N/A"
        print(f"  {name:25s}: {n_true:4d}件  WR={wr:>6s}  avg={avg}")

    # ==========================================
    # ⑧ AI_Model_Version 分布
    # ==========================================
    print("\n" + "="*70)
    print("  ⑧ AI_Model_Version 分布（short_a行のみ）")
    print("="*70)
    if df["_ver"].astype(str).ne("").any():
        ver_counts = df["_ver"].value_counts()
        for v, c in ver_counts.head(10).items():
            sub = df[df["_ver"] == v]
            wins = sub["_win"].dropna()
            wr = f"{wins.mean()*100:.1f}%" if len(wins) > 0 else "N/A"
            print(f"  {v if v else '(空白)'}: {c}件  WR={wr}")

    # ==========================================
    # ⑨ ltf_aligned の確認
    # ==========================================
    print("\n" + "="*70)
    print("  ⑨ ltf_aligned 関連")
    print("="*70)
    if df["_ltf"].astype(str).ne("").any():
        ltf_counts = df["_ltf"].value_counts()
        for v, c in ltf_counts.head(10).items():
            print(f"  {v if v else '(空白)'}: {c}件")
    else:
        print("  LTF列なし。AI_Note内のltf_aligned=0/1を検索:")
        ltf0 = df["_note_lower"].str.contains(r"ltf_aligned\s*=\s*0", regex=True, na=False).sum()
        ltf1 = df["_note_lower"].str.contains(r"ltf_aligned\s*=\s*1", regex=True, na=False).sum()
        print(f"    ltf_aligned=0: {ltf0}件")
        print(f"    ltf_aligned=1: {ltf1}件")

    # ==========================================
    # ⑩ AI_Note サンプル（最初の10件）
    # ==========================================
    print("\n" + "="*70)
    print("  ⑩ AI_Note サンプル（先頭10件、各300字）")
    print("="*70)
    for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
        date = row["_date"] if pd.notna(row.get("_date", None)) else "?"
        rsi  = f"{row['_rsi']:.1f}" if pd.notna(row.get("_rsi")) else "?"
        ai   = f"{row['_ai']:.3f}" if pd.notna(row.get("_ai")) else "?"
        win  = "WIN" if row["_win"] == 1.0 else ("LOSE" if row["_win"] == 0.0 else "?")
        print(f"\n  [{idx:2d}] {date}  RSI={rsi}  AI={ai}  result={win}")
        note_text = str(row["_note"])[:300]
        print(f"      note: {note_text}")

    # ==========================================
    # ⑪ 結論サマリー
    # ==========================================
    print("\n" + "="*70)
    print("  ⑪ 条件適合判定サマリー")
    print("="*70)
    if df["_rsi"].notna().any():
        in_new   = ((df["_rsi"] >= 35) & (df["_rsi"] < 45)).sum()
        in_old   = ((df["_rsi"] >= 50) & (df["_rsi"] < 55)).sum()
        total_rsi = df["_rsi"].notna().sum()
        print(f"  RSI [35-45): {in_new}/{total_rsi} ({in_new/total_rsi*100:.1f}%)")
        print(f"  RSI [50-55): {in_old}/{total_rsi} ({in_old/total_rsi*100:.1f}%)")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
