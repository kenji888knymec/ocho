"""
long_e_verify.py — LONG_E候補 詳細検証

候補条件: side=LONG;mode=UP;rsi_min=40;rsi_max=50;ai_max=0.2

確認項目:
  1. 全期間成績 (30日/20日/10日)
  2. 日付分布 — n=69がどの期間に集中しているか
  3. train/test 時系列分割 (古い60% = train / 新しい40% = test)
  4. LONG_D条件との重複行数
  5. active_days / zero_days
  6. AI_Prob_Win < 0.2 の分布 (いつ集中したか)

実行: python long_e_verify.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
オプション:
  LOOKBACK_DAYS   : 集計日数 (default: 30)
  TRAIN_RATIO     : train比率 (default: 0.6)
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
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "30"))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.6"))

# 検証候補条件
CAND_RSI_MIN = 40.0
CAND_RSI_MAX = 50.0
CAND_AI_MAX  = 0.2

# LONG_D現行条件 (重複確認用)
# side=LONG;mode=UP;p1_max=1.0;volratio_min=2.0;volconfirmed=1
LONGD_P1_MAX       = 1.0
LONGD_VOLRATIO_MIN = 2.0
LONGD_VOLCONFIRMED = 1


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


def _get_col(df: pd.DataFrame, *candidates) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([""] * len(df), index=df.index)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "Datetime_JST" not in df.columns:
        raise RuntimeError("Datetime_JST 列が見つかりません")
    df["_dt"]   = pd.to_datetime(df["Datetime_JST"], errors="coerce", utc=False)
    df["_date"] = df["_dt"].dt.date

    df["_dir"]      = _get_col(df, "Direction").astype(str).str.strip().str.upper()
    df["_btc_mode"] = _get_col(df, "BTC_Mode_Compat", "BTC_Mode", "BTCMode").astype(str).str.strip().str.upper()

    df["_win"] = _to_win(_get_col(df, "WinLose"))
    df["_pnl"] = _to_num(_get_col(df, "PnL_Pct"))

    df["_rsi"]  = _to_num(_get_col(df, "RSI"))
    df["_ai"]   = _to_num(_get_col(df, "AI_Prob_Win"))
    df["_p1"]   = _to_num(_get_col(df, "P1_TrendScore"))
    df["_volr"] = _to_num(_get_col(df, "VolRatio"))

    vc_raw = _get_col(df, "VolConfirmed")
    df["_volc"] = _to_num(vc_raw)

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


def _fmt(s: dict) -> str:
    n = s["n"]
    if n == 0:
        return f"n={n:4d}  (データなし)"
    wr  = f"{s['wr']*100:.1f}%" if np.isfinite(s.get("wr", float("nan"))) else "  N/A"
    pm  = f"{s['pnl_mean']:+.3f}" if np.isfinite(s.get("pnl_mean", float("nan"))) else "  N/A"
    md  = f"{s['pnl_med']:+.3f}" if np.isfinite(s.get("pnl_med", float("nan"))) else "  N/A"
    return f"n={n:4d}  wr={wr}  avg={pm}  med={md}"


# ==========================================
# フィルタ関数
# ==========================================

def apply_cand(df: pd.DataFrame) -> pd.DataFrame:
    """LONG_E候補条件でフィルタ"""
    mask = (
        (df["_dir"] == "LONG")
        & (df["_btc_mode"] == "UP")
        & df["_rsi"].notna() & (df["_rsi"] >= CAND_RSI_MIN) & (df["_rsi"] < CAND_RSI_MAX)
        & df["_ai"].notna()  & (df["_ai"] < CAND_AI_MAX)
    )
    return df[mask].copy()


def apply_longd(df: pd.DataFrame) -> pd.DataFrame:
    """LONG_D現行条件でフィルタ（重複確認用）"""
    # side=LONG;mode=UP;p1_max=1.0;volratio_min=2.0;volconfirmed=1
    mask = (
        (df["_dir"] == "LONG")
        & (df["_btc_mode"] == "UP")
        & df["_p1"].notna()  & (df["_p1"] <= LONGD_P1_MAX)
        & df["_volr"].notna() & (df["_volr"] >= LONGD_VOLRATIO_MIN)
        & df["_volc"].notna() & (df["_volc"] == LONGD_VOLCONFIRMED)
    )
    return df[mask].copy()


# ==========================================
# メイン分析
# ==========================================

def main():
    print("=== LONG_E候補 詳細検証 ===")
    print(f"候補条件: side=LONG;mode=UP;rsi_min={CAND_RSI_MIN};rsi_max={CAND_RSI_MAX};ai_max={CAND_AI_MAX}")
    print(f"対象: 直近{LOOKBACK_DAYS}日 EvalStatus=DONE")
    print(f"train/test分割: 時系列 先頭{TRAIN_RATIO*100:.0f}% = train / 残り = test\n")

    df_raw = fetch_done()
    df = preprocess(df_raw)

    # 日付フィルタ
    now_jst = datetime.now(JST)
    cutoff = (now_jst - timedelta(days=LOOKBACK_DAYS)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cutoff)].copy()
    all_days = set(pd.date_range(cutoff, now_jst.date()).date)
    print(f"[INFO] 直近{LOOKBACK_DAYS}日 全DONE: {len(df_recent)}件", flush=True)

    # 候補条件フィルタ
    df_cand = apply_cand(df_recent)
    print(f"[INFO] 候補DONE: {len(df_cand)}件\n", flush=True)

    # ==========================================
    # Section 1: 時間帯別成績 (30日/20日/10日)
    # ==========================================
    print(f"\n{'='*70}")
    print("  Section 1: 時間帯別成績")
    print(f"{'='*70}")
    for days in [30, 20, 10, 5]:
        cut = (now_jst - timedelta(days=days)).date()
        sub = df_cand[df_cand["_date"] >= cut]
        s = _stats(sub)
        print(f"  直近{days:2d}日: {_fmt(s)}")

    # ==========================================
    # Section 2: 日付分布 — n=? がどの期間に集中しているか
    # ==========================================
    print(f"\n{'='*70}")
    print("  Section 2: 日付分布（候補が出た日）")
    print(f"{'='*70}")
    if len(df_cand) == 0:
        print("  データなし")
    else:
        day_counts = df_cand.groupby("_date").size().sort_index()
        active_days = len(day_counts)
        zero_days = len(all_days) - active_days
        print(f"  active_days: {active_days}日 / {len(all_days)}日  (ゼロ日: {zero_days}日)")
        print(f"\n  日別件数（全日）:")
        for d, cnt in day_counts.items():
            g = df_cand[df_cand["_date"] == d]
            s = _stats(g)
            wr_s = f"{s['wr']*100:.0f}%" if np.isfinite(s.get("wr", float("nan"))) else "N/A"
            pm_s = f"{s['pnl_mean']:+.3f}" if np.isfinite(s.get("pnl_mean", float("nan"))) else "N/A"
            print(f"    {d}  n={cnt:3d}  wr={wr_s}  avg={pm_s}")

    # ==========================================
    # Section 3: train/test 時系列分割
    # ==========================================
    print(f"\n{'='*70}")
    print("  Section 3: train/test 時系列分割")
    print(f"{'='*70}")
    if len(df_cand) < 2:
        print("  データ不足 — 分割不可")
    else:
        df_sorted = df_cand.sort_values("_dt").reset_index(drop=True)
        n_total = len(df_sorted)
        n_train = max(1, int(n_total * TRAIN_RATIO))
        df_train = df_sorted.iloc[:n_train]
        df_test  = df_sorted.iloc[n_train:]

        s_train = _stats(df_train)
        s_test  = _stats(df_test)

        print(f"\n  train ({n_train}件, 古い{TRAIN_RATIO*100:.0f}%):")
        print(f"    {_fmt(s_train)}")
        if len(df_train) > 0:
            print(f"    期間: {df_train['_date'].min()} 〜 {df_train['_date'].max()}")

        print(f"\n  test  ({len(df_test)}件, 新しい{(1-TRAIN_RATIO)*100:.0f}%):")
        print(f"    {_fmt(s_test)}")
        if len(df_test) > 0:
            print(f"    期間: {df_test['_date'].min()} 〜 {df_test['_date'].max()}")

        if np.isfinite(s_train.get("wr", float("nan"))) and np.isfinite(s_test.get("wr", float("nan"))):
            diff = (s_test["wr"] - s_train["wr"]) * 100
            verdict = "崩れなし" if diff >= -10 else ("やや崩れ" if diff >= -20 else "崩れあり — 要注意")
            print(f"\n  train→test 勝率変化: {diff:+.1f}%  → {verdict}")

    # ==========================================
    # Section 4: LONG_D との重複
    # ==========================================
    print(f"\n{'='*70}")
    print("  Section 4: LONG_D現行条件との重複")
    print(f"{'='*70}")
    print(f"  LONG_D条件: side=LONG;mode=UP;p1_max={LONGD_P1_MAX};volratio_min={LONGD_VOLRATIO_MIN};volconfirmed={LONGD_VOLCONFIRMED}")

    df_longd = apply_longd(df_recent)
    s_ld = _stats(df_longd)
    print(f"\n  LONG_D全体: {_fmt(s_ld)}")

    # 重複：両方の条件を満たす行
    df_both = apply_cand(df_longd)
    n_both = len(df_both)
    n_cand_total = len(df_cand)
    pct = n_both / n_cand_total * 100 if n_cand_total > 0 else 0.0
    print(f"\n  重複行数: {n_both}件 / 候補{n_cand_total}件 ({pct:.1f}%)")
    if n_both > 0:
        s_both = _stats(df_both)
        print(f"  重複行成績: {_fmt(s_both)}")

    # 候補からLONG_D重複を除いた場合
    idx_both = df_both.index
    df_cand_only = df_cand[~df_cand.index.isin(idx_both)]
    s_only = _stats(df_cand_only)
    print(f"\n  重複除外後の候補のみ: {_fmt(s_only)}")

    # ==========================================
    # Section 5: AI_Prob_Win < 0.2 の分布詳細
    # ==========================================
    print(f"\n{'='*70}")
    print("  Section 5: AI_Prob_Win < 0.2 の詳細分布")
    print(f"{'='*70}")
    print("  ※ AIが低い(悲観)のに実際は勝つパターンの期間依存性を確認")

    # LONG×UP×RSI40-50 の中で AI帯別成績
    base_mask = (
        (df_recent["_dir"] == "LONG")
        & (df_recent["_btc_mode"] == "UP")
        & df_recent["_rsi"].notna()
        & (df_recent["_rsi"] >= CAND_RSI_MIN)
        & (df_recent["_rsi"] < CAND_RSI_MAX)
    )
    df_rsi_base = df_recent[base_mask].copy()
    print(f"\n  RSI {CAND_RSI_MIN}-{CAND_RSI_MAX} × UP 全体: n={len(df_rsi_base)}")

    ai_bins = [
        ("AI blank",   df_rsi_base["_ai"].isna()),
        ("AI 0.0-0.2", df_rsi_base["_ai"].notna() & (df_rsi_base["_ai"] < 0.2)),
        ("AI 0.2-0.4", (df_rsi_base["_ai"] >= 0.2) & (df_rsi_base["_ai"] < 0.4)),
        ("AI 0.4-0.6", (df_rsi_base["_ai"] >= 0.4) & (df_rsi_base["_ai"] < 0.6)),
        ("AI 0.6-0.8", (df_rsi_base["_ai"] >= 0.6) & (df_rsi_base["_ai"] < 0.8)),
        ("AI 0.8+",    df_rsi_base["_ai"] >= 0.8),
    ]
    for label, mask in ai_bins:
        g = df_rsi_base[mask]
        s = _stats(g)
        print(f"  {label:12s}: {_fmt(s)}")

    # AI<0.2 の月別集中確認
    if len(df_cand) > 0:
        print(f"\n  AI<{CAND_AI_MAX} 候補の月別分布:")
        df_cand2 = df_cand.copy()
        df_cand2["_ym"] = df_cand2["_dt"].dt.to_period("M").astype(str)
        for ym, g in df_cand2.groupby("_ym"):
            s = _stats(g)
            print(f"    {ym}: {_fmt(s)}")

    # ==========================================
    # Section 6: 現状のA候補推計（条件適合行がOPEN中に存在するか）
    # ==========================================
    print(f"\n{'='*70}")
    print("  Section 6: OPEN行の候補適合推計（A候補確認）")
    print(f"{'='*70}")
    df_open = df[df["EvalStatus"].astype(str).str.strip().str.upper() == "OPEN"].copy() if "EvalStatus" in df.columns else pd.DataFrame()
    if len(df_open) == 0:
        print("  OPEN行データなし（EvalStatus=OPEN の行が取得できていない可能性あり）")
    else:
        df_open_pp = preprocess(df_open) if "_dir" not in df_open.columns else df_open
        df_open_cand = apply_cand(df_open_pp)
        print(f"  現在OPEN中の候補条件適合行: {len(df_open_cand)}件")
        if len(df_open_cand) > 0:
            for _, row in df_open_cand.iterrows():
                sym  = row.get("Symbol", "?")
                dt_s = str(row.get("Datetime_JST", ""))[:16]
                rsi  = f"{row['_rsi']:.1f}" if pd.notna(row.get("_rsi")) else "N/A"
                ai   = f"{row['_ai']:.3f}" if pd.notna(row.get("_ai")) else "N/A"
                print(f"    {dt_s}  {sym}  RSI={rsi}  AI={ai}")

    # ==========================================
    # Section 7: 総合判定
    # ==========================================
    print(f"\n{'='*70}")
    print("  Section 7: 総合判定")
    print(f"{'='*70}")

    s_30 = _stats(df_cand)
    cut20 = (now_jst - timedelta(days=20)).date()
    cut10 = (now_jst - timedelta(days=10)).date()
    s_20 = _stats(df_cand[df_cand["_date"] >= cut20])
    s_10 = _stats(df_cand[df_cand["_date"] >= cut10])

    checks = []

    # 件数
    ok_n30 = s_30["n"] >= 10
    checks.append(("30日件数 >=10",   ok_n30,   f"n={s_30['n']}"))

    # 勝率
    ok_wr30 = np.isfinite(s_30.get("wr", float("nan"))) and s_30["wr"] >= 0.55
    checks.append(("30日勝率 >=55%",  ok_wr30,  f"wr={s_30['wr']*100:.1f}%" if np.isfinite(s_30.get("wr", float("nan"))) else "N/A"))

    # avgPnL
    ok_pm30 = np.isfinite(s_30.get("pnl_mean", float("nan"))) and s_30["pnl_mean"] > 0
    checks.append(("30日avgPnL >0",   ok_pm30,  f"{s_30['pnl_mean']:+.3f}" if np.isfinite(s_30.get("pnl_mean", float("nan"))) else "N/A"))

    # medPnL
    ok_md30 = np.isfinite(s_30.get("pnl_med", float("nan"))) and s_30["pnl_med"] > 0
    checks.append(("30日medPnL >0",   ok_md30,  f"{s_30['pnl_med']:+.3f}" if np.isfinite(s_30.get("pnl_med", float("nan"))) else "N/A"))

    # 10日崩れなし
    ok_10 = s_10["n"] == 0 or (np.isfinite(s_10.get("wr", float("nan"))) and s_10["wr"] >= 0.50)
    checks.append(("直近10日崩れなし (wr>=50% or n=0)", ok_10, f"n={s_10['n']} wr={s_10['wr']*100:.1f}%" if s_10["n"] > 0 and np.isfinite(s_10.get("wr", float("nan"))) else f"n={s_10['n']}"))

    print()
    all_ok = True
    for label, ok, val in checks:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {label}: {val}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  → 全チェック通過。train/testが崩れていなければ本番投入候補に昇格。")
    else:
        print("  → 未通過項目あり。研究候補として継続観察。")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
