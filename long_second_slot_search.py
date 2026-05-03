"""
long_second_slot_search.py — LONG 2本目枠の候補探索

LONG_Eの補完になる2本目のLONG本番通知枠を探す。
30日データでグリッドサーチし、7d/14d/30d比較と現LONG_Eとの重複チェックを実施。

実行: python long_second_slot_search.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
オプション:
  LOOKBACK_DAYS   : 集計日数 (default: 30)
  MIN_DONE_30D    : 30日最低DONE件数 (default: 10)
  MIN_WR          : 最低勝率 (default: 0.55)
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
MIN_DONE_30D   = int(os.environ.get("MIN_DONE_30D", "10"))
MIN_WR         = float(os.environ.get("MIN_WR", "0.55"))

# 現LONG_E条件（重複チェック用）
LONG_E_RSI_MIN = 40.0
LONG_E_RSI_MAX = 50.0
LONG_E_P3_MIN  = 0.2


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

    col_a = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A:A",
    ).execute().get("values", [])
    last_row = len(col_a)
    start_row = max(2, last_row - 60000)

    data_res = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A{start_row}:{HEADER_COL_END}",
    ).execute()
    rows = data_res.get("values", []) or []
    if not rows:
        raise RuntimeError("データ行が空です")

    n_cols = len(headers)
    fixed = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
    df_all = pd.DataFrame(fixed, columns=headers)
    print(f"[INFO] 取得行数: {len(df_all)}", flush=True)

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
    df["_dt"]   = pd.to_datetime(df["Datetime_JST"], errors="coerce", utc=False)
    df["_date"] = df["_dt"].dt.date
    df["_hour"] = df["_dt"].dt.hour

    df["_dir"]      = _get_col(df, "Direction").astype(str).str.strip().str.upper()
    df["_btc_mode"] = _get_col(df, "BTC_Mode_Compat", "BTC_Mode").astype(str).str.strip().str.upper()
    df["_win"]      = _to_win(_get_col(df, "WinLose"))
    df["_pnl"]      = _to_num(_get_col(df, "PnL_Pct"))
    df["_rsi"]      = _to_num(_get_col(df, "RSI"))
    df["_p1"]       = _to_num(_get_col(df, "P1_TrendScore"))
    df["_p3"]       = _to_num(_get_col(df, "P3_VolumeScore"))
    df["_volr"]     = _to_num(_get_col(df, "VolRatio"))
    return df


# ==========================================
# 集計ヘルパー
# ==========================================

def _stats(sub: pd.DataFrame) -> dict:
    win = sub["_win"].dropna()
    pnl = sub["_pnl"].dropna()
    n = len(win)
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg": np.nan, "med": np.nan,
                "active_days": 0, "zero_days": 0, "max_day": 0}
    days = sub["_date"].dropna()
    day_counts = days.value_counts()
    return {
        "n": n,
        "wr": float(win.mean()),
        "avg": float(pnl.mean()) if len(pnl) > 0 else np.nan,
        "med": float(pnl.median()) if len(pnl) > 0 else np.nan,
        "active_days": len(day_counts),
        "zero_days": max(0, 30 - len(day_counts)),
        "max_day": int(day_counts.max()) if len(day_counts) > 0 else 0,
    }


def _fmt_stats(s: dict, period_label: str) -> str:
    n = s["n"]
    if n == 0:
        return f"  {period_label}: n=   0  (データなし)"
    wr  = f"{s['wr']*100:.1f}%" if np.isfinite(s.get("wr", float("nan"))) else " N/A"
    pm  = f"{s['avg']:+.3f}" if np.isfinite(s.get("avg", float("nan"))) else " N/A"
    md  = f"{s['med']:+.3f}" if np.isfinite(s.get("med", float("nan"))) else " N/A"
    ad  = s.get("active_days", 0)
    zd  = s.get("zero_days", 0)
    mx  = s.get("max_day", 0)
    return (f"  {period_label}: n={n:4d}  wr={wr}  avg={pm}  med={md}"
            f"  active={ad}d  zero={zd}d  max/d={mx}")


def _passes_criteria(s30: dict, s7: dict) -> tuple[bool, list]:
    reasons = []
    ok = True

    if s30["n"] < MIN_DONE_30D:
        ok = False
        reasons.append(f"30d_n={s30['n']}<{MIN_DONE_30D}")
    if not np.isfinite(s30.get("wr", float("nan"))) or s30["wr"] < MIN_WR:
        ok = False
        wr_s = f"{s30['wr']*100:.1f}%" if np.isfinite(s30.get("wr", float("nan"))) else "N/A"
        reasons.append(f"30d_wr={wr_s}<{MIN_WR*100:.0f}%")
    if not np.isfinite(s30.get("avg", float("nan"))) or s30["avg"] <= 0:
        ok = False
        reasons.append(f"30d_avg={s30.get('avg', 'N/A'):.3f}<=0")
    if not np.isfinite(s30.get("med", float("nan"))) or s30["med"] <= 0:
        ok = False
        reasons.append(f"30d_med={s30.get('med', 'N/A'):.3f}<=0")
    if s30.get("active_days", 0) < 5:
        ok = False
        reasons.append(f"30d_active={s30['active_days']}<5")
    if s30.get("max_day", 0) > 15:
        ok = False
        reasons.append(f"max/day={s30['max_day']}>15(burst)")

    # 直近7日で崩れていないか
    if s7["n"] >= 3:
        if not np.isfinite(s7.get("wr", float("nan"))) or s7["wr"] < 0.45:
            ok = False
            wr_s = f"{s7['wr']*100:.1f}%" if np.isfinite(s7.get("wr", float("nan"))) else "N/A"
            reasons.append(f"7d_wr={wr_s}<45%(崩れ)")

    return ok, reasons


def _overlap_with_long_e(rsi_min, rsi_max, p3_min) -> str:
    """現LONG_Eとの重複度チェック"""
    # RSI範囲が完全に内包または一致
    rsi_overlap = (rsi_min >= LONG_E_RSI_MIN and rsi_max <= LONG_E_RSI_MAX)
    p3_stricter = (p3_min is not None and p3_min >= LONG_E_P3_MIN)

    if rsi_overlap and p3_stricter:
        return "HIGH(LONG_E内包)"
    if rsi_overlap:
        return "MED(RSI同一)"
    if rsi_min >= LONG_E_RSI_MIN and rsi_min < LONG_E_RSI_MAX:
        return "LOW(RSI一部重複)"
    return "NONE"


# ==========================================
# グリッドサーチ
# ==========================================

RSI_RANGES = [
    (40, 50),
    (42, 50),
    (40, 48),
    (45, 52),
    (50, 60),
    (50, 55),
    (55, 65),
    (40, 55),
]

P3_MINS = [0.2, 0.5, 0.7, 1.0]

P1_RANGES = [
    (None, None),
    (1.5, None),
    (1.5, 2.5),
    (1.0, 2.0),
    (2.0, None),
    (0.5, 1.5),
]

VOLR_RANGES = [
    (None, None),
    (1.0, 2.0),
    (1.0, 2.5),
    (1.5, None),
    (None, 2.0),
]

HOUR_RANGES = [
    (None, None),
    (0, 10),
    (6, 14),
    (8, 18),
    (10, 18),
    (16, 24),
]


def run_search(df_base: pd.DataFrame, now_jst) -> list:
    cut7  = (now_jst - timedelta(days=7)).date()
    cut14 = (now_jst - timedelta(days=14)).date()
    cut30 = (now_jst - timedelta(days=30)).date()

    results = []
    total_combos = (len(RSI_RANGES) * len(P3_MINS) * len(P1_RANGES)
                    * len(VOLR_RANGES) * len(HOUR_RANGES))
    print(f"[INFO] 組み合わせ数: {total_combos}", flush=True)

    checked = 0
    for rsi_min, rsi_max in RSI_RANGES:
        for p3_min in P3_MINS:
            for p1_min, p1_max in P1_RANGES:
                for vol_min, vol_max in VOLR_RANGES:
                    for hour_min, hour_max in HOUR_RANGES:
                        checked += 1
                        if checked % 200 == 0:
                            print(f"[INFO] {checked}/{total_combos} 処理中...", flush=True)

                        # フィルタ適用
                        m = (
                            df_base["_rsi"].notna()
                            & (df_base["_rsi"] >= rsi_min)
                            & (df_base["_rsi"] < rsi_max)
                            & df_base["_p3"].notna()
                            & (df_base["_p3"] >= p3_min)
                        )
                        if p1_min is not None:
                            m &= df_base["_p1"].notna() & (df_base["_p1"] >= p1_min)
                        if p1_max is not None:
                            m &= df_base["_p1"].notna() & (df_base["_p1"] < p1_max)
                        if vol_min is not None:
                            m &= df_base["_volr"].notna() & (df_base["_volr"] >= vol_min)
                        if vol_max is not None:
                            m &= df_base["_volr"].notna() & (df_base["_volr"] < vol_max)
                        if hour_min is not None:
                            m &= df_base["_hour"] >= hour_min
                        if hour_max is not None:
                            m &= df_base["_hour"] < hour_max

                        sub = df_base[m]

                        # 30日・14日・7日サブセット
                        sub30 = sub[sub["_date"] >= cut30]
                        sub14 = sub[sub["_date"] >= cut14]
                        sub7  = sub[sub["_date"] >= cut7]

                        s30 = _stats(sub30)
                        if s30["n"] < MIN_DONE_30D:
                            continue

                        s14 = _stats(sub14)
                        s7  = _stats(sub7)

                        ok, fail_reasons = _passes_criteria(s30, s7)

                        parts = [
                            "side=LONG", "mode=UP",
                            f"rsi_min={rsi_min}", f"rsi_max={rsi_max}",
                            f"p3_min={p3_min}",
                        ]
                        if p1_min is not None:
                            parts.append(f"p1_min={p1_min}")
                        if p1_max is not None:
                            parts.append(f"p1_max={p1_max}")
                        if vol_min is not None:
                            parts.append(f"volratio_min={vol_min}")
                        if vol_max is not None:
                            parts.append(f"volratio_max={vol_max}")
                        if hour_min is not None:
                            parts.append(f"hour_min={hour_min}")
                        if hour_max is not None:
                            parts.append(f"hour_max={hour_max}")
                        cond = ";".join(parts)

                        overlap = _overlap_with_long_e(rsi_min, rsi_max, p3_min)

                        results.append({
                            "condition": cond,
                            "ok": ok,
                            "fail": "|".join(fail_reasons),
                            "overlap": overlap,
                            "n30": s30["n"],
                            "wr30": s30["wr"],
                            "avg30": s30["avg"],
                            "med30": s30["med"],
                            "active30": s30["active_days"],
                            "zero30": s30["zero_days"],
                            "max_day30": s30["max_day"],
                            "n14": s14["n"],
                            "wr14": s14["wr"],
                            "avg14": s14["avg"],
                            "n7": s7["n"],
                            "wr7": s7["wr"],
                            "avg7": s7["avg"],
                            "s30": s30, "s14": s14, "s7": s7,
                        })
    return results


# ==========================================
# メイン
# ==========================================

def main():
    print("=== LONG 2本目枠 候補探索 ===")
    print(f"対象: 直近{LOOKBACK_DAYS}日 EvalStatus=DONE / Direction=LONG / BTC_Mode=UP")
    print(f"候補基準: 30d DONE>={MIN_DONE_30D} / 勝率>={MIN_WR*100:.0f}% / avg>0 / med>0 / active_days>=5")
    print(f"除外: 直近7d 勝率<45%(崩れ検出)\n")

    df_raw = fetch_done()
    df = preprocess(df_raw)

    now_jst = datetime.now(JST)
    cut30 = (now_jst - timedelta(days=30)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cut30)].copy()
    print(f"[INFO] 直近30日 DONE: {len(df_recent)}件", flush=True)

    df_base = df_recent[
        (df_recent["_dir"] == "LONG") & (df_recent["_btc_mode"] == "UP")
    ].copy()
    print(f"[INFO] LONG×UP DONE: {len(df_base)}件\n", flush=True)

    if len(df_base) == 0:
        print("[WARN] 対象データ0件")
        return

    results = run_search(df_base, now_jst)
    print(f"\n[INFO] 集計完了 — 候補数（DONE>={MIN_DONE_30D}）: {len(results)}件", flush=True)

    if not results:
        print("候補なし")
        return

    df_res = pd.DataFrame(results)

    # ==========================================
    # Section A: 実用候補（全基準通過 + LONG_E重複なし）
    # ==========================================
    print(f"\n{'='*75}")
    print("  Section A: 実用候補（全基準通過 + LONG_E重複なし）")
    print(f"{'='*75}")

    good_no_overlap = df_res[
        df_res["ok"] & ~df_res["overlap"].isin(["HIGH(LONG_E内包)", "MED(RSI同一)"])
    ].copy()
    good_no_overlap = good_no_overlap.sort_values(
        by=["wr30", "avg30", "med30", "active30"],
        ascending=False
    )

    if good_no_overlap.empty:
        print("  実用候補なし（LONG_Eと非重複かつ全基準通過の条件が見つからない）")
    else:
        for _, r in good_no_overlap.head(15).iterrows():
            print(f"\n  ★ {r['condition']}")
            print(f"     overlap={r['overlap']}")
            print(_fmt_stats(r["s30"], "30d"))
            print(_fmt_stats(r["s14"], "14d"))
            print(_fmt_stats(r["s7"],  " 7d"))

    # ==========================================
    # Section B: 実用候補（全基準通過、LONG_E重複あり）
    # ==========================================
    good_overlap = df_res[
        df_res["ok"] & df_res["overlap"].isin(["HIGH(LONG_E内包)", "MED(RSI同一)"])
    ].copy()
    if not good_overlap.empty:
        print(f"\n{'='*75}")
        print("  Section B: 全基準通過（ただしLONG_E重複あり — 参考値）")
        print(f"{'='*75}")
        good_overlap = good_overlap.sort_values(by=["wr30", "avg30"], ascending=False)
        for _, r in good_overlap.head(10).iterrows():
            print(f"\n  (参考) {r['condition']}")
            print(f"     overlap={r['overlap']}")
            print(_fmt_stats(r["s30"], "30d"))
            print(_fmt_stats(r["s7"],  " 7d"))

    # ==========================================
    # Section C: 基準未達だが近い候補
    # ==========================================
    near = df_res[
        ~df_res["ok"]
        & (df_res["n30"] >= MIN_DONE_30D)
        & (df_res["wr30"] >= MIN_WR - 0.05)
        & ~df_res["overlap"].isin(["HIGH(LONG_E内包)", "MED(RSI同一)"])
    ].sort_values(by=["wr30", "avg30"], ascending=False)

    if not near.empty:
        print(f"\n{'='*75}")
        print("  Section C: 基準に近い候補（実用候補未達、参考）")
        print(f"{'='*75}")
        for _, r in near.head(10).iterrows():
            print(f"\n  △ {r['condition']}")
            print(f"     fail={r['fail']}  overlap={r['overlap']}")
            print(_fmt_stats(r["s30"], "30d"))
            print(_fmt_stats(r["s7"],  " 7d"))

    # ==========================================
    # Section D: 総合サマリ
    # ==========================================
    print(f"\n{'='*75}")
    print("  Section D: 総合サマリ")
    print(f"{'='*75}")
    n_good_no_ov = len(good_no_overlap)
    n_good_ov    = len(good_overlap) if not good_overlap.empty else 0
    print(f"  全基準通過（LONG_E非重複）: {n_good_no_ov}件")
    print(f"  全基準通過（LONG_E重複あり）: {n_good_ov}件")
    print(f"  基準に近い候補（非重複）: {len(near)}件")

    if n_good_no_ov == 0:
        if n_good_ov > 0:
            print("\n  → LONG_Eと異なる条件の実用候補なし。LONG_E一本で継続。")
        else:
            print("\n  → 実用候補なし。LONG_E一本で継続するか、条件軸を変えて再探索。")
    else:
        best = good_no_overlap.iloc[0]
        print(f"\n  → 最有力候補: {best['condition']}")
        print(f"     30d: wr={best['wr30']*100:.1f}% avg={best['avg30']:+.3f} med={best['med30']:+.3f}")
        print(f"      7d: wr={best['wr7']*100:.1f}% avg={best['avg7']:+.3f}" if best["n7"] > 0 else "      7d: データなし")
        print("\n  次のステップ: この候補を long_e_verify.py 相当でtrain/test確認後、")
        print("               LONG_D または LONG_F 枠に入れる。ALLOW/BYPASSに追加。")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
