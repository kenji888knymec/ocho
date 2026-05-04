"""
long_d_replacement_search.py — LONG_D枠 代替条件探索

現LONG_D条件を無効化するのではなく、LONG_D枠に入れる勝てる代替条件を探索する。
LONG_E/Fとの重複チェックを行い、独立した新しい枠として機能する候補を見つける。

現行条件（参考）:
  LONG_D: side=LONG;mode=UP;p1_max=1.0;volratio_min=2.0;volconfirmed=1
  LONG_E: side=LONG;mode=UP;rsi_min=40;rsi_max=50;p3_min=0.2
  LONG_F: side=LONG;mode=UP;rsi_min=55;rsi_max=65;p3_min=0.7;p1_min=0.5;p1_max=1.5;
           volratio_min=1.0;volratio_max=2.5;hour_min=6;hour_max=14

実行: python long_d_replacement_search.py
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
    "SPREADSHEET_ID", "1XwWkzijIwRlafg2zDgPHQ4tgjYModapFI3T_wbYS9_8"
)
V2_SHADOW_SHEET = os.environ.get("V2_SHADOW_SHEET", "v2_shadow_ai")
HEADER_COL_END = "AZ"
JST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "30"))
MIN_DONE_30D  = int(os.environ.get("MIN_DONE_30D", "10"))
MIN_WR        = float(os.environ.get("MIN_WR", "0.55"))
TRAIN_RATIO   = float(os.environ.get("TRAIN_RATIO", "0.6"))

# 現行条件定義（重複チェック用）
LONG_E_RSI = (40.0, 50.0);  LONG_E_P3 = 0.2
LONG_F_RSI = (55.0, 65.0);  LONG_F_P3 = 0.7; LONG_F_P1 = (0.5, 1.5)
LONG_F_VOLR = (1.0, 2.5);   LONG_F_HOUR = (6, 14)


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

    col_a = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=f"{V2_SHADOW_SHEET}!A:A",
    ).execute().get("values", [])
    last_row = len(col_a)
    start_row = max(2, last_row - 60000)

    rows = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A{start_row}:{HEADER_COL_END}",
    ).execute().get("values", []) or []

    n_cols = len(headers)
    fixed = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
    df = pd.DataFrame(fixed, columns=headers)
    print(f"[INFO] 取得行数: {len(df)}", flush=True)

    status = df["EvalStatus"].astype(str).str.strip().str.upper()
    df_done = df[status == "DONE"].copy()
    print(f"[INFO] DONE件数: {len(df_done)}", flush=True)
    return df_done


# ==========================================
# 前処理
# ==========================================

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce"
    )


def _to_win(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.strip().str.lower()
    return v.map(lambda x: 1.0 if x in {"win","w","1","true","yes"}
                           else (0.0 if x in {"lose","l","0","false","no"} else np.nan))


def _get(df, *cols):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series([""] * len(df), index=df.index)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["_dt"]   = pd.to_datetime(df["Datetime_JST"], errors="coerce", utc=False)
    df["_date"] = df["_dt"].dt.date
    df["_hour"] = df["_dt"].dt.hour
    df["_dir"]  = _get(df, "Direction").astype(str).str.strip().str.upper()
    df["_mode"] = _get(df, "BTC_Mode_Compat", "BTC_Mode").astype(str).str.strip().str.upper()
    df["_win"]  = _to_win(_get(df, "WinLose"))
    df["_pnl"]  = _to_num(_get(df, "PnL_Pct"))
    df["_rsi"]  = _to_num(_get(df, "RSI"))
    df["_p1"]   = _to_num(_get(df, "P1_TrendScore"))
    df["_p2"]   = _to_num(_get(df, "P2_FundingScore"))
    df["_p3"]   = _to_num(_get(df, "P3_VolumeScore"))
    df["_volr"] = _to_num(_get(df, "VolRatio"))
    df["_ai"]   = _to_num(_get(df, "AI_Prob_Win"))
    vc = _get(df, "VolConfirmed")
    df["_volc"] = vc.astype(str).str.strip().str.upper().isin({"1", "TRUE", "YES"})
    fra = _get(df, "FR_Available")
    df["_fra"]  = fra.astype(str).str.strip().str.upper().isin({"1", "TRUE", "YES"})
    return df


# ==========================================
# 集計ヘルパー
# ==========================================

def _stats(sub: pd.DataFrame, n_days_window: int = 30) -> dict:
    win = sub["_win"].dropna()
    pnl = sub["_pnl"].dropna()
    n = len(win)
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg": np.nan, "med": np.nan,
                "active_days": 0, "zero_days": n_days_window, "max_day": 0, "med_day": 0}
    dc = sub["_date"].dropna().value_counts()
    return {
        "n": n, "wr": float(win.mean()),
        "avg": float(pnl.mean()) if len(pnl) > 0 else np.nan,
        "med": float(pnl.median()) if len(pnl) > 0 else np.nan,
        "active_days": len(dc),
        "zero_days": max(0, n_days_window - len(dc)),
        "max_day": int(dc.max()) if len(dc) > 0 else 0,
        "med_day": float(dc.median()) if len(dc) > 0 else 0,
    }


def _fmt(s: dict, label: str) -> str:
    if s["n"] == 0:
        return f"  {label}: n=   0  (データなし)"
    wr  = f"{s['wr']*100:.1f}%" if np.isfinite(s.get("wr", np.nan)) else " N/A"
    pm  = f"{s['avg']:+.3f}" if np.isfinite(s.get("avg", np.nan)) else " N/A"
    md  = f"{s['med']:+.3f}" if np.isfinite(s.get("med", np.nan)) else " N/A"
    return (f"  {label}: n={s['n']:4d}  wr={wr}  avg={pm}  med={md}"
            f"  active={s['active_days']}d  zero={s['zero_days']}d  max/d={s['max_day']}")


def _train_test(sub: pd.DataFrame) -> str:
    if len(sub) < 5:
        return "  train/test: サンプル不足"
    ss = sub.sort_values("_dt").reset_index(drop=True)
    n_tr = max(1, int(len(ss) * TRAIN_RATIO))
    tr = ss.iloc[:n_tr]; te = ss.iloc[n_tr:]
    st = _stats(tr); se = _stats(te)
    wr_tr = f"{st['wr']*100:.1f}%" if np.isfinite(st.get("wr", np.nan)) else "N/A"
    wr_te = f"{se['wr']*100:.1f}%" if np.isfinite(se.get("wr", np.nan)) else "N/A"
    diff = ""
    if np.isfinite(st.get("wr", np.nan)) and np.isfinite(se.get("wr", np.nan)):
        d = (se["wr"] - st["wr"]) * 100
        vrd = "崩れなし" if d >= -10 else ("やや崩れ" if d >= -20 else "崩れあり")
        diff = f"  変化={d:+.1f}% {vrd}"
    return (f"  train: n={n_tr} wr={wr_tr} avg={st['avg']:+.3f}" if np.isfinite(st.get("avg", np.nan)) else f"  train: n={n_tr} wr={wr_tr}") + "\n" \
         + (f"  test:  n={len(te)} wr={wr_te} avg={se['avg']:+.3f}" if np.isfinite(se.get("avg", np.nan)) else f"  test:  n={len(te)} wr={wr_te}") + "\n" \
         + f"  {diff}"


def _passes(s30: dict, s7: dict) -> tuple:
    ok = True; fail = []
    if s30["n"] < MIN_DONE_30D: ok = False; fail.append(f"n30={s30['n']}<{MIN_DONE_30D}")
    if not np.isfinite(s30.get("wr", np.nan)) or s30["wr"] < MIN_WR:
        ok = False; wr_s = f"{s30['wr']*100:.1f}%" if np.isfinite(s30.get("wr", np.nan)) else "N/A"
        fail.append(f"wr30={wr_s}")
    if not np.isfinite(s30.get("avg", np.nan)) or s30["avg"] <= 0: ok = False; fail.append(f"avg30={s30.get('avg','N/A'):.3f}<=0")
    if not np.isfinite(s30.get("med", np.nan)) or s30["med"] <= 0: ok = False; fail.append(f"med30={s30.get('med','N/A'):.3f}<=0")
    if s30.get("active_days", 0) < 5: ok = False; fail.append(f"active={s30['active_days']}<5")
    if s30.get("max_day", 0) > 15: ok = False; fail.append(f"max/d={s30['max_day']}>15")
    if s7["n"] >= 3 and (not np.isfinite(s7.get("wr", np.nan)) or s7["wr"] < 0.45):
        ok = False; wr7 = f"{s7['wr']*100:.1f}%" if np.isfinite(s7.get("wr", np.nan)) else "N/A"; fail.append(f"wr7={wr7}<45%")
    return ok, "|".join(fail)


def _overlap_ef(rsi_min, rsi_max, p3_min, p1_min, p1_max, volr_min, volr_max, hour_min, hour_max):
    """LONG_E/F との重複度判定"""
    # LONG_E: RSI 40-50, p3>=0.2
    e_rsi_ov = not (rsi_max <= LONG_E_RSI[0] or rsi_min >= LONG_E_RSI[1])
    e_p3_ov  = (p3_min is None or p3_min < LONG_E_P3 + 0.3)
    if e_rsi_ov and e_p3_ov:
        if rsi_min >= LONG_E_RSI[0] and rsi_max <= LONG_E_RSI[1]:
            e_level = "HIGH(LONG_E完全内包)"
        else:
            e_level = "MED(LONG_E一部重複)"
    elif e_rsi_ov:
        e_level = "LOW(RSI重複のみ)"
    else:
        e_level = "NONE"

    # LONG_F: RSI 55-65, p3>=0.7, p1 0.5-1.5, volr 1.0-2.5, hour 6-14
    f_rsi_ov = not (rsi_max <= LONG_F_RSI[0] or rsi_min >= LONG_F_RSI[1])
    f_p3_ov  = (p3_min is None or p3_min >= LONG_F_P3 - 0.2)
    f_p1_ov  = True  # 簡易: p1が指定なしなら重複可能性あり
    if p1_min is not None and p1_max is not None:
        f_p1_ov = not (p1_max <= LONG_F_P1[0] or p1_min >= LONG_F_P1[1])
    if f_rsi_ov and f_p3_ov and f_p1_ov:
        f_level = "MED(LONG_F重複)"
    elif f_rsi_ov:
        f_level = "LOW(LONG_F RSI重複)"
    else:
        f_level = "NONE"

    return e_level, f_level


# ==========================================
# 現LONG_D ベースライン
# ==========================================

def show_baseline(df_long: pd.DataFrame, now_jst) -> None:
    print(f"\n{'='*70}")
    print("  現LONG_D ベースライン (approximate)")
    print("  条件: side=LONG;mode=UP;p1_max=1.0;volratio_min=2.0;volconfirmed=1")
    print(f"{'='*70}")
    print("  ※ ALLOW/BYPASSに入っていないため全数DONE。実通知D=0件。")

    cut30 = (now_jst - timedelta(days=30)).date()
    cut14 = (now_jst - timedelta(days=14)).date()
    cut7  = (now_jst - timedelta(days=7)).date()

    for mode_label, mode_val in [("UP", "UP"), ("RANGE", "RANGE"), ("ALL", None)]:
        m = (
            df_long["_dir"] == "LONG"
        ) & (
            df_long["_p1"].notna() & (df_long["_p1"] <= 1.0)
        ) & (
            df_long["_volr"].notna() & (df_long["_volr"] >= 2.0)
        ) & (
            df_long["_volc"] == True
        )
        if mode_val:
            m &= df_long["_mode"] == mode_val

        sub = df_long[m]
        print(f"\n  [mode={mode_label}]")
        print(_fmt(_stats(sub[sub["_date"] >= cut30], 30), "30d"))
        print(_fmt(_stats(sub[sub["_date"] >= cut14], 14), "14d"))
        print(_fmt(_stats(sub[sub["_date"] >= cut7],   7), " 7d"))


# ==========================================
# グリッドサーチ
# ==========================================

RSI_RANGES = [
    # LONG_E(40-50)とLONG_F(55-65)の間・外を優先
    (50, 60), (50, 55), (55, 60),        # E-F間
    (65, 75), (70, 80), (65, 80),        # F上
    (30, 40), (35, 45),                  # E下
    (40, 55), (45, 60),                  # E-F跨ぎ
    (50, 70), (55, 75),                  # 広め
]

P3_MINS = [None, 0.0, 0.3, 0.5, 0.7]

P1_SPECS = [
    (None, None),
    (0.5, 1.5), (1.0, 2.0), (1.5, 2.5),
    (2.0, None), (0.5, None),
    (None, 1.0), (None, 2.0),
]

VOLR_SPECS = [
    (None, None),
    (1.0, 2.0), (1.5, 2.5), (1.0, 2.5),
    (None, 2.0), (2.0, None),
]

VOLC_SPECS = [None, True]  # None=問わず, True=VolConfirmed=1

HOUR_SPECS = [
    (None, None), (0, 10), (6, 14), (10, 18), (14, 24),
]

AI_SPECS = [None, 0.4, 0.6]  # None=問わず, 数値=ai_max


def run_grid(df_base: pd.DataFrame, now_jst) -> list:
    cut30 = (now_jst - timedelta(days=30)).date()
    cut14 = (now_jst - timedelta(days=14)).date()
    cut7  = (now_jst - timedelta(days=7)).date()

    total = (len(RSI_RANGES) * len(P3_MINS) * len(P1_SPECS)
             * len(VOLR_SPECS) * len(VOLC_SPECS) * len(HOUR_SPECS) * len(AI_SPECS))
    print(f"[INFO] 組み合わせ数: {total}", flush=True)

    results = []
    checked = 0
    for rsi_min, rsi_max in RSI_RANGES:
        for p3_min in P3_MINS:
            for p1_min, p1_max in P1_SPECS:
                for volr_min, volr_max in VOLR_SPECS:
                    for volc in VOLC_SPECS:
                        for hour_min, hour_max in HOUR_SPECS:
                            for ai_max in AI_SPECS:
                                checked += 1
                                if checked % 500 == 0:
                                    print(f"[INFO] {checked}/{total}...", flush=True)

                                m = (
                                    df_base["_rsi"].notna()
                                    & (df_base["_rsi"] >= rsi_min)
                                    & (df_base["_rsi"] < rsi_max)
                                )
                                if p3_min is not None:
                                    m &= df_base["_p3"].notna() & (df_base["_p3"] >= p3_min)
                                if p1_min is not None:
                                    m &= df_base["_p1"].notna() & (df_base["_p1"] >= p1_min)
                                if p1_max is not None:
                                    m &= df_base["_p1"].notna() & (df_base["_p1"] < p1_max)
                                if volr_min is not None:
                                    m &= df_base["_volr"].notna() & (df_base["_volr"] >= volr_min)
                                if volr_max is not None:
                                    m &= df_base["_volr"].notna() & (df_base["_volr"] < volr_max)
                                if volc is not None:
                                    m &= df_base["_volc"] == volc
                                if hour_min is not None:
                                    m &= df_base["_hour"] >= hour_min
                                if hour_max is not None:
                                    m &= df_base["_hour"] < hour_max
                                if ai_max is not None:
                                    m &= df_base["_ai"].notna() & (df_base["_ai"] < ai_max)

                                sub = df_base[m]
                                s30 = _stats(sub[sub["_date"] >= cut30], 30)
                                if s30["n"] < MIN_DONE_30D:
                                    continue

                                s14 = _stats(sub[sub["_date"] >= cut14], 14)
                                s7  = _stats(sub[sub["_date"] >= cut7],   7)
                                ok, fail = _passes(s30, s7)

                                parts = ["side=LONG", "mode=UP",
                                         f"rsi_min={rsi_min}", f"rsi_max={rsi_max}"]
                                if p3_min is not None: parts.append(f"p3_min={p3_min}")
                                if p1_min is not None: parts.append(f"p1_min={p1_min}")
                                if p1_max is not None: parts.append(f"p1_max={p1_max}")
                                if volr_min is not None: parts.append(f"volratio_min={volr_min}")
                                if volr_max is not None: parts.append(f"volratio_max={volr_max}")
                                if volc: parts.append("volconfirmed=1")
                                if hour_min is not None: parts.append(f"hour_min={hour_min}")
                                if hour_max is not None: parts.append(f"hour_max={hour_max}")
                                if ai_max is not None: parts.append(f"ai_max={ai_max}")
                                cond = ";".join(parts)

                                e_lv, f_lv = _overlap_ef(
                                    rsi_min, rsi_max, p3_min, p1_min, p1_max,
                                    volr_min, volr_max, hour_min, hour_max
                                )
                                results.append({
                                    "condition": cond, "ok": ok, "fail": fail,
                                    "overlap_e": e_lv, "overlap_f": f_lv,
                                    "s30": s30, "s14": s14, "s7": s7,
                                    "sub": sub,
                                })
    return results


# ==========================================
# メイン
# ==========================================

def main():
    print("=== LONG_D 代替条件探索 ===")
    print(f"対象: 直近{LOOKBACK_DAYS}日 EvalStatus=DONE / Direction=LONG")
    print(f"候補基準: 30d DONE>={MIN_DONE_30D} / wr>={MIN_WR*100:.0f}% / avg>0 / med>0 / active_days>=5")
    print(f"          + 7d wr>=45% (崩れ検出)\n")

    df_raw = fetch_done()
    df = preprocess(df_raw)

    now_jst = datetime.now(JST)
    cut30 = (now_jst - timedelta(days=30)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cut30)].copy()
    df_long = df_recent[df_recent["_dir"] == "LONG"].copy()
    print(f"[INFO] 直近30日 LONG DONE: {len(df_long)}件\n", flush=True)

    # ベースライン
    show_baseline(df_long, now_jst)

    # グリッドサーチ (UP mode)
    df_up = df_long[df_long["_mode"] == "UP"].copy()
    df_range = df_long[df_long["_mode"] == "RANGE"].copy()
    print(f"\n[INFO] LONG UP: {len(df_up)}件 / LONG RANGE: {len(df_range)}件\n", flush=True)

    print("[INFO] UP mode グリッドサーチ中...", flush=True)
    results_up = run_grid(df_up, now_jst)

    print(f"\n[INFO] UP 候補数（DONE>={MIN_DONE_30D}）: {len(results_up)}", flush=True)

    # RANGE mode でも基本集計
    print(f"\n{'='*70}")
    print("  RANGE mode LONG 基本成績（参考）")
    print(f"{'='*70}")
    if len(df_range) > 0:
        cut14 = (now_jst - timedelta(days=14)).date()
        cut7  = (now_jst - timedelta(days=7)).date()
        print(_fmt(_stats(df_range[df_range["_date"] >= cut30], 30), "30d"))
        print(_fmt(_stats(df_range[df_range["_date"] >= cut14], 14), "14d"))
        print(_fmt(_stats(df_range[df_range["_date"] >= cut7],   7), " 7d"))
        print("  ※ RANGE mode LONGの詳細グリッドは別途実施")
    else:
        print("  RANGE mode LONG DONE: 0件")

    def _show_section(label: str, items: list, with_tt: bool = True):
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        if not items:
            print("  候補なし")
            return
        for r in items[:15]:
            print(f"\n  {'★' if r['ok'] else '△'} {r['condition']}")
            print(f"     overlap_E={r['overlap_e']}  overlap_F={r['overlap_f']}")
            if r["fail"]:
                print(f"     fail={r['fail']}")
            print(_fmt(r["s30"], "30d"))
            print(_fmt(r["s14"], "14d"))
            print(_fmt(r["s7"],  " 7d"))
            if with_tt:
                print(_train_test(r["sub"]))

    # Section A: 実用候補 (全基準通過 + E/F非重複)
    def _no_major_overlap(r):
        return r["overlap_e"] not in ("HIGH(LONG_E完全内包)", "MED(LONG_E一部重複)") \
               and r["overlap_f"] not in ("MED(LONG_F重複)",)

    sec_a = sorted(
        [r for r in results_up if r["ok"] and _no_major_overlap(r)],
        key=lambda r: (-r["s30"]["wr"], -r["s30"].get("avg", 0))
    )
    _show_section("Section A: 実用候補（全基準通過 + LONG_E/F非重複）", sec_a)

    # Section B: 全基準通過 + 重複あり
    sec_b = sorted(
        [r for r in results_up if r["ok"] and not _no_major_overlap(r)],
        key=lambda r: (-r["s30"]["wr"], -r["s30"].get("avg", 0))
    )
    _show_section("Section B: 全基準通過（LONG_E/F重複あり — 参考）", sec_b, with_tt=False)

    # Section C: 基準近い・未達
    sec_c = sorted(
        [r for r in results_up if not r["ok"] and r["s30"]["n"] >= MIN_DONE_30D
         and np.isfinite(r["s30"].get("wr", np.nan)) and r["s30"]["wr"] >= MIN_WR - 0.05
         and _no_major_overlap(r)],
        key=lambda r: (-r["s30"]["wr"], -r["s30"].get("avg", 0))
    )
    _show_section("Section C: 基準に近い候補（非重複）", sec_c, with_tt=False)

    # Section D: サマリ
    print(f"\n{'='*70}")
    print("  Section D: 総合サマリ")
    print(f"{'='*70}")
    print(f"  実用候補（A）: {len(sec_a)}件")
    print(f"  参考候補（B）: {len(sec_b)}件")
    print(f"  近い候補（C）: {len(sec_c)}件")

    if sec_a:
        best = sec_a[0]
        print(f"\n  ✅ LONG_D枠の差し替え候補あり")
        print(f"  最有力: {best['condition']}")
        print(f"  30d: wr={best['s30']['wr']*100:.1f}%  avg={best['s30']['avg']:+.3f}  med={best['s30']['med']:+.3f}")
        print(f"  次: この候補を train/test 確認後、LONG_D 条件を差し替えて ALLOW/BYPASS に追加")
    else:
        print(f"\n  ❌ LONG_D枠の差し替え候補なし（現時点）")
        print(f"  → LONG_D枠を空のまま（rsi_min=101等）にして、LONG_E/F 2本で継続")
        print(f"  → または条件軸を変えて再探索（RANGE mode LONG / AI帯追加など）")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
