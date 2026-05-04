"""
short_b_replacement_search.py — SHORT_B枠 代替条件探索

SHORT_B枠に入れる勝てる代替条件を探索する。
SHORT_A/Cとの重複チェックと、short_recent_pause/profile no_match の分離も実施。

現行条件（参考）:
  SHORT_B: side=SHORT;mode=RANGE;ltf_aligned=0;p2_min=0;rsi_min=40;rsi_max=45
  SHORT_A: side=SHORT;mode=RANGE;rsi_min=50;rsi_max=55;p2_min=0;volratio_max=1.2;ai_max=0.6
  SHORT_C: side=SHORT;mode=DOWN;rsi_min=50;rsi_max=60;volratio_min=1.2;volratio_max=2.5;ai_max=0.6

実行: python short_b_replacement_search.py
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
HEADER_COL_END  = "AZ"
JST = timezone(timedelta(hours=9))
LOOKBACK_DAYS   = int(os.environ.get("LOOKBACK_DAYS", "30"))
MIN_DONE_30D    = int(os.environ.get("MIN_DONE_30D", "10"))
MIN_WR          = float(os.environ.get("MIN_WR", "0.55"))
TRAIN_RATIO     = float(os.environ.get("TRAIN_RATIO", "0.6"))

# 現行条件定義（重複チェック用）
SHORT_A_RSI  = (50.0, 55.0); SHORT_A_VOLR_MAX = 1.2; SHORT_A_AI_MAX = 0.6; SHORT_A_P2_MIN = 0.0
SHORT_B_RSI  = (40.0, 45.0)
SHORT_C_RSI  = (50.0, 60.0); SHORT_C_MODE = "DOWN"
SHORT_C_VOLR = (1.2, 2.5);   SHORT_C_AI_MAX = 0.6


# ==========================================
# Google Sheets 取得
# ==========================================

def _sheets_service(creds):
    http = httplib2.Http(timeout=120)
    authed_http = AuthorizedHttp(creds, http=http)
    return build("sheets", "v4", http=authed_http, cache_discovery=False)


def fetch_all() -> pd.DataFrame:
    """DONE + OPEN 両方取得（funnel確認用）"""
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
    return df


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
    df["_status"] = df["EvalStatus"].astype(str).str.strip().str.upper() if "EvalStatus" in df.columns else "UNKNOWN"
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

    # ltf_aligned: 0 or 1
    ltf = _get(df, "ltf_aligned", "LTF_Aligned", "LTFAligned")
    ltf_num = _to_num(ltf)
    df["_ltf"] = ltf_num

    # notify_sent
    ns = _get(df, "notify_sent", "_notify_sent")
    df["_notify"] = ns.astype(str).str.strip() == "1"

    # Notify_Reason / AI_Note から rejection 理由を抽出
    nr = _get(df, "Notify_Reason", "notify_reason").astype(str).str.lower()
    ai_note = _get(df, "AI_Note", "ai_note").astype(str).str.lower()
    df["_recent_pause"] = nr.str.contains("recent_pause", na=False) | ai_note.str.contains("recent_pause", na=False)
    df["_profile_miss"]  = nr.str.contains("profile_miss|no_match|not_whitelisted", na=False) | \
                           ai_note.str.contains("profile_miss|no_match|not_whitelisted", na=False)

    # feature_profile / allow_hits
    fp = _get(df, "feature_profile", "Feature_Profile").astype(str).str.lower()
    ah = _get(df, "Feature_Allow_Hits", "allow_hits").astype(str).str.lower()
    bp = _get(df, "Feature_Bypass_Profile", "bypass_profile").astype(str).str.lower()
    df["_short_b_route"] = (
        fp.str.contains("short_b", na=False)
        | ah.str.contains("short_b", na=False)
        | bp.str.contains("short_b", na=False)
    )
    return df


# ==========================================
# 集計ヘルパー
# ==========================================

def _stats(sub: pd.DataFrame, window_days: int = 30) -> dict:
    done = sub[sub["_status"] == "DONE"] if "_status" in sub.columns else sub
    win = done["_win"].dropna()
    pnl = done["_pnl"].dropna()
    n = len(win)
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg": np.nan, "med": np.nan,
                "active_days": 0, "zero_days": window_days, "max_day": 0}
    dc = done["_date"].dropna().value_counts()
    return {
        "n": n, "wr": float(win.mean()),
        "avg": float(pnl.mean()) if len(pnl) > 0 else np.nan,
        "med": float(pnl.median()) if len(pnl) > 0 else np.nan,
        "active_days": len(dc),
        "zero_days": max(0, window_days - len(dc)),
        "max_day": int(dc.max()) if len(dc) > 0 else 0,
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
    done = sub[sub["_status"] == "DONE"] if "_status" in sub.columns else sub
    if len(done) < 5:
        return "  train/test: サンプル不足"
    ss = done.sort_values("_dt").reset_index(drop=True)
    n_tr = max(1, int(len(ss) * TRAIN_RATIO))
    tr, te = ss.iloc[:n_tr], ss.iloc[n_tr:]
    st, se = _stats(tr), _stats(te)
    wr_tr = f"{st['wr']*100:.1f}%" if np.isfinite(st.get("wr", np.nan)) else "N/A"
    wr_te = f"{se['wr']*100:.1f}%" if np.isfinite(se.get("wr", np.nan)) else "N/A"
    diff = ""
    if np.isfinite(st.get("wr", np.nan)) and np.isfinite(se.get("wr", np.nan)):
        d = (se["wr"] - st["wr"]) * 100
        vrd = "崩れなし" if d >= -10 else ("やや崩れ" if d >= -20 else "崩れあり")
        diff = f"  変化={d:+.1f}% {vrd}"
    avgt = f"{st['avg']:+.3f}" if np.isfinite(st.get("avg", np.nan)) else "N/A"
    avge = f"{se['avg']:+.3f}" if np.isfinite(se.get("avg", np.nan)) else "N/A"
    return (f"  train(n={n_tr}): wr={wr_tr} avg={avgt}\n"
            f"  test (n={len(te)}): wr={wr_te} avg={avge}{diff}")


def _passes(s30: dict, s7: dict) -> tuple:
    ok = True; fail = []
    if s30["n"] < MIN_DONE_30D: ok = False; fail.append(f"n30={s30['n']}<{MIN_DONE_30D}")
    if not np.isfinite(s30.get("wr", np.nan)) or s30["wr"] < MIN_WR:
        ok = False
        wr_s = f"{s30['wr']*100:.1f}%" if np.isfinite(s30.get("wr", np.nan)) else "N/A"
        fail.append(f"wr30={wr_s}")
    if not np.isfinite(s30.get("avg", np.nan)) or s30["avg"] <= 0: ok = False; fail.append(f"avg30<=0")
    if not np.isfinite(s30.get("med", np.nan)) or s30["med"] <= 0: ok = False; fail.append(f"med30<=0")
    if s30.get("active_days", 0) < 5: ok = False; fail.append(f"active={s30['active_days']}<5")
    if s30.get("max_day", 0) > 15: ok = False; fail.append(f"max/d={s30['max_day']}>15")
    if s7["n"] >= 3 and (not np.isfinite(s7.get("wr", np.nan)) or s7["wr"] < 0.45):
        ok = False
        wr7 = f"{s7['wr']*100:.1f}%" if np.isfinite(s7.get("wr", np.nan)) else "N/A"
        fail.append(f"wr7={wr7}<45%")
    return ok, "|".join(fail)


def _overlap_ac(rsi_min, rsi_max, mode_val, volr_min, volr_max, ai_max, p2_min) -> tuple:
    """SHORT_A / SHORT_C との重複度判定"""
    # SHORT_A: RANGE, RSI 50-55, p2>=0, volr<1.2, ai<0.6
    a_rsi_ov = not (rsi_max <= SHORT_A_RSI[0] or rsi_min >= SHORT_A_RSI[1])
    a_mode_ov = (mode_val is None or mode_val == "RANGE")
    if a_rsi_ov and a_mode_ov:
        a_level = "HIGH(SHORT_A重複)"
    elif a_rsi_ov:
        a_level = "LOW(RSI重複)"
    else:
        a_level = "NONE"

    # SHORT_C: DOWN, RSI 50-60, volr 1.2-2.5, ai<0.6
    c_rsi_ov = not (rsi_max <= SHORT_C_RSI[0] or rsi_min >= SHORT_C_RSI[1])
    c_mode_ov = (mode_val is None or mode_val == "DOWN")
    if c_rsi_ov and c_mode_ov:
        c_level = "HIGH(SHORT_C重複)"
    elif c_rsi_ov:
        c_level = "LOW(RSI重複)"
    else:
        c_level = "NONE"

    return a_level, c_level


# ==========================================
# 現SHORT_B ベースライン + funnel
# ==========================================

def show_baseline(df_all: pd.DataFrame, now_jst) -> None:
    print(f"\n{'='*70}")
    print("  現SHORT_B ベースライン")
    print("  条件: side=SHORT;mode=RANGE;ltf_aligned=0;p2_min=0;rsi_min=40;rsi_max=45")
    print(f"{'='*70}")

    cut30 = (now_jst - timedelta(days=30)).date()
    cut14 = (now_jst - timedelta(days=14)).date()
    cut7  = (now_jst - timedelta(days=7)).date()

    m = (
        (df_all["_dir"] == "SHORT")
        & (df_all["_mode"] == "RANGE")
        & df_all["_rsi"].notna() & (df_all["_rsi"] >= 40) & (df_all["_rsi"] < 45)
        & df_all["_p2"].notna() & (df_all["_p2"] >= 0)
        # ltf_aligned=0: 値が0か欠損
        & (df_all["_ltf"].isna() | (df_all["_ltf"] == 0))
    )
    sub = df_all[m]
    sub_done = sub[sub["_status"] == "DONE"]

    print(f"\n  全行（DONE+OPEN合計）: {len(sub)}件")
    print(f"  DONE: {len(sub_done)}件  OPEN: {len(sub[sub['_status']=='OPEN'])}件")

    for label, cut in [("30d", cut30), ("14d", cut14), ("7d", cut7)]:
        s = _stats(sub_done[sub_done["_date"] >= cut], 30)
        print(_fmt(s, label))

    # funnel breakdown
    sub_recent = sub[sub["_date"].notna() & (sub["_date"] >= cut7)]
    pause_n    = sub_recent["_recent_pause"].sum()
    miss_n     = sub_recent["_profile_miss"].sum()
    route_n    = sub_recent["_short_b_route"].sum()
    notify_n   = sub_recent["_notify"].sum()
    done_n     = len(sub_recent[sub_recent["_status"] == "DONE"])
    print(f"\n  [直近7日 funnel (全行={len(sub_recent)}件)]")
    print(f"    recent_pause で落ちた行  : {pause_n}件")
    print(f"    profile no_match で落ちた: {miss_n}件")
    print(f"    short_b route乗った行    : {route_n}件")
    print(f"    notify_sent=1            : {notify_n}件")
    print(f"    DONE                     : {done_n}件")

    # モード別 SHORT 全体
    print(f"\n  [直近30日 SHORT 全体 mode別]")
    df_sh30 = df_all[df_all["_dir"] == "SHORT"]
    for mv in ["RANGE", "DOWN", "UP"]:
        sub_m = df_sh30[df_sh30["_mode"] == mv]
        sub_md = sub_m[sub_m["_status"] == "DONE"]
        s = _stats(sub_md[sub_md["_date"] >= cut30], 30)
        print(f"  mode={mv}: {_fmt(s, 'DONE')}")


# ==========================================
# グリッドサーチ
# ==========================================

RSI_RANGES = [
    # SHORT_A(50-55)/SHORT_B(40-45)/SHORT_C(50-60) と異なる範囲を優先
    (40, 45), (42, 48), (40, 50),        # SHORT_B周辺
    (45, 55), (45, 50),                  # SHORT_B-A間
    (55, 65), (60, 70), (65, 75),        # SHORT_A/C上
    (35, 45), (35, 50),                  # SHORT_B下
    (50, 60), (50, 55),                  # SHORT_A/C（重複確認用）
]

MODE_VALS = ["RANGE", "DOWN"]  # UPでSHORTは基本弱いが参考で含める

P2_MINS  = [None, 0.0, 0.5, 1.0]
P3_SPECS = [None, 0.0, 0.3, 0.5]
VOLR_SPECS = [
    (None, None), (None, 1.2), (None, 1.5), (None, 2.0),
    (1.0, 2.5), (1.2, 2.5),
]
AI_MAXES = [None, 0.4, 0.6]
LTF_SPECS = [None, 0, 1]  # None=問わず
HOUR_SPECS = [
    (None, None), (0, 10), (6, 14), (10, 18), (14, 24),
]


def run_grid(df_base: pd.DataFrame, now_jst) -> list:
    cut30 = (now_jst - timedelta(days=30)).date()
    cut14 = (now_jst - timedelta(days=14)).date()
    cut7  = (now_jst - timedelta(days=7)).date()

    total = (len(RSI_RANGES) * len(MODE_VALS) * len(P2_MINS) * len(P3_SPECS)
             * len(VOLR_SPECS) * len(AI_MAXES) * len(LTF_SPECS) * len(HOUR_SPECS))
    print(f"[INFO] 組み合わせ数: {total}", flush=True)

    results = []
    checked = 0
    for rsi_min, rsi_max in RSI_RANGES:
        for mode_val in MODE_VALS:
            for p2_min in P2_MINS:
                for p3_min in P3_SPECS:
                    for volr_min, volr_max in VOLR_SPECS:
                        for ai_max in AI_MAXES:
                            for ltf_val in LTF_SPECS:
                                for hour_min, hour_max in HOUR_SPECS:
                                    checked += 1
                                    if checked % 1000 == 0:
                                        print(f"[INFO] {checked}/{total}...", flush=True)

                                    m = (
                                        (df_base["_dir"] == "SHORT")
                                        & (df_base["_mode"] == mode_val)
                                        & df_base["_rsi"].notna()
                                        & (df_base["_rsi"] >= rsi_min)
                                        & (df_base["_rsi"] < rsi_max)
                                    )
                                    if p2_min is not None:
                                        m &= df_base["_p2"].notna() & (df_base["_p2"] >= p2_min)
                                    if p3_min is not None:
                                        m &= df_base["_p3"].notna() & (df_base["_p3"] >= p3_min)
                                    if volr_min is not None:
                                        m &= df_base["_volr"].notna() & (df_base["_volr"] >= volr_min)
                                    if volr_max is not None:
                                        m &= df_base["_volr"].notna() & (df_base["_volr"] < volr_max)
                                    if ai_max is not None:
                                        m &= df_base["_ai"].notna() & (df_base["_ai"] < ai_max)
                                    if ltf_val is not None:
                                        if ltf_val == 0:
                                            m &= (df_base["_ltf"].isna() | (df_base["_ltf"] == 0))
                                        else:
                                            m &= df_base["_ltf"] == ltf_val
                                    if hour_min is not None:
                                        m &= df_base["_hour"] >= hour_min
                                    if hour_max is not None:
                                        m &= df_base["_hour"] < hour_max

                                    sub = df_base[m]
                                    sub_done = sub[sub["_status"] == "DONE"]

                                    s30 = _stats(sub_done[sub_done["_date"] >= cut30], 30)
                                    if s30["n"] < MIN_DONE_30D:
                                        continue

                                    s14 = _stats(sub_done[sub_done["_date"] >= cut14], 14)
                                    s7  = _stats(sub_done[sub_done["_date"] >= cut7],   7)
                                    ok, fail = _passes(s30, s7)

                                    parts = ["side=SHORT", f"mode={mode_val}",
                                             f"rsi_min={rsi_min}", f"rsi_max={rsi_max}"]
                                    if ltf_val is not None: parts.append(f"ltf_aligned={ltf_val}")
                                    if p2_min is not None:  parts.append(f"p2_min={p2_min}")
                                    if p3_min is not None:  parts.append(f"p3_min={p3_min}")
                                    if volr_min is not None: parts.append(f"volratio_min={volr_min}")
                                    if volr_max is not None: parts.append(f"volratio_max={volr_max}")
                                    if ai_max is not None:  parts.append(f"ai_max={ai_max}")
                                    if hour_min is not None: parts.append(f"hour_min={hour_min}")
                                    if hour_max is not None: parts.append(f"hour_max={hour_max}")
                                    cond = ";".join(parts)

                                    a_lv, c_lv = _overlap_ac(
                                        rsi_min, rsi_max, mode_val, volr_min, volr_max, ai_max, p2_min
                                    )
                                    results.append({
                                        "condition": cond, "ok": ok, "fail": fail,
                                        "overlap_a": a_lv, "overlap_c": c_lv,
                                        "s30": s30, "s14": s14, "s7": s7,
                                        "sub_done": sub_done,
                                    })
    return results


# ==========================================
# メイン
# ==========================================

def main():
    print("=== SHORT_B 代替条件探索 ===")
    print(f"対象: 直近{LOOKBACK_DAYS}日 / Direction=SHORT")
    print(f"候補基準: 30d DONE>={MIN_DONE_30D} / wr>={MIN_WR*100:.0f}% / avg>0 / med>0 / active_days>=5")
    print(f"          + 7d wr>=45%(崩れ検出)\n")

    df_raw = fetch_all()
    df = preprocess(df_raw)

    now_jst = datetime.now(JST)
    cut30 = (now_jst - timedelta(days=30)).date()
    df_recent = df[df["_date"].notna() & (df["_date"] >= cut30)].copy()
    print(f"[INFO] 直近30日 全行: {len(df_recent)}件", flush=True)

    # ベースライン
    show_baseline(df_recent, now_jst)

    # グリッドサーチ
    print(f"\n[INFO] SHORT グリッドサーチ中...", flush=True)
    results = run_grid(df_recent, now_jst)
    print(f"\n[INFO] 候補数（DONE>={MIN_DONE_30D}）: {len(results)}", flush=True)

    def _no_major_overlap(r):
        return r["overlap_a"] != "HIGH(SHORT_A重複)" and r["overlap_c"] != "HIGH(SHORT_C重複)"

    # Section A: 全基準通過 + SHORT_A/C非重複
    sec_a = sorted(
        [r for r in results if r["ok"] and _no_major_overlap(r)],
        key=lambda r: (-r["s30"]["wr"], -r["s30"].get("avg", 0))
    )
    print(f"\n{'='*70}")
    print("  Section A: 実用候補（全基準通過 + SHORT_A/C非重複）")
    print(f"{'='*70}")
    if not sec_a:
        print("  実用候補なし")
    for r in sec_a[:15]:
        print(f"\n  ★ {r['condition']}")
        print(f"     overlap_A={r['overlap_a']}  overlap_C={r['overlap_c']}")
        print(_fmt(r["s30"], "30d"))
        print(_fmt(r["s14"], "14d"))
        print(_fmt(r["s7"],  " 7d"))
        print(_train_test(r["sub_done"]))

    # Section B: 全基準通過 + 重複あり
    sec_b = sorted(
        [r for r in results if r["ok"] and not _no_major_overlap(r)],
        key=lambda r: (-r["s30"]["wr"], -r["s30"].get("avg", 0))
    )
    print(f"\n{'='*70}")
    print("  Section B: 全基準通過（SHORT_A/C重複あり — 参考）")
    print(f"{'='*70}")
    if not sec_b:
        print("  なし")
    for r in sec_b[:10]:
        print(f"\n  (参考) {r['condition']}")
        print(f"     overlap_A={r['overlap_a']}  overlap_C={r['overlap_c']}")
        print(_fmt(r["s30"], "30d"))
        print(_fmt(r["s7"],  " 7d"))

    # Section C: 基準に近い候補
    sec_c = sorted(
        [r for r in results if not r["ok"]
         and r["s30"]["n"] >= MIN_DONE_30D
         and np.isfinite(r["s30"].get("wr", np.nan))
         and r["s30"]["wr"] >= MIN_WR - 0.05
         and _no_major_overlap(r)],
        key=lambda r: (-r["s30"]["wr"], -r["s30"].get("avg", 0))
    )
    print(f"\n{'='*70}")
    print("  Section C: 基準に近い候補（非重複）")
    print(f"{'='*70}")
    if not sec_c:
        print("  なし")
    for r in sec_c[:10]:
        print(f"\n  △ {r['condition']}")
        print(f"     fail={r['fail']}  overlap_A={r['overlap_a']}  overlap_C={r['overlap_c']}")
        print(_fmt(r["s30"], "30d"))
        print(_fmt(r["s7"],  " 7d"))

    # Section D: サマリ
    print(f"\n{'='*70}")
    print("  Section D: 総合サマリ")
    print(f"{'='*70}")
    print(f"  実用候補（A）: {len(sec_a)}件")
    print(f"  参考候補（B）: {len(sec_b)}件")
    print(f"  近い候補（C）: {len(sec_c)}件")

    if sec_a:
        best = sec_a[0]
        print(f"\n  ✅ SHORT_B枠の差し替え候補あり")
        print(f"  最有力: {best['condition']}")
        print(f"  30d: wr={best['s30']['wr']*100:.1f}%  avg={best['s30']['avg']:+.3f}  med={best['s30']['med']:+.3f}")
        print(f"  次: train/test 確認後、SHORT_B 条件を差し替えて ALLOW/BYPASS に追加")
    else:
        print(f"\n  ❌ SHORT_B枠の差し替え候補なし（現時点）")
        if sec_c:
            print(f"  → Section C の近い候補を参考に、条件緩和を検討")
        print(f"  → または mode=DOWN / UP でのSHORT候補を別途探索")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
