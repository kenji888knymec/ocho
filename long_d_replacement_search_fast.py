"""
long_d_replacement_search_fast.py — LONG_D枠 代替条件探索（高速版）

現LONG_Dを無効化するのではなく、LONG_D枠に入れる勝てる代替条件を高速に探索する。
総当たりグリッドサーチを廃止し、事前設計した候補リスト（< 10,000通り）で評価する。

現行条件（参考）:
  LONG_D: side=LONG;mode=UP;p1_max=1.0;volratio_min=2.0;volconfirmed=1
  LONG_E: side=LONG;mode=UP;rsi_min=40;rsi_max=50;p3_min=0.2
  LONG_F: side=LONG;mode=UP;rsi_min=55;rsi_max=65;p3_min=0.7;p1_min=0.5;p1_max=1.5;
           volratio_min=1.0;volratio_max=2.5;hour_min=6;hour_max=14

実行: python long_d_replacement_search_fast.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値）
read-only: デプロイ・env変更なし
"""

import os
import sys
import warnings
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import google.auth
import httplib2
import google_auth_httplib2
from googleapiclient.discovery import build

warnings.filterwarnings("ignore")

SPREADSHEET_ID = os.environ.get(
    "SPREADSHEET_ID", "1XwWkzijIwRlafg2zDgPHQ4tgjYModapFI3T_wbYS9_8"
)
SHEET = os.environ.get("V2_SHADOW_SHEET", "v2_shadow_ai")
HEADER_COL_END = "AZ"
MIN_DONE_30D = int(os.environ.get("MIN_DONE_30D", "10"))
MIN_WR = float(os.environ.get("MIN_WR", "0.55"))

t0 = time.time()


# ==========================================
# Google Sheets 取得
# ==========================================

def build_service():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    http = httplib2.Http(timeout=120)
    authed_http = google_auth_httplib2.AuthorizedHttp(creds, http=http)
    return build("sheets", "v4", http=authed_http, cache_discovery=False)


def fetch_sheet(svc):
    hdr = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=f"{SHEET}!A1:{HEADER_COL_END}1"
    ).execute().get("values", [[]])[0]
    headers = [str(h).strip() for h in hdr]

    col_a = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=f"{SHEET}!A:A"
    ).execute().get("values", [])
    last_row = len(col_a)
    start_row = max(2, last_row - 60000)

    rows = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET}!A{start_row}:{HEADER_COL_END}",
    ).execute().get("values", []) or []

    n = len(headers)
    fixed = [r[:n] + [""] * max(0, n - len(r)) for r in rows]
    return pd.DataFrame(fixed, columns=headers)


# ==========================================
# ユーティリティ
# ==========================================

def col(df, name, default=""):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)

def num(df, name):
    return pd.to_numeric(col(df, name), errors="coerce")

def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "on"])

def done_mask(df):
    return col(df, "EvalStatus").astype(str).str.upper().eq("DONE")

def win_mask(df):
    return col(df, "WinLose").astype(str).str.lower().eq("win")


# ==========================================
# 統計計算
# ==========================================

def stat_block(done_sub, latest_dt, days):
    sub = done_sub[done_sub["_dt"] >= latest_dt - pd.Timedelta(days=days)].copy()
    n = len(sub)
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg": np.nan, "med": np.nan,
                "active": 0, "zero": days, "median_day": 0, "max_day": 0}
    wins = win_mask(sub).sum()
    pnl = pd.to_numeric(col(sub, "PnL_Pct"), errors="coerce")
    daily = sub["_dt"].dt.date.value_counts()
    return {
        "n": n,
        "wr": wins / n,
        "avg": pnl.mean(),
        "med": pnl.median(),
        "active": int(daily.shape[0]),
        "zero": max(0, days - int(daily.shape[0])),
        "median_day": float(daily.median()) if len(daily) else 0,
        "max_day": int(daily.max()) if len(daily) else 0,
    }


def fmt_stat(s, label):
    if s["n"] == 0:
        return f"{label}: n=0"
    return (
        f"{label}: n={s['n']:4d}  wr={s['wr']*100:5.1f}%  "
        f"avg={s['avg']:+.3f}  med={s['med']:+.3f}  "
        f"active={s['active']}d  zero={s['zero']}d  "
        f"median/d={s['median_day']:.1f}  max/d={s['max_day']}"
    )


def train_test_split_stats(done_30, latest_dt):
    sub = done_30.sort_values("_dt").copy()
    if len(sub) < 6:
        return None, None
    mid = len(sub) // 2
    tr = sub.iloc[:mid]
    te = sub.iloc[mid:]
    def mini_stat(d):
        n = len(d)
        if n == 0:
            return {"n": 0, "wr": np.nan, "avg": np.nan}
        pnl = pd.to_numeric(col(d, "PnL_Pct"), errors="coerce")
        return {"n": n, "wr": win_mask(d).sum() / n, "avg": pnl.mean()}
    return mini_stat(tr), mini_stat(te)


def fmt_train_test(tr, te):
    if tr is None:
        return "  train/test: n<6 (not split)"
    delta = (te["wr"] - tr["wr"]) * 100 if (
        np.isfinite(tr["wr"]) and np.isfinite(te["wr"])
    ) else np.nan
    flag = ""
    if np.isfinite(delta):
        if delta < -20:
            flag = " ⚠ 崩れ(-20%超)"
        elif delta < -10:
            flag = " △ やや低下"
    return (
        f"  train: n={tr['n']} wr={tr['wr']*100:.1f}% avg={tr['avg']:+.3f} | "
        f"test: n={te['n']} wr={te['wr']*100:.1f}% avg={te['avg']:+.3f} | "
        f"delta={delta:+.1f}%{flag}"
    )


# ==========================================
# 重複チェック
# ==========================================

def overlap_level(cand_done_idx, ref_done_idx):
    if len(cand_done_idx) == 0:
        return "NONE", 0.0
    inter = cand_done_idx.intersection(ref_done_idx)
    ratio = len(inter) / len(cand_done_idx)
    if ratio >= 0.70:
        return "HIGH", ratio
    if ratio >= 0.30:
        return "MED", ratio
    if ratio > 0:
        return "LOW", ratio
    return "NONE", 0.0


# ==========================================
# 候補評価
# ==========================================

def pass_criteria(s30, s7):
    reasons = []
    if s30["n"] < MIN_DONE_30D:
        reasons.append(f"30d_n<{MIN_DONE_30D}")
    if not np.isfinite(s30["wr"]) or s30["wr"] < MIN_WR:
        reasons.append(f"30d_wr<{MIN_WR*100:.0f}%")
    if not np.isfinite(s30["avg"]) or s30["avg"] <= 0:
        reasons.append("30d_avg<=0")
    if not np.isfinite(s30["med"]) or s30["med"] <= 0:
        reasons.append("30d_med<=0")
    if s30["active"] < 5:
        reasons.append("active_days<5")
    if s30["max_day"] > 15:
        reasons.append("max/day>15")
    if s7["n"] >= 3 and np.isfinite(s7["wr"]) and s7["wr"] < 0.45:
        reasons.append("7d_wr<45%")
    return len(reasons) == 0, ",".join(reasons)


def near_pass(s30):
    return (
        s30["n"] >= MIN_DONE_30D
        and np.isfinite(s30["wr"]) and s30["wr"] >= 0.48
        and np.isfinite(s30["avg"]) and s30["avg"] > -0.1
    )


def evaluate_long_candidates(df, latest_dt, candidates, long_e_done_idx, long_f_done_idx):
    done_base = df[done_mask(df)].copy()
    results = []
    for label, mask in candidates:
        cand_done = df[mask & done_mask(df)].copy()
        if len(cand_done) < 3:
            continue
        cand_done_30 = cand_done[cand_done["_dt"] >= latest_dt - pd.Timedelta(days=30)]
        if len(cand_done_30) < 3:
            continue

        s30 = stat_block(cand_done, latest_dt, 30)
        s14 = stat_block(cand_done, latest_dt, 14)
        s7  = stat_block(cand_done, latest_dt, 7)

        tr, te = train_test_split_stats(cand_done_30, latest_dt)

        ok, fail = pass_criteria(s30, s7)

        ov_e, rat_e = overlap_level(cand_done_30.index, long_e_done_idx)
        ov_f, rat_f = overlap_level(cand_done_30.index, long_f_done_idx)
        overlap_text = f"LONG_E:{ov_e}({rat_e:.0%}) LONG_F:{ov_f}({rat_f:.0%})"
        max_rank = max(
            {"NONE": 0, "LOW": 1, "MED": 2, "HIGH": 3}[ov_e],
            {"NONE": 0, "LOW": 1, "MED": 2, "HIGH": 3}[ov_f],
        )

        results.append({
            "label": label, "s30": s30, "s14": s14, "s7": s7,
            "tr": tr, "te": te,
            "ok": ok, "fail": fail,
            "overlap_text": overlap_text, "max_rank": max_rank,
            "near": near_pass(s30),
        })

    results.sort(key=lambda r: (
        r["ok"],
        -r["max_rank"],
        r["s30"]["wr"] if np.isfinite(r["s30"]["wr"]) else -1,
        r["s30"]["avg"] if np.isfinite(r["s30"]["avg"]) else -999,
        r["s30"]["n"],
    ), reverse=True)
    return results


def print_candidate(r):
    print(f"\n  ★ {r['label']}")
    print(f"     overlap: {r['overlap_text']}")
    print("  " + fmt_stat(r["s30"], "30d"))
    print("  " + fmt_stat(r["s14"], "14d"))
    print("  " + fmt_stat(r["s7"],  " 7d"))
    print(fmt_train_test(r["tr"], r["te"]))


def print_sections(results, slot_name, ref_names):
    sec_a = [r for r in results if r["ok"] and r["max_rank"] <= 1]
    sec_b = [r for r in results if r["ok"] and r["max_rank"] >= 2]
    sec_c = [r for r in results if not r["ok"] and r["near"]]

    print(f"\n{'='*78}")
    print(f"Section A: 実用候補（{ref_names}と非重複 or LOW重複）")
    print(f"{'='*78}")
    if sec_a:
        for r in sec_a[:10]:
            print_candidate(r)
    else:
        print("  候補なし")

    print(f"\n{'='*78}")
    print(f"Section B: 勝っているが{ref_names}と重複大（参考）")
    print(f"{'='*78}")
    if sec_b:
        for r in sec_b[:8]:
            print_candidate(r)
    else:
        print("  候補なし")

    print(f"\n{'='*78}")
    print(f"Section C: 近いが基準未達")
    print(f"{'='*78}")
    if sec_c:
        for r in sec_c[:8]:
            print(f"\n  △ {r['label']}")
            print(f"     fail={r['fail']}  overlap={r['overlap_text']}")
            print("  " + fmt_stat(r["s30"], "30d"))
            print("  " + fmt_stat(r["s7"],  " 7d"))
    else:
        print("  候補なし")

    print(f"\n{'='*78}")
    print(f"Section D: 総合サマリ — {slot_name}枠")
    print(f"{'='*78}")
    print(f"Section A: {len(sec_a)}件  Section B: {len(sec_b)}件  Section C: {len(sec_c)}件")
    if sec_a:
        best = sec_a[0]
        print(f"\n{slot_name}枠を差し替える候補: あり")
        print(f"最有力候補条件:\n  {best['label']}")
        print("  " + fmt_stat(best["s30"], "30d"))
        print("  " + fmt_stat(best["s7"],  " 7d"))
        print(fmt_train_test(best["tr"], best["te"]))
    else:
        print(f"\n{slot_name}枠を差し替える候補: なし")
        print("  現時点では差し替え見送り または 探索条件の再設計が必要。")

    print()


# ==========================================
# メイン
# ==========================================

def main():
    print("=" * 78)
    print("long_d_replacement_search_fast.py")
    print("LONG_D枠 代替条件探索（高速版）  read-only")
    print("=" * 78)

    print("\n[1/3] Fetching data from Google Sheets...")
    svc = build_service()
    df_raw = fetch_sheet(svc)

    df_raw["_dt"] = pd.to_datetime(col(df_raw, "Datetime_JST"), errors="coerce")
    df_raw = df_raw.dropna(subset=["_dt"]).copy()
    latest_dt = df_raw["_dt"].max()
    df = df_raw[df_raw["_dt"] >= latest_dt - pd.Timedelta(days=30)].copy()

    df["_dir"]  = col(df, "Direction").astype(str).str.upper()
    df["_mode"] = col(df, "BTC_Mode_Compat").astype(str).str.upper()
    df["_rsi"]  = num(df, "RSI")
    df["_p1"]   = num(df, "P1_TrendScore")
    df["_p2"]   = num(df, "P2_FundingScore")
    df["_p3"]   = num(df, "P3_VolumeScore")
    df["_vol"]  = num(df, "VolRatio")
    df["_ai"]   = num(df, "AI_Prob_Win")
    df["_hour"] = num(df, "Hour_JST")
    df["_vc"]   = truthy(col(df, "VolConfirmed"))
    df["_fr"]   = truthy(col(df, "FR_Available"))

    print(f"  date_range : {df['_dt'].min().date()} → {latest_dt.date()}")
    print(f"  rows_30d   : {len(df)}")
    print(f"  DONE_30d   : {int(done_mask(df).sum())}")

    # --------------------------------------------------
    # 現行プロファイルのマスク
    # --------------------------------------------------
    long_d_mask = (
        (df["_dir"] == "LONG") & (df["_mode"] == "UP")
        & (df["_p1"] <= 1.0) & (df["_vol"] >= 2.0) & df["_vc"]
    )
    long_e_mask = (
        (df["_dir"] == "LONG") & (df["_mode"] == "UP")
        & (df["_rsi"] >= 40) & (df["_rsi"] < 50) & (df["_p3"] >= 0.2)
    )
    long_f_mask = (
        (df["_dir"] == "LONG") & (df["_mode"] == "UP")
        & (df["_rsi"] >= 55) & (df["_rsi"] < 65)
        & (df["_p3"] >= 0.7)
        & (df["_p1"] >= 0.5) & (df["_p1"] < 1.5)
        & (df["_vol"] >= 1.0) & (df["_vol"] < 2.5)
        & (df["_hour"] >= 6) & (df["_hour"] < 14)
    )

    long_e_done_idx = df[long_e_mask & done_mask(df)].index
    long_f_done_idx = df[long_f_mask & done_mask(df)].index

    # --------------------------------------------------
    # [2/3] ベースライン表示
    # --------------------------------------------------
    print("\n[2/3] Baselines")

    for name, mask, note in [
        ("LONG_D (current)", long_d_mask,
         "side=LONG;mode=UP;p1_max=1.0;volratio_min=2.0;volconfirmed=1"),
        ("LONG_E (active)",  long_e_mask,
         "side=LONG;mode=UP;rsi_min=40;rsi_max=50;p3_min=0.2"),
        ("LONG_F (active)",  long_f_mask,
         "side=LONG;mode=UP;rsi_min=55;rsi_max=65;p3_min=0.7;p1_min=0.5;p1_max=1.5;"
         "volratio_min=1.0;volratio_max=2.5;hour_min=6;hour_max=14"),
    ]:
        d = df[mask & done_mask(df)].copy()
        print(f"\n  {name}")
        print(f"  条件: {note}")
        print("  " + fmt_stat(stat_block(d, latest_dt, 30), "30d"))
        print("  " + fmt_stat(stat_block(d, latest_dt, 14), "14d"))
        print("  " + fmt_stat(stat_block(d, latest_dt,  7), " 7d"))

    # LONG_D の BTC_Mode 別内訳
    print("\n  LONG_D BTC_Mode breakdown (all LONG, 30d):")
    for mode in ["UP", "RANGE", "DOWN"]:
        m = (df["_dir"] == "LONG") & (df["_mode"] == mode) & done_mask(df)
        d = df[m].copy()
        s = stat_block(d, latest_dt, 30)
        print(f"    mode={mode}: " + fmt_stat(s, "30d"))

    # --------------------------------------------------
    # [3/3] 候補探索
    # --------------------------------------------------
    print("\n[3/3] Searching candidates (pre-defined list, no grid)...")
    IS_LONG = df["_dir"] == "LONG"

    # LONG_E/Fと被らないRSI帯を中心に設計
    # LONG_E = UP RSI40-50 p3>=0.2
    # LONG_F = UP RSI55-65 p3>=0.7 p1[0.5,1.5] vol[1.0,2.5] hour[6,14]

    UP = df["_mode"] == "UP"
    RNG = df["_mode"] == "RANGE"

    candidates = []

    # ---- Group 1: mode=UP, RSI帯 35-45（LONG_E下端・未探索）----
    for p3 in [0.2, 0.5, 0.7]:
        for p1_lo, p1_hi in [(0.0, 1.5), (1.5, 3.0), (2.0, 4.0)]:
            for ai_max in [0.6, 0.4]:
                lbl = (
                    f"side=LONG;mode=UP;rsi_min=35;rsi_max=45;"
                    f"p3_min={p3};p1_min={p1_lo};p1_max={p1_hi};"
                    f"ai_max={ai_max}"
                )
                msk = (
                    IS_LONG & UP
                    & (df["_rsi"] >= 35) & (df["_rsi"] < 45)
                    & (df["_p3"] >= p3)
                    & (df["_p1"] >= p1_lo) & (df["_p1"] < p1_hi)
                    & (df["_ai"] <= ai_max)
                )
                candidates.append((lbl, msk))

    # ---- Group 2: mode=UP, RSI帯 45-52（LONG_E上端と微重複）----
    for p3 in [0.2, 0.5, 0.7, 1.0]:
        for p1_lo, p1_hi in [(1.5, 3.0), (2.0, 4.0), (0.5, 1.5)]:
            for vol_lo, vol_hi in [(1.0, 2.0), (1.0, 2.5)]:
                for hour_lo, hour_hi in [(0, 24), (8, 18), (10, 20)]:
                    lbl = (
                        f"side=LONG;mode=UP;rsi_min=45;rsi_max=52;"
                        f"p3_min={p3};p1_min={p1_lo};p1_max={p1_hi};"
                        f"volratio_min={vol_lo};volratio_max={vol_hi};"
                        f"hour_min={hour_lo};hour_max={hour_hi}"
                    )
                    msk = (
                        IS_LONG & UP
                        & (df["_rsi"] >= 45) & (df["_rsi"] < 52)
                        & (df["_p3"] >= p3)
                        & (df["_p1"] >= p1_lo) & (df["_p1"] < p1_hi)
                        & (df["_vol"] >= vol_lo) & (df["_vol"] < vol_hi)
                        & (df["_hour"] >= hour_lo) & (df["_hour"] < hour_hi)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 3: mode=UP, RSI帯 50-55（LONG_EとFの間）----
    for p3 in [0.5, 0.7, 1.0]:
        for p1_lo, p1_hi in [(1.5, 3.0), (2.0, 4.0)]:
            for hour_lo, hour_hi in [(0, 24), (16, 24), (8, 18)]:
                lbl = (
                    f"side=LONG;mode=UP;rsi_min=50;rsi_max=55;"
                    f"p3_min={p3};p1_min={p1_lo};p1_max={p1_hi};"
                    f"hour_min={hour_lo};hour_max={hour_hi}"
                )
                msk = (
                    IS_LONG & UP
                    & (df["_rsi"] >= 50) & (df["_rsi"] < 55)
                    & (df["_p3"] >= p3)
                    & (df["_p1"] >= p1_lo) & (df["_p1"] < p1_hi)
                    & (df["_hour"] >= hour_lo) & (df["_hour"] < hour_hi)
                )
                candidates.append((lbl, msk))

    # ---- Group 4: mode=UP, RSI帯 60-70（LONG_F上端より上）----
    for p3 in [0.5, 0.7, 1.0]:
        for p1_lo, p1_hi in [(0.5, 1.5), (1.0, 2.0), (1.5, 2.5)]:
            for vol_lo, vol_hi in [(1.0, 2.0), (1.0, 2.5)]:
                for hour_lo, hour_hi in [(0, 24), (6, 14), (8, 18)]:
                    lbl = (
                        f"side=LONG;mode=UP;rsi_min=60;rsi_max=70;"
                        f"p3_min={p3};p1_min={p1_lo};p1_max={p1_hi};"
                        f"volratio_min={vol_lo};volratio_max={vol_hi};"
                        f"hour_min={hour_lo};hour_max={hour_hi}"
                    )
                    msk = (
                        IS_LONG & UP
                        & (df["_rsi"] >= 60) & (df["_rsi"] < 70)
                        & (df["_p3"] >= p3)
                        & (df["_p1"] >= p1_lo) & (df["_p1"] < p1_hi)
                        & (df["_vol"] >= vol_lo) & (df["_vol"] < vol_hi)
                        & (df["_hour"] >= hour_lo) & (df["_hour"] < hour_hi)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 5: mode=RANGE（LONG_E/Fは全てUPのため基本非重複）----
    for rsi_lo, rsi_hi in [(35, 45), (40, 50), (45, 55), (50, 60), (55, 65)]:
        for p3 in [0.2, 0.5, 0.7]:
            for p1_lo, p1_hi in [(0.5, 1.5), (1.0, 2.0), (1.5, 3.0)]:
                lbl = (
                    f"side=LONG;mode=RANGE;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                    f"p3_min={p3};p1_min={p1_lo};p1_max={p1_hi}"
                )
                msk = (
                    IS_LONG & RNG
                    & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                    & (df["_p3"] >= p3)
                    & (df["_p1"] >= p1_lo) & (df["_p1"] < p1_hi)
                )
                candidates.append((lbl, msk))

    # ---- Group 6: mode=UP, 高P1（LONG_D的な高ボラ候補の代替）----
    for rsi_lo, rsi_hi in [(35, 50), (45, 55), (50, 60)]:
        for p1_lo in [2.0, 2.5, 3.0]:
            for p3 in [0.2, 0.5]:
                for ai_max in [0.6, 0.4, 0.2]:
                    lbl = (
                        f"side=LONG;mode=UP;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                        f"p1_min={p1_lo};p3_min={p3};ai_max={ai_max}"
                    )
                    msk = (
                        IS_LONG & UP
                        & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                        & (df["_p1"] >= p1_lo)
                        & (df["_p3"] >= p3)
                        & (df["_ai"] <= ai_max)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 7: VolConfirmed=1（LONG_D的特徴の代替）----
    for rsi_lo, rsi_hi in [(35, 50), (45, 55), (50, 65), (55, 70)]:
        for p3 in [0.2, 0.5, 0.7]:
            for mode_mask, mode_str in [(UP, "UP"), (RNG, "RANGE")]:
                lbl = (
                    f"side=LONG;mode={mode_str};rsi_min={rsi_lo};rsi_max={rsi_hi};"
                    f"p3_min={p3};volconfirmed=1"
                )
                msk = (
                    IS_LONG & mode_mask
                    & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                    & (df["_p3"] >= p3)
                    & df["_vc"]
                )
                candidates.append((lbl, msk))

    print(f"  候補数: {len(candidates)}")

    results = evaluate_long_candidates(
        df, latest_dt, candidates, long_e_done_idx, long_f_done_idx
    )

    print_sections(results, "LONG_D", "LONG_E/F")

    elapsed = time.time() - t0
    print(f"経過時間: {elapsed:.1f}秒")
    print("\nread-only完了。デプロイ・env変更なし。")


if __name__ == "__main__":
    main()
