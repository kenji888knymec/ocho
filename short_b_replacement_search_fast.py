"""
short_b_replacement_search_fast.py — SHORT_B枠 代替条件探索（高速版）

SHORT_B枠に入れる勝てる代替条件を高速に探索する。
SHORT_A/Cとの重複チェックと、short_recent_pause/profile no_matchの分離も実施。

現行条件（参考）:
  SHORT_B: side=SHORT;mode=RANGE;ltf_aligned=0;p2_min=0;rsi_min=40;rsi_max=45
  SHORT_A: side=SHORT;mode=RANGE;rsi_min=50;rsi_max=55;p2_min=0;volratio_max=1.2;ai_max=0.6
  SHORT_C: side=SHORT;mode=DOWN;rsi_min=50;rsi_max=60;volratio_min=1.2;volratio_max=2.5;ai_max=0.6

実行: python short_b_replacement_search_fast.py
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

def notify_sent_mask(df):
    direct = col(df, "Notify_Sent").astype(str).str.lower().isin(["1", "true", "yes"])
    note   = col(df, "AI_Note").astype(str)
    return direct | note.str.contains("notify_sent=1", case=False, na=False)

def route_mask(df, profile_name):
    masks = []
    for c in ["Feature_Allow_Hits", "Feature_Bypass_Profile", "AI_Note", "Notify_Reason"]:
        if c in df.columns:
            masks.append(df[c].astype(str).str.contains(profile_name, case=False, na=False))
    if not masks:
        return pd.Series(False, index=df.index)
    out = masks[0].copy()
    for m in masks[1:]:
        out = out | m
    return out


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


def train_test_split_stats(done_30):
    sub = done_30.sort_values("_dt").copy()
    if len(sub) < 6:
        return None, None
    mid = len(sub) // 2
    tr, te = sub.iloc[:mid], sub.iloc[mid:]
    def mini(d):
        n = len(d)
        if n == 0:
            return {"n": 0, "wr": np.nan, "avg": np.nan}
        pnl = pd.to_numeric(col(d, "PnL_Pct"), errors="coerce")
        return {"n": n, "wr": win_mask(d).sum() / n, "avg": pnl.mean()}
    return mini(tr), mini(te)


def fmt_train_test(tr, te):
    if tr is None:
        return "  train/test: n<6"
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

def overlap_level(cand_idx, ref_idx):
    if len(cand_idx) == 0:
        return "NONE", 0.0
    inter = cand_idx.intersection(ref_idx)
    ratio = len(inter) / len(cand_idx)
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
    if s30["max_day"] > 20:
        reasons.append("max/day>20")
    if s7["n"] >= 3 and np.isfinite(s7["wr"]) and s7["wr"] < 0.45:
        reasons.append("7d_wr<45%")
    return len(reasons) == 0, ",".join(reasons)


def near_pass(s30):
    return (
        s30["n"] >= MIN_DONE_30D
        and np.isfinite(s30["wr"]) and s30["wr"] >= 0.48
        and np.isfinite(s30["avg"]) and s30["avg"] > -0.1
    )


def evaluate_short_candidates(df, latest_dt, candidates, short_a_done_idx, short_c_done_idx):
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

        tr, te = train_test_split_stats(cand_done_30)

        ok, fail = pass_criteria(s30, s7)

        ov_a, rat_a = overlap_level(cand_done_30.index, short_a_done_idx)
        ov_c, rat_c = overlap_level(cand_done_30.index, short_c_done_idx)
        overlap_text = f"SHORT_A:{ov_a}({rat_a:.0%}) SHORT_C:{ov_c}({rat_c:.0%})"
        max_rank = max(
            {"NONE": 0, "LOW": 1, "MED": 2, "HIGH": 3}[ov_a],
            {"NONE": 0, "LOW": 1, "MED": 2, "HIGH": 3}[ov_c],
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
    print("short_b_replacement_search_fast.py")
    print("SHORT_B枠 代替条件探索（高速版）  read-only")
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
    df["_ltf"]  = truthy(col(df, "LTF_Aligned"))
    df["_vc"]   = truthy(col(df, "VolConfirmed"))
    df["_note"] = col(df, "AI_Note").astype(str)

    print(f"  date_range : {df['_dt'].min().date()} → {latest_dt.date()}")
    print(f"  rows_30d   : {len(df)}")
    print(f"  DONE_30d   : {int(done_mask(df).sum())}")

    # --------------------------------------------------
    # 現行プロファイルのマスク
    # --------------------------------------------------
    IS_SHORT = df["_dir"] == "SHORT"
    RNG  = df["_mode"] == "RANGE"
    DOWN = df["_mode"] == "DOWN"
    UP   = df["_mode"] == "UP"

    short_a_mask = (
        IS_SHORT & RNG
        & (df["_rsi"] >= 50) & (df["_rsi"] < 55)
        & (df["_p2"] >= 0)
        & (df["_vol"] <= 1.2)
        & (df["_ai"] <= 0.6)
    )
    short_b_mask = (
        IS_SHORT & RNG
        & (~df["_ltf"])
        & (df["_p2"] >= 0)
        & (df["_rsi"] >= 40) & (df["_rsi"] < 45)
    )
    short_c_mask = (
        IS_SHORT & DOWN
        & (df["_rsi"] >= 50) & (df["_rsi"] < 60)
        & (df["_vol"] >= 1.2) & (df["_vol"] < 2.5)
        & (df["_ai"] <= 0.6)
    )

    short_a_done_idx = df[short_a_mask & done_mask(df)].index
    short_c_done_idx = df[short_c_mask & done_mask(df)].index

    # --------------------------------------------------
    # [2/3] ベースライン表示
    # --------------------------------------------------
    print("\n[2/3] Baselines")

    for name, mask, note in [
        ("SHORT_A (active)", short_a_mask,
         "side=SHORT;mode=RANGE;rsi_min=50;rsi_max=55;p2_min=0;volratio_max=1.2;ai_max=0.6"),
        ("SHORT_B (current)", short_b_mask,
         "side=SHORT;mode=RANGE;ltf_aligned=0;p2_min=0;rsi_min=40;rsi_max=45"),
        ("SHORT_C (active)", short_c_mask,
         "side=SHORT;mode=DOWN;rsi_min=50;rsi_max=60;volratio_min=1.2;volratio_max=2.5;ai_max=0.6"),
    ]:
        d = df[mask & done_mask(df)].copy()
        print(f"\n  {name}")
        print(f"  条件: {note}")
        print("  " + fmt_stat(stat_block(d, latest_dt, 30), "30d"))
        print("  " + fmt_stat(stat_block(d, latest_dt, 14), "14d"))
        print("  " + fmt_stat(stat_block(d, latest_dt,  7), " 7d"))

    # SHORT_B の BTC_Mode 別内訳
    print("\n  SHORT BTC_Mode breakdown (all SHORT, 30d DONE):")
    for mode_mask, mode_str in [(RNG, "RANGE"), (DOWN, "DOWN"), (UP, "UP")]:
        d = df[IS_SHORT & mode_mask & done_mask(df)].copy()
        s = stat_block(d, latest_dt, 30)
        print(f"    mode={mode_str}: " + fmt_stat(s, "30d"))

    # SHORT_B ファネル（直近7日）
    print("\n  SHORT_B recent funnel (7d):")
    sb7 = df[short_b_mask & (df["_dt"] >= latest_dt - pd.Timedelta(days=7))].copy()
    note7 = sb7["_note"]
    recent_pause = note7.str.contains("short_recent_pause", case=False, na=False).sum()
    no_match     = note7.str.contains("feature_profile_only_no_match", case=False, na=False).sum()
    route_cnt    = route_mask(sb7, "short_b").sum()
    notify_cnt   = notify_sent_mask(sb7).sum()
    done_cnt     = (notify_sent_mask(sb7) & done_mask(sb7)).sum()
    print(f"    A候補(7d)        : {len(sb7)}")
    print(f"    recent_pause落ち : {int(recent_pause)}")
    print(f"    profile_no_match : {int(no_match)}")
    print(f"    route通過        : {int(route_cnt)}")
    print(f"    notify_sent      : {int(notify_cnt)}")
    print(f"    D (DONE)         : {int(done_cnt)}")

    # --------------------------------------------------
    # [3/3] 候補探索
    # --------------------------------------------------
    print("\n[3/3] Searching candidates (pre-defined list, no grid)...")
    candidates = []

    # ---- Group 1: mode=RANGE, RSI 35-45（SHORT_B現条件周辺・条件改善）----
    # 現SHORT_B: RANGE, RSI 40-45, ltf=0, p2>=0 → 勝率27% 弱い
    for rsi_lo, rsi_hi in [(35, 45), (38, 46), (40, 48)]:
        for p2_min in [0.0, 0.2]:
            for p3_min in [0.2, 0.5]:
                for ai_max in [0.4, 0.2]:
                    lbl = (
                        f"side=SHORT;mode=RANGE;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                        f"p2_min={p2_min};p3_min={p3_min};ai_max={ai_max}"
                    )
                    msk = (
                        IS_SHORT & RNG
                        & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                        & (df["_p2"] >= p2_min)
                        & (df["_p3"] >= p3_min)
                        & (df["_ai"] <= ai_max)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 2: mode=RANGE, RSI 45-55（SHORT_A下端・隣接帯）----
    # SHORT_Aは RSI 50-55 → RSI 45-50 or 46-54 は非重複候補
    for rsi_lo, rsi_hi in [(45, 50), (46, 54), (45, 55)]:
        for vol_max in [1.0, 1.2, 2.0]:
            for p3_min in [0.2, 0.5, 0.7]:
                for ai_max in [0.6, 0.4]:
                    lbl = (
                        f"side=SHORT;mode=RANGE;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                        f"p3_min={p3_min};volratio_max={vol_max};ai_max={ai_max}"
                    )
                    msk = (
                        IS_SHORT & RNG
                        & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                        & (df["_p3"] >= p3_min)
                        & (df["_vol"] <= vol_max)
                        & (df["_ai"] <= ai_max)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 3: mode=RANGE, RSI 55-65（SHORT_A上端より上）----
    for rsi_lo, rsi_hi in [(55, 65), (58, 68)]:
        for p1_lo in [1.5, 2.0]:
            for p3_min in [0.5, 0.7]:
                for vol_lo, vol_hi in [(1.0, 2.5), (0.5, 1.5)]:
                    for ai_max in [0.6, 0.4]:
                        lbl = (
                            f"side=SHORT;mode=RANGE;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                            f"p1_min={p1_lo};p3_min={p3_min};"
                            f"volratio_min={vol_lo};volratio_max={vol_hi};ai_max={ai_max}"
                        )
                        msk = (
                            IS_SHORT & RNG
                            & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                            & (df["_p1"] >= p1_lo)
                            & (df["_p3"] >= p3_min)
                            & (df["_vol"] >= vol_lo) & (df["_vol"] < vol_hi)
                            & (df["_ai"] <= ai_max)
                        )
                        candidates.append((lbl, msk))

    # ---- Group 4: mode=DOWN, RSI 35-50（SHORT_C下端より低いRSI）----
    # SHORT_CはDOWN RSI 50-60 → RSI 35-50はSHORT_Cと重複なし
    for rsi_lo, rsi_hi in [(35, 45), (40, 50), (45, 52)]:
        for vol_lo, vol_hi in [(1.0, 2.5), (1.2, 2.5), (0.5, 1.5)]:
            for p2_min in [0.0, -0.5]:
                for ai_max in [0.6, 0.4]:
                    lbl = (
                        f"side=SHORT;mode=DOWN;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                        f"p2_min={p2_min};"
                        f"volratio_min={vol_lo};volratio_max={vol_hi};ai_max={ai_max}"
                    )
                    msk = (
                        IS_SHORT & DOWN
                        & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                        & (df["_p2"] >= p2_min)
                        & (df["_vol"] >= vol_lo) & (df["_vol"] < vol_hi)
                        & (df["_ai"] <= ai_max)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 5: mode=DOWN, RSI 55-70（SHORT_C上端より高い）----
    # SHORT_CはDOWN RSI 50-60 → RSI 60-70はSHORT_Cと最小重複
    for rsi_lo, rsi_hi in [(55, 65), (60, 70), (58, 68)]:
        for p1_lo in [1.0, 1.5, 2.0]:
            for vol_lo, vol_hi in [(1.0, 2.5), (1.2, 2.5)]:
                for ai_max in [0.6, 0.4]:
                    lbl = (
                        f"side=SHORT;mode=DOWN;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                        f"p1_min={p1_lo};"
                        f"volratio_min={vol_lo};volratio_max={vol_hi};ai_max={ai_max}"
                    )
                    msk = (
                        IS_SHORT & DOWN
                        & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                        & (df["_p1"] >= p1_lo)
                        & (df["_vol"] >= vol_lo) & (df["_vol"] < vol_hi)
                        & (df["_ai"] <= ai_max)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 6: mode=UP（SHORT_A/CはRANGE/DOWN → UP全体はほぼ非重複）----
    for rsi_lo, rsi_hi in [(50, 60), (55, 65), (60, 70)]:
        for p1_lo in [1.5, 2.0, 2.5]:
            for p3_min in [0.5, 0.7]:
                for ai_max in [0.6, 0.4]:
                    lbl = (
                        f"side=SHORT;mode=UP;rsi_min={rsi_lo};rsi_max={rsi_hi};"
                        f"p1_min={p1_lo};p3_min={p3_min};ai_max={ai_max}"
                    )
                    msk = (
                        IS_SHORT & UP
                        & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                        & (df["_p1"] >= p1_lo)
                        & (df["_p3"] >= p3_min)
                        & (df["_ai"] <= ai_max)
                    )
                    candidates.append((lbl, msk))

    # ---- Group 7: LTF_Alignedフィルター（SHORT_Bの特徴的な条件）----
    for rsi_lo, rsi_hi in [(40, 50), (45, 55), (35, 45)]:
        for mode_mask, mode_str in [(RNG, "RANGE"), (DOWN, "DOWN")]:
            for p3_min in [0.2, 0.5]:
                for ai_max in [0.4, 0.2]:
                    # ltf=0（SHORT_B現条件と同じ方向）
                    lbl = (
                        f"side=SHORT;mode={mode_str};rsi_min={rsi_lo};rsi_max={rsi_hi};"
                        f"p3_min={p3_min};ai_max={ai_max};ltf_aligned=0"
                    )
                    msk = (
                        IS_SHORT & mode_mask
                        & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                        & (df["_p3"] >= p3_min)
                        & (df["_ai"] <= ai_max)
                        & (~df["_ltf"])
                    )
                    candidates.append((lbl, msk))

    # ---- Group 8: P2 negative（資金調達コスト逆流を狙う）----
    for rsi_lo, rsi_hi in [(40, 55), (45, 60)]:
        for mode_mask, mode_str in [(RNG, "RANGE"), (DOWN, "DOWN")]:
            for p3_min in [0.2, 0.5, 0.7]:
                lbl = (
                    f"side=SHORT;mode={mode_str};rsi_min={rsi_lo};rsi_max={rsi_hi};"
                    f"p2_max=0;p3_min={p3_min}"
                )
                msk = (
                    IS_SHORT & mode_mask
                    & (df["_rsi"] >= rsi_lo) & (df["_rsi"] < rsi_hi)
                    & (df["_p2"] <= 0)
                    & (df["_p3"] >= p3_min)
                )
                candidates.append((lbl, msk))

    print(f"  候補数: {len(candidates)}")

    results = evaluate_short_candidates(
        df, latest_dt, candidates, short_a_done_idx, short_c_done_idx
    )

    print_sections(results, "SHORT_B", "SHORT_A/C")

    elapsed = time.time() - t0
    print(f"経過時間: {elapsed:.1f}秒")
    print("\nread-only完了。デプロイ・env変更なし。")


if __name__ == "__main__":
    main()
