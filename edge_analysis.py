"""
edge_analysis.py — v2_shadow_ai データのエッジ存在確認（一回限りの分析用）
実行: python edge_analysis.py
必要環境変数: SPREADSHEET_ID（省略時はデフォルト値を使用）
"""

import os
import sys
import json
import warnings
from datetime import timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import google.auth
import httplib2
from googleapiclient.discovery import build
from google_auth_httplib2 import AuthorizedHttp

warnings.filterwarnings("ignore")

# ---- 設定 ----
SPREADSHEET_ID = os.environ.get(
    "SPREADSHEET_ID",
    "1XwWkzijIwRlafg2zDgPHQ4tgjYModapFI3T_wbYS9_8",
)
V2_SHADOW_SHEET = os.environ.get("V2_SHADOW_SHEET", "v2_shadow_ai")
HEADER_COL_END = "AZ"   # 十分広く取る
MIN_SAMPLES = 20         # 統計的に扱う最低件数

JST = timezone(timedelta(hours=9))


# ==========================================
# Google Sheets 取得
# ==========================================

def _sheets_service(creds):
    http = httplib2.Http(timeout=120)
    authed_http = AuthorizedHttp(creds, http=http)
    return build("sheets", "v4", http=authed_http, cache_discovery=False)


def fetch_v2_shadow_done() -> pd.DataFrame:
    """v2_shadow_ai の EvalStatus=DONE 行のみ取得して DataFrame で返す。"""
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds, project = google.auth.default(scopes=scopes)
    svc = _sheets_service(creds)

    # ヘッダー取得
    hdr_res = svc.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{V2_SHADOW_SHEET}!A1:{HEADER_COL_END}1",
    ).execute()
    raw_hdr = (hdr_res.get("values", [[]]) or [[]])[0]
    headers = [str(h).strip() for h in raw_hdr]

    if not headers:
        raise RuntimeError("ヘッダーが空です")

    # データ取得
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

    print(f"[INFO] 総行数(ヘッダー除く): {len(df_all)}", flush=True)

    # EvalStatus == DONE に絞る
    if "EvalStatus" not in df_all.columns:
        raise RuntimeError("EvalStatus 列が見つかりません")

    status = df_all["EvalStatus"].astype(str).str.strip().str.upper()
    df_done = df_all[status == "DONE"].copy()
    print(f"[INFO] DONE件数: {len(df_done)}", flush=True)

    return df_done


# ==========================================
# 前処理ユーティリティ
# ==========================================

def _to_win(series: pd.Series) -> pd.Series:
    """WinLose列 → 1=Win / 0=Lose / NaN=不明"""
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


def _win_rate(sub: pd.Series) -> Optional[float]:
    """1/0 の Series から勝率を返す。有効値がなければ None"""
    valid = sub.dropna()
    if len(valid) == 0:
        return None
    return float(valid.mean() * 100)


def _chi2_pvalue(n_win_a, n_lose_a, n_win_b, n_lose_b) -> Optional[float]:
    """2x2 カイ二乗検定の p 値（簡易実装）"""
    try:
        from scipy.stats import chi2_contingency
        table = [[n_win_a, n_lose_a], [n_win_b, n_lose_b]]
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        return float(p)
    except ImportError:
        # scipy がなければ手計算
        try:
            obs = np.array([[n_win_a, n_lose_a], [n_win_b, n_lose_b]], dtype=float)
            row_sum = obs.sum(axis=1, keepdims=True)
            col_sum = obs.sum(axis=0, keepdims=True)
            total = obs.sum()
            if total == 0:
                return None
            expected = row_sum * col_sum / total
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.nansum((obs - expected) ** 2 / expected))
            # p値近似（自由度1）
            import math
            # 正規近似 (p ~ 2*(1-Φ(sqrt(chi2))))
            z = math.sqrt(chi2)
            # erfc 近似
            p_approx = math.erfc(z / math.sqrt(2))
            return float(p_approx)
        except Exception:
            return None


def _edge_verdict(diff: float, p: Optional[float], n_a: int, n_b: int) -> str:
    if n_a < MIN_SAMPLES or n_b < MIN_SAMPLES:
        return "⚠️ サンプル不足（判定保留）"
    if abs(diff) < 5.0:
        return "❌ エッジなし（差が小さい）"
    if p is not None and p < 0.05:
        return "✅ エッジあり（統計的有意）"
    if p is not None and p < 0.10:
        return f"△ 弱いエッジ（p={p:.3f}, 差={diff:+.1f}%）"
    p_str = f"{p:.3f}" if p is not None else "?"
    return f"△ 差あり（p={p_str}, 差={diff:+.1f}%）"


# ==========================================
# 分析1: TotalScore 高低で勝率差
# ==========================================

def analyze_totalscore(df: pd.DataFrame) -> dict:
    print("\n=== 分析1: TotalScore 高低（3以上 vs 3未満） ===")

    if "TotalScore" not in df.columns or "WinLose" not in df.columns:
        print("  → 必要列なし (TotalScore or WinLose)")
        return {}

    df = df.copy()
    df["_score"] = _to_num(df["TotalScore"])
    df["_win"]   = _to_win(df["WinLose"])

    # 閾値を 3.0 に固定
    THRESHOLD = 3.0
    high = df[df["_score"] >= THRESHOLD]["_win"]
    low  = df[df["_score"] <  THRESHOLD]["_win"]

    wr_high = _win_rate(high)
    wr_low  = _win_rate(low)
    n_high  = int(high.notna().sum())
    n_low   = int(low.notna().sum())

    diff = (wr_high or 0) - (wr_low or 0)

    # p値
    n_win_h  = int((high == 1).sum())
    n_lose_h = int((high == 0).sum())
    n_win_l  = int((low  == 1).sum())
    n_lose_l = int((low  == 0).sum())
    p = _chi2_pvalue(n_win_h, n_lose_h, n_win_l, n_lose_l)

    # スコア分布も見る（10段階）
    score_stats = df["_score"].describe()

    print(f"  TotalScore >= {THRESHOLD}: n={n_high}, 勝率={wr_high:.1f}%" if wr_high else f"  TotalScore >= {THRESHOLD}: n={n_high}, 勝率=N/A")
    print(f"  TotalScore <  {THRESHOLD}: n={n_low},  勝率={wr_low:.1f}%" if wr_low else f"  TotalScore <  {THRESHOLD}: n={n_low}, 勝率=N/A")
    print(f"  差: {diff:+.1f}%  p値: {p:.4f}" if p is not None else f"  差: {diff:+.1f}%  p値: N/A")
    print(f"  スコア統計: min={score_stats['min']:.1f} mean={score_stats['mean']:.2f} max={score_stats['max']:.1f}")

    # 閾値をスライドして最大差も探す
    print("\n  [参考] スコア閾値別 勝率:")
    for thr in [1, 2, 3, 4, 5]:
        h = df[df["_score"] >= thr]["_win"]
        l = df[df["_score"] <  thr]["_win"]
        wh = _win_rate(h)
        wl = _win_rate(l)
        nh = int(h.notna().sum())
        nl = int(l.notna().sum())
        if wh and wl and nh >= MIN_SAMPLES and nl >= MIN_SAMPLES:
            print(f"    >= {thr}: {wh:.1f}% (n={nh}) | < {thr}: {wl:.1f}% (n={nl}) | 差={wh-wl:+.1f}%")

    verdict = _edge_verdict(diff, p, n_high, n_low)
    print(f"\n  → 判定: {verdict}")

    return {"threshold": THRESHOLD, "wr_high": wr_high, "wr_low": wr_low,
            "diff": diff, "p": p, "n_high": n_high, "n_low": n_low, "verdict": verdict}


# ==========================================
# 分析2: P3_VolumeScore 正負で勝率差
# ==========================================

def analyze_p3volume(df: pd.DataFrame) -> dict:
    print("\n=== 分析2: P3_VolumeScore (正 vs ゼロ以下) ===")

    if "P3_VolumeScore" not in df.columns or "WinLose" not in df.columns:
        print("  → 必要列なし")
        return {}

    df = df.copy()
    df["_p3"]  = _to_num(df["P3_VolumeScore"])
    df["_win"] = _to_win(df["WinLose"])

    pos  = df[df["_p3"] >  0]["_win"]
    zero = df[df["_p3"] <= 0]["_win"]

    wr_pos  = _win_rate(pos)
    wr_zero = _win_rate(zero)
    n_pos   = int(pos.notna().sum())
    n_zero  = int(zero.notna().sum())

    diff = (wr_pos or 0) - (wr_zero or 0)

    n_win_p  = int((pos  == 1).sum())
    n_lose_p = int((pos  == 0).sum())
    n_win_z  = int((zero == 1).sum())
    n_lose_z = int((zero == 0).sum())
    p = _chi2_pvalue(n_win_p, n_lose_p, n_win_z, n_lose_z)

    # 内訳もプラス/ゼロ/マイナス別に
    neg = df[df["_p3"] < 0]["_win"]
    wr_neg = _win_rate(neg)
    wr_exactly_zero = _win_rate(df[df["_p3"] == 0]["_win"])

    print(f"  P3 > 0 :  n={n_pos},  勝率={wr_pos:.1f}%" if wr_pos is not None else f"  P3 > 0: n={n_pos}, 勝率=N/A")
    print(f"  P3 <= 0:  n={n_zero}, 勝率={wr_zero:.1f}%" if wr_zero is not None else f"  P3 <= 0: n={n_zero}, 勝率=N/A")
    print(f"  (内訳) P3 < 0: n={int(neg.notna().sum())}, 勝率={wr_neg:.1f}%" if wr_neg is not None else f"  (内訳) P3 < 0: n={int(neg.notna().sum())}, 勝率=N/A")
    print(f"  (内訳) P3 = 0: 勝率={wr_exactly_zero:.1f}%" if wr_exactly_zero is not None else "  (内訳) P3 = 0: N/A")
    print(f"  差(正 vs ≤0): {diff:+.1f}%  p値: {p:.4f}" if p is not None else f"  差: {diff:+.1f}%  p値: N/A")
    print(f"  目標: 差10%以上{'  ✅ クリア' if abs(diff) >= 10 else '  ❌ 未達'}")

    verdict = _edge_verdict(diff, p, n_pos, n_zero)
    print(f"\n  → 判定: {verdict}")

    return {"wr_pos": wr_pos, "wr_zero_or_neg": wr_zero, "diff": diff,
            "p": p, "n_pos": n_pos, "n_zero_or_neg": n_zero, "verdict": verdict}


# ==========================================
# 分析3: FundingRate 絶対値と勝率の相関
# ==========================================

def analyze_funding_rate(df: pd.DataFrame) -> dict:
    print("\n=== 分析3: FundingRate 絶対値 × 勝率 ===")

    if "FundingRate" not in df.columns or "WinLose" not in df.columns:
        print("  → 必要列なし")
        return {}

    df = df.copy()
    df["_fr"]  = _to_num(df["FundingRate"]).abs()
    df["_win"] = _to_win(df["WinLose"])

    # FR が入っている行だけ
    valid = df[df["_fr"].notna() & df["_win"].notna()].copy()
    n_valid = len(valid)
    print(f"  FundingRate 有効件数: {n_valid}")

    if n_valid < MIN_SAMPLES:
        print(f"  → サンプル不足（{n_valid} < {MIN_SAMPLES}）")
        return {}

    # ---- 点二列相関（phi: biserial）
    corr = float(valid["_fr"].corr(valid["_win"]))
    print(f"  点双列相関(FR|win): {corr:+.4f}")

    # ---- 四分位数でバケット化
    try:
        quantiles = valid["_fr"].quantile([0.25, 0.5, 0.75]).to_dict()
        q25 = quantiles[0.25]
        q50 = quantiles[0.50]
        q75 = quantiles[0.75]
    except Exception:
        q25 = q50 = q75 = 0.0

    print(f"  FR絶対値 分位数: Q25={q25:.6f}  Q50={q50:.6f}  Q75={q75:.6f}")
    print("\n  [FR絶対値 バケット別 勝率]")

    buckets = [
        (f"極小(< Q25={q25:.5f})", valid["_fr"] <  q25),
        (f"小  (Q25-Q50)"        , (valid["_fr"] >= q25) & (valid["_fr"] < q50)),
        (f"中  (Q50-Q75)"        , (valid["_fr"] >= q50) & (valid["_fr"] < q75)),
        (f"大  (>= Q75={q75:.5f})", valid["_fr"] >= q75),
    ]
    wr_list = []
    for label, mask in buckets:
        sub = valid[mask]["_win"]
        wr = _win_rate(sub)
        n = int(sub.notna().sum())
        wr_list.append(wr or 0)
        if wr is not None:
            print(f"    {label}: n={n}, 勝率={wr:.1f}%")
        else:
            print(f"    {label}: n={n}, 勝率=N/A")

    spread = max(wr_list) - min(wr_list) if wr_list else 0
    print(f"\n  バケット間最大差: {spread:.1f}%")

    # 相関の強さ判定
    if abs(corr) >= 0.15:
        corr_verdict = "✅ 有意な相関あり"
    elif abs(corr) >= 0.08:
        corr_verdict = "△ 弱い相関あり"
    else:
        corr_verdict = "❌ ほぼ相関なし"

    print(f"\n  → 相関判定: {corr_verdict} (r={corr:+.4f})")

    return {"corr": corr, "n_valid": n_valid, "spread": spread, "verdict": corr_verdict}


# ==========================================
# 分析4: Hour_JST 別勝率のばらつき
# ==========================================

def analyze_hour_jst(df: pd.DataFrame) -> dict:
    print("\n=== 分析4: Hour_JST 別 勝率のばらつき ===")

    if "Hour_JST" not in df.columns or "WinLose" not in df.columns:
        print("  → 必要列なし")
        return {}

    df = df.copy()
    df["_hour"] = _to_num(df["Hour_JST"])
    df["_win"]  = _to_win(df["WinLose"])

    valid = df[df["_hour"].notna() & df["_win"].notna()].copy()
    valid["_hour_int"] = valid["_hour"].astype(int)

    print(f"  Hour_JST 有効件数: {len(valid)}")

    hours = sorted(valid["_hour_int"].unique())
    rows = []
    for h in hours:
        sub = valid[valid["_hour_int"] == h]["_win"]
        n = int(sub.notna().sum())
        if n < MIN_SAMPLES:
            continue
        wr = _win_rate(sub)
        if wr is not None:
            rows.append({"hour": h, "n": n, "wr": wr,
                         "n_win": int((sub == 1).sum()),
                         "n_lose": int((sub == 0).sum())})

    if not rows:
        print("  → 基準件数以上の時間帯なし")
        return {}

    rows.sort(key=lambda x: x["wr"], reverse=True)
    wr_values = [r["wr"] for r in rows]
    spread = max(wr_values) - min(wr_values)

    print(f"\n  [時間帯別 勝率] (n >= {MIN_SAMPLES} のみ)")
    print(f"  {'時間':>4}  {'件数':>5}  {'勝率':>6}  バー")
    for r in sorted(rows, key=lambda x: x["hour"]):
        bar = "█" * int(r["wr"] / 5)
        marker = " ← 高" if r["wr"] == max(wr_values) else (" ← 低" if r["wr"] == min(wr_values) else "")
        print(f"  {r['hour']:>4}時  {r['n']:>5}件  {r['wr']:>5.1f}%  {bar}{marker}")

    top3 = sorted(rows, key=lambda x: x["wr"], reverse=True)[:3]
    bot3 = sorted(rows, key=lambda x: x["wr"])[:3]

    top3_str = [(r["hour"], f"{r['wr']:.1f}%") for r in top3]
    bot3_str = [(r["hour"], f"{r['wr']:.1f}%") for r in bot3]
    print(f"\n  勝率TOP3: {top3_str}")
    print(f"  勝率BOT3: {bot3_str}")
    print(f"  最大差(max-min): {spread:.1f}%  目標: 15%以上 {'✅ クリア' if spread >= 15 else '❌ 未達'}")

    if spread >= 15:
        verdict = "✅ エッジあり（時間帯による明確な差）"
    elif spread >= 8:
        verdict = "△ 弱いエッジ（時間帯差は存在するが小さい）"
    else:
        verdict = "❌ エッジなし（時間帯による差なし）"

    print(f"\n  → 判定: {verdict}")

    # カイ二乗: TOP3時間 vs BOT3時間
    top_hours = {r["hour"] for r in top3}
    bot_hours  = {r["hour"] for r in bot3}
    top_sub = valid[valid["_hour_int"].isin(top_hours)]["_win"]
    bot_sub = valid[valid["_hour_int"].isin(bot_hours)]["_win"]
    p = _chi2_pvalue(
        int((top_sub == 1).sum()), int((top_sub == 0).sum()),
        int((bot_sub == 1).sum()), int((bot_sub == 0).sum()),
    )
    if p is not None:
        print(f"  (TOP3 vs BOT3 カイ二乗 p値: {p:.4f})")

    return {"spread": spread, "n_hours": len(rows), "top3": top3, "bot3": bot3,
            "p_top_vs_bot": p, "verdict": verdict}


# ==========================================
# 総合エッジ判定
# ==========================================

def overall_verdict(results: dict) -> None:
    print("\n" + "=" * 60)
    print("=== 総合エッジ判定 ===")
    print("=" * 60)

    edge_count  = 0
    weak_count  = 0
    total_count = 0

    checks = {
        "分析1 TotalScore高低":    results.get("score",   {}).get("verdict", ""),
        "分析2 P3_VolumeScore":    results.get("p3",      {}).get("verdict", ""),
        "分析3 FundingRate相関":   results.get("funding", {}).get("verdict", ""),
        "分析4 Hour_JST時間帯":    results.get("hour",    {}).get("verdict", ""),
    }

    for label, v in checks.items():
        if not v:
            print(f"  {label}: データなし")
            continue
        total_count += 1
        if "✅" in v:
            edge_count += 1
        elif "△" in v:
            weak_count += 1
        print(f"  {label}: {v}")

    print()
    if total_count == 0:
        print("  → 判定不能（データ不足）")
        return

    if edge_count >= 2:
        overall = "✅✅ 方向B（絞り込みフィルタ）は有望。エッジが複数確認された。"
    elif edge_count == 1 or weak_count >= 2:
        overall = "△  方向Bは条件付きで可能。エッジは弱いが存在する。慎重に実装を検討。"
    else:
        overall = "❌  現時点では方向Bは早い。まず方向A（ブロックリスト整理）を優先。"

    print(f"  ▶ {overall}")
    print()

    # 追加インサイト
    score_diff  = results.get("score",   {}).get("diff")
    p3_diff     = results.get("p3",      {}).get("diff")
    hour_spread = results.get("hour",    {}).get("spread")
    fr_corr     = results.get("funding", {}).get("corr")

    print("  [方向B実装に向けた優先フィルター候補]")
    candidates = []
    if score_diff is not None and abs(score_diff) >= 5:
        candidates.append(f"  1. TotalScore閾値フィルタ（差 {score_diff:+.1f}%）")
    if p3_diff is not None and abs(p3_diff) >= 5:
        candidates.append(f"  2. P3_VolumeScore > 0 フィルタ（差 {p3_diff:+.1f}%）")
    if hour_spread is not None and hour_spread >= 8:
        candidates.append(f"  3. Hour_JSTフィルタ（時間帯差 {hour_spread:.1f}%）")
    if fr_corr is not None and abs(fr_corr) >= 0.08:
        candidates.append(f"  4. FundingRate閾値フィルタ（相関 r={fr_corr:+.4f}）")

    if candidates:
        for c in candidates:
            print(c)
    else:
        print("  → 有効なフィルター候補なし（現時点ではエッジが確認できない）")

    print()
    print("  [ロードマップ整合性]")
    if edge_count >= 2:
        print("  → 方向B（第6段階）を前倒し検討可能な状況。")
        print("     まず方向A継続 + 方向B PoC（threshold試し実装）を並行推奨。")
    else:
        print("  → 現在の第4段階（方向A: ブロックリスト整理）を継続。")
        print("     方向Bはデータ充実後（DONE 3,000件超）に再評価。")


# ==========================================
# メイン
# ==========================================

def main():
    print("=" * 60)
    print("v2_shadow_ai エッジ分析レポート")
    print("=" * 60)

    try:
        df_done = fetch_v2_shadow_done()
    except Exception as e:
        print(f"[FATAL] データ取得失敗: {e}")
        return 1

    if len(df_done) < MIN_SAMPLES:
        print(f"[ERROR] DONE件数 {len(df_done)} < {MIN_SAMPLES}（分析不能）")
        return 1

    print(f"\n使用列: {list(df_done.columns)}")
    print(f"DONE件数（分析対象）: {len(df_done)}")

    # 4分析を実行
    results = {}
    results["score"]   = analyze_totalscore(df_done)
    results["p3"]      = analyze_p3volume(df_done)
    results["funding"] = analyze_funding_rate(df_done)
    results["hour"]    = analyze_hour_jst(df_done)

    # 総合判定
    overall_verdict(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
