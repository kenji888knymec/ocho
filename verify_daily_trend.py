#!/usr/bin/env python3
"""
verify_daily_trend.py  【日次トレンドフォロー仮説検証】
==========================================================
仮説（事前固定・後から変更しない）:
  暗号資産は15分足ではノイズが強すぎた。
  日足レベルでは中期トレンドが持続する可能性がある。
  Donchian 20日高値ブレイクアウトの翌日にLONGエントリーし、
  10日保有または5% SL到達で決済する。

パラメータ（事前固定）:
  対象    : BTC / ETH（2銘柄のみ。SOL/DOGEは参考扱い）
  足      : 日足（1h OHLCVを日次集計）
  エントリー: 過去20日高値を終値で上抜けた翌日の始値
  ポジション: LONG のみ
  保有    : 10日 または SL 5% のどちらか早い方
  コスト  : 往復0.15%

train : 2024-01-01 〜 2025-03-31（15ヶ月）
test  : 2025-04-01 〜 2026-06-21（15ヶ月）

合格基準（事前固定）:
  test trades >= 10（件数不足は勝率が良くても保留・不合格）
  test avg PnL_Net > 0
  test median PnL_Net > -3%
  test active_months >= 5
  test maxDD > -35%
  BTC と ETH 両方合格（片方だけはNG）
  top3ヶ月集中度 < 60%
  trainで合格・testで符号逆転は不合格

SL判定の定義（事前固定）:
  エントリー当日を含め、日足 Low が entry_price × 0.95 以下でSL到達。
  （前日終値ブレイク → 当日始値エントリーのため、当日のintraday下落もSL対象）
  SL到達日: exit_price = entry_price × 0.95（通常ケース）
  ギャップダウン: 当日 Open がすでに SL 以下 → exit_price = 当日 Open（保守的）
                  ※始値=entry_priceの当日は対象外。翌日以降のみ判定
  SL と 10日保有終了が同日の場合: SL 優先

やらないこと（厳守）:
  20日を18日・22日に変更しない
  SL幅・保有日数を変更しない
  ETH/SOL/DOGEで再検証して合格を探さない
  「少し変えると合格」は一切しない

データ: /tmp/ohlcv_long/ohlcv_2024_2026/{SYM}_1h.csv（OKX swap 1h）
実行  : python3 verify_daily_trend.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ── 定数（事前固定） ──────────────────────────────────────────────
OHLCV_DIR   = Path("/tmp/ohlcv_long/ohlcv_2024_2026")
SYMBOLS     = ["BTC", "ETH"]
REF_SYMS    = ["SOL", "DOGE"]   # 参考のみ。合否判定に使わない

DONCHIAN_N  = 20       # ブレイクアウト日数
HOLD_DAYS   = 10       # 最大保有日数
SL_PCT      = 0.05     # ストップロス 5%
COST_RT     = 0.0015   # 往復コスト 0.15%

TRAIN_START = "2024-01-01"
TRAIN_END   = "2025-03-31"
TEST_START  = "2025-04-01"
TEST_END    = "2026-06-21"

# 合格基準（事前固定）
PASS_MIN_N  = 10     # test trades >= 10
PASS_AVG    = 0.0    # avg > 0
PASS_MED    = -3.0   # median > -3%
PASS_MONTHS = 5      # active_months >= 5
PASS_DD     = -35.0  # maxDD > -35%
PASS_TOP3   = 60.0   # top3集中 < 60%


def load_daily(sym: str) -> pd.DataFrame | None:
    path = OHLCV_DIR / f"{sym}_1h.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df["date"] = df["datetime_utc"].dt.date
    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.reset_index(drop=True)


def simulate(daily: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Donchian ブレイクアウト LONG シミュレーション（as-of厳守）"""
    d = daily.copy().reset_index(drop=True)
    # 過去DONCHIAN_N日の最高値（当日を除く = shift(1)してから rolling）
    d["dc_hi"] = d["close"].shift(1).rolling(DONCHIAN_N).max()
    d["signal"] = d["close"] > d["dc_hi"]

    trades = []
    in_trade = False
    entry_price = 0.0
    entry_date  = None
    entry_idx   = 0

    for i in range(1, len(d)):
        # ① エントリー判定（前日シグナル発生 → 当日始値でエントリー）
        if not in_trade:
            if d.at[i - 1, "signal"] and np.isfinite(d.at[i - 1, "dc_hi"]):
                in_trade    = True
                entry_price = d.at[i, "open"]
                entry_date  = d.at[i, "date"]
                entry_idx   = i

        # ② 決済判定（エントリー当日 days_held=0 を含む。当日のintraday SLも捕捉）
        if in_trade:
            days_held   = i - entry_idx
            sl_price    = entry_price * (1.0 - SL_PCT)
            # ギャップダウン(始値がSL以下)は翌日以降のみ。当日は open==entry_price のため対象外
            gap_down    = (i > entry_idx) and (d.at[i, "open"] <= sl_price)
            hit_sl_low  = d.at[i, "low"]  <= sl_price   # 日中にSL到達（当日含む）
            hit_sl      = gap_down or hit_sl_low
            hit_hold    = days_held >= HOLD_DAYS

            if hit_sl or hit_hold:
                if gap_down:
                    exit_price = d.at[i, "open"]   # ギャップダウン→始値でexit（保守的）
                    reason     = "SL_GAP"
                elif hit_sl_low:
                    exit_price = sl_price           # 日中SL到達→SL価格でexit
                    reason     = "SL"
                else:
                    exit_price = d.at[i, "close"]  # 保有期間終了
                    reason     = "HOLD"
                pnl_gross  = (exit_price / entry_price - 1.0) * 100.0
                pnl_net    = pnl_gross - COST_RT * 100.0
                trades.append({
                    "symbol":      sym,
                    "entry_date":  entry_date,
                    "exit_date":   d.at[i, "date"],
                    "days_held":   days_held,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "pnl_gross_%": round(pnl_gross, 4),
                    "pnl_net_%":   round(pnl_net,   4),
                    "exit_reason": reason,
                })
                in_trade = False

    return pd.DataFrame(trades)


def evaluate(trades: pd.DataFrame, label: str, sym: str) -> dict:
    if len(trades) == 0:
        return {"label": label, "sym": sym, "n": 0}

    pnl = trades["pnl_net_%"]
    trades = trades.copy()
    trades["month"] = pd.to_datetime(trades["entry_date"]).dt.to_period("M")
    monthly     = trades.groupby("month")["pnl_net_%"].sum()
    active_m    = int((monthly != 0).sum())
    top3_sum    = monthly.nlargest(3).sum()
    total       = pnl.sum()
    top3_pct    = top3_sum / total * 100.0 if total != 0 else 0.0

    # 最大ドローダウン（累積PnLベース）
    cum      = pnl.cumsum()
    roll_max = cum.cummax()
    maxdd    = float((cum - roll_max).min())

    return {
        "label":         label,
        "sym":           sym,
        "n":             len(trades),
        "active_months": active_m,
        "WR_%":          round(float((pnl > 0).mean() * 100), 1),
        "avg_%":         round(float(pnl.mean()),   3),
        "median_%":      round(float(pnl.median()), 3),
        "maxDD_%":       round(maxdd, 2),
        "top3_conc_%":   round(top3_pct, 1),
        "total_%":       round(float(total), 2),
    }


def check_pass(r: dict) -> list[str]:
    if r.get("n", 0) == 0:
        return ["データなし"]
    fails = []
    if r["n"] < PASS_MIN_N:
        fails.append(f"件数不足 n={r['n']} < {PASS_MIN_N}（勝率・avgに関係なく保留）")
    if r["avg_%"] <= PASS_AVG:
        fails.append(f"avg {r['avg_%']:.3f}% ≤ 0")
    if r["median_%"] <= PASS_MED:
        fails.append(f"median {r['median_%']:.3f}% ≤ {PASS_MED}%")
    if r["active_months"] < PASS_MONTHS:
        fails.append(f"active_months {r['active_months']} < {PASS_MONTHS}")
    if r["maxDD_%"] < PASS_DD:
        fails.append(f"maxDD {r['maxDD_%']:.1f}% < {PASS_DD}%")
    if r["top3_conc_%"] > PASS_TOP3:
        fails.append(f"top3集中 {r['top3_conc_%']:.1f}% > {PASS_TOP3}%")
    return fails


def print_row(r: dict):
    if r.get("n", 0) == 0:
        print(f"    [{r['label']}] データなし")
        return
    print(f"    [{r['label']}] n={r['n']:3d}  WR={r['WR_%']:4.1f}%  "
          f"avg={r['avg_%']:+6.3f}%  med={r['median_%']:+6.3f}%  "
          f"maxDD={r['maxDD_%']:+6.1f}%  top3={r['top3_conc_%']:5.1f}%  "
          f"months={r['active_months']}")


def main():
    print("=" * 68)
    print("▶ 日次トレンドフォロー仮説検証（事前固定・変更不可）")
    print(f"  Donchian {DONCHIAN_N}日ブレイクアウト LONG "
          f"/ 保有{HOLD_DAYS}日 / SL{SL_PCT*100:.0f}% / コスト往復{COST_RT*100:.2f}%")
    print(f"  train: {TRAIN_START} 〜 {TRAIN_END}   "
          f"test: {TEST_START} 〜 {TEST_END}")
    print("=" * 68)

    all_results: list[tuple] = []
    all_test_trades: list[pd.DataFrame] = []

    for sym in SYMBOLS + REF_SYMS:
        daily = load_daily(sym)
        if daily is None:
            print(f"\n{sym}: ファイルなし（スキップ）")
            continue

        trades_all = simulate(daily, sym)
        if len(trades_all) == 0:
            print(f"\n{sym}: シグナルなし")
            continue

        train_end_dt  = pd.Timestamp(TRAIN_END)
        test_start_dt = pd.Timestamp(TEST_START)
        test_end_dt   = pd.Timestamp(TEST_END)

        train = trades_all[
            pd.to_datetime(trades_all["entry_date"]) <= train_end_dt
        ].copy()
        test = trades_all[
            (pd.to_datetime(trades_all["entry_date"]) >= test_start_dt) &
            (pd.to_datetime(trades_all["entry_date"]) <= test_end_dt)
        ].copy()

        r_train = evaluate(train, "TRAIN", sym)
        r_test  = evaluate(test,  "TEST",  sym)

        is_ref = sym in REF_SYMS
        tag = " (参考のみ・合否判定外)" if is_ref else ""
        print(f"\n── {sym}{tag} ────────────────────────────────────")
        print_row(r_train)
        print_row(r_test)

        if not is_ref:
            fails = check_pass(r_test)
            # train/test 符号逆転チェック
            if r_train.get("avg_%", 0) > 0 and r_test.get("avg_%", 0) <= 0:
                fails.append("train合格・test逆転（過学習リスク）")
            if fails:
                print(f"  ❌ 不合格: {' / '.join(fails)}")
            else:
                print(f"  ✅ 合格")
            all_results.append((sym, r_train, r_test, fails))
            if len(test) > 0:
                all_test_trades.append(test)

    # ── 総合判定 ────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("▶ 総合判定")
    main_syms = {sym: (r_train, r_test, fails)
                 for sym, r_train, r_test, fails in all_results}

    btc_pass = "BTC" in main_syms and len(main_syms["BTC"][2]) == 0
    eth_pass = "ETH" in main_syms and len(main_syms["ETH"][2]) == 0

    if btc_pass and eth_pass:
        print("  ✅ BTC / ETH 両方合格 → 仮説支持")
        print("  次のステップ: 月別偏り・相場局面別を確認してから採用可否を判断")
        print("  ※ 条件変更・閾値調整は禁止。現状のまま評価すること")
    else:
        not_pass = []
        if not btc_pass:
            btc_fails = main_syms.get("BTC", (None, None, ["データなし"]))[2]
            not_pass.append(f"BTC: {', '.join(btc_fails)}")
        if not eth_pass:
            eth_fails = main_syms.get("ETH", (None, None, ["データなし"]))[2]
            not_pass.append(f"ETH: {', '.join(eth_fails)}")
        print(f"  ❌ 不合格 → 仮説棄却")
        for msg in not_pass:
            print(f"     {msg}")
        print("  やること: 条件変更しない。この仮説はここで終了。")

    # ── 月別詳細（TEST期間・主要2銘柄）────────────────────────────
    if all_test_trades:
        combined = pd.concat(all_test_trades, ignore_index=True)
        combined["month"] = pd.to_datetime(combined["entry_date"]).dt.to_period("M")
        monthly_det = combined.groupby(["symbol", "month"]).agg(
            n=("pnl_net_%", "count"),
            avg=("pnl_net_%", "mean"),
            total=("pnl_net_%", "sum"),
            WR=("pnl_net_%", lambda x: (x > 0).mean() * 100),
        ).round(3)
        print(f"\n{'─' * 68}")
        print("▶ TEST 月別明細（主要2銘柄）")
        with pd.option_context("display.width", 200, "display.max_rows", 60):
            print(monthly_det.to_string())

    print(f"\n{'─' * 68}")
    print("▶ 結果のまとめ（合否のみ）")
    print(f"  BTC: {'✅ 合格' if btc_pass else '❌ 不合格'}")
    print(f"  ETH: {'✅ 合格' if eth_pass else '❌ 不合格'}")
    if btc_pass and eth_pass:
        print("  総合: ✅ 両方合格")
    else:
        print("  総合: ❌ 棄却")


if __name__ == "__main__":
    main()
