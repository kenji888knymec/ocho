#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_unlock_event_study.py
============================
【検証B / Step B3】トークンアンロック（供給ジャンプ）への価格反応・イベントスタディ

★このコードは「データを見る前」に事前登録(verify_unlock_PREREG.md)どおり固定する。
  supply_coingecko.csv 投入後に窓・閾値・基準を動かさない（後出し禁止）。

入力:
  supply_coingecko.csv  : Macで取得（列 symbol,date_utc,price,market_cap）
  /tmp/ohlcv_long/ohlcv_2024_2026/{SYM}_1h.csv : 既存OHLCV（日次終値に集約）

処理（事前登録 §4・§5）:
  1. circ_supply = market_cap / price（欠損は np.nan・補完しない）
  2. イベント = 単日で supply が ≥1.0% 増え、かつ その水準が持続（偽ジャンプ除外）
     - jump = supply[t]/supply[t-1]-1 >= 0.01
     - 持続 = median(supply[t..t+3]) / supply[t-1] - 1 >= 0.005
     - ±5日以内の近接イベントは jump 最大の1件に統合
  3. market-model 超過リターン: β を [-90,-11]日 でOLS推定（α=0固定・vs BTC日次）
     AR_t = r_token,t − β·r_BTC,t
  4. CAR_pre[-10,-2] / CAR_event[-1,+1] / CAR_post[+2,+10]
  5. 集計: 平均/中央値/負割合/t統計量/イベント数。≥5%はサブ集計。
     頑健性: train/test(イベント日中央値で前後分割) / leave-one-out(最悪1件除外で符号維持)

合格基準（§6・全て満たすこと）:
  ① n ≥ 30
  ② CAR_pre または CAR_event の平均が 明確に負
  ③ その窓で |t| ≥ 2.0 かつ 中央値も負
  ④ train/test 両方で同符号（負）
  ⑤ leave-one-out で符号反転しない

本番Bot・main.py・deploy・merge には一切関与しない研究専用スクリプト。
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

# ───────── 事前固定パラメータ（変更禁止） ─────────
SUPPLY_CSV   = "supply_coingecko.csv"
OHLCV_DIR    = "/tmp/ohlcv_long/ohlcv_2024_2026"
JUMP_MIN     = 0.010      # 単日供給増 ≥1.0% をイベントとする
PERSIST_MIN  = 0.005      # 持続条件: 数日後も ≥0.5% 上にとどまる（偽ジャンプ除外）
MERGE_WIN    = 5          # ±5日以内の近接イベントは統合
BIG_MIN      = 0.050      # サブ集計: ≥5.0%（規模依存確認用）
BETA_LO, BETA_HI = -90, -11   # β推定窓
BETA_MIN_OBS = 40             # β推定に必要な最低日数
PRE_LO, PRE_HI   = -10, -2
EVT_LO, EVT_HI   = -1, 1
POST_LO, POST_HI = 2, 10
BTC_SYM      = "BTC"
# ──────────────────────────────────────────────


def load_daily_close(sym: str) -> pd.Series:
    path = f"{OHLCV_DIR}/{sym}_1h.csv"
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("dt")
    df["date"] = df["dt"].dt.normalize()
    daily = df.groupby("date")["close"].last()
    daily.name = sym
    return daily


def load_supply() -> pd.DataFrame:
    if not os.path.exists(SUPPLY_CSV):
        print(f"❌ {SUPPLY_CSV} が見つかりません。Macで fetch_supply_coingecko.py を実行し配置してください。")
        sys.exit(1)
    df = pd.read_csv(SUPPLY_CSV, parse_dates=["date_utc"])
    df = df.rename(columns={"date_utc": "date"})
    # 供給復元（欠損補完なし）
    price = pd.to_numeric(df["price"], errors="coerce")
    mcap  = pd.to_numeric(df["market_cap"], errors="coerce")
    df["supply"] = np.where((price > 0) & (mcap > 0), mcap / price, np.nan)
    df["date"] = df["date"].dt.normalize()
    return df[["symbol", "date", "supply"]].dropna(subset=["supply"])


def detect_events(sup: pd.Series) -> list:
    """単日≥1%増＋持続のイベント日リスト（jump_size付き）。±MERGE_WINで統合。"""
    sup = sup.sort_index()
    s = sup.values
    idx = sup.index
    raw = []
    for t in range(1, len(s) - 3):
        if s[t-1] <= 0:
            continue
        jump = s[t] / s[t-1] - 1.0
        if jump < JUMP_MIN:
            continue
        post = np.nanmedian(s[t:t+4])
        if post / s[t-1] - 1.0 < PERSIST_MIN:
            continue  # 偽ジャンプ（戻る）除外
        raw.append((idx[t], jump))
    # 近接統合: jump最大を残す
    raw.sort(key=lambda x: x[0])
    merged = []
    for dt0, jp in raw:
        if merged and (dt0 - merged[-1][0]).days <= MERGE_WIN:
            if jp > merged[-1][1]:
                merged[-1] = (dt0, jp)
        else:
            merged.append((dt0, jp))
    return merged


def market_model_ar(tok_ret: pd.Series, btc_ret: pd.Series, ev_date) -> dict | None:
    """イベント前後のAR/CARを返す。窓が満たせなければ None。"""
    # 共通日でそろえる
    df = pd.DataFrame({"tok": tok_ret, "btc": btc_ret}).dropna()
    if ev_date not in df.index:
        # イベント日が取引日でない場合、最も近い後続営業日に寄せる
        after = df.index[df.index >= ev_date]
        if len(after) == 0:
            return None
        ev_date = after[0]
    pos = df.index.get_loc(ev_date)
    if isinstance(pos, slice) or isinstance(pos, np.ndarray):
        return None
    # 相対オフセット
    def win(lo, hi):
        lo_i, hi_i = pos + lo, pos + hi
        if lo_i < 0 or hi_i >= len(df):
            return None
        return df.iloc[lo_i:hi_i+1]
    beta_w = win(BETA_LO, BETA_HI)
    if beta_w is None or len(beta_w) < BETA_MIN_OBS:
        return None
    # β: α=0固定のOLS（保守的）→ β = Σ(x·y)/Σ(x²)
    x = beta_w["btc"].values
    y = beta_w["tok"].values
    denom = np.sum(x * x)
    if denom <= 0:
        return None
    beta = np.sum(x * y) / denom
    # 各窓のCAR
    def car(lo, hi):
        w = win(lo, hi)
        if w is None:
            return np.nan
        ar = w["tok"].values - beta * w["btc"].values
        return np.sum(ar)
    car_pre  = car(PRE_LO, PRE_HI)
    car_evt  = car(EVT_LO, EVT_HI)
    car_post = car(POST_LO, POST_HI)
    if not np.isfinite(car_pre) or not np.isfinite(car_evt) or not np.isfinite(car_post):
        return None
    return {"beta": beta, "car_pre": car_pre, "car_event": car_evt, "car_post": car_post}


def agg(label: str, vals: np.ndarray):
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        print(f"   {label:10s}: n=0")
        return None
    mean = vals.mean(); med = np.median(vals)
    negpct = (vals < 0).mean() * 100
    t = mean / (vals.std(ddof=1) / np.sqrt(n)) if n > 1 and vals.std(ddof=1) > 0 else np.nan
    print(f"   {label:10s}: n={n:3d}  平均CAR={mean*100:+6.2f}%  中央={med*100:+6.2f}%  "
          f"負={negpct:4.0f}%  t={t:+5.2f}")
    return {"n": n, "mean": mean, "med": med, "t": t}


def main():
    print("="*72)
    print("【検証B/B3】トークンアンロック（供給ジャンプ）への価格反応・イベントスタディ")
    print(f"  イベント: 単日供給増≥{JUMP_MIN*100:.0f}%＋持続  /  β窓[{BETA_LO},{BETA_HI}]  "
          f"CAR pre[{PRE_LO},{PRE_HI}] event[{EVT_LO},{EVT_HI}] post[{POST_LO},{POST_HI}]")
    print("="*72)

    sup_df = load_supply()
    syms = sorted(sup_df["symbol"].unique())
    print(f"供給データ銘柄: {len(syms)}  期間: {sup_df['date'].min().date()}〜{sup_df['date'].max().date()}")

    # 価格（OHLCV）
    closes = {s: load_daily_close(s) for s in set(syms) | {BTC_SYM}}
    btc_close = closes.get(BTC_SYM)
    if btc_close is None or btc_close.empty:
        print("❌ BTCのOHLCVが無い。β推定不能。中止。")
        sys.exit(1)
    btc_ret = btc_close.pct_change()

    # イベント収集
    rows = []  # dict per event
    per_sym_counts = {}
    for s in syms:
        sup = sup_df[sup_df["symbol"] == s].set_index("date")["supply"].sort_index()
        if len(sup) < 30:
            continue
        events = detect_events(sup)
        per_sym_counts[s] = len(events)
        tok_close = closes.get(s)
        if tok_close is None or tok_close.empty:
            continue
        tok_ret = tok_close.pct_change()
        for ev_date, jump in events:
            res = market_model_ar(tok_ret, btc_ret, ev_date)
            if res is None:
                continue
            res.update({"symbol": s, "date": ev_date, "jump": jump})
            rows.append(res)

    print("\n■ 銘柄別イベント検出数（≥1%・持続・統合後）:")
    for s in syms:
        c = per_sym_counts.get(s, 0)
        if c > 0:
            print(f"   {s:5s}: {c}")
    ev = pd.DataFrame(rows)
    if ev.empty:
        print("\n❌ 有効イベント0件（OHLCV窓を満たすものなし）。検証不能。")
        sys.exit(0)
    ev = ev.sort_values("date").reset_index(drop=True)
    print(f"\n有効イベント総数（β窓・CAR窓を満たす）: {len(ev)}")
    print(f"  期間: {ev['date'].min().date()}〜{ev['date'].max().date()}  "
          f"≥5%大型: {(ev['jump']>=BIG_MIN).sum()}件")

    # ── 全体集計 ──
    print("\n■ 全イベント CAR（market-model 超過リターン）:")
    a_pre  = agg("pre[-10,-2]",  ev["car_pre"].values)
    a_evt  = agg("event[-1,+1]", ev["car_event"].values)
    a_post = agg("post[+2,+10]", ev["car_post"].values)

    # ── ≥5% サブ集計 ──
    big = ev[ev["jump"] >= BIG_MIN]
    print(f"\n■ ≥5%大型のみ（サブ集計・n={len(big)}）:")
    if len(big) > 0:
        agg("pre",   big["car_pre"].values)
        agg("event", big["car_event"].values)
        agg("post",  big["car_post"].values)

    # ── 合格基準 ──
    print("\n" + "="*72)
    print("■ 合格基準チェック（事前固定・5つ全て）")
    n = len(ev)
    c1 = n >= 30
    # ②③: pre または event のどちらかで「平均負」かつ「|t|>=2 かつ 中央値負」
    def neg_strong(a):
        return a is not None and a["mean"] < 0 and abs(a["t"]) >= 2.0 and a["med"] < 0
    cand = []
    if a_pre and a_pre["mean"] < 0: cand.append(("pre", a_pre))
    if a_evt and a_evt["mean"] < 0: cand.append(("event", a_evt))
    c2 = len(cand) > 0
    strong = [(name, a) for name, a in cand if neg_strong(a)]
    c3 = len(strong) > 0
    win_name = strong[0][0] if strong else (cand[0][0] if cand else None)

    # ④ train/test（イベント日中央値で前後分割）
    c4 = False
    if win_name:
        col = {"pre": "car_pre", "event": "car_event"}[win_name]
        mid = ev["date"].quantile(0.5)
        tr = ev[ev["date"] <= mid][col].values
        te = ev[ev["date"] >  mid][col].values
        tr = tr[np.isfinite(tr)]; te = te[np.isfinite(te)]
        if len(tr) > 0 and len(te) > 0:
            c4 = (tr.mean() < 0) and (te.mean() < 0)
            print(f"  train/test({win_name}): train平均={tr.mean()*100:+.2f}%(n={len(tr)}) "
                  f"test平均={te.mean()*100:+.2f}%(n={len(te)})")

    # ⑤ leave-one-out（最悪=最も負に寄与する1件を除外して符号維持）
    c5 = False
    if win_name:
        col = {"pre": "car_pre", "event": "car_event"}[win_name]
        v = ev[col].values
        v = v[np.isfinite(v)]
        if len(v) > 1:
            # 平均を最も押し下げている=最小値を1つ除外
            worst = np.argmin(v)
            v2 = np.delete(v, worst)
            c5 = v2.mean() < 0
            print(f"  leave-one-out({win_name}): 最悪1件除外後 平均={v2.mean()*100:+.2f}%")

    t = lambda b: "✅" if b else "❌"
    print(f"\n  ① n≥30                         : {t(c1)}  (n={n})")
    print(f"  ② pre/event の平均が負          : {t(c2)}" + (f"  ({win_name})" if win_name else ""))
    print(f"  ③ その窓で|t|≥2 かつ 中央値も負 : {t(c3)}")
    print(f"  ④ train/test 両方で負           : {t(c4)}")
    print(f"  ⑤ leave-one-out で符号維持      : {t(c5)}")

    allpass = c1 and c2 and c3 and c4 and c5
    print("\n" + "="*72)
    print(f"■ 総合判定: {'✅ 合格（アンロック前後に超過反応あり → B5検討へ）' if allpass else '❌ 不合格'}")
    print("="*72)
    print("\n※ 反応が出ても本番実装はしない。B4でCLAUDE.md記録 → 明確ならB5(小さな実運用設計)。")


if __name__ == "__main__":
    main()
