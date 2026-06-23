"""
eval_cs_mom_shadow.py
=====================
cs_mom_shadow（クロスセクション・モメンタム市場ニュートラルバスケット）の
オフライン評価専用スクリプト。

【重要・前提】
- bot本体(main.py)は cs_mom_shadow を「記録のみ」する設計で、評価(OPEN→PnL算出)は
  行わない（main.py ヘッダコメント: "eval_status # OPEN（評価は後段/オフライン）"）。
- 本スクリプトはその「後段/オフライン評価」を担う。
- 本スクリプトは EntryRecord を一切書き換えない。結果は別CSVに出力する。
- 本番(Cloud Run)・main.py・ENV・通知には一切影響しない。検証専用。

【計算は bot の記録ロジックと完全一致させる（main.py:13298-13356）】
- エントリー参照価格 ref = ランク確定1h足の close（= シート long_ref_prices / short_ref_prices）
  → 既にシートに記録済みなので再取得しない（スケール不一致を避ける）。
- 決済eval価格 = ランク確定足の open から H時間後に open する1h足の close
  （entry: open=rank_candle_ts_utc の足の close / exit: open=rank_candle_ts_utc+H の足の close）
  expected_eval_time_jst(=rank_candle_jst+H) を UTC に直した時刻が exit足の open timestamp。
- LONG leg(各銘柄): gross=(eval/ref-1)*100 ; net=gross-cost
- SHORT leg(各銘柄): gross=(ref-eval)/ref*100 ; net=gross-cost   （弱い銘柄を空売り）
- leg = 5銘柄の net 平均
- spread_pnl_net = (long_leg + short_leg) / 2   （= 市場ニュートラル本体）
  （CLAUDE.md実測: LONG leg -0.120 / SHORT leg +0.778 → SPREAD +0.329 = 平均で一致）
- cost = 各行の cost_pct_per_leg（既定0.15・片レッグ往復コスト%）

【欠損補完は一切しない（CLAUDE.md厳守）】
- ref または eval が取得できない銘柄はその銘柄を leg から除外（0/平均で埋めない）。
- leg内に有効銘柄が n 未満になった行は SKIPPED（spread を出さない）。
- eval足がまだ存在しない（expected_eval が未来 or OHLCV未取得）行は PENDING。

【OHLCV eval価格の取得元】
本スクリプトは2系統をサポート（--ohlcv-source）:
  fetch : ccxt.okx(swap) から直接取得（bot と同じ "{BASE}/USDT" / defaultType=swap）
          → スケールが bot の ref と自動一致。OKXに到達できる環境（賢治さんローカル）で実行。
  csv   : 事前取得した 1h OHLCV CSV を読む（列: timestamp[ms],open,high,low,close,...）
          → ただしSHIB/BONK等のスケールが ref と一致している必要がある（ref整合チェックで検出）。

【ref整合チェック（スケール/市場の取り違え検出）】
exit足の取得時、同時に entry足(open=rank_candle_ts_utc)の close も取得し、
シート記録の ref と相対誤差を比較する。誤差 > --ref-tol（既定0.5%）なら、その銘柄は
市場/スケール不一致の疑いとして leg から除外し、warn を出す（補完しない）。

【使い方（賢治さんローカル・OKX到達環境）】
  python3 eval_cs_mom_shadow.py \
      --entryrecord /path/to/EntryRecord_4.xlsx \
      --out /tmp/cs_mom_eval_result.csv \
      --ohlcv-source fetch

【再利用】
  本スクリプトは毎回シート全行を読み、評価可能になった行を再計算して別CSVに出す。
  行が増えても同じコマンドで「溜まった分まとめて」評価できる（冪等）。

【サンドボックス自己テスト（OKX不要・計算配線の確認）】
  python3 eval_cs_mom_shadow.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CS_MOM_SHEET = "cs_mom_shadow"
JST = timezone(timedelta(hours=9))


# --------------------------------------------------------------------------
# 純粋関数: leg / spread PnL 計算（bot ロジックと一致・OKX不要・自己テスト対象）
# --------------------------------------------------------------------------
def compute_leg_net(ref_prices: List[float], eval_prices: List[float],
                    side: str, cost: float) -> Tuple[Optional[float], int, int]:
    """1レッグ(複数銘柄)の net PnL平均を返す。

    Returns: (leg_net_or_None, n_valid, n_total)
    - 欠損(ref/eval が NaN/不正)は除外（補完しない）。
    - 有効銘柄が0なら None。
    """
    side = side.upper()
    nets: List[float] = []
    n_total = len(ref_prices)
    for ref, ev in zip(ref_prices, eval_prices):
        if ref is None or ev is None:
            continue
        if not (np.isfinite(ref) and np.isfinite(ev)) or abs(ref) <= 1e-18:
            continue
        if side == "LONG":
            gross = (ev / ref - 1.0) * 100.0
        elif side == "SHORT":
            gross = (ref - ev) / ref * 100.0
        else:
            raise ValueError(f"side must be LONG/SHORT, got {side}")
        nets.append(gross - cost)
    if not nets:
        return None, 0, n_total
    return float(np.mean(nets)), len(nets), n_total


def compute_spread(long_leg_net: Optional[float],
                   short_leg_net: Optional[float]) -> Optional[float]:
    """spread = (long_leg + short_leg)/2 。どちらか欠ければ None。"""
    if long_leg_net is None or short_leg_net is None:
        return None
    return (long_leg_net + short_leg_net) / 2.0


# --------------------------------------------------------------------------
# 時刻・パース
# --------------------------------------------------------------------------
def parse_jst_naive(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def jst_naive_to_utc_open_ms(jst_naive: datetime) -> int:
    """JST naive（足のopen時刻として解釈）→ UTC open timestamp(ms)。"""
    utc = jst_naive - timedelta(hours=9)
    return int(utc.replace(tzinfo=timezone.utc).timestamp() * 1000)


def parse_price_list(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            out.append(float("nan"))
            continue
        try:
            out.append(float(tok))
        except Exception:
            out.append(float("nan"))
    return out


def parse_symbol_list(s: str) -> List[str]:
    return [t.strip().upper() for t in str(s).split(",") if t.strip()]


# --------------------------------------------------------------------------
# OHLCV eval価格の取得（fetch / csv）
# --------------------------------------------------------------------------
def _base_to_bot_symbol(base: str) -> str:
    """bot と同じシンボル表記（"{BASE}/USDT", defaultType=swap で解決）。
    bot は main.py:13643- の "SHIB/USDT" 等をそのまま使う → スケールも一致。"""
    return f"{base}/USDT"


class CloseLookup:
    """open timestamp(ms, UTC) → close を引く。fetch / csv の両対応。"""

    def __init__(self, source: str, ohlcv_dir: Optional[str]):
        self.source = source
        self.ohlcv_dir = ohlcv_dir
        self._exchange = None
        self._csv_cache: Dict[str, pd.DataFrame] = {}

    def _ensure_exchange(self):
        if self._exchange is None:
            import ccxt  # local import: csvモードでは不要
            self._exchange = ccxt.okx({
                "options": {"defaultType": "swap"},
                "enableRateLimit": True,
                "timeout": 15000,
            })
        return self._exchange

    def max_available_open_ms(self, base: str) -> Optional[int]:
        """その銘柄で取得可能な最大 open timestamp(ms)。PENDING判定に使う。"""
        if self.source == "csv":
            df = self._load_csv(base)
            if df is None or df.empty:
                return None
            return int(df["timestamp"].max())
        # fetch: 直近足の open を見る
        try:
            ex = self._ensure_exchange()
            bars = ex.fetch_ohlcv(_base_to_bot_symbol(base), "1h", limit=3)
            if not bars:
                return None
            return int(bars[-1][0])
        except Exception as e:
            print(f"  [WARN] {base} max_available fetch失敗: {type(e).__name__}: {str(e)[:80]}")
            return None

    def _load_csv(self, base: str) -> Optional[pd.DataFrame]:
        if base in self._csv_cache:
            return self._csv_cache[base]
        if not self.ohlcv_dir:
            return None
        import os
        path = os.path.join(self.ohlcv_dir, f"{base}_1h.csv")
        if not os.path.exists(path):
            self._csv_cache[base] = None
            return None
        df = pd.read_csv(path)
        # timestamp列(ms)を必須とする
        if "timestamp" not in df.columns:
            print(f"  [WARN] {path} に timestamp列なし → スキップ")
            self._csv_cache[base] = None
            return None
        df = df[["timestamp", "close"]].dropna().copy()
        df["timestamp"] = df["timestamp"].astype("int64")
        self._csv_cache[base] = df
        return df

    def close_at_open(self, base: str, open_ms: int) -> Optional[float]:
        """open timestamp(ms)がちょうど open_ms の1h足の close を返す。無ければ None。"""
        if self.source == "csv":
            df = self._load_csv(base)
            if df is None:
                return None
            hit = df[df["timestamp"] == open_ms]
            if hit.empty:
                return None
            return float(hit["close"].iloc[0])
        # fetch: since=open_ms から数本取り、ちょうど一致する足を探す
        try:
            ex = self._ensure_exchange()
            bars = ex.fetch_ohlcv(_base_to_bot_symbol(base), "1h", since=open_ms, limit=3)
            for b in bars or []:
                if int(b[0]) == open_ms:
                    return float(b[4])
            return None
        except Exception as e:
            print(f"  [WARN] {base} close_at_open fetch失敗: {type(e).__name__}: {str(e)[:80]}")
            return None


# --------------------------------------------------------------------------
# 1行(バスケット)の評価
# --------------------------------------------------------------------------
def evaluate_row(row: pd.Series, lookup: CloseLookup, ref_tol: float,
                 now_utc_ms: int) -> Dict:
    rank_jst = parse_jst_naive(row.get("rank_candle_time_jst", ""))
    eval_jst = parse_jst_naive(row.get("expected_eval_time_jst", ""))
    H = int(float(row.get("hold_h", 48)))
    n_side = int(float(row.get("n_per_side", 5)))
    cost = float(row.get("cost_pct_per_leg", 0.15))

    long_syms = parse_symbol_list(row.get("long_symbols", ""))
    short_syms = parse_symbol_list(row.get("short_symbols", ""))
    long_refs = parse_price_list(row.get("long_ref_prices", ""))
    short_refs = parse_price_list(row.get("short_ref_prices", ""))

    result = {
        "rank_candle_time_jst": row.get("rank_candle_time_jst", ""),
        "expected_eval_time_jst": row.get("expected_eval_time_jst", ""),
        "hold_h": H,
        "n_per_side": n_side,
        "cost_pct_per_leg": cost,
        "long_symbols": ",".join(long_syms),
        "short_symbols": ",".join(short_syms),
        "long_leg_pnl_net": "",
        "short_leg_pnl_net": "",
        "spread_pnl_net": "",
        "long_valid": 0,
        "short_valid": 0,
        "eval_status": "",
        "note": "",
    }

    if rank_jst is None or eval_jst is None:
        result["eval_status"] = "SKIPPED"
        result["note"] = "rank/eval時刻パース失敗"
        return result

    # entry足(open=rank_candle_ts_utc) と exit足(open=eval_ts_utc) の open timestamp(ms)
    entry_open_ms = jst_naive_to_utc_open_ms(rank_jst)
    exit_open_ms = jst_naive_to_utc_open_ms(eval_jst)

    # exit足がまだ存在しない（未来 or 未取得）なら PENDING
    if exit_open_ms > now_utc_ms:
        result["eval_status"] = "PENDING"
        result["note"] = f"exit足(open={eval_jst} JST)が未到来"
        return result

    def _resolve_eval(base: str, ref: float) -> Tuple[Optional[float], str]:
        """exit足のclose（eval価格）を返す。同時にentry足closeでref整合チェック。"""
        if not (np.isfinite(ref) and abs(ref) > 1e-18):
            return None, "ref欠損"
        ev = lookup.close_at_open(base, exit_open_ms)
        if ev is None:
            return None, "eval足なし"
        # ref整合チェック（スケール/市場取り違え検出・補完はしない）
        entry_close = lookup.close_at_open(base, entry_open_ms)
        if entry_close is not None and np.isfinite(entry_close) and abs(entry_close) > 1e-18:
            rel = abs(entry_close - ref) / abs(ref)
            if rel > ref_tol:
                return None, f"ref不整合(rel={rel:.3f}>{ref_tol})"
        return ev, "ok"

    # まず exit足の存在を1銘柄で確認（全銘柄PENDING/未取得の切り分け）
    long_evals: List[Optional[float]] = []
    short_evals: List[Optional[float]] = []
    notes: List[str] = []
    any_eval_found = False

    for base, ref in zip(long_syms, long_refs):
        ev, st = _resolve_eval(base, ref)
        long_evals.append(ev)
        if ev is not None:
            any_eval_found = True
        elif st != "ok":
            notes.append(f"L:{base}:{st}")
    for base, ref in zip(short_syms, short_refs):
        ev, st = _resolve_eval(base, ref)
        short_evals.append(ev)
        if ev is not None:
            any_eval_found = True
        elif st != "ok":
            notes.append(f"S:{base}:{st}")

    if not any_eval_found:
        # exit時刻は過ぎているのに1銘柄もeval取得できない → OHLCV未取得の可能性大
        result["eval_status"] = "PENDING"
        result["note"] = "eval価格が1銘柄も取得できず（OHLCV未取得の可能性）; " + ";".join(notes[:6])
        return result

    long_leg, l_valid, l_tot = compute_leg_net(long_refs, long_evals, "LONG", cost)
    short_leg, s_valid, s_tot = compute_leg_net(short_refs, short_evals, "SHORT", cost)
    result["long_valid"] = l_valid
    result["short_valid"] = s_valid

    # 有効銘柄が n_side 未満のレッグがあれば spread は出さない（補完しない）
    if long_leg is None or short_leg is None or l_valid < n_side or s_valid < n_side:
        result["eval_status"] = "SKIPPED"
        result["long_leg_pnl_net"] = "" if long_leg is None else f"{long_leg:.4f}"
        result["short_leg_pnl_net"] = "" if short_leg is None else f"{short_leg:.4f}"
        result["note"] = (f"有効銘柄不足 L={l_valid}/{l_tot} S={s_valid}/{s_tot}; "
                          + ";".join(notes[:6]))
        return result

    spread = compute_spread(long_leg, short_leg)
    result["long_leg_pnl_net"] = f"{long_leg:.4f}"
    result["short_leg_pnl_net"] = f"{short_leg:.4f}"
    result["spread_pnl_net"] = f"{spread:.4f}"
    result["eval_status"] = "EVALUATED"
    result["note"] = ("ok" if not notes else "ok; warn:" + ";".join(notes[:6]))
    return result


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------
def run(entryrecord: str, out_csv: str, ohlcv_source: str,
        ohlcv_dir: Optional[str], ref_tol: float) -> None:
    xl = pd.ExcelFile(entryrecord)
    if CS_MOM_SHEET not in xl.sheet_names:
        print(f"[ERR] シート {CS_MOM_SHEET} が {entryrecord} に無い。sheets={xl.sheet_names}")
        sys.exit(1)
    cs = xl.parse(CS_MOM_SHEET)
    print(f"[INFO] cs_mom_shadow 行数: {len(cs)}")

    lookup = CloseLookup(ohlcv_source, ohlcv_dir)
    now_utc_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows_out: List[Dict] = []
    for i, row in cs.iterrows():
        res = evaluate_row(row, lookup, ref_tol, now_utc_ms)
        rows_out.append(res)
        sp = res["spread_pnl_net"]
        print(f"  row{i}: {res['eval_status']:9s} "
              f"rank={res['rank_candle_time_jst']} eval={res['expected_eval_time_jst']} "
              f"long_leg={res['long_leg_pnl_net'] or '-':>8} "
              f"short_leg={res['short_leg_pnl_net'] or '-':>8} "
              f"spread={sp or '-':>8}  {res['note']}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)
    print(f"\n[OK] 結果を {out_csv} に出力（EntryRecordは書き換えていない）")

    ev = out_df[out_df["eval_status"] == "EVALUATED"]
    print(f"\n=== 集計 ===")
    print(f"  EVALUATED: {len(ev)} / PENDING: {(out_df['eval_status']=='PENDING').sum()} / "
          f"SKIPPED: {(out_df['eval_status']=='SKIPPED').sum()}")
    if len(ev) >= 1:
        sp = pd.to_numeric(ev["spread_pnl_net"], errors="coerce").dropna()
        if len(sp) >= 1:
            print(f"  SPREAD: n={len(sp)} avg={sp.mean():+.4f}% median={sp.median():+.4f}% "
                  f"WR={(sp>0).mean():.1%}")
            if len(sp) < 8:
                print(f"  [注意] n={len(sp)} は統計的判断に不十分。"
                      f"GO条件(leave-one-week-out/閾値安定)には複数週分の蓄積が必要。")


def selftest() -> None:
    """OKX不要。計算配線の確認（ref=eval なら spread=-cost 等）。"""
    print("=== self-test: leg/spread 計算配線 ===")
    cost = 0.15
    # ケース1: eval=ref（無変動）→ 各 net = -cost、spread=-cost
    refs = [10.0, 20.0, 30.0, 40.0, 50.0]
    long_leg, lv, lt = compute_leg_net(refs, refs, "LONG", cost)
    short_leg, sv, st = compute_leg_net(refs, refs, "SHORT", cost)
    sp = compute_spread(long_leg, short_leg)
    print(f"  [1] 無変動: long_leg={long_leg:+.4f} short_leg={short_leg:+.4f} spread={sp:+.4f} "
          f"(期待: 全て {-cost:+.4f})")
    assert abs(long_leg - (-cost)) < 1e-9 and abs(short_leg - (-cost)) < 1e-9 and abs(sp - (-cost)) < 1e-9

    # ケース2: LONG銘柄が+2%上昇, SHORT銘柄が-2%下落（理想的な市場ニュートラル勝ち）
    long_ref = [100.0]; long_ev = [102.0]   # +2%
    short_ref = [100.0]; short_ev = [98.0]  # -2% → SHORTで+2%
    ll, _, _ = compute_leg_net(long_ref, long_ev, "LONG", cost)
    sl, _, _ = compute_leg_net(short_ref, short_ev, "SHORT", cost)
    sp2 = compute_spread(ll, sl)
    print(f"  [2] L+2%/S-2%: long_leg={ll:+.4f}(期待{2-cost:+.4f}) "
          f"short_leg={sl:+.4f}(期待{2-cost:+.4f}) spread={sp2:+.4f}(期待{2-cost:+.4f})")
    assert abs(ll - (2 - cost)) < 1e-9 and abs(sl - (2 - cost)) < 1e-9

    # ケース3: 欠損除外（補完しない）。LONG 1銘柄欠損 → n_valid=4
    refs4 = [10, 20, float("nan"), 40, 50]
    evs4 = [10, 20, 30, 40, 50]
    ll3, v3, t3 = compute_leg_net(refs4, evs4, "LONG", cost)
    print(f"  [3] 欠損除外: n_valid={v3}/{t3} (期待 4/5・補完なし)")
    assert v3 == 4 and t3 == 5

    # ケース4: 時刻変換（JST open → UTC open ms）
    jst = datetime(2026, 6, 22, 21, 0, 0)
    ms = jst_naive_to_utc_open_ms(jst)
    back = datetime.utcfromtimestamp(ms / 1000.0)
    print(f"  [4] 6/22 21:00 JST → UTC {back} (期待 2026-06-22 12:00:00)")
    assert back == datetime(2026, 6, 22, 12, 0, 0)

    print("=== self-test: 全ケース PASS ===")


def main():
    ap = argparse.ArgumentParser(description="cs_mom_shadow オフライン評価（記録のみのbotを補完）")
    ap.add_argument("--entryrecord", help="EntryRecord xlsx パス")
    ap.add_argument("--out", default="/tmp/cs_mom_eval_result.csv", help="出力CSVパス")
    ap.add_argument("--ohlcv-source", choices=["fetch", "csv"], default="fetch",
                    help="eval価格の取得元（fetch=ccxt okx swap / csv=事前取得CSV）")
    ap.add_argument("--ohlcv-dir", default="/tmp/ohlcv_okx",
                    help="--ohlcv-source csv のときのCSVディレクトリ（{BASE}_1h.csv）")
    ap.add_argument("--ref-tol", type=float, default=0.005,
                    help="ref整合チェックの許容相対誤差（既定0.5%%）")
    ap.add_argument("--selftest", action="store_true", help="OKX不要の計算自己テスト")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.entryrecord:
        ap.error("--entryrecord が必要（--selftest 以外）")
    run(args.entryrecord, args.out, args.ohlcv_source, args.ohlcv_dir, args.ref_tol)


if __name__ == "__main__":
    main()
