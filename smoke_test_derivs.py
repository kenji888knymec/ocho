#!/usr/bin/env python3
"""
smoke_test_derivs.py
====================
【無料・足切り専用の煙テスト】
OKX無料APIで「今取れる範囲だけ」Funding/OI/LSRとOHLCVを取得し、
将来リターンとの関係に "煙が立つか" だけを見る。採用判断には使わない。

重要な前提（賢治さん指示・厳守）:
  - 目的は採用判断ではなく「完全に無関係そうか / 少しは見る価値がありそうか」の足切り
  - OI/LSRは直近約30日、Fundingは直近約3か月しか無料APIで取れない（実測済み）
  - 30日級データは統計的に弱い。煙が出ても「勝てる」とは判断しない
  - 本番変更なし / main.py変更なし / deployなし / mergeなし / 有料データなし

実行: python3 smoke_test_derivs.py   （OKXに到達できるローカルで実行）
依存: 標準ライブラリのみ（pandas/numpy不要 → pip install不要）

確認する観点:
  1. Funding極端値 と 将来リターンの関係（ロング過熱→反落の煙か）
  2. OI急増/急減 と 将来リターンの関係
  3. Long/Short Ratioの偏り と 逆行/順行の関係
  （4. 既存LONG_F・停止中SHORTとの相性は EntryRecord が要るので別途。本スクリプト対象外）
  5. 何も傾向が無ければ この路線は一旦見送り
"""
import urllib.request, json, ssl, time
from datetime import datetime, timezone
from collections import defaultdict
from statistics import mean

BASE = "https://www.okx.com"
CTX  = ssl.create_default_context()
HDRS = {"User-Agent": "Mozilla/5.0"}
SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "BONK"]
HOUR_MS = 3600 * 1000


def _get(path, params, retries=3):
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v != "")
    url = f"{BASE}{path}?{qs}"
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                d = json.loads(r.read().decode())
                if str(d.get("code")) != "0":
                    return None, f"code={d.get('code')} msg={d.get('msg')}"
                return d.get("data", []), None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"; time.sleep(1.0)
    return None, last


def _floor_hour(ms):
    return (int(ms) // HOUR_MS) * HOUR_MS


# ───────── 取得 ─────────
def fetch_candles_1h(sym, max_pages=30):
    """1H OHLCV を新→旧にページング。ts(hour)->close の dict を返す。"""
    inst = f"{sym}-USDT-SWAP"
    out = {}
    d0, e = _get("/api/v5/market/candles", {"instId": inst, "bar": "1H", "limit": "300"})
    if not d0:
        # history-candles にフォールバック
        d0, e = _get("/api/v5/market/history-candles", {"instId": inst, "bar": "1H", "limit": "100"})
        if not d0:
            return out, e
    for row in d0:
        out[_floor_hour(row[0])] = float(row[4])
    oldest = min(out)
    pages = 0
    while pages < max_pages:
        d2, e2 = _get("/api/v5/market/history-candles",
                      {"instId": inst, "bar": "1H", "after": str(oldest), "limit": "100"})
        if not d2:
            break
        for row in d2:
            out[_floor_hour(row[0])] = float(row[4])
        no = min(_floor_hour(r[0]) for r in d2)
        if no >= oldest:
            break
        oldest = no; pages += 1; time.sleep(0.1)
    return out, None


def fetch_funding(sym, max_pages=10):
    """fundingTime(ms)->rate の dict。"""
    inst = f"{sym}-USDT-SWAP"
    out = {}
    d0, e = _get("/api/v5/public/funding-rate-history", {"instId": inst, "limit": "100"})
    if not d0:
        return out, e
    for r in d0:
        out[int(r["fundingTime"])] = float(r["fundingRate"])
    oldest = min(out); pages = 0
    while pages < max_pages:
        d2, e2 = _get("/api/v5/public/funding-rate-history",
                      {"instId": inst, "after": str(oldest), "limit": "100"})
        if not d2:
            break
        for r in d2:
            out[int(r["fundingTime"])] = float(r["fundingRate"])
        no = min(int(r["fundingTime"]) for r in d2)
        if no >= oldest:
            break
        oldest = no; pages += 1; time.sleep(0.1)
    return out, None


def fetch_rubik(sym, path, val_idx):
    """ts(hour)->value の dict。直近約30日。"""
    d, e = _get(path, {"ccy": sym, "period": "1H"})
    if not d:
        return {}, e
    return {_floor_hour(r[0]): float(r[val_idx]) for r in d}, None


# ───────── 分析 ─────────
def fwd_ret(close_by_ts, ts, hours):
    a = close_by_ts.get(ts)
    b = close_by_ts.get(ts + hours * HOUR_MS)
    if a is None or b is None or a <= 0:
        return None
    return (b / a - 1) * 100.0


def tercile_table(pairs):
    """pairs=[(feature, fwd_ret)] を3分位し各バケットの平均リターン・件数を返す。"""
    pairs = [(f, r) for f, r in pairs if f is not None and r is not None]
    if len(pairs) < 9:
        return None
    pairs.sort(key=lambda x: x[0])
    n = len(pairs); k = n // 3
    lo, mid, hi = pairs[:k], pairs[k:n-k], pairs[n-k:]
    def stat(b):
        rs = [r for _, r in b]; fs = [f for f, _ in b]
        return len(b), mean(rs), min(fs), max(fs)
    return {"low": stat(lo), "mid": stat(mid), "high": stat(hi)}


def show(title, table, interp):
    print(f"\n[{title}]")
    if table is None:
        print("  データ不足（n<9）"); return
    print(f"  {'bucket':<6} {'n':>5} {'平均fwd_ret%':>12} {'特徴量レンジ'}")
    for k in ("low", "mid", "high"):
        n, mr, fmin, fmax = table[k]
        print(f"  {k:<6} {n:>5} {mr:>+12.3f}  [{fmin:+.5f}, {fmax:+.5f}]")
    diff = table["high"][1] - table["low"][1]
    print(f"  high-low 差: {diff:+.3f}%  → {interp}")


def main():
    print("=" * 72)
    print("SMOKE TEST: Funding / OI / LSR → 将来リターン （無料・足切り専用）")
    print("⚠ 30日〜3か月級・統計的に弱い・採用判断不可・煙の有無だけ見る")
    print("=" * 72)

    t, e = _get("/api/v5/public/time", {})
    if t is None:
        print(f"\n[!] OKX未到達: {e} → OKXに繋がるローカルで実行してください。")
        return

    # 全銘柄プール用
    fund_pairs8, fund_pairs24 = [], []
    oi_pairs24, lsr_pairs24 = [], []
    persym = {}

    for s in SYMBOLS:
        print(f"\n--- {s} 取得中 ---")
        close, ce = fetch_candles_1h(s)
        fund, fe = fetch_funding(s)
        oi, oie  = fetch_rubik(s, "/api/v5/rubik/stat/contracts/open-interest-volume", 1)
        lsr, le  = fetch_rubik(s, "/api/v5/rubik/stat/contracts/long-short-account-ratio", 1)
        print(f"  OHLCV={len(close)}  funding={len(fund)}  OI={len(oi)}  LSR={len(lsr)}")
        if not close:
            print(f"  [skip] OHLCV取得失敗: {ce}"); continue

        # 1. Funding（8h settlement時点で評価）
        fp8 = [(rate, fwd_ret(close, _floor_hour(ft), 8))  for ft, rate in fund.items()]
        fp24 = [(rate, fwd_ret(close, _floor_hour(ft), 24)) for ft, rate in fund.items()]
        fund_pairs8 += fp8; fund_pairs24 += fp24

        # 2. OI 24h変化率 → fwd24
        oip = []
        for ts, v in oi.items():
            prev = oi.get(ts - 24 * HOUR_MS)
            if prev and prev > 0:
                oip.append((v / prev - 1, fwd_ret(close, ts, 24)))
        oi_pairs24 += oip

        # 3. LSR水準 → fwd24
        lp = [(v, fwd_ret(close, ts, 24)) for ts, v in lsr.items()]
        lsr_pairs24 += lp

        # per-symbol high-low（funding24 / lsr24）
        ft = tercile_table(fp24); lt = tercile_table(lp)
        persym[s] = {
            "fund24_hl": (ft["high"][1] - ft["low"][1]) if ft else None,
            "lsr24_hl":  (lt["high"][1] - lt["low"][1]) if lt else None,
        }
        time.sleep(0.2)

    print("\n" + "=" * 72)
    print("■ プール集計（5銘柄まとめ）")
    show("1. Funding率 → 8h先リターン",  tercile_table(fund_pairs8),
         "highでマイナス=ロング過熱後の反落(逆張りSHORTの煙)")
    show("1. Funding率 → 24h先リターン", tercile_table(fund_pairs24),
         "highでマイナス=同上")
    show("2. OI 24h変化 → 24h先リターン", tercile_table(oi_pairs24),
         "OI急増(high)が順張り/逆張りどちらに効くか")
    show("3. LSR水準 → 24h先リターン", tercile_table(lsr_pairs24),
         "highでマイナス=ロング偏り後の逆行(逆張りの煙)")

    print("\n■ 銘柄別 high-low 差（符号が揃うか＝偶然でない手がかり）")
    print(f"  {'sym':<6} {'funding24 h-l':>14} {'lsr24 h-l':>12}")
    for s in SYMBOLS:
        if s in persym:
            f = persym[s]["fund24_hl"]; l = persym[s]["lsr24_hl"]
            fs = f"{f:+.3f}" if f is not None else "n/a"
            ls = f"{l:+.3f}" if l is not None else "n/a"
            print(f"  {s:<6} {fs:>14} {ls:>12}")

    print("\n" + "=" * 72)
    print("判定の見方（足切り）:")
    print("  - high-lowに明確な差があり、銘柄間で符号が揃う → 煙あり（前向き蓄積 or 有料検討）")
    print("  - バケットがバラバラ・銘柄で符号バラバラ → 無関係寄り（この路線は一旦見送り）")
    print("  ※ 30日級なので、煙が出ても『勝てる』判断はしない。あくまで足切り。")
    print("=" * 72)


if __name__ == "__main__":
    main()
