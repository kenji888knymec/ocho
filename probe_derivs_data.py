#!/usr/bin/env python3
"""
probe_derivs_data.py  (v2: 保持期間の境界を実測で探す)
====================================================
OKX の「価格以外の入力」3種が、どの銘柄・どこまで過去・どの粒度で取れるかを
実測で確認するプローブ。取得可否レポートのみ。本番変更・保存・学習なし。

対象:
  1. Funding Rate History       /api/v5/public/funding-rate-history   (instId別・8h settle)
  2. Open Interest & Volume     /api/v5/rubik/stat/contracts/open-interest-volume (ccy別)
  3. Long/Short Account Ratio   /api/v5/rubik/stat/contracts/long-short-account-ratio (ccy別)

v2の変更点（賢治さんのローカル実測を受けて）:
  - Funding: ページングを深く回し「どこで打ち切られるか(=API保持限界)」を実測
  - OI/LSR : begin=2024 を一発で投げると code=50030(Illegal time range) になるため、
             短い窓を 直近7d→30d→90d→2025年の1ヶ月→2024年の1ヶ月 と刻んで
             「合法に取れる最古の窓」を探す
  - 各データの 取得可能最古 / 粒度 / 銘柄差 を表で出す

注意:
  - Claude remote環境では取引所APIが403。賢治さんのローカルで実行すること。
  - rubik/stat は ccy 単位(BTC/ETH...)。instId単位ではない。
  - Funding は 8h ごと(=480分)。1h足には forward-fill で合わせる想定（このスクリプトは生の可否のみ）。

実行: python3 probe_derivs_data.py   (依存: 標準ライブラリのみ)
"""
import urllib.request
import json
import ssl
import time
from datetime import datetime, timezone, timedelta
from collections import Counter

BASE = "https://www.okx.com"
CTX  = ssl.create_default_context()
HDRS = {"User-Agent": "Mozilla/5.0"}

PROBE_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "BONK"]

OI_PATH  = "/api/v5/rubik/stat/contracts/open-interest-volume"
LSR_PATH = "/api/v5/rubik/stat/contracts/long-short-account-ratio"


def _get(path, params, retries=3):
    qs  = "&".join(f"{k}={v}" for k, v in params.items() if v != "")
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
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.0)
    return None, last


def _d(ms):
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def _ms(dt):
    return str(int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000))


def _gran_min(ts_list):
    if len(ts_list) < 3:
        return None
    ts = sorted(int(x) for x in ts_list)
    diffs = [round((ts[i+1] - ts[i]) / 60000.0) for i in range(len(ts) - 1)]
    diffs = [x for x in diffs if x > 0]
    return Counter(diffs).most_common(1)[0][0] if diffs else None


# ═══════════════════════════════════════════════════════════════════
# 1. Funding: 深くページングして API保持限界(最古)を実測
# ═══════════════════════════════════════════════════════════════════
def probe_funding(sym, max_pages=200):
    inst = f"{sym}-USDT-SWAP"
    d0, err = _get("/api/v5/public/funding-rate-history", {"instId": inst, "limit": "100"})
    if not d0:
        return {"ok": False, "err": err or "empty"}
    ts_all = [int(r["fundingTime"]) for r in d0]
    newest = max(ts_all)
    cur_oldest = min(ts_all)
    total = len(d0)
    gran = _gran_min(ts_all)
    pages = 0
    stopped = "max_pages"
    while pages < max_pages:
        d2, e2 = _get("/api/v5/public/funding-rate-history",
                      {"instId": inst, "after": str(cur_oldest), "limit": "100"})
        if e2:
            stopped = f"error:{e2}"; break
        if not d2:
            stopped = "empty(=API保持限界)"; break
        ts2 = [int(r["fundingTime"]) for r in d2]
        no = min(ts2)
        if no >= cur_oldest:
            stopped = "no_progress(=API保持限界)"; break
        cur_oldest = no
        total += len(d2)
        pages += 1
        time.sleep(0.12)
    return {
        "ok": True, "newest": _d(newest), "oldest": _d(cur_oldest),
        "gran_min": gran, "records": total, "pages": pages, "stopped": stopped,
    }


# ═══════════════════════════════════════════════════════════════════
# 2&3. rubik: 短い窓を新→旧に刻んで「合法に取れる最古の窓」を探す
# ═══════════════════════════════════════════════════════════════════
def probe_rubik_windows(sym, path):
    now = datetime.now(timezone.utc)
    # (ラベル, 窓のbegin, 窓のend)
    windows = [
        ("直近7日",      now - timedelta(days=7),    now),
        ("直近30日",     now - timedelta(days=30),   now),
        ("直近90日",     now - timedelta(days=90),   now),
        ("2025-12月",    datetime(2025, 12, 1),      datetime(2025, 12, 8)),
        ("2025-06月",    datetime(2025, 6, 1),       datetime(2025, 6, 8)),
        ("2024-06月",    datetime(2024, 6, 1),       datetime(2024, 6, 8)),
        ("2024-01月",    datetime(2024, 1, 1),       datetime(2024, 1, 8)),
    ]
    # まず時間指定なし（デフォルト窓の深さ）
    base, berr = _get(path, {"ccy": sym, "period": "1H"})
    rows = []
    if base:
        ts = [r[0] for r in base]
        rows.append(("指定なし", "OK", len(base), _d(max(int(t) for t in ts)),
                     _d(min(int(t) for t in ts)), _gran_min(ts)))
    else:
        rows.append(("指定なし", f"NG:{berr}", 0, "-", "-", None))

    for label, b, e in windows:
        data, err = _get(path, {"ccy": sym, "period": "1H", "begin": _ms(b), "end": _ms(e)})
        if data:
            ts = [r[0] for r in data]
            rows.append((label, "OK", len(data), _d(max(int(t) for t in ts)),
                         _d(min(int(t) for t in ts)), _gran_min(ts)))
        else:
            rows.append((label, f"NG:{err}", 0, "-", "-", None))
        time.sleep(0.12)
    return rows


def main():
    print("=" * 74)
    print("OKX デリバ補助データ 保持期間プローブ v2")
    print(f"確認銘柄: {PROBE_SYMBOLS}")
    print("=" * 74)

    t, e = _get("/api/v5/public/time", {})
    if t is None:
        print(f"\n[!] OKX API へ到達できません: {e}")
        print("    → OKXに到達できるローカル環境で実行してください。")
        return

    # 1. Funding
    print("\n■ 1. Funding Rate History — API保持限界を実測")
    print(f"  {'sym':<6} {'最新':<17} {'最古':<17} {'粒度分':<7} {'件数':<6} {'打切理由'}")
    for s in PROBE_SYMBOLS:
        r = probe_funding(s)
        if r["ok"]:
            print(f"  {s:<6} {r['newest']:<17} {r['oldest']:<17} "
                  f"{str(r['gran_min']):<7} {str(r['records']):<6} {r['stopped']}")
        else:
            print(f"  {s:<6} 取得不可: {r['err']}")

    # 2. OI
    print("\n■ 2. Open Interest & Volume — 窓を刻んで最古を実測")
    for s in PROBE_SYMBOLS:
        print(f"  --- {s} ---")
        print(f"    {'窓':<10} {'結果':<28} {'件数':<5} {'新':<17} {'旧':<17} {'粒度分'}")
        for label, status, n, newest, oldest, g in probe_rubik_windows(s, OI_PATH):
            print(f"    {label:<10} {status:<28} {str(n):<5} {newest:<17} {oldest:<17} {g}")

    # 3. LSR
    print("\n■ 3. Long/Short Account Ratio — 窓を刻んで最古を実測")
    for s in PROBE_SYMBOLS:
        print(f"  --- {s} ---")
        print(f"    {'窓':<10} {'結果':<28} {'件数':<5} {'新':<17} {'旧':<17} {'粒度分'}")
        for label, status, n, newest, oldest, g in probe_rubik_windows(s, LSR_PATH):
            print(f"    {label:<10} {status:<28} {str(n):<5} {newest:<17} {oldest:<17} {g}")

    print("\n" + "=" * 74)
    print("見どころ:")
    print("  - Funding: 打切理由が「API保持限界」なら、それより前はAPIでは取れない")
    print("             (2024まで要るなら OKX公式の履歴DLファイル or Tardis等の別ソース)")
    print("  - OI/LSR : どの窓までOKでどこからNGか = 実際の保持境界")
    print("             指定なしの『旧』も保持の手がかり")
    print("  - LSR    : contract-level は 2024-08-08 以降との情報あり → 2024前半は出ない想定")
    print("=" * 74)


if __name__ == "__main__":
    main()
