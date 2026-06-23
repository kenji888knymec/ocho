#!/usr/bin/env python3
"""
probe_derivs_data.py
====================
OKX の「価格以外の入力」3種が、どの銘柄・どの期間・どの粒度で取れるかを
実測で確認するプローブ。取得可否レポートのみ。本番変更・保存・学習なし。

対象:
  1. Funding Rate History       /api/v5/public/funding-rate-history   (instId別・8h settle)
  2. Open Interest & Volume     /api/v5/rubik/stat/contracts/open-interest-volume (ccy別)
  3. Long/Short Account Ratio   /api/v5/rubik/stat/contracts/long-short-account-ratio (ccy別)

注意:
  - Claude実行環境(remote)では取引所APIが403でブロックされる。
    → 賢治さんのローカル(OKXに到達できる環境)で実行すること。
  - rubik/stat 系は ccy 単位(BTC/ETH/SOL...)。instId単位ではない。
    → アルト全28銘柄でrubik統計が揃うとは限らない（プローブで確認）。
  - Funding は 8時間ごと(1日3回)。1h足には forward-fill で合わせる想定。
    → このスクリプトは「生の取得可否」だけ確認する。補完はしない。

実行: python3 probe_derivs_data.py
依存: 標準ライブラリのみ(urllib)。ccxtは不要。
"""
import urllib.request
import json
import ssl
import time
from datetime import datetime, timezone

BASE = "https://www.okx.com"
CTX  = ssl.create_default_context()
HDRS = {"User-Agent": "Mozilla/5.0"}

# 確認対象（まずは少数で取得可否を確認 → 取れたら全銘柄へ拡張）
PROBE_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "BONK"]
TARGET_START  = "2024-01-01"  # 揃えたい履歴の起点
TARGET_END    = "2026-06-21"


def _get(path, params, retries=3):
    """OKX GET。失敗時はNoneとエラーをタプルでReturn。"""
    qs  = "&".join(f"{k}={v}" for k, v in params.items() if v != "")
    url = f"{BASE}{path}?{qs}"
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                d = json.loads(r.read().decode())
                if d.get("code") not in ("0", 0):
                    return None, f"code={d.get('code')} msg={d.get('msg')}"
                return d.get("data", []), None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.0)
    return None, last


def _ts_to_date(ms):
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def _infer_granularity(ts_list_ms):
    """連続するtsの最頻間隔(分)を返す。"""
    if len(ts_list_ms) < 3:
        return None
    ts = sorted(int(x) for x in ts_list_ms)
    diffs = [(ts[i+1] - ts[i]) / 60000.0 for i in range(len(ts) - 1)]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return None
    # 最頻値
    from collections import Counter
    c = Counter(round(d) for d in diffs)
    return c.most_common(1)[0][0]


# ═══════════════════════════════════════════════════════════════════
# 1. Funding Rate History (instId別・backward pagination で最古を探す)
# ═══════════════════════════════════════════════════════════════════
def probe_funding(sym):
    inst = f"{sym}-USDT-SWAP"
    # まず最新100件
    data, err = _get("/api/v5/public/funding-rate-history",
                     {"instId": inst, "limit": "100"})
    if data is None:
        return {"ok": False, "err": err}
    if not data:
        return {"ok": False, "err": "empty"}

    all_ts = [r["fundingTime"] for r in data]
    newest = max(int(t) for t in all_ts)
    oldest = min(int(t) for t in all_ts)
    gran   = _infer_granularity(all_ts)

    # backward pagination で最古まで（最大30ページ=3000件で打ち切り・プローブ用）
    pages = 0
    cur_oldest = oldest
    while pages < 30:
        d2, e2 = _get("/api/v5/public/funding-rate-history",
                      {"instId": inst, "after": str(cur_oldest), "limit": "100"})
        if not d2:
            break
        ts2 = [int(r["fundingTime"]) for r in d2]
        new_oldest = min(ts2)
        if new_oldest >= cur_oldest:
            break
        cur_oldest = new_oldest
        pages += 1
        time.sleep(0.15)

    return {
        "ok": True,
        "newest": _ts_to_date(newest),
        "oldest_reached": _ts_to_date(cur_oldest),
        "granularity_min": gran,
        "pages_walked": pages,
    }


# ═══════════════════════════════════════════════════════════════════
# 2 & 3. rubik/stat (ccy別) — begin=TARGET_START を指定して最古到達を見る
# ═══════════════════════════════════════════════════════════════════
def probe_rubik(sym, path, value_keys):
    begin_ms = str(int(datetime.strptime(TARGET_START, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000))
    # period=1H で TARGET_START 起点を要求
    data, err = _get(path, {"ccy": sym, "period": "1H", "begin": begin_ms, "limit": "100"})
    if data is None:
        return {"ok": False, "err": err}
    if not data:
        # begin指定なしで最新を取ってみる（履歴が短い可能性）
        data2, err2 = _get(path, {"ccy": sym, "period": "1H", "limit": "100"})
        if not data2:
            return {"ok": False, "err": err2 or "empty(begin), empty(latest)"}
        ts = [r[0] for r in data2]
        return {
            "ok": True,
            "note": "begin=2024指定では空。最新のみ取得可",
            "newest": _ts_to_date(max(int(t) for t in ts)),
            "oldest_in_page": _ts_to_date(min(int(t) for t in ts)),
            "granularity_min": _infer_granularity(ts),
        }
    ts = [r[0] for r in data]
    return {
        "ok": True,
        "newest": _ts_to_date(max(int(t) for t in ts)),
        "oldest_reached_with_begin2024": _ts_to_date(min(int(t) for t in ts)),
        "granularity_min": _infer_granularity(ts),
        "n_first_page": len(data),
    }


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("OKX デリバティブ補助データ 取得可否プローブ")
    print(f"目標期間: {TARGET_START} 〜 {TARGET_END} / 目標粒度: 1h")
    print(f"確認銘柄: {PROBE_SYMBOLS}")
    print("=" * 70)

    # 到達性チェック
    test, err = _get("/api/v5/public/time", {})
    if test is None:
        print(f"\n[!] OKX API へ到達できません: {err}")
        print("    → Claude remote環境では取引所APIが403でブロックされます。")
        print("    → このスクリプトは OKX に到達できるローカル環境で実行してください。")
        print("    (構文・ロジックは検証済み。ローカルなら下記の表が出力されます)")
        return

    print("\n■ 1. Funding Rate History (instId別)")
    print(f"  {'sym':<6} {'最新':<17} {'到達した最古':<17} {'粒度(分)':<8} {'ページ'}")
    for s in PROBE_SYMBOLS:
        r = probe_funding(s)
        if r["ok"]:
            print(f"  {s:<6} {r['newest']:<17} {r['oldest_reached']:<17} "
                  f"{str(r['granularity_min']):<8} {r['pages_walked']}")
        else:
            print(f"  {s:<6} 取得不可: {r['err']}")
        time.sleep(0.2)

    print("\n■ 2. Open Interest & Volume (rubik/stat, ccy別)")
    print(f"  {'sym':<6} 結果")
    for s in PROBE_SYMBOLS:
        r = probe_rubik(s, "/api/v5/rubik/stat/contracts/open-interest-volume", [1, 2])
        print(f"  {s:<6} {r}")
        time.sleep(0.2)

    print("\n■ 3. Long/Short Account Ratio (rubik/stat, ccy別)")
    print(f"  {'sym':<6} 結果")
    for s in PROBE_SYMBOLS:
        r = probe_rubik(s, "/api/v5/rubik/stat/contracts/long-short-account-ratio", [1])
        print(f"  {s:<6} {r}")
        time.sleep(0.2)

    print("\n" + "=" * 70)
    print("確認の見どころ:")
    print("  - Funding: 8h粒度(=480分)が普通。最古が2024-01まで届くか")
    print("  - OI/LSR : begin=2024指定で2024-01まで遡れるか / 履歴が短くないか")
    print("  - LSR    : contract-level は 2024-08-08 以降の可能性(要実測)")
    print("  - 全銘柄: rubikはccy単位。アルトでデータ無し/短い銘柄が出るか")
    print("=" * 70)


if __name__ == "__main__":
    main()
