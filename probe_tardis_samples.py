#!/usr/bin/env python3
"""
probe_tardis_samples.py  【Tardis 購入前調査・無料サンプル形式確認】
====================================================================
目的（賢治さん指示・厳守）:
  「煙が出たから買う」ではなく「煙が出たから "購入前調査" に進む」。
  Tardis の無料サンプル（各月1日分・APIキー不要）を実際に落として、
  欲しいデータ（Funding / Open Interest）が本当に取れるか・列形式・粒度・
  銘柄対応・年代カバレッジを実測で確認する。購入判断はこの結果を見てから。

確認すること（購入前チェックリスト）:
  1. okex-swap / derivative_ticker の無料サンプルが落ちるか
  2. 列に funding_rate / open_interest が含まれるか（←今回の煙の本体）
  3. 粒度（ティック秒単位か）と1日の行数
  4. BTC/ETH/SOL/DOGE/BONK が対応しているか（特にBONKと命名）
  5. 2024-01 / 2024-06 / 2025-01 / 2025-06 / 2026-01 / 2026-06 が落ちるか
     （= 無料サンプルだけで 2.5年・複数レジームの "符号確認" ができるか）

重要な前提:
  - 無料サンプルは「各月1日だけ」。連続データは有料契約が必要。
  - これは "形式と入手可否の確認" であって、本番採用・有料購入ではない。
  - Long/Short Ratio(LSR) は Tardis derivative_ticker には通常含まれない
    （OKX websocketフィードベースのため）。LSRの可否も末尾で軽く試す。
  - 本番変更なし / main.py変更なし / deployなし / mergeなし / 有料契約なし。

実行: python3 probe_tardis_samples.py   （OKX/Tardisに到達できるローカルで）
依存: 標準ライブラリのみ（urllib / gzip / csv）。pip不要。
"""
import urllib.request, gzip, io, csv, ssl, time
from collections import Counter

BASE = "https://datasets.tardis.dev/v1"
EXCHANGE = "okex-swap"
CTX  = ssl.create_default_context()
HDRS = {"User-Agent": "Mozilla/5.0"}

# OKX USDT無期限の instId 形式。BONK が別命名なら下で代替を試す。
SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
           "DOGE-USDT-SWAP", "BONK-USDT-SWAP"]
BONK_ALTS = ["1000BONK-USDT-SWAP"]  # OKXで1000系命名の可能性

# 2.5年を薄く横断（無料は各月1日のみ）
MONTHS = [(2024, 1), (2024, 6), (2025, 1), (2025, 6), (2026, 1), (2026, 6)]

DTYPE = "derivative_ticker"   # funding + open_interest を含むチャンネル


def url_for(dtype, y, m, sym):
    return f"{BASE}/{EXCHANGE}/{dtype}/{y}/{m:02d}/01/{sym}.csv.gz"


def fetch_csv_gz(u, timeout=30):
    req = urllib.request.Request(u, headers=HDRS)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        raw = r.read()
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    return text


def inspect_one(y, m, sym):
    u = url_for(DTYPE, y, m, sym)
    try:
        text = fetch_csv_gz(u)
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": f"HTTP {e.code}", "url": u}
    except Exception as e:
        return {"ok": False, "status": f"{type(e).__name__}: {e}", "url": u}

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 2:
        return {"ok": False, "status": "空CSV", "url": u}
    header = rows[0]
    body = rows[1:]

    # 列インデックス検出（Tardis正規化列名のゆらぎに耐える）
    def find(*names):
        for nm in names:
            for i, h in enumerate(header):
                if h.strip().lower() == nm:
                    return i
        return None

    i_ts   = find("timestamp")
    i_fund = find("funding_rate")
    i_oi   = find("open_interest")
    i_mark = find("mark_price", "last_price")

    # 粒度（timestamp は μs か ms。差分の最頻値を秒で）
    gran = None
    if i_ts is not None and len(body) >= 5:
        ts = []
        for r in body[:3000]:
            try:
                ts.append(int(r[i_ts]))
            except Exception:
                pass
        if len(ts) >= 5:
            ts.sort()
            # μs/ms 自動判定
            span = ts[-1] - ts[0]
            unit = 1_000_000 if span > 10**12 * 10 else 1000
            diffs = [round((ts[i+1]-ts[i]) / unit, 2) for i in range(len(ts)-1)]
            diffs = [d for d in diffs if d > 0]
            if diffs:
                gran = Counter(diffs).most_common(1)[0][0]

    return {
        "ok": True, "url": u, "rows": len(body), "cols": header,
        "has_funding": i_fund is not None,
        "has_oi": i_oi is not None,
        "has_mark": i_mark is not None,
        "gran_sec": gran,
    }


def main():
    print("=" * 78)
    print("Tardis 無料サンプル 形式確認（購入前調査・各月1日のみ・無料）")
    print(f"exchange={EXCHANGE}  dataType={DTYPE}")
    print("=" * 78)

    # 到達確認（remote環境では 403 になる想定）
    probe = inspect_one(2024, 1, "BTC-USDT-SWAP")
    if not probe["ok"] and "403" in probe["status"]:
        print(f"\n[!] Tardis未到達/403: {probe['status']}")
        print("    → OKX/Tardisに到達できるローカル（Mac）で実行してください。")
        print(f"    例URL: {probe['url']}")
        return

    # 1) 代表1ファイルで列形式を表示
    print("\n■ 1. 列形式の確認（2024-01-01 BTC-USDT-SWAP）")
    if probe["ok"]:
        print(f"  行数: {probe['rows']}  粒度(秒): {probe['gran_sec']}")
        print(f"  funding_rate列: {probe['has_funding']}  open_interest列: {probe['has_oi']}  mark/last: {probe['has_mark']}")
        print(f"  全列: {probe['cols']}")
    else:
        print(f"  取得不可: {probe['status']}  ({probe['url']})")

    # 2) 銘柄 × 年代カバレッジ表
    print("\n■ 2. 銘柄 × 月 カバレッジ（OK=行数 / NG=理由）")
    print(f"  {'symbol':<20} " + " ".join(f"{y%100:02d}/{m:02d}" for y, m in MONTHS))
    bonk_ok = False
    for sym in SYMBOLS:
        cells = []
        for (y, m) in MONTHS:
            r = inspect_one(y, m, sym)
            time.sleep(0.15)
            if r["ok"]:
                cells.append(f"{r['rows']:>5}")
                if sym.startswith("BONK"):
                    bonk_ok = True
            else:
                cells.append(" NG  ")
        print(f"  {sym:<20} " + " ".join(cells))

    # BONK代替命名（USDT-SWAPで全滅なら1000系を試す）
    if not bonk_ok:
        print("\n  [BONK] USDT-SWAPで取得不可 → 代替命名を試す")
        for alt in BONK_ALTS:
            r = inspect_one(2026, 1, alt)
            print(f"    {alt:<20} 2026-01: " + ("OK rows=%d" % r["rows"] if r["ok"] else "NG " + r["status"]))
            time.sleep(0.15)

    # 3) LSR（Long/Short Ratio）が Tardis に別dataTypeであるか軽く確認
    print("\n■ 3. LSR(Long/Short Ratio)の入手可否（参考・derivative_tickerには通常無い）")
    for dt in ("long_short_ratio", "liquidations"):  # 命名は当て・存在確認のみ
        u = f"{BASE}/{EXCHANGE}/{dt}/2026/01/01/BTC-USDT-SWAP.csv.gz"
        try:
            txt = fetch_csv_gz(u)
            hdr = txt.splitlines()[0] if txt else ""
            print(f"  {dt:<18} OK  先頭列: {hdr[:80]}")
        except urllib.error.HTTPError as e:
            print(f"  {dt:<18} NG  HTTP {e.code}（このdataTypeは無い可能性）")
        except Exception as e:
            print(f"  {dt:<18} NG  {type(e).__name__}")
        time.sleep(0.15)

    print("\n" + "=" * 78)
    print("判定の見方（購入前調査）:")
    print("  - funding_rate列とopen_interest列が両方True、5銘柄×6ヶ月が全部OK")
    print("    → 無料サンプル(各月1日)だけで 2024-2026 の符号確認ができる。まず無料で横断検証。")
    print("  - 連続データ(全日)が必要なら有料契約。Tardisは:")
    print("      * 月契約=直近のみ / 四半期=12ヶ月 / 年契約Pro=4年分 / Business=2019〜全部")
    print("      → 2024まで遡るには 年契約Pro 以上が必要（月契約では届かない）")
    print("  - LSRが両dataTypeともNGなら、LSRの2024履歴はTardisでは取れない")
    print("    （OKX純正rubikは直近30日のみ＝実測済み。LSRだけは別ソースか前向き蓄積）")
    print("=" * 78)


if __name__ == "__main__":
    main()
