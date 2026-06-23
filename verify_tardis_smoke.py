#!/usr/bin/env python3
"""
verify_tardis_smoke.py  【Tardis 無料サンプル × 30ヶ月 OOS横断検証】
===================================================================
目的:
  煙テスト（直近3ヶ月・無料API）でFunding/OI/LSRの傾向が出た。
  Tardis 無料サンプル（各月1日・APIキー不要）を 2024-01〜2026-06 の
  30ヶ月分 × 5銘柄 使い、「煙の傾向が複数レジームで符号安定するか」を確認する。

  ★ 採用判断ではなく「OOS横断で符号が70%以上の月でマイナスか」の足切り確認。
  ★ 有料契約なし / main.py変更なし / deployなし / mergeなし / 本番変更なし。

手法:
  ① Funding:
     - Tardis derivative_ticker で funding_timestamp 変化点 = 決済イベントを検出
     - 変化直前の funding_rate = 「決済されたレート」（精度高い）
     - 決済時の last_price → 8h後の last_price（月1日内完結分のみ）
  ② OI:
     - 月1日の0:00 UTC基準のOI → 各時間帯でのOI変化率（日内変化）
     - 煙テストの「前日比24h」とは定義が違う。「日内OI変化→8h先」として読む。
  ③ LSR: Tardisに存在しない（HTTP 400確認済み）→ 対象外

キャッシュ:
  ~/Downloads/tardis_cache/  (再実行時は再DLしない。初回は全150ファイル/10〜30分)

実行: python3 verify_tardis_smoke.py   (Tardisに到達できるMacで)
依存: 標準ライブラリのみ (pip install 不要)
"""
import gzip, io, csv, ssl, time, sys
import urllib.request
from pathlib import Path
from datetime import timezone
from statistics import mean, stdev
from collections import defaultdict

CACHE_DIR = Path.home() / "Downloads" / "tardis_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://datasets.tardis.dev/v1"
EXCHANGE = "okex-swap"
DTYPE    = "derivative_ticker"
CTX      = ssl.create_default_context()
HDRS     = {"User-Agent": "Mozilla/5.0"}

SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
           "DOGE-USDT-SWAP", "BONK-USDT-SWAP"]

# BONK は 2024-06 以降のみ（実測済み: 2024-01 = NG）
MONTHS = [(y, m) for y in range(2024, 2027) for m in range(1, 13)
          if (2024, 1) <= (y, m) <= (2026, 6)]

HOUR_MS = 3_600_000


def url_for(y, m, sym):
    return f"{BASE_URL}/{EXCHANGE}/{DTYPE}/{y}/{m:02d}/01/{sym}.csv.gz"


def cache_path(y, m, sym):
    return CACHE_DIR / f"{EXCHANGE}_{DTYPE}_{y}_{m:02d}_{sym}.csv.gz"


def download_with_cache(y, m, sym):
    cp = cache_path(y, m, sym)
    if cp.exists():
        return cp
    u = url_for(y, m, sym)
    for attempt in range(3):
        try:
            req = urllib.request.Request(u, headers=HDRS)
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                data = r.read()
            cp.write_bytes(data)
            return cp
        except urllib.error.HTTPError:
            return None   # 404など = データなし
        except Exception as e:
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
    return None


def _f(v):
    try: return float(v)
    except: return None


def process_one(y, m, sym):
    """
    1ファイル（月1日分ティック）を処理し以下を返す:
      funding_pairs: [(funding_rate_at_settlement, fwd_8h_pct)]
      oi_pairs:      [(oi_intraday_chg_rate, fwd_8h_pct)]
    """
    cp = download_with_cache(y, m, sym)
    if cp is None:
        return [], []

    try:
        with gzip.open(cp, "rt", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception:
        cp.unlink(missing_ok=True)   # キャッシュ壊れ → 削除して次回再DL
        return [], []

    if len(rows) < 100:
        return [], []

    # 数値化 (ts: ms, ft: ms, fr: float, oi: float, price: float)
    records = []
    for r in rows:
        ts    = _f(r.get("timestamp") or "")
        ft    = _f(r.get("funding_timestamp") or "")
        fr    = _f(r.get("funding_rate") or "")
        oi    = _f(r.get("open_interest") or "")
        price = _f(r.get("last_price") or "")
        if ts is None or price is None or price <= 0:
            continue
        records.append((int(ts), int(ft) if ft else None, fr, oi, price))

    if len(records) < 50:
        return [], []

    records.sort(key=lambda x: x[0])
    ts_max = records[-1][0]

    # ── ① Funding 決済イベント検出 ──────────────────────────────
    # funding_timestamp が変化した瞬間 = 直前の決済予定で決済完了
    # 変化直前の funding_rate = 「決済されたレート」（OKX確定レート）
    settlement_events = []   # (settle_ts_ms, settled_funding_rate, price)
    prev_ft  = records[0][1]
    prev_fr  = records[0][2]
    prev_price = records[0][4]
    for ts, ft, fr, oi, price in records[1:]:
        if ft is not None and ft != prev_ft and prev_ft is not None:
            if prev_fr is not None:
                settlement_events.append((ts, prev_fr, price))
            prev_ft = ft
        elif ft is not None:
            prev_ft = ft
        if fr is not None:
            prev_fr = fr

    # 8h先の価格 lookup: ts(ms) → price。±5分(300000ms)の最近傍
    def price_at(target_ms):
        best_p, best_d = None, 300_001
        for t, _, _, _, p in records:
            d = abs(t - target_ms)
            if d < best_d:
                best_d = d; best_p = p
        return best_p

    funding_pairs = []
    for settle_ts, fr, p0 in settlement_events:
        fwd_ts = settle_ts + 8 * HOUR_MS
        if fwd_ts > ts_max + 300_000:
            continue   # 月1日内完結しない → スキップ
        p8 = price_at(fwd_ts)
        if p8 is None: continue
        fwd_ret = (p8 / p0 - 1) * 100.0
        funding_pairs.append((fr, fwd_ret))

    # ── ② OI 日内変化率 → 8h先リターン ─────────────────────────
    # 月1日の0:00 UTC を基準にした OI変化率（定義: 煙テストの「前日比24h」とは別）
    hour_oi    = defaultdict(list)
    hour_price = defaultdict(list)
    for ts, _, _, oi, price in records:
        hk = ts // HOUR_MS
        if oi is not None:    hour_oi[hk].append(oi)
        if price is not None: hour_price[hk].append(price)

    # 0:00 UTC の時間バケット
    day_start_ms = records[0][0] // (24 * HOUR_MS) * (24 * HOUR_MS)
    hk_start = day_start_ms // HOUR_MS

    # 基準OI: 0:00か1:00の最初の値
    oi_baseline = None
    for hk in sorted(hour_oi.keys()):
        if hour_oi[hk]:
            oi_baseline = hour_oi[hk][0]
            break

    oi_pairs = []
    if oi_baseline and oi_baseline > 0:
        for hk in sorted(hour_oi.keys()):
            oi_cur = hour_oi[hk][-1]
            if oi_cur is None: continue
            oi_chg = oi_cur / oi_baseline - 1
            hk8 = hk + 8
            if not hour_price.get(hk8) or not hour_price.get(hk): continue
            p0 = hour_price[hk][-1]
            p8 = hour_price[hk8][-1]
            if p0 is None or p0 <= 0: continue
            fwd_ret = (p8 / p0 - 1) * 100.0
            oi_pairs.append((oi_chg, fwd_ret))

    return funding_pairs, oi_pairs


def tercile_table(pairs):
    pairs = [(f, r) for f, r in pairs if f is not None and r is not None]
    if len(pairs) < 9:
        return None
    pairs.sort(key=lambda x: x[0])
    n = len(pairs); k = n // 3
    lo, mid, hi = pairs[:k], pairs[k:n-k], pairs[n-k:]
    def stat(b):
        rs = [r for _, r in b]
        return len(b), mean(rs)
    return {"low": stat(lo), "mid": stat(mid), "high": stat(hi)}


def show(title, table, interp):
    print(f"\n[{title}]")
    if table is None:
        print("  データ不足（n<9）"); return
    for k in ("low", "mid", "high"):
        n, mr = table[k]
        print(f"  {k:<6} n={n:>5}  avg={mr:>+8.3f}%")
    diff = table["high"][1] - table["low"][1]
    print(f"  high-low差: {diff:>+.3f}%  → {interp}")
    return diff


def main():
    print("=" * 78)
    print("Tardis 無料サンプル × 30ヶ月 OOS横断検証（各月1日・APIキー不要・$0）")
    print("⚠ 各月「1日分のみ」のサンプル。採用判断ではなく符号安定性の足切り確認。")
    print(f"キャッシュ先: {CACHE_DIR}")
    print("=" * 78)

    # 到達確認（remote環境では403になる）
    probe_cp = cache_path(2026, 6, "BTC-USDT-SWAP")
    if not probe_cp.exists():
        try:
            req = urllib.request.Request(url_for(2026, 6, "BTC-USDT-SWAP"), headers=HDRS)
            urllib.request.urlopen(req, timeout=15, context=CTX).close()
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"\n[!] Tardis 403: APIキーなしでもアクセスできる環境で実行してください（Mac推奨）")
                print(f"    URL例: {url_for(2026, 6, 'BTC-USDT-SWAP')}")
                sys.exit(1)
        except Exception as e:
            print(f"\n[!] 接続エラー: {e}")
            sys.exit(1)

    all_fund, all_oi = [], []
    month_results   = {}
    sym_fund        = defaultdict(list)
    total_files = 0

    print(f"\nダウンロード・処理中 (初回: 150ファイル × 5〜20MB = 10〜30分)")
    print("-" * 78)

    for (y, m) in MONTHS:
        mfp, moi = [], []
        for sym in SYMBOLS:
            if sym.startswith("BONK") and (y, m) < (2024, 6):
                continue
            fp, op = process_one(y, m, sym)
            mfp += fp; moi += op
            all_fund += fp; all_oi += op
            sym_fund[sym] += fp
            total_files += 1

        ft = tercile_table(mfp)
        ot = tercile_table(moi)
        f_hl = (ft["high"][1] - ft["low"][1]) if ft else None
        o_hl = (ot["high"][1] - ot["low"][1]) if ot else None
        month_results[(y, m)] = {"f_hl": f_hl, "o_hl": o_hl,
                                  "n_f": len(mfp), "n_o": len(moi)}
        sign = "✓" if (f_hl is not None and f_hl < 0) else ("✗" if f_hl is not None else "-")
        print(f"  {y}-{m:02d}  fund_n={len(mfp):>3}  fund_hl={f_hl:+.3f}%  {sign}"
              f"  oi_n={len(moi):>4}  oi_hl={o_hl:+.3f}%" if f_hl is not None and o_hl is not None
              else f"  {y}-{m:02d}  fund_n={len(mfp):>3}  (データ不足)", flush=True)

    print("\n" + "=" * 78)
    print("■ 30ヶ月プール集計")
    f_diff = show("1. Funding率 → 8h先リターン（30ヶ月×5銘柄）",
                  tercile_table(all_fund),
                  "highでマイナス → 煙テストと同方向（OOS安定）")
    o_diff = show("2. OI日内変化率 → 8h先リターン（※煙テストの前日比24hとは別定義）",
                  tercile_table(all_oi),
                  "highでプラス → OI増加時はSHORT不利（煙テストと同方向）")

    # ── 月別符号表 ────────────────────────────────────────────────
    print("\n■ 月別 Funding high-low 差（符号が揃うか）")
    print(f"  {'月':<9} {'fund_hl':>9} {'n_f':>5}  {'oi_hl':>9} {'n_o':>5}  符号")
    neg_months = 0; valid_months = 0
    for (y, m), v in sorted(month_results.items()):
        fhl = v["f_hl"]; ohl = v["o_hl"]
        nf  = v["n_f"];  no  = v["n_o"]
        if nf == 0 and fhl is None:
            print(f"  {y}-{m:02d}    {'--':>9} {'--':>5}  {'--':>9} {'--':>5}")
            continue
        fhl_s = f"{fhl:+.3f}" if fhl is not None else "  n/a"
        ohl_s = f"{ohl:+.3f}" if ohl is not None else "  n/a"
        sign  = ("✓" if fhl is not None and fhl < 0
                 else ("✗" if fhl is not None and fhl > 0 else "-"))
        print(f"  {y}-{m:02d}    {fhl_s:>9} {nf:>5}  {ohl_s:>9} {no:>5}  {sign}")
        if fhl is not None:
            valid_months += 1
            if fhl < 0: neg_months += 1

    pct = neg_months / valid_months * 100 if valid_months else 0
    print(f"\n  Fundingマイナス月: {neg_months}/{valid_months} = {pct:.0f}%")

    # ── 銘柄別 ───────────────────────────────────────────────────
    print("\n■ 銘柄別 Funding high-low 差（符号の一貫性）")
    print(f"  {'symbol':<22} {'fund_hl':>9} {'n':>5}")
    neg_syms = 0
    for sym, pairs in sorted(sym_fund.items()):
        t = tercile_table(pairs)
        if t:
            hl = t["high"][1] - t["low"][1]
            sign = "✓" if hl < 0 else "✗"
            print(f"  {sym:<22} {hl:>+9.3f}% {len(pairs):>5}  {sign}")
            if hl < 0: neg_syms += 1
        else:
            print(f"  {sym:<22} {'n/a':>9}  {'--':>5}")

    # ── 最終判定 ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("■ OOS横断 最終判定")
    print(f"  Funding h-l マイナス月: {neg_months}/{valid_months} = {pct:.0f}%")

    if pct >= 70:
        verdict = "🟢 煙あり（OOS安定）"
        comment = ("Fundingの逆張りシグナルは複数レジームで符号が揃う。\n"
                   "  → 有料連続データ（Tardis Pro）で日次OOS確認に進む価値がある。\n"
                   "  → ただし「各月1日」サンプルでの確認。連続日次では異なる可能性あり。")
    elif pct >= 50:
        verdict = "🟡 弱い煙（方向はあるが不安定）"
        comment = ("符号はある程度一致するが、レジーム依存の可能性が残る。\n"
                   "  → 有料データへの投資は判断保留。前向き無料蓄積を3ヶ月続けてから再判断。")
    else:
        verdict = "🔴 煙なし（符号が揃わない）"
        comment = ("Fundingの傾向は直近3ヶ月の偶然だった可能性が高い。\n"
                   "  → この路線は一旦見送り。有料データへの投資は不要。")

    print(f"\n  判定: {verdict}")
    print(f"  {comment}")
    print("\n禁止事項（変更なし）:")
    print("  main.py変更なし / deployなし / mergeなし / 有料契約まだなし")
    print("=" * 78)


if __name__ == "__main__":
    main()
