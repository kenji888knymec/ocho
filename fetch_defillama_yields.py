#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_defillama_yields.py  —  【Macローカルで実行】DeFi利回り取得（検証⑤⑥・無料・キー不要）
==============================================================================
DeFiLlama の yields API（★無料・キー不要・カード不要。有料化したemissionsとは別物）から、
  ⑤ DeFi貸し出し（ステーブルコイン lending APY）
  ⑥ DEX LP（流動性提供の APY と インパーマネントロス・リスク）
の現在地スナップショット＋30日平均を取得して保存する。

★賢治さんのMac（DeFiLlamaに接続できる環境）で実行する。リモートはプロキシで403。

使い方（Macのターミナル・1行ずつ）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/fetch_defillama_yields.py?v=1" -o fetch_defillama_yields.py
    head -3 fetch_defillama_yields.py
    python3 fetch_defillama_yields.py

  → ~/Downloads に defillama_yields.csv が出来る（列は下記）。
    画面の要約 or CSV をチャットに貼れば、こちらで⑤⑥を判定する。

依存: 標準ライブラリのみ（pip不要・キー不要）。売買・送金・Bot接続は一切なし。取得・保存のみ。
"""

from __future__ import annotations
import urllib.request
import json

POOLS_URL = "https://yields.llama.fi/pools"

# ⑤ 貸し出し: 主要lendingプロトコル（ステーブル）
LEND_PROJECTS = {
    "aave-v3", "aave-v2", "compound-v3", "compound", "morpho-blue", "morpho-aave",
    "spark", "fluid-lending", "venus-core-pool", "maker-dsr", "sky-lending",
}
# 対象チェーン（メジャーのみ・極端に小さいL2の高APYワナを避ける）
MAJOR_CHAINS = {"Ethereum", "Arbitrum", "Base", "Optimism", "Polygon", "Avalanche", "BSC", "Solana"}
MIN_TVL = 20_000_000   # $20M 未満は除外（薄いプール=持続性低い）


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    print("=" * 64)
    print("DeFiLlama yields 取得  検証⑤(貸出) ⑥(DEX LP)  無料・キー不要")
    print("=" * 64)
    try:
        j = get_json(POOLS_URL)
    except Exception as e:
        print(f"❌ 取得失敗: {e}")
        print("   → DeFiLlama yieldsも無料で叩けない場合、⑤⑥は『データ取得コストあり候補』として保留に。")
        return
    data = j.get("data", [])
    print(f"  全プール数: {len(data)}")

    rows = []  # category, project, symbol, chain, apy, apyBase, apyReward, apyMean30d, ilRisk, stablecoin, tvlUsd, pool

    # ⑤ ステーブル貸出
    lend = []
    for p in data:
        if p.get("project") in LEND_PROJECTS and p.get("stablecoin") is True:
            tvl = num(p.get("tvlUsd")) or 0
            if tvl >= MIN_TVL and p.get("chain") in MAJOR_CHAINS:
                lend.append(p)
    lend.sort(key=lambda p: -(num(p.get("tvlUsd")) or 0))
    print(f"\n=== ⑤ ステーブル貸出（{len(lend)}件・TVL≥$20M・メジャーチェーン）===")
    print(f"  {'project':12s} {'sym':6s} {'chain':9s} {'APY%':>6} {'30dMean%':>8} {'TVL$M':>7}")
    for p in lend[:20]:
        apy = num(p.get("apy")); m30 = num(p.get("apyMean30d")); tvl = (num(p.get("tvlUsd")) or 0)/1e6
        print(f"  {str(p.get('project')):12s} {str(p.get('symbol'))[:6]:6s} {str(p.get('chain')):9s} "
              f"{apy if apy is not None else 0:6.2f} {m30 if m30 is not None else 0:8.2f} {tvl:7.0f}")
        rows.append(("lending", p.get("project"), p.get("symbol"), p.get("chain"),
                     apy, num(p.get("apyBase")), num(p.get("apyReward")), m30,
                     p.get("ilRisk"), p.get("stablecoin"), num(p.get("tvlUsd")), p.get("pool")))

    # ⑥ DEX LP（multi-asset=ペア、IL有り得る）
    lp = []
    for p in data:
        if p.get("exposure") == "multi" and (num(p.get("tvlUsd")) or 0) >= 50_000_000:
            if p.get("chain") in MAJOR_CHAINS:
                lp.append(p)
    lp.sort(key=lambda p: -(num(p.get("apy")) or 0))
    print(f"\n=== ⑥ DEX LP（multi-asset・TVL≥$50M・メジャー・APY降順上位）===")
    print(f"  {'project':14s} {'symbol':16s} {'APY%':>6} {'base%':>6} {'ilRisk':6s} {'stable':6s} {'TVL$M':>7}")
    for p in lp[:25]:
        apy = num(p.get("apy")); base = num(p.get("apyBase")); tvl = (num(p.get("tvlUsd")) or 0)/1e6
        print(f"  {str(p.get('project'))[:14]:14s} {str(p.get('symbol'))[:16]:16s} "
              f"{apy if apy is not None else 0:6.2f} {base if base is not None else 0:6.2f} "
              f"{str(p.get('ilRisk')):6s} {str(p.get('stablecoin')):6s} {tvl:7.0f}")
        rows.append(("dex_lp", p.get("project"), p.get("symbol"), p.get("chain"),
                     apy, base, num(p.get("apyReward")), num(p.get("apyMean30d")),
                     p.get("ilRisk"), p.get("stablecoin"), num(p.get("tvlUsd")), p.get("pool")))

    path = "defillama_yields.csv"
    with open(path, "w") as f:
        f.write("category,project,symbol,chain,apy,apyBase,apyReward,apyMean30d,ilRisk,stablecoin,tvlUsd,pool\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print("\n" + "=" * 64)
    print(f"✅ {path} 保存: {len(rows)}行（lending {len(lend[:20])} / dex_lp {len(lp[:25])}）")
    print("→ 画面要約 or defillama_yields.csv をチャットに貼ってください。")
    print("=" * 64)


if __name__ == "__main__":
    main()
