#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_unlock_data.py  —  【Macローカルで実行】DeFiLlama emissions スキーマ確認
==============================================================================
検証B（トークンアンロック）の本取得に入る前に、DeFiLlamaの無料 emissions API の
実際のデータ構造を確認する。スキーマを推測でハードコードせず、生JSONを保存して
中身を見てから本パーサを書くための偵察スクリプト。

★賢治さんのMac（DeFiLlama に接続できる環境）で実行する。
  リモート（Claude側）はプロキシで api.llama.fi が403。

使い方（Macのターミナル）:
    cd ~/Downloads
    curl -L "https://raw.githubusercontent.com/kenji888knymec/ocho/claude/crypto-bot-assistant-QlA5G/probe_unlock_data.py" -o probe_unlock_data.py
    python3 probe_unlock_data.py

  → 画面に構造サマリーが出る。unlock_probe_list.json と
    unlock_probe_samples.json が ~/Downloads に保存される。
    この2ファイル（または画面出力）をチャットに貼れば、本パーサを書く。

依存: 標準ライブラリのみ（pip不要）。何も売買しない・取得して保存するだけ。
"""

from __future__ import annotations
import urllib.request
import json

LIST_URL   = "https://api.llama.fi/emissions"
DETAIL_URL = "https://api.llama.fi/emission/{name}"
# 反応分析で価格データが揃いやすい代表銘柄（スキーマ確認用サンプル。合否には無関係）
SAMPLE_HINTS = ["aptos", "arbitrum", "optimism", "sui", "sei", "celestia"]


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())


def describe(obj, depth=0, max_depth=3, prefix=""):
    """JSONの構造（キー・型・先頭要素）を浅く表示。値の中身は出しすぎない。"""
    ind = "  " * depth
    if depth > max_depth:
        print(f"{ind}{prefix}...(深さ上限)")
        return
    if isinstance(obj, dict):
        print(f"{ind}{prefix}dict keys: {list(obj.keys())[:20]}")
        for k in list(obj.keys())[:6]:
            describe(obj[k], depth+1, max_depth, prefix=f"{k}: ")
    elif isinstance(obj, list):
        print(f"{ind}{prefix}list len={len(obj)}")
        if obj:
            describe(obj[0], depth+1, max_depth, prefix="[0]: ")
    else:
        s = str(obj)
        print(f"{ind}{prefix}{type(obj).__name__} = {s[:80]}")


def main():
    print("="*60)
    print("DeFiLlama emissions スキーマ偵察")
    print("="*60)

    # ── 一覧 ──
    print("\n■ 一覧 (api.llama.fi/emissions) 取得中...")
    try:
        lst = get_json(LIST_URL)
    except Exception as e:
        print(f"❌ 一覧取得失敗: {e}")
        print("   → 第2候補(token.unlocks.app)の検討が必要。賢治さんに報告。")
        return

    with open("unlock_probe_list.json", "w") as f:
        json.dump(lst, f)
    print("✅ unlock_probe_list.json 保存")
    print("\n--- 一覧の構造 ---")
    describe(lst, max_depth=2)

    # 一覧から名前候補を拾う（スキーマ未知なので緩く探索）
    names = []
    container = lst
    if isinstance(lst, dict):
        # よくある形: {"protocols":[...]} 等。最初のlist値を使う
        for v in lst.values():
            if isinstance(v, list) and v:
                container = v
                break
    if isinstance(container, list):
        for item in container[:2000]:
            if isinstance(item, dict):
                for key in ("name", "gecko_id", "protocolId", "slug"):
                    if key in item and isinstance(item[key], str):
                        names.append(item[key])
                        break
    print(f"\n一覧から拾えた識別子例（先頭30）: {names[:30]}")
    print(f"総数（拾えた分）: {len(names)}")

    # ── 個別詳細サンプル ──
    print("\n■ 個別詳細サンプル取得（スキーマ確認用）...")
    tried = []
    for hint in SAMPLE_HINTS:
        match = next((nm for nm in names if hint.lower() in nm.lower()), hint)
        tried.append(match)
    # 拾えた名前があれば先頭も足す
    if names:
        tried = list(dict.fromkeys(tried + names[:3]))

    samples = {}
    for nm in tried[:6]:
        url = DETAIL_URL.format(name=nm)
        try:
            d = get_json(url)
            samples[nm] = d
            print(f"\n--- 詳細: {nm} ---")
            describe(d, max_depth=2)
        except Exception as e:
            print(f"  {nm}: 取得失敗 {e}")

    if samples:
        with open("unlock_probe_samples.json", "w") as f:
            json.dump(samples, f)
        print("\n✅ unlock_probe_samples.json 保存")

    print("\n" + "="*60)
    print("完了。以下をチャットに貼ってください:")
    print("  1) この画面の『構造』出力")
    print("  2) unlock_probe_list.json / unlock_probe_samples.json（重ければ画面出力だけでも可）")
    print("→ 実スキーマを見てから、事前登録の窓・基準のまま本取得パーサを書きます。")
    print("="*60)


if __name__ == "__main__":
    main()
