"""
Z2 Bypass バックテストフレームワーク
対象: 6/17-6/19 bypass通知データ（short_a, notify_sent=1）
比較: 日次cap=3 + daily loss-stop=-3.0 の各組み合わせ
評価軸: 日次PnL_Net、通知件数、WR
"""
import pandas as pd
import numpy as np
import sys
from datetime import date

EXCEL_PATH = "/root/.claude/uploads/9ce18743-e04a-5f8a-a82e-32ed9d65046e/caae4e3d-EntryRecord_2.xlsx"

# ---- データ読み込み ----
df = pd.read_excel(EXCEL_PATH)
short_done = df[(df["Direction"] == "SHORT") & (df["EvalStatus"] == "DONE")].copy()
short_done["notify_sent"] = short_done["AI_Note"].str.contains("notify_sent=1", na=False).astype(int)
short_done["Date_JST"] = pd.to_datetime(short_done["Datetime_JST"]).dt.date
short_done["rank_score"] = short_done["AI_Note"].str.extract(r"rank_score=([0-9.e+\-]+)")[0].astype(float)

# bypass notified 6/17-6/19
bypass_all = short_done[short_done["Feature_Bypass_Profile"].notna()].copy()
TARGET_DATES = [date(2026, 6, 17), date(2026, 6, 18), date(2026, 6, 19)]
june = bypass_all[bypass_all["Date_JST"].isin(TARGET_DATES)].copy()
notified = june[june["notify_sent"] == 1].sort_values("Datetime_JST").reset_index(drop=True)

def run_scenario(signals, daily_cap=999, daily_loss_stop=float("-inf"),
                 use_g2_block=False, sort_by_rank=False, label=""):
    """
    シナリオシミュレーション
    signals: DataFrame (notified bypass signals, sort済み)
    daily_cap: 日次最大通知件数
    daily_loss_stop: 日次PnL_Netがこの値以下になったら当日通知停止
    use_g2_block: G2=BLOCK の場合にスキップ
    sort_by_rank: True=rank_scoreでソート, False=時系列順
    """
    results = []
    for d, grp in signals.groupby("Date_JST"):
        if sort_by_rank:
            grp = grp.sort_values("rank_score", ascending=False).reset_index(drop=True)
        else:
            grp = grp.sort_values("Datetime_JST").reset_index(drop=True)

        daily_pnl = 0.0
        daily_count = 0
        blocked_cap = 0
        blocked_loss = 0
        blocked_g2 = 0
        sent_rows = []

        for _, row in grp.iterrows():
            # G2 gate
            if use_g2_block and row.get("G2_Gate") == "BLOCK":
                blocked_g2 += 1
                continue
            # daily cap
            if daily_count >= daily_cap:
                blocked_cap += 1
                continue
            # daily loss stop
            if daily_pnl <= daily_loss_stop:
                blocked_loss += 1
                continue
            # pass
            sent_rows.append(row)
            pnl = row["PnL_Net"] if np.isfinite(row["PnL_Net"]) else 0.0
            daily_pnl += pnl
            daily_count += 1

        wins = sum(1 for r in sent_rows if r["WinLose"] == "Win")
        n = len(sent_rows)
        wr = wins / n if n > 0 else float("nan")
        results.append({
            "date": d, "sent": n, "blocked_cap": blocked_cap,
            "blocked_loss": blocked_loss, "blocked_g2": blocked_g2,
            "wins": wins, "wr": wr, "pnl_net": daily_pnl
        })
    return results

def print_scenario(label, results, total_in):
    total_sent = sum(r["sent"] for r in results)
    total_blocked_cap = sum(r["blocked_cap"] for r in results)
    total_blocked_loss = sum(r["blocked_loss"] for r in results)
    total_blocked_g2 = sum(r["blocked_g2"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_pnl = sum(r["pnl_net"] for r in results)
    total_wr = total_wins / total_sent if total_sent > 0 else float("nan")

    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'Date':<12} {'sent':>5} {'b_cap':>6} {'b_loss':>7} {'b_g2':>6} {'WR':>7} {'PnL_Net':>9}")
    for r in results:
        wr_str = f"{r['wr']:.1%}" if np.isfinite(r["wr"]) else "  N/A"
        print(f"{str(r['date']):<12} {r['sent']:>5} {r['blocked_cap']:>6} {r['blocked_loss']:>7} {r['blocked_g2']:>6} {wr_str:>7} {r['pnl_net']:>9.3f}")
    print(f"{'TOTAL':<12} {total_sent:>5} {total_blocked_cap:>6} {total_blocked_loss:>7} {total_blocked_g2:>6} {total_wr:>7.1%} {total_pnl:>9.3f}")
    print(f"  (input pool: {total_in}件)")


total_in = len(notified)

# ---- シナリオ一覧 ----
scenarios = [
    ("A: 実績（cap無し）",               dict(daily_cap=999, daily_loss_stop=float("-inf"), use_g2_block=False, sort_by_rank=False)),
    ("B: cap=3 時系列順",                dict(daily_cap=3,   daily_loss_stop=float("-inf"), use_g2_block=False, sort_by_rank=False)),
    ("C: cap=3 rank順",                  dict(daily_cap=3,   daily_loss_stop=float("-inf"), use_g2_block=False, sort_by_rank=True)),
    ("D: cap=3 + loss_stop=-3.0 時系列", dict(daily_cap=3,   daily_loss_stop=-3.0,          use_g2_block=False, sort_by_rank=False)),
    ("E: cap=3 + loss_stop=-3.0 rank順", dict(daily_cap=3,   daily_loss_stop=-3.0,          use_g2_block=False, sort_by_rank=True)),
    ("F: G2 block のみ（cap無し）",       dict(daily_cap=999, daily_loss_stop=float("-inf"), use_g2_block=True,  sort_by_rank=False)),
    ("G: G2 block + cap=3 rank順",       dict(daily_cap=3,   daily_loss_stop=float("-inf"), use_g2_block=True,  sort_by_rank=True)),
    ("H: G2 block + cap=3 + loss_stop",  dict(daily_cap=3,   daily_loss_stop=-3.0,          use_g2_block=True,  sort_by_rank=True)),
]

print("=" * 60)
print("Z2 Bypass バックテスト 6/17-6/19")
print(f"対象: bypass notified SHORT DONE {total_in}件")
print(f"G2 BLOCK数: {(notified['G2_Gate']=='BLOCK').sum()}件 / PASS数: {(notified['G2_Gate']=='PASS').sum()}件")

for label, kwargs in scenarios:
    results = run_scenario(notified, **kwargs, label=label)
    print_scenario(label, results, total_in)

# ---- G2 の詳細統計 ----
print("\n" + "=" * 60)
print("[G2 Gate 詳細統計 in notified bypass]")
for gate, grp in notified.groupby("G2_Gate"):
    wins = (grp["WinLose"] == "Win").sum()
    n = len(grp)
    wr = wins/n
    pnl = grp["PnL_Net"].sum()
    avg = grp["PnL_Net"].mean()
    print(f"  G2={gate}: n={n}, WR={wr:.1%}, PnL_Net_sum={pnl:.3f}, avgPnL={avg:.3f}")
