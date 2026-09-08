# -*- coding: utf-8 -*-
"""
A-6: Panic群(VIX>40)が実質何個の独立エピソードで構成されているかを確認し、
2008年GFC・2020年COVIDのそれぞれを除外しても結果が生き残るかを検証する（leave-one-crisis-out）。
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from data_pipeline import DATA_DIR, build_dataset, analysis_frame, VIX_PHASE_LABELS

RESULTS = {}

df_raw = build_dataset()
df = analysis_frame(df_raw, ["VIX_Phase", "Return_6M"])

# --- 1. VIX>40（Panic）のエピソードをクラスタリング ---
# 「63営業日（約3ヶ月）以上VIX>40の日が空いたら別エピソード」とみなす
panic_dates = df.index[df["VIX_Phase"] == "4_Panic (>40)"]
gap_threshold_days = 63

episodes = []
if len(panic_dates) > 0:
    start = panic_dates[0]
    prev = panic_dates[0]
    # 営業日インデックス上のギャップで判定（df全体のインデックス位置を使う）
    pos = {d: i for i, d in enumerate(df.index)}
    for d in panic_dates[1:]:
        if pos[d] - pos[prev] > gap_threshold_days:
            episodes.append((start, prev))
            start = d
        prev = d
    episodes.append((start, prev))

print("=" * 70)
print(f"VIX>40（Panic）のエピソード検出（{gap_threshold_days}営業日以上の空白で別エピソードとみなす）")
print("=" * 70)
for i, (s, e) in enumerate(episodes, 1):
    n_days = ((df.index >= s) & (df.index <= e) & (df["VIX_Phase"] == "4_Panic (>40)")).sum()
    print(f"  エピソード{i}: {s.date()} 〜 {e.date()}  (Panic日数={n_days})")

print(f"\n独立エピソード数: {len(episodes)}  (対して単純な観測日数: {len(panic_dates)})")
RESULTS["panic_episodes"] = [
    {"start": str(s.date()), "end": str(e.date())} for s, e in episodes
]
RESULTS["n_panic_days"] = int(len(panic_dates))
RESULTS["n_panic_episodes"] = len(episodes)

# 同様にFear群(30-40)のエピソードも数える
fear_dates = df.index[df["VIX_Phase"] == "3_Fear (30-40)"]
fear_episodes = []
if len(fear_dates) > 0:
    start = fear_dates[0]
    prev = fear_dates[0]
    for d in fear_dates[1:]:
        if pos[d] - pos[prev] > gap_threshold_days:
            fear_episodes.append((start, prev))
            start = d
        prev = d
    fear_episodes.append((start, prev))
print(f"\n参考: Fear(30-40)群の独立エピソード数: {len(fear_episodes)} (観測日数={len(fear_dates)})")
for i, (s, e) in enumerate(fear_episodes, 1):
    print(f"  エピソード{i}: {s.date()} 〜 {e.date()}")
RESULTS["n_fear_episodes"] = len(fear_episodes)
RESULTS["n_fear_days"] = int(len(fear_dates))

# --- 2. Leave-one-crisis-out ---
# エピソードを危機名で分類（年で判定。1990-2023の実データに基づき自動でラベル付け）


def label_episode(s, e):
    year = s.year
    if 2007 <= year <= 2009:
        return "2008_GFC"
    if year == 2020:
        return "2020_COVID"
    return f"Other_{year}"


episode_labels = [label_episode(s, e) for s, e in episodes]
print("\nエピソードのラベル付け:", list(zip(episode_labels, [(s.date(), e.date()) for s, e in episodes])))
RESULTS["episode_labels"] = episode_labels

print("\n" + "=" * 70)
print("Leave-one-crisis-out: 危機を1つずつ除外してPanic群の効果を再検証")
print("=" * 70)


def run_phase_tests(sub_df, label):
    sub_groups_dict = {lab: sub_df.loc[sub_df["VIX_Phase"] == lab, "Return_6M"].values for lab in VIX_PHASE_LABELS}
    counts = {lab: len(v) for lab, v in sub_groups_dict.items()}
    sub_groups = list(sub_groups_dict.values())
    if any(len(g) < 2 for g in sub_groups):
        print(f"[{label}] 群内nが不足しているため一部検定を省略。 N別={counts}")
        return {"label": label, "group_counts": counts, "insufficient": True}

    f_s, p_s = stats.f_oneway(*sub_groups)
    welch_s = anova_oneway(sub_groups, use_var="unequal", welch_correction=True)
    # PanicとNormalの平均差（Welch t検定）
    normal = sub_groups_dict["1_Normal (<20)"]
    panic = sub_groups_dict["4_Panic (>40)"]
    t_stat, p_t = stats.ttest_ind(panic, normal, equal_var=False)
    mean_diff = panic.mean() - normal.mean()

    print(f"[{label}] N={len(sub_df)}  群別N={counts}")
    print(f"   ANOVA F={f_s:.4f} p={p_s:.4e} | Welch F={welch_s.statistic:.4f} p={welch_s.pvalue:.4e}")
    print(f"   Panic vs Normal: 平均差={mean_diff:+.4f}  Welch-t p={p_t:.4e}  (Panic平均={panic.mean():.4f}, N_panic={len(panic)})")
    return {
        "label": label, "group_counts": counts,
        "anova": {"stat": f_s, "p": p_s}, "welch": {"stat": welch_s.statistic, "p": welch_s.pvalue},
        "panic_vs_normal": {"mean_diff": mean_diff, "welch_t_p": p_t, "panic_mean": float(panic.mean()), "n_panic": len(panic)},
        "insufficient": False,
    }


loco_results = []
loco_results.append(run_phase_tests(df, "Baseline(全期間)"))

for crisis in sorted(set(episode_labels)):
    exclude_ranges = [(s, e) for (s, e), lab in zip(episodes, episode_labels) if lab == crisis]
    mask = pd.Series(True, index=df.index)
    for s, e in exclude_ranges:
        mask &= ~((df.index >= s) & (df.index <= e))
    sub = df[mask]
    loco_results.append(run_phase_tests(sub, f"{crisis}を除外"))

# 両方除外
both_ranges = [(s, e) for (s, e), lab in zip(episodes, episode_labels) if lab in ("2008_GFC", "2020_COVID")]
mask = pd.Series(True, index=df.index)
for s, e in both_ranges:
    mask &= ~((df.index >= s) & (df.index <= e))
sub_both = df[mask]
loco_results.append(run_phase_tests(sub_both, "2008_GFCと2020_COVIDの両方を除外"))

RESULTS["leave_one_crisis_out"] = loco_results

# ---------------------------------------------------------------
# 重要な注意（本分析の解釈上の制約）
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("【解釈上の注意】上記 Welch-t の p 値について")
print("=" * 70)
print("""上記のp値は日次でスライドさせた（重複する）6ヶ月先リターンに対して計算されている。
本リポジトリの analysis_01 が示す通り、この標本は強い正の自己相関を持ち（DW≈0.03）、
p値は独立標本を前提とした場合より大幅に過大評価される。
したがって、これらのp値を「有意性が維持された」根拠として用いてはならない。
LOCOで確認できるのは『効果の符号と大きさが、特定の危機イベントの除外に対して
安定しているかどうか』に限られる。以下にエピソード単位の推論を併記する。""")

# ---------------------------------------------------------------
# エピソード単位の推論（重複の影響を受けにくいブロック単位の集計）
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("エピソード単位の集計（1エピソード = 1観測）")
print("=" * 70)
normal_mean = df.loc[df["VIX_Phase"] == "1_Normal (<20)", "Return_6M"].mean()
print(f"比較基準: Normal群(<20)の平均6ヶ月先リターン = {normal_mean:+.4f}")

episode_means = []
for (s, e), lab in zip(episodes, episode_labels):
    mask_ep = (df.index >= s) & (df.index <= e) & (df["VIX_Phase"] == "4_Panic (>40)")
    vals = df.loc[mask_ep, "Return_6M"]
    if len(vals) == 0:
        continue
    episode_means.append({
        "episode": f"{s.date()}〜{e.date()}", "label": lab,
        "n_panic_days": int(len(vals)),
        "mean_return_6m": float(vals.mean()),
        "exceeds_normal": bool(vals.mean() > normal_mean),
    })
    print(f"  {s.date()}〜{e.date()} ({lab:12s}, n={len(vals):3d}日): "
          f"平均={vals.mean():+.4f}  Normal超={'○' if vals.mean() > normal_mean else '×'}")

n_ep = len(episode_means)
n_exceed = sum(r["exceeds_normal"] for r in episode_means)
ep_vals = np.array([r["mean_return_6m"] for r in episode_means])

# 符号検定（両側二項検定）: エピソードを独立単位とみなした最も保守的な推論
sign_p = float(stats.binomtest(n_exceed, n_ep, 0.5).pvalue)
print(f"\n  {n_ep}エピソード中 {n_exceed}個 でNormal平均を上回った")
print(f"  符号検定（両側二項検定, H0: 上回る確率=0.5）: p = {sign_p:.4f}")
print(f"  エピソード平均リターンの中央値 = {np.median(ep_vals):+.4f}, "
      f"範囲 = [{ep_vals.min():+.4f}, {ep_vals.max():+.4f}]")
print(f"""
  【この検定の限界】n={n_ep} は極めて小さく、検出力は低い。
  符号検定は「エピソード内部の重複」を回避する代わりに情報の大半を捨てているため、
  日次検定（過大評価）と符号検定（過小評価に近い保守側）の間に真の不確実性がある。
  本研究はどちらか一方を「正しい有意性」として採用しない。""")

RESULTS["episode_level_inference"] = {
    "normal_group_mean": float(normal_mean),
    "episodes": episode_means,
    "n_episodes": n_ep,
    "n_exceeding_normal": int(n_exceed),
    "sign_test_two_sided_p": sign_p,
    "episode_mean_median": float(np.median(ep_vals)),
    "episode_mean_min": float(ep_vals.min()),
    "episode_mean_max": float(ep_vals.max()),
    "caveat": "n=9は小標本であり検出力は低い。日次p値は自己相関により過大評価される。",
}

with open(DATA_DIR / "results_02_leave_one_crisis_out.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
print("\n結果を data/results_02_leave_one_crisis_out.json に保存しました")
