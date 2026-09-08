# -*- coding: utf-8 -*-
"""
A-1: README に記載されているが欠落していた分析をコードとして復元する。
- 標準ANOVA / Welch ANOVA / Kruskal-Wallis
- Levene(mean) / Levene(median, Brown-Forsythe) / Bartlett
- Tukey HSD / Welch型ペアワイズ + Holm補正
- 間引きANOVA（21/63/126営業日）+ 群別N + 効果量(eta^2)
- ANOVA型残差のDurbin-Watson

このスクリプトは data_pipeline.build_dataset() の単一データソースのみを使う（A-2対応）。
"""
import json
from scipy import stats
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf

from data_pipeline import DATA_DIR, build_dataset, analysis_frame, VIX_PHASE_LABELS

RESULTS = {}

print("=" * 70)
print("データ取得（単一パイプライン）")
print("=" * 70)
df_raw = build_dataset()
df = analysis_frame(df_raw, ["VIX_Phase", "Return_6M"])
print(f"N = {len(df)}  期間: {df.index.min().date()} 〜 {df.index.max().date()}")

group_counts = df["VIX_Phase"].value_counts().reindex(VIX_PHASE_LABELS)
print("\n群別N:")
print(group_counts)
RESULTS["group_counts_daily"] = group_counts.to_dict()

groups = [df.loc[df["VIX_Phase"] == lab, "Return_6M"].values for lab in VIX_PHASE_LABELS]

# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("1. 標準ANOVA / Welch ANOVA / Kruskal-Wallis")
print("=" * 70)
f_stat, p_anova = stats.f_oneway(*groups)
print(f"Standard one-way ANOVA: F={f_stat:.4f}, p={p_anova:.4e}")

welch_res = anova_oneway(groups, use_var="unequal", welch_correction=True)
print(f"Welch ANOVA: F={welch_res.statistic:.4f}, p={welch_res.pvalue:.4e}")

h_stat, p_kw = stats.kruskal(*groups)
print(f"Kruskal-Wallis: H={h_stat:.4f}, p={p_kw:.4e}")

RESULTS["anova_daily"] = {
    "standard": {"stat": f_stat, "p": p_anova},
    "welch": {"stat": welch_res.statistic, "p": welch_res.pvalue},
    "kruskal": {"stat": h_stat, "p": p_kw},
    "N": len(df),
}

# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("2. 等分散性の検証")
print("=" * 70)
lev_mean = stats.levene(*groups, center="mean")
lev_med = stats.levene(*groups, center="median")
bart = stats.bartlett(*groups)
print(f"Levene(center=mean): stat={lev_mean.statistic:.4f}, p={lev_mean.pvalue:.4e}")
print(f"Levene(center=median / Brown-Forsythe): stat={lev_med.statistic:.4f}, p={lev_med.pvalue:.4e}")
print(f"Bartlett: stat={bart.statistic:.4f}, p={bart.pvalue:.4e}")

RESULTS["variance_tests"] = {
    "levene_mean": {"stat": lev_mean.statistic, "p": lev_mean.pvalue},
    "levene_median": {"stat": lev_med.statistic, "p": lev_med.pvalue},
    "bartlett": {"stat": bart.statistic, "p": bart.pvalue},
}

# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3. Tukey HSD と Welch型ペアワイズ(Holm補正)")
print("=" * 70)
tukey = pairwise_tukeyhsd(endog=df["Return_6M"], groups=df["VIX_Phase"], alpha=0.05)
print(tukey)

pairs = []
raw_p = []
for i in range(len(VIX_PHASE_LABELS)):
    for j in range(i + 1, len(VIX_PHASE_LABELS)):
        a, b = VIX_PHASE_LABELS[i], VIX_PHASE_LABELS[j]
        ga = df.loc[df["VIX_Phase"] == a, "Return_6M"].values
        gb = df.loc[df["VIX_Phase"] == b, "Return_6M"].values
        t, p = stats.ttest_ind(gb, ga, equal_var=False)  # Welch's t-test
        pairs.append((a, b, gb.mean() - ga.mean(), p))
        raw_p.append(p)

reject, p_holm, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
print("\nWelch型ペアワイズ + Holm補正:")
welch_pairwise = []
for (a, b, diff, p_raw), p_adj, rej in zip(pairs, p_holm, reject):
    print(f"  {a} vs {b}: meandiff={diff:+.4f}  Holm-p={p_adj:.4e}  reject={rej}")
    welch_pairwise.append({"group1": a, "group2": b, "meandiff": diff, "holm_p": p_adj, "reject": bool(rej)})
RESULTS["welch_pairwise_holm"] = welch_pairwise

# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("4. 間引きANOVA（重複リターンの影響確認）+ 効果量(eta^2) + 群別N")
print("=" * 70)


def eta_squared(f_stat, groups):
    k = len(groups)
    n = sum(len(g) for g in groups)
    df_between = k - 1
    df_within = n - k
    return (f_stat * df_between) / (f_stat * df_between + df_within)


thinning = {"Daily": 1, "Monthly_21d": 21, "Quarterly_63d": 63, "Semiannual_126d": 126}
thin_results = []
for name, step in thinning.items():
    sub = df.iloc[::step]
    sub_groups = [sub.loc[sub["VIX_Phase"] == lab, "Return_6M"].values for lab in VIX_PHASE_LABELS]
    sub_counts = {lab: len(g) for lab, g in zip(VIX_PHASE_LABELS, sub_groups)}
    min_n = min(len(g) for g in sub_groups)

    # 標準ANOVA・Kruskal-Wallisはn=1の群があっても計算自体は可能（群内自由度が減るだけ）
    f_s, p_s = stats.f_oneway(*sub_groups)
    h_s, p_kw_s = stats.kruskal(*sub_groups)
    eta2 = eta_squared(f_s, sub_groups)

    entry = {
        "sampling": name, "N": len(sub), "group_counts": sub_counts, "min_group_n": min_n,
        "anova": {"stat": f_s, "p": p_s},
        "kruskal": {"stat": h_s, "p": p_kw_s},
        "eta_sq": eta2,
    }

    print(f"{name}: N={len(sub)}  群別N={sub_counts}")
    if min_n < 2:
        # Welchは群内分散を要求するため、n=1の群があると計算不能。正直に「計算不可」と明示する。
        print(f"   ANOVA F={f_s:.4f} p={p_s:.4e} | Welch: 計算不可(最小群n={min_n}<2のため群内分散が推定不能) | "
              f"Kruskal H={h_s:.4f} p={p_kw_s:.4e} | eta^2={eta2:.4f}")
        print(f"   [注意] 最小群nが{min_n}のため、この行のANOVA/eta^2も統計的に不安定（参考値）")
        entry["welch"] = None
        entry["reliable"] = False
    else:
        welch_s = anova_oneway(sub_groups, use_var="unequal", welch_correction=True)
        print(f"   ANOVA F={f_s:.4f} p={p_s:.4e} | Welch F={welch_s.statistic:.4f} p={welch_s.pvalue:.4e} | "
              f"Kruskal H={h_s:.4f} p={p_kw_s:.4e} | eta^2={eta2:.4f}")
        entry["welch"] = {"stat": welch_s.statistic, "p": welch_s.pvalue}
        entry["reliable"] = True
    thin_results.append(entry)
RESULTS["thinned_anova"] = thin_results

# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("5. Durbin-Watson（ANOVA型モデルの残差の自己相関）")
print("=" * 70)
# VIX_Phaseをダミー変数化したOLSの残差でDWを計算（ANOVAと数学的に等価なモデル）
dummy_model = smf.ols("Return_6M ~ C(VIX_Phase)", data=df).fit()
dw_anova = durbin_watson(dummy_model.resid)
print(f"ANOVA型モデル(Return_6M ~ C(VIX_Phase))残差のDurbin-Watson: {dw_anova:.4f}")
RESULTS["durbin_watson_anova_model"] = dw_anova

# 保存
with open(DATA_DIR / "results_01_phase_tests.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
print("\n結果を data/results_01_phase_tests.json に保存しました")
