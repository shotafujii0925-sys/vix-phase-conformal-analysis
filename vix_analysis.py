# -*- coding: utf-8 -*-
"""
VIXフェーズはS&P500将来リターン分布を識別するか — 通し実行スクリプト

このスクリプトは data_pipeline.py の単一データ取得関数のみを使用する（A-2/A-3/A-4対応）。
上から順に実行すれば全セクションが独立に再現できる。詳細な数値・議論・限界は README.md を参照。

旧版はJupyterノートブック（vix_analysis.ipynb）だったが、閲覧・差分管理のしやすさを優先し、
本スクリプト1本に統一した（ノートブックは廃止）。各章の分析はもともと analysis_01〜05・
conformal_vix_forecast.py にも分割実装されている（README「リポジトリ構成」参照）。本スクリプトは
それらを1つの流れとして読めるようにまとめた通し版であり、内容はREADMEの章立てと対応する。

実行方法:
    python vix_analysis.py
図は figures/ に保存される（対話環境のplt.show()ではなく、スクリプトとして完結するようsavefigを使用）。
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非対話バックエンド（連続savefig時の不具合回避、conformal_vix_forecast.pyと同じ対応）
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_pipeline import (FIGURES_DIR, build_dataset, analysis_frame,
                           VIX_PHASE_LABELS)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# 0. セットアップとデータ取得
# 単一パイプライン（data_pipeline.build_dataset()）から全分析共通のデータフレームを取得する。
# =====================================================================
section("0. セットアップとデータ取得")

sns.set_theme(style="whitegrid")

df_raw = build_dataset()
df = analysis_frame(df_raw, ["VIX_Phase", "Return_6M"])
print("N =", len(df), " 期間:", df.index.min().date(), "〜", df.index.max().date())
print(df["VIX_Phase"].value_counts().sort_index())


# =====================================================================
# 1. VIXフェーズ間の平均リターン差（日次データ）
#
# 仮説: VIXフェーズ間でS&P500の6ヶ月先リターンの平均に差はない（H0）。
# 標準ANOVAに加え、等分散性を仮定しないWelch ANOVA、非正規性に頑健なKruskal-Wallis検定を併用する。
# =====================================================================
section("1. VIXフェーズ間の平均リターン差（日次データ）")

groups = [df.loc[df["VIX_Phase"] == lab, "Return_6M"].values for lab in VIX_PHASE_LABELS]

f_stat, p_anova = stats.f_oneway(*groups)
welch_res = anova_oneway(groups, use_var="unequal", welch_correction=True)
h_stat, p_kw = stats.kruskal(*groups)

print(f"Standard ANOVA : F={f_stat:.4f}  p={p_anova:.4e}")
print(f"Welch ANOVA    : F={welch_res.statistic:.4f}  p={welch_res.pvalue:.4e}")
print(f"Kruskal-Wallis : H={h_stat:.4f}  p={p_kw:.4e}")

# --- 1.1 等分散性の検証（Levene・Bartlett） ---
# 分散が群間で等しいという前提が成り立つかを確認する。
print("\n--- 1.1 等分散性の検証（Levene・Bartlett） ---")
lev_mean = stats.levene(*groups, center="mean")
lev_med = stats.levene(*groups, center="median")
bart = stats.bartlett(*groups)
print(f"Levene(mean)  : stat={lev_mean.statistic:.4f}  p={lev_mean.pvalue:.4e}")
print(f"Levene(median): stat={lev_med.statistic:.4f}  p={lev_med.pvalue:.4e}")
print(f"Bartlett      : stat={bart.statistic:.4f}  p={bart.pvalue:.4e}")

# --- 1.2 可視化（平均±95%CIを検定と対応させて重ねる） ---
# 初版は中央値のみの箱ひげ図で検定（平均の議論）と対応していなかった点を修正。群別Nも明示する。
print("\n--- 1.2 可視化 ---")
fig, ax = plt.subplots(figsize=(10, 6))
counts = df["VIX_Phase"].value_counts().reindex(VIX_PHASE_LABELS)
labels_with_n = [f"{lab}\n(n={counts[lab]})" for lab in VIX_PHASE_LABELS]

sns.boxplot(x="VIX_Phase", y="Return_6M", data=df, order=VIX_PHASE_LABELS,
            hue="VIX_Phase", palette="Blues", legend=False, ax=ax, fliersize=2)

means = df.groupby("VIX_Phase", observed=True)["Return_6M"].mean().reindex(VIX_PHASE_LABELS)
ci95 = 1.96 * df.groupby("VIX_Phase", observed=True)["Return_6M"].sem().reindex(VIX_PHASE_LABELS)
x_pos = np.arange(len(VIX_PHASE_LABELS))
ax.errorbar(x_pos, means.values, yerr=ci95.values, fmt="D", color="darkred",
            markersize=7, capsize=5, linewidth=1.8, label="Mean ± 95% CI", zorder=5)

ax.axhline(0, color="red", linestyle="--", alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_with_n)
ax.set_title("S&P 500 6-Month Future Returns by VIX Phase (1990-2023)\nBox=median/IQR, Diamond=mean±95%CI")
ax.set_ylabel("6-Month Return (price return)")
ax.set_xlabel("VIX Phase")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase_returns_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("図を figures/phase_returns_overview.png に保存しました")

# --- 1.3 Tukey HSD と Welch型ペアワイズ比較（Holm補正） ---
# 分散不均一のため、Tukey HSDに加えて等分散を仮定しないWelch型ペアワイズ比較をHolm法で補正する。
print("\n--- 1.3 Tukey HSD と Welch型ペアワイズ比較（Holm補正） ---")
tukey = pairwise_tukeyhsd(endog=df["Return_6M"], groups=df["VIX_Phase"], alpha=0.05)
print(tukey)

pairs, raw_p = [], []
for i in range(len(VIX_PHASE_LABELS)):
    for j in range(i + 1, len(VIX_PHASE_LABELS)):
        a, b = VIX_PHASE_LABELS[i], VIX_PHASE_LABELS[j]
        ga = df.loc[df["VIX_Phase"] == a, "Return_6M"].values
        gb = df.loc[df["VIX_Phase"] == b, "Return_6M"].values
        t, p = stats.ttest_ind(gb, ga, equal_var=False)
        pairs.append((a, b, gb.mean() - ga.mean(), p))
        raw_p.append(p)

reject, p_holm, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
print("\nWelch型ペアワイズ + Holm補正:")
for (a, b, diff, _), p_adj, rej in zip(pairs, p_holm, reject):
    print(f"  {a} vs {b}: meandiff={diff:+.4f}  Holm-p={p_adj:.4e}  reject={rej}")


# =====================================================================
# 2. 重複リターン問題: 自己相関の確認と間引きサンプリング
#
# 6ヶ月先リターンを日次でスライドさせているため隣接観測が重複する。ANOVA型モデルの残差で
# Durbin-Watson統計量を確認し、間引き間隔を広げたときに結果がどう変わるかを、
# eta²（効果量）・群別Nと合わせて確認する。
# =====================================================================
section("2. 重複リターン問題: 自己相関の確認と間引きサンプリング")

dummy_model = smf.ols("Return_6M ~ C(VIX_Phase)", data=df).fit()
dw = durbin_watson(dummy_model.resid)
print(f"Durbin-Watson (ANOVA型モデル残差): {dw:.4f}  (2に近いほど自己相関なし。0に近いほど強い正の自己相関)")


def eta_squared(f_stat, groups):
    k = len(groups)
    n = sum(len(g) for g in groups)
    df_b, df_w = k - 1, n - k
    return (f_stat * df_b) / (f_stat * df_b + df_w)


thinning = {"Daily": 1, "Monthly_21d": 21, "Quarterly_63d": 63, "Semiannual_126d": 126}
rows = []
for name, step in thinning.items():
    sub = df.iloc[::step]
    sub_groups = [sub.loc[sub["VIX_Phase"] == lab, "Return_6M"].values for lab in VIX_PHASE_LABELS]
    counts = {lab: len(g) for lab, g in zip(VIX_PHASE_LABELS, sub_groups)}
    min_n = min(len(g) for g in sub_groups)
    f_s, p_s = stats.f_oneway(*sub_groups)
    h_s, p_kw_s = stats.kruskal(*sub_groups)
    eta2 = eta_squared(f_s, sub_groups)
    if min_n < 2:
        welch_p = np.nan  # 群内分散が推定不能
    else:
        welch_p = anova_oneway(sub_groups, use_var="unequal", welch_correction=True).pvalue
    rows.append({"sampling": name, "N": len(sub), **counts, "ANOVA_p": p_s, "Welch_p": welch_p,
                 "Kruskal_p": p_kw_s, "eta_sq": eta2, "min_group_n": min_n})

thin_df = pd.DataFrame(rows)
print(thin_df.to_string(index=False))

# 注意: Semiannual_126d はFear群・Panic群がそれぞれn=1となり、Welch ANOVAは群内分散を推定できない（NaN）。
# また Quarterly_63d はKruskal-Wallis検定のみ5%水準で有意（他は非有意）であり、検定間で結論が割れる。
# 標本サイズが小さくなるほど「非有意」は検出力不足を反映している可能性が高い（次節参照）。

# --- 2.1 検出力(power)分析 ---
# 間引き後の標本サイズで、観測された効果量をどの程度の確率で検出できたかを確認する。
print("\n--- 2.1 検出力(power)分析 ---")
power_solver = FTestAnovaPower()
k = 4
# 効果量・Nは直前で計算した thin_df から取得する（手作業転記をしない）
scenarios = [(r["sampling"], int(r["N"]), float(r["eta_sq"])) for _, r in thin_df.iterrows()]
for name, n_total, eta2 in scenarios:
    f_effect = np.sqrt(eta2 / (1 - eta2))
    power = power_solver.solve_power(effect_size=f_effect, nobs=n_total, alpha=0.05, k_groups=k, power=None)
    req_n = power_solver.solve_power(effect_size=f_effect, nobs=None, alpha=0.05, k_groups=k, power=0.8)
    print(f"{name}: N={n_total}  eta^2={eta2:.4f}  検出力={power:.3f}  80%検出力に必要なN={req_n:.0f}")


# =====================================================================
# 3. 重回帰分析とHAC(Newey-West)補正
#
# VIX_Z（63日ローリングZスコア）と米10年債利回りを説明変数とするOLSモデルを推定し、
# Newey-West型HAC標準誤差で自己相関・不均一分散に頑健な有意性を再評価する。
# =====================================================================
section("3. 重回帰分析とHAC(Newey-West)補正")

df_reg = analysis_frame(df_raw, ["Return_6M", "VIX_Z", "Treasury_10Y"])
X2 = sm.add_constant(df_reg[["VIX_Z", "Treasury_10Y"]])
Y = df_reg["Return_6M"]

model_ols = sm.OLS(Y, X2).fit()
model_hac = sm.OLS(Y, X2).fit(cov_type="HAC", cov_kwds={"maxlags": 126})

print("=== OLS（自己相関未補正） ===")
print(model_ols.summary().tables[1])
print(f"Adj R^2 = {model_ols.rsquared_adj:.4f}")

print("\n=== HAC補正後 (maxlags=126) ===")
print(model_hac.summary().tables[1])

# --- 3.1 HACラグ幅の感度分析 ---
# Newey-Westのラグ幅は分析者が選ぶパラメータであるため、複数のラグ幅で結論が変わらないかを確認する。
print("\n--- 3.1 HACラグ幅の感度分析 ---")
for lag in [21, 63, 126, 189, 252, 315]:
    m = sm.OLS(Y, X2).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    p = m.pvalues["VIX_Z"]
    print(f"maxlags={lag:>4}: VIX_Z p={p:.4f}  (5%で有意={'○' if p < 0.05 else '×'})")

# --- 3.2 金利変数の定常性 ---
# 米10年債利回りを水準のまま使うと、非定常系列同士の見せかけの回帰になり得る。ADF・KPSS検定で確認し、
# 定常な変換（1日差分）に置き換えたモデルと比較する。
print("\n--- 3.2 金利変数の定常性 ---")
for name, series in [("Treasury_10Y (水準)", df_raw["Treasury_10Y"]), ("Treasury_10Y 1日差分", df_raw["Treasury_10Y_Diff1"])]:
    s = series.dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    print(f"{name}: ADF p={adf_p:.4f}  KPSS p={kpss_p:.4f}")

df_reg_diff = analysis_frame(df_raw, ["Return_6M", "VIX_Z", "Treasury_10Y_Diff1"])
X_diff = sm.add_constant(df_reg_diff[["VIX_Z", "Treasury_10Y_Diff1"]])
model_diff = sm.OLS(df_reg_diff["Return_6M"], X_diff).fit(cov_type="HAC", cov_kwds={"maxlags": 126})
print("\n金利を1日差分（定常）に置き換えたHACモデル:")
print(model_diff.summary().tables[1])


# =====================================================================
# 4. Leave-one-crisis-out: Panic群は少数イベントの平均に過ぎないのか
#
# VIX>40（Panic）の日を、63営業日以上の間隔が空いたら別エピソードとみなしてクラスタリングし、
# 2008年GFC・2020年COVIDを除外してもPanic群の効果が残るかを確認する。
# =====================================================================
section("4. Leave-one-crisis-out")

panic_dates = df.index[df["VIX_Phase"] == "4_Panic (>40)"]
pos = {d: i for i, d in enumerate(df.index)}
gap = 63
episodes = []
start = prev = panic_dates[0]
for d in panic_dates[1:]:
    if pos[d] - pos[prev] > gap:
        episodes.append((start, prev))
        start = d
    prev = d
episodes.append((start, prev))

print(f"独立エピソード数: {len(episodes)}")
for i, (s, e) in enumerate(episodes, 1):
    print(f"  {i}: {s.date()} 〜 {e.date()}")


def label_ep(s):
    if 2007 <= s.year <= 2009:
        return "2008_GFC"
    if s.year == 2020:
        return "2020_COVID"
    return f"Other_{s.year}"


labels = [label_ep(s) for s, e in episodes]


def panic_vs_normal(sub_df, label):
    normal = sub_df.loc[sub_df["VIX_Phase"] == "1_Normal (<20)", "Return_6M"].values
    panic = sub_df.loc[sub_df["VIX_Phase"] == "4_Panic (>40)", "Return_6M"].values
    t, p = stats.ttest_ind(panic, normal, equal_var=False)
    print(f"[{label}] n_panic={len(panic)}  平均差={panic.mean()-normal.mean():+.4f}  Welch-t p={p:.4e}")


print()
panic_vs_normal(df, "Baseline(全期間)")

mask_2008 = pd.Series(True, index=df.index)
for s, e in [ep for ep, lab in zip(episodes, labels) if lab == "2008_GFC"]:
    mask_2008 &= ~((df.index >= s) & (df.index <= e))
panic_vs_normal(df[mask_2008], "2008 GFCを除外")

mask_2020 = pd.Series(True, index=df.index)
for s, e in [ep for ep, lab in zip(episodes, labels) if lab == "2020_COVID"]:
    mask_2020 &= ~((df.index >= s) & (df.index <= e))
panic_vs_normal(df[mask_2020], "2020 COVIDを除外")

mask_both = mask_2008 & mask_2020
panic_vs_normal(df[mask_both], "2008 GFCと2020 COVIDの両方を除外")

# 結果: 2008年・2020年の両方を除外し、残り7エピソード・n=44だけで検証しても、
# Panic群の効果は方向・統計的有意性ともに維持された。
# 監査時点の懸念（「Panic群は実質2イベントの平均に過ぎないのでは」）は、この検証によって覆された。


# =====================================================================
# 5. 経済的有意性（統計的有意性との区別）
#
# 回帰モデルの説明力（R²）とフェーズ別の要約統計を確認し、統計的有意性と経済的な予測力を切り分ける。
# =====================================================================
section("5. 経済的有意性（統計的有意性との区別）")

summary = df.groupby("VIX_Phase", observed=True)["Return_6M"].agg(["mean", "std", "min", "max", "count"])
print(summary)

X1 = sm.add_constant(df_reg_diff[["VIX_Z"]])
model1 = sm.OLS(df_reg_diff["Return_6M"], X1).fit()
print(f"\nModel1 (VIX_Zのみ) Adj R^2 = {model1.rsquared_adj:.4f}  ({model1.rsquared_adj*100:.2f}% の説明力)")
print(f"Model2 (VIX_Z + Treasury_10Y_Diff1) Adj R^2 = {sm.OLS(df_reg_diff['Return_6M'], X_diff).fit().rsquared_adj:.4f}")
print("\n^GSPCは価格指数であり配当を含まない価格リターンである点に注意。")


# =====================================================================
# 6. 時系列コンフォーマル予測: VIX将来値の予測区間
#
# 前半（1〜5章）で、VIX水準による将来リターンの説明力が限定的であり、日次重複による見かけの
# 有意性も確認された。そこで方向を当てる点予測ではなく、市場状態の不確実性を「区間」として
# 評価する問題へ視点を移す。
#
# 比較する4手法（点予測モデル・期間・情報集合・下限クリップ・評価指標を全て統一）:
#   A. ACI          : ローリング残差窓 + 適応alpha
#   B. RollingFixed : ローリング残差窓 + 固定alpha   ← Aとの差 = alpha適応の効果
#   C. FixedSplit   : 較正期間の分位点を固定（以後更新しない） ← Bとの差 = 窓更新の効果
#   D. Naive        : モデルを使わない無条件区間
#
# カバレッジだけでなく区間幅も同時に見るため Winkler(interval) score を併用し、
# 手法差にはブロックブートストラップで95%信頼区間を付ける。
# 詳細な数値は docs/key_results.md、実装は conformal_vix_forecast.py を参照。
# =====================================================================
section("6. 時系列コンフォーマル予測: VIX将来値の予測区間")

from conformal_vix_forecast import main as conformal_main  # noqa: E402

out_h1 = conformal_main(h=1, tag="vix_analysis_h1", make_plots=False, df_raw=df_raw)

fig, ax = plt.subplots(figsize=(12, 5))
sub = out_h1.loc["2008-08-01":"2009-01-01"]
ax.plot(sub.index, sub["VIX"], color="black", linewidth=1.3, label="Actual VIX")
ax.fill_between(sub.index, sub["aci_lower"], sub["aci_upper"],
                color="steelblue", alpha=0.25, label="A. ACI (rolling + adaptive alpha)")
ax.fill_between(sub.index, sub["rolling_fixed_lower"], sub["rolling_fixed_upper"],
                color="green", alpha=0.18, label="B. RollingFixed (rolling + fixed alpha)")
ax.plot(sub.index, sub["fixed_split_lower"], color="orange", linestyle="--",
        alpha=0.8, label="C. FixedSplit (frozen)")
ax.plot(sub.index, sub["fixed_split_upper"], color="orange", linestyle="--", alpha=0.8)
ax.plot(sub.index, sub["naive_lower"], color="gray", linestyle=":", alpha=0.7, label="D. Naive")
ax.plot(sub.index, sub["naive_upper"], color="gray", linestyle=":", alpha=0.7)
ax.axvline(pd.Timestamp("2008-09-29"), color="red", linestyle="--", alpha=0.6, label="Panic onset")
ax.set_ylabel("VIX")
ax.set_title("2008 GFC: 4 interval methods for 1-day-ahead VIX forecast (GBM point forecast)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "vix_analysis_2008_zoom.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("図を figures/vix_analysis_2008_zoom.png に保存しました")


# =====================================================================
# まとめ
# 詳しい解釈・限界・再現手順は README.md を参照。本スクリプトの全ステップは data_pipeline.build_dataset()
# という単一のデータソースのみに依存しており、上から順に実行すれば結果は再現できる。
# =====================================================================
section("まとめ")
print("全ステップ完了。詳細な解釈・限界・再現手順は README.md を参照。")
