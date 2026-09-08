# -*- coding: utf-8 -*-
"""
B-4: HAC(Newey-West)補正のラグ幅を振って、VIX_Zの有意性がラグ選択に対してどれだけ頑健かを確認する。
A-5: 間引きANOVAで「有意差なし」となった標本サイズにおける検出力(power)・最小検出効果量(MDE)を確認する。
"""
import json
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.power import FTestAnovaPower

from data_pipeline import DATA_DIR, build_dataset, analysis_frame

RESULTS = {}

df_raw = build_dataset()
df_reg = analysis_frame(df_raw, ["Return_6M", "VIX_Z", "Treasury_10Y"])
print(f"回帰用N = {len(df_reg)}")

X2 = sm.add_constant(df_reg[["VIX_Z", "Treasury_10Y"]])
Y = df_reg["Return_6M"]

print("\n" + "=" * 70)
print("B-4: HAC補正 ラグ幅感度分析")
print("=" * 70)
lag_candidates = [21, 63, 126, 189, 252, 315]  # 126(ホライズンそのもの)を基準に前後に振る
hac_sensitivity = []
for lag in lag_candidates:
    model = sm.OLS(Y, X2).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    vix_p = model.pvalues["VIX_Z"]
    vix_coef = model.params["VIX_Z"]
    tnx_p = model.pvalues["Treasury_10Y"]
    print(f"maxlags={lag:>4}: VIX_Z coef={vix_coef:+.5f}  p={vix_p:.4f}  (5%で有意={'○' if vix_p < 0.05 else '×'})  "
          f"| Treasury_10Y p={tnx_p:.4f}")
    hac_sensitivity.append({"maxlags": lag, "vix_z_coef": vix_coef, "vix_z_p": vix_p,
                            "treasury_p": tnx_p, "vix_z_significant_5pct": bool(vix_p < 0.05)})
RESULTS["hac_lag_sensitivity"] = hac_sensitivity

n_sig = sum(1 for r in hac_sensitivity if r["vix_z_significant_5pct"])
print(f"\n{len(lag_candidates)}通りのラグ幅のうち、VIX_Zが5%水準で有意だったのは{n_sig}通り")
RESULTS["hac_lag_sensitivity_summary"] = f"{n_sig}/{len(lag_candidates)} lags significant at 5%"

print("\n" + "=" * 70)
print("A-5: 間引きANOVAの検出力(power)分析")
print("=" * 70)
# 効果量(eta^2)と標本数Nは analysis_01 の出力JSONから読み込む（手作業転記を廃止）。
# これにより、データが更新されたときに検出力分析だけ古い値で走ることを防ぐ。
power_solver = FTestAnovaPower()
k_groups = 4

upstream_path = DATA_DIR / "results_01_phase_tests.json"
if not upstream_path.exists():
    raise FileNotFoundError(
        f"{upstream_path} がありません。先に analysis_01_phase_tests.py を実行してください。"
    )
with open(upstream_path, encoding="utf-8") as f:
    upstream = json.load(f)

scenarios = [
    {"name": row["sampling"], "N": int(row["N"]), "observed_eta_sq": float(row["eta_sq"]),
     "min_group_n": int(row["min_group_n"])}
    for row in upstream["thinned_anova"]
]
print(f"（効果量・Nは {upstream_path.name} から読み込み）")

power_results = []
for sc in scenarios:
    eta2 = sc["observed_eta_sq"]
    n_total = sc["N"]
    f_effect = np.sqrt(eta2 / (1 - eta2))  # Cohen's f
    achieved_power = power_solver.solve_power(effect_size=f_effect, nobs=n_total,
                                              alpha=0.05, k_groups=k_groups, power=None)
    required_n_total = power_solver.solve_power(effect_size=f_effect, nobs=None,
                                                alpha=0.05, k_groups=k_groups, power=0.8)
    note = ""
    if sc["min_group_n"] < 2:
        note = "  ※最小群n<2のためeta^2自体が参考値"
    print(f"{sc['name']}: N={n_total}  観測eta^2={eta2:.4f} (Cohen's f={f_effect:.4f}){note}")
    print(f"   -> この効果量・この標本サイズでの検出力(power) = {achieved_power:.3f}")
    print(f"   -> 80%検出力に必要な合計N = {required_n_total:.0f} (群あたり約{required_n_total/k_groups:.0f})")
    power_results.append({
        "scenario": sc["name"], "N": n_total, "observed_eta_sq": eta2,
        "cohens_f": float(f_effect), "achieved_power": float(achieved_power),
        "required_n_for_80pct_power": float(required_n_total),
        "min_group_n": sc["min_group_n"],
    })
RESULTS["power_analysis"] = power_results
RESULTS["upstream_source"] = upstream_path.name

out_path = DATA_DIR / "results_03_hac_power.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
print(f"\n結果を data/{out_path.name} に保存しました")
