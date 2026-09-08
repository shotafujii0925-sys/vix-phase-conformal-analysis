# -*- coding: utf-8 -*-
"""
A-7: 米国10年債利回りの水準が非定常かどうかをADF/KPSSで確認し、
     定常な変換（1日差分・63日変化）に置き換えたモデルを再推定する。
B-5: 経済的有意性（R^2、価格リターンである点、下方リスク）を定量化する。
"""
import json
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

from data_pipeline import DATA_DIR, build_dataset, analysis_frame

RESULTS = {}
df_raw = build_dataset()

print("=" * 70)
print("A-7: Treasury_10Y の定常性検定（水準 vs 定常化変換）")
print("=" * 70)


def run_stationarity(series, name):
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    print(f"{name}: N={len(s)}")
    print(f"   ADF   : stat={adf_stat:.4f}  p={adf_p:.4f}  ({'定常(帰無棄却)' if adf_p < 0.05 else '非定常の可能性(帰無棄却できず)'})")
    print(f"   KPSS  : stat={kpss_stat:.4f}  p={kpss_p:.4f}  ({'非定常の可能性(帰無棄却)' if kpss_p < 0.05 else '定常(帰無棄却できず)'})")
    return {"name": name, "N": len(s), "adf_stat": adf_stat, "adf_p": adf_p, "kpss_stat": kpss_stat, "kpss_p": kpss_p}


stationarity_results = []
stationarity_results.append(run_stationarity(df_raw["Treasury_10Y"], "Treasury_10Y (水準)"))
stationarity_results.append(run_stationarity(df_raw["Treasury_10Y_Diff1"], "Treasury_10Y 1日差分"))
RESULTS["stationarity"] = stationarity_results

print("\n" + "=" * 70)
print("A-7続き: 定常化した金利変数でモデルを再推定")
print("=" * 70)
df_reg = analysis_frame(df_raw, ["Return_6M", "VIX_Z", "Treasury_10Y_Diff1"])
print(f"N = {len(df_reg)}")

X_level = sm.add_constant(df_raw.dropna(subset=["Return_6M", "VIX_Z", "Treasury_10Y"])[["VIX_Z", "Treasury_10Y"]])
Y_level = df_raw.dropna(subset=["Return_6M", "VIX_Z", "Treasury_10Y"])["Return_6M"]
model_level = sm.OLS(Y_level, X_level).fit(cov_type="HAC", cov_kwds={"maxlags": 126})

X_diff = sm.add_constant(df_reg[["VIX_Z", "Treasury_10Y_Diff1"]])
Y_diff = df_reg["Return_6M"]
model_diff = sm.OLS(Y_diff, X_diff).fit(cov_type="HAC", cov_kwds={"maxlags": 126})

print("\n[水準] Treasury_10Y（非定常の可能性）を使ったHACモデル:")
print(f"   VIX_Z: coef={model_level.params['VIX_Z']:+.5f} p={model_level.pvalues['VIX_Z']:.4f}")
print(f"   Treasury_10Y: coef={model_level.params['Treasury_10Y']:+.5f} p={model_level.pvalues['Treasury_10Y']:.4f}")
print(f"   Adj R^2 = {model_level.rsquared_adj:.4f}")

print("\n[1日差分] Treasury_10Y_Diff1（定常化）を使ったHACモデル:")
print(f"   VIX_Z: coef={model_diff.params['VIX_Z']:+.5f} p={model_diff.pvalues['VIX_Z']:.4f}")
print(f"   Treasury_10Y_Diff1: coef={model_diff.params['Treasury_10Y_Diff1']:+.5f} p={model_diff.pvalues['Treasury_10Y_Diff1']:.4f}")
print(f"   Adj R^2 = {model_diff.rsquared_adj:.4f}")

RESULTS["model_level_hac"] = {
    "vix_z_coef": model_level.params["VIX_Z"], "vix_z_p": model_level.pvalues["VIX_Z"],
    "treasury_coef": model_level.params["Treasury_10Y"], "treasury_p": model_level.pvalues["Treasury_10Y"],
    "adj_r2": model_level.rsquared_adj, "N": int(model_level.nobs),
}
RESULTS["model_diff_hac"] = {
    "vix_z_coef": model_diff.params["VIX_Z"], "vix_z_p": model_diff.pvalues["VIX_Z"],
    "treasury_diff_coef": model_diff.params["Treasury_10Y_Diff1"], "treasury_diff_p": model_diff.pvalues["Treasury_10Y_Diff1"],
    "adj_r2": model_diff.rsquared_adj, "N": int(model_diff.nobs),
}

print("\n" + "=" * 70)
print("B-5: 経済的有意性の定量化")
print("=" * 70)

df_phase = analysis_frame(df_raw, ["VIX_Phase", "Return_6M"])
by_phase = df_phase.groupby("VIX_Phase", observed=True)["Return_6M"].agg(["mean", "std", "min", "max", "count"])
print("\nフェーズ別 Return_6M の要約統計（価格リターン、配当再投資は含まない点に注意）:")
print(by_phase)
RESULTS["phase_summary_stats"] = by_phase.to_dict(orient="index")

# VIX_Z単体モデルとVIX_Z+金利差分モデルのR^2比較（OLS, 参考値）
X1 = sm.add_constant(df_reg[["VIX_Z"]])
model1 = sm.OLS(Y_diff, X1).fit()
print(f"\nModel1 (VIX_Zのみ, OLS): Adj R^2 = {model1.rsquared_adj:.4f}  (=説明力 {model1.rsquared_adj*100:.2f}%)")
print(f"Model2 (VIX_Z + Treasury_10Y_Diff1, OLS): Adj R^2 = {sm.OLS(Y_diff, X_diff).fit().rsquared_adj:.4f}")
RESULTS["model1_ols_adj_r2"] = model1.rsquared_adj

# ANOVAのeta^2（既にanalysis_01で計算済みだが経済的有意性の文脈でも参照するため再掲）
RESULTS["note"] = "ANOVAのeta^2は analysis_01 の results_01_phase_tests.json を参照（数値の転記はしない）"

with open(DATA_DIR / "results_04_stationarity_economic.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
print("\n結果を data/results_04_stationarity_economic.json に保存しました")
