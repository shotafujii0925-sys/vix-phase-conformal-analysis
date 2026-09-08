# -*- coding: utf-8 -*-
"""
A-8: 可視化の修正
- 中央値だけでなく平均+95%CIを重ねる（検定は平均の議論のため）
- 群別Nをラベルに明示
- 「日次（重複あり）」と「21営業日間引き」を並べ、重複リターンの影響を図でも示す
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import FIGURES_DIR, build_dataset, analysis_frame, VIX_PHASE_LABELS

sns.set_theme(style="whitegrid")

df_raw = build_dataset()
df = analysis_frame(df_raw, ["VIX_Phase", "Return_6M"])

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

for ax, (title, sub) in zip(
    axes,
    [("Daily sampling (overlapping 6M windows, pseudo-replicated)", df),
     ("21-trading-day thinned sampling (reduces overlap)", df.iloc[::21])],
):
    counts = sub["VIX_Phase"].value_counts().reindex(VIX_PHASE_LABELS)
    labels_with_n = [f"{lab}\n(n={counts[lab]})" for lab in VIX_PHASE_LABELS]

    sns.boxplot(x="VIX_Phase", y="Return_6M", data=sub, order=VIX_PHASE_LABELS,
                palette="Blues", ax=ax, showmeans=False, fliersize=2, hue="VIX_Phase", legend=False)

    # 平均 + 95%CI（正規近似）を重ねる
    means = sub.groupby("VIX_Phase", observed=True)["Return_6M"].mean().reindex(VIX_PHASE_LABELS)
    sems = sub.groupby("VIX_Phase", observed=True)["Return_6M"].sem().reindex(VIX_PHASE_LABELS)
    ci95 = 1.96 * sems
    x_pos = np.arange(len(VIX_PHASE_LABELS))
    ax.errorbar(x_pos, means.values, yerr=ci95.values, fmt="D", color="darkred",
                markersize=7, capsize=5, linewidth=1.8,
                label="Mean ± 95% CI", zorder=5)

    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_with_n, fontsize=9)
    ax.set_xlabel("VIX Phase")
    ax.set_ylabel("6-Month Forward Return (price return)")
    ax.legend(loc="upper left", fontsize=9)

fig.suptitle(
    "S&P 500 6-Month Future Returns by VIX Phase (1990-2023)\n"
    "Box = median/IQR; diamond = mean ± 95% CI (matches the ANOVA/t-test claims)",
    fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase_returns_daily_vs_thinned.png", dpi=150, bbox_inches="tight")
print("図を figures/phase_returns_daily_vs_thinned.png に保存しました")

# 数値も確認用に出力
print("\n日次サンプルの平均・95%CI:")
for lab in VIX_PHASE_LABELS:
    sub_lab = df.loc[df["VIX_Phase"] == lab, "Return_6M"]
    m = sub_lab.mean()
    ci = 1.96 * sub_lab.sem()
    print(f"  {lab}: mean={m:+.4f}  95%CI=[{m-ci:+.4f}, {m+ci:+.4f}]  n={len(sub_lab)}")
