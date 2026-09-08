# -*- coding: utf-8 -*-
"""
各分析スクリプトが出力した results_*.json から、主要数値のMarkdown表を自動生成する。

出力: docs/key_results.md
READMEはこのファイルの数値を参照する（手作業転記による不整合を防ぐため）。
tests/test_repo.py::test_readme_key_numbers_match_results がREADMEとJSONの一致を検査する。

    python build_key_results.py
"""
from __future__ import annotations

import json

from data_pipeline import DATA_DIR, DOCS_DIR


def load(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません。先に該当の分析スクリプトを実行してください。")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_p(p: float) -> str:
    return f"{p:.4e}" if p < 1e-3 else f"{p:.4f}"


def main() -> None:
    r01 = load("results_01_phase_tests.json")
    r02 = load("results_02_leave_one_crisis_out.json")
    r03 = load("results_03_hac_power.json")
    r04 = load("results_04_stationarity_economic.json")
    r05 = load("results_05_conformal_vix.json")

    L: list[str] = []
    L.append("# 主要結果（自動生成）\n")
    L.append("このファイルは `build_key_results.py` が `data/results_*.json` から生成しています。")
    L.append("手で編集しないでください。READMEの数値はここと一致している必要があります")
    L.append("（`tests/test_repo.py::test_readme_key_numbers_match_results` が検査）。\n")

    # --- 1. VIXフェーズ間検定 ---
    a = r01["anova_daily"]
    L.append("## 1. VIXフェーズ間の平均リターン差（日次、N=%d）\n" % a["N"])
    L.append("| Test | Statistic | p-value |")
    L.append("|---|---:|---:|")
    L.append(f"| Standard one-way ANOVA | {a['standard']['stat']:.4f} | {fmt_p(a['standard']['p'])} |")
    L.append(f"| Welch ANOVA | {a['welch']['stat']:.4f} | {fmt_p(a['welch']['p'])} |")
    L.append(f"| Kruskal-Wallis | {a['kruskal']['stat']:.4f} | {fmt_p(a['kruskal']['p'])} |")
    L.append("")
    v = r01["variance_tests"]
    L.append("### 等分散性\n")
    L.append("| Test | Statistic | p-value |")
    L.append("|---|---:|---:|")
    L.append(f"| Levene (center=mean) | {v['levene_mean']['stat']:.4f} | {fmt_p(v['levene_mean']['p'])} |")
    L.append(f"| Levene (center=median) | {v['levene_median']['stat']:.4f} | {fmt_p(v['levene_median']['p'])} |")
    L.append(f"| Bartlett | {v['bartlett']['stat']:.4f} | {fmt_p(v['bartlett']['p'])} |")
    L.append("")
    L.append("### Welch型ペアワイズ（Holm補正）\n")
    L.append("| Comparison | Mean Diff | Holm-adjusted p |")
    L.append("|---|---:|---:|")
    for row in r01["welch_pairwise_holm"]:
        g1 = row["group1"].split("_")[1].split(" ")[0]
        g2 = row["group2"].split("_")[1].split(" ")[0]
        L.append(f"| {g1} vs {g2} | {row['meandiff']:+.4f} | {fmt_p(row['holm_p'])} |")
    L.append("")

    # --- 2. 重複と間引き ---
    L.append(f"## 2. 重複リターンと間引き（ANOVA型残差 Durbin-Watson = "
             f"{r01['durbin_watson_anova_model']:.4f}）\n")
    L.append("| Sampling | N | 最小群n | ANOVA p | Welch p | Kruskal p | eta² | 検出力 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    power_by_name = {p["scenario"]: p for p in r03["power_analysis"]}
    for row in r01["thinned_anova"]:
        welch = "計算不可" if row.get("welch") is None else fmt_p(row["welch"]["p"])
        pw = power_by_name.get(row["sampling"], {}).get("achieved_power")
        pw_s = f"{pw:.3f}" if pw is not None else "-"
        L.append(f"| {row['sampling']} | {row['N']} | {row['min_group_n']} | "
                 f"{fmt_p(row['anova']['p'])} | {welch} | {fmt_p(row['kruskal']['p'])} | "
                 f"{row['eta_sq']:.4f} | {pw_s} |")
    L.append("")

    # --- 3. HACラグ感度 ---
    L.append("## 3. HAC(Newey-West)ラグ幅の感度\n")
    L.append("| maxlags | VIX_Z coef | VIX_Z p | 5%で有意 |")
    L.append("|---:|---:|---:|:---:|")
    for row in r03["hac_lag_sensitivity"]:
        sig = "○" if row["vix_z_significant_5pct"] else "×"
        L.append(f"| {row['maxlags']} | {row['vix_z_coef']:+.5f} | {row['vix_z_p']:.4f} | {sig} |")
    L.append("")

    # --- 4. 定常性・経済的有意性 ---
    L.append("## 4. 金利の定常性と経済的有意性\n")
    L.append("| 系列 | ADF p | KPSS p |")
    L.append("|---|---:|---:|")
    for row in r04["stationarity"]:
        L.append(f"| {row['name']} | {row['adf_p']:.4f} | {row['kpss_p']:.4f} |")
    L.append("")
    ml, md = r04["model_level_hac"], r04["model_diff_hac"]
    L.append("| モデル(HAC, maxlags=126) | VIX_Z coef | VIX_Z p | 金利 coef | 金利 p | Adj R² |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(f"| 金利=水準 | {ml['vix_z_coef']:+.5f} | {ml['vix_z_p']:.4f} | "
             f"{ml['treasury_coef']:+.5f} | {ml['treasury_p']:.4f} | {ml['adj_r2']:.4f} |")
    L.append(f"| 金利=1日差分 | {md['vix_z_coef']:+.5f} | {md['vix_z_p']:.4f} | "
             f"{md['treasury_diff_coef']:+.5f} | {md['treasury_diff_p']:.4f} | {md['adj_r2']:.4f} |")
    L.append("")
    L.append(f"- VIX_Z単体モデルの調整済みR² = **{r04['model1_ols_adj_r2']:.4f}** "
             f"（説明力 {r04['model1_ols_adj_r2']*100:.2f}%）")
    L.append("")
    L.append("### フェーズ別 6ヶ月先リターン（価格リターン、配当を含まない）\n")
    L.append("| VIX_Phase | 平均 | 標準偏差 | 最小 | 最大 | N |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for phase, s in r04["phase_summary_stats"].items():
        L.append(f"| {phase} | {s['mean']*100:+.2f}% | {s['std']*100:.2f}% | "
                 f"{s['min']*100:+.1f}% | {s['max']*100:+.1f}% | {int(s['count'])} |")
    L.append("")

    # --- 5. LOCO ---
    L.append(f"## 5. Leave-one-crisis-out（独立Panicエピソード = {r02['n_panic_episodes']}個 / "
             f"Panic日数 {r02['n_panic_days']}日）\n")
    L.append("**注**: 以下のWelch-t p値は日次重複標本に基づくため過大評価されている。"
             "有意性の根拠としては用いず、効果の符号と大きさの安定性のみを読む。\n")
    L.append("| 除外対象 | 残りPanic N | 平均差 | Welch-t p（過大評価） |")
    L.append("|---|---:|---:|---:|")
    for row in r02["leave_one_crisis_out"]:
        if row.get("insufficient"):
            continue
        pv = row["panic_vs_normal"]
        L.append(f"| {row['label']} | {pv['n_panic']} | {pv['mean_diff']:+.4f} | {fmt_p(pv['welch_t_p'])} |")
    L.append("")
    ep = r02["episode_level_inference"]
    L.append("### エピソード単位の推論（1エピソード = 1観測、重複を回避）\n")
    L.append(f"- 比較基準（Normal群平均）: {ep['normal_group_mean']:+.4f}")
    L.append(f"- {ep['n_episodes']}エピソード中 **{ep['n_exceeding_normal']}個** がNormal平均を上回った")
    L.append(f"- 符号検定（両側二項検定）: **p = {ep['sign_test_two_sided_p']:.4f}**")
    L.append(f"- エピソード平均の中央値 = {ep['episode_mean_median']:+.4f}、"
             f"範囲 = [{ep['episode_mean_min']:+.4f}, {ep['episode_mean_max']:+.4f}]")
    L.append(f"- {ep['caveat']}")
    L.append("")

    # --- 6. コンフォーマル予測 ---
    for tag, title in [("h1_primary", "h=1（1営業日先）"), ("h21_robustness", "h=21（21営業日先）")]:
        block = r05[tag]
        per = block["periods"]
        L.append(f"## 6. 時系列コンフォーマル予測 — {title}\n")
        L.append(f"- Train: {per['train'][0]}〜{per['train'][1]} (N={per['n_train']})")
        L.append(f"- Calibration: {per['calibration'][0]}〜{per['calibration'][1]} (N={per['n_calibration']})")
        L.append(f"- Test: {per['test'][0]}〜{per['test'][1]} (N={per['n_test']})")
        L.append("")
        L.append("### 点予測モデル（Test期間MAE, VIXポイント）\n")
        L.append("| Persistence | Linear | GBM |")
        L.append("|---:|---:|---:|")
        m = block["model_mae"]
        L.append(f"| {m['persistence']:.3f} | {m['linear']:.3f} | {m['gbm']:.3f} |")
        L.append("")
        L.append("### 4手法 × 3モデル（目標カバレッジ90%）\n")
        L.append("| モデル | 手法 | カバレッジ | 平均幅 | Winkler |")
        L.append("|---|---|---:|---:|---:|")
        for model_name, methods in block["method_comparison"].items():
            for meth, r in methods.items():
                L.append(f"| {model_name} | {meth} | {r['coverage']*100:.2f}% | "
                         f"{r['mean_width']:.3f} | {r['winkler_score']:.3f} |")
        L.append("")
        if "bootstrap" in block:
            L.append("### 手法差の不確実性（moving block bootstrap）\n")
            L.append("| モデル | 比較 | Winkler差 | 95%CI | 判定 | カバレッジ差 | 95%CI | 判定 |")
            L.append("|---|---|---:|---|---|---:|---|---|")
            for model_name, pairs in block["bootstrap"].items():
                for pair, d in pairs.items():
                    w, c = d["winkler_diff"], d["coverage_diff"]
                    wv = "差を確認" if w["significant"] else "**明確な差なし**"
                    cv = "差を確認" if c["significant"] else "**明確な差なし**"
                    L.append(
                        f"| {model_name} | {pair.replace('_vs_', ' − ')} | {w['point_diff']:+.3f} | "
                        f"[{w['ci95_low']:+.3f}, {w['ci95_high']:+.3f}] | {wv} | "
                        f"{c['point_diff']*100:+.2f}pt | "
                        f"[{c['ci95_low']*100:+.2f}, {c['ci95_high']*100:+.2f}] | {cv} |")
            b0 = next(iter(block["bootstrap"].values()))
            meta = next(iter(b0.values()))["winkler_diff"]
            L.append("")
            L.append(f"ブートストラップ設定: B={meta['n_boot']}回、ブロック長={meta['block_len']}営業日、"
                     f"評価標本数={meta['n_obs']}")
            L.append("")
        L.append("### 相場局面別カバレッジ（GBMベース）\n")
        L.append("| VIXフェーズ | N | 手法 | カバレッジ | 平均幅 | Winkler |")
        L.append("|---|---:|---|---:|---:|---:|")
        for phase, methods in block["phase_coverage"].items():
            for meth, r in methods.items():
                L.append(f"| {phase} | {r['n']} | {meth} | "
                         f"{r['coverage']*100:.1f}% ({r['n_covered']}/{r['n']}) | "
                         f"{r['mean_width']:.3f} | {r['winkler_score']:.3f} |")
        L.append("")
        if "crisis_pre_post" in block:
            cp = block["crisis_pre_post"]
            L.append("### Panic突入前後21営業日（GBMベース、予測日基準）\n")
            L.append("| エピソード開始 | " + " | ".join(
                f"{m} pre/post" for m in ["aci", "rolling_fixed", "fixed_split", "naive"]) + " |")
            L.append("|---|" + "---|" * 4)
            for row in cp["episodes"]:
                cells = []
                for meth in ["aci", "rolling_fixed", "fixed_split", "naive"]:
                    pre, post = row[f"{meth}_pre"], row[f"{meth}_post"]
                    cells.append(f"{pre['n_covered']}/{pre['n']} → {post['n_covered']}/{post['n']}")
                L.append(f"| {row['episode_start']} | " + " | ".join(cells) + " |")
            avg = cp["average"]
            L.append("| **平均** | " + " | ".join(
                f"{avg[m]['pre_coverage']*100:.1f}% → {avg[m]['post_coverage']*100:.1f}%"
                for m in ["aci", "rolling_fixed", "fixed_split", "naive"]) + " |")
            L.append("")
            L.append(f"エピソード数 = {avg['aci']['n_episodes']}、各窓のN=21（1エピソードあたりの標本が小さく、"
                     "個別の値には大きな不確実性がある）")
            L.append("")
        if "crisis_pre_post_target_aligned" in block:
            av = block["crisis_pre_post_target_aligned"]["average"]
            L.append("### Panic突入前後（対象日=target date基準で揃えた再集計）\n")
            L.append("| 手法 | pre カバレッジ | post カバレッジ |")
            L.append("|---|---:|---:|")
            for meth, r in av.items():
                L.append(f"| {meth} | {r['pre_coverage']*100:.1f}% | {r['post_coverage']*100:.1f}% |")
            L.append("")

    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "key_results.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"主要結果を {out.relative_to(DOCS_DIR.parent)} に生成しました（{len(L)}行）")


if __name__ == "__main__":
    main()
