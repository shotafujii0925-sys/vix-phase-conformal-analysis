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


def render_readme(r01, r02, r03, r04, r05):
    """Generate headline numbers and verdicts from the same JSON as the tables."""
    ep = r02["episode_level_inference"]
    h1 = r05["h1_primary"]
    mae = h1["model_mae"]
    methods = h1["method_comparison"]["gbm"]
    lines = [
        "# VIXは何を予測できて、何を予測できないのか", "",
        "S&P500将来リターンの識別力の検証と、VIX将来値の時系列コンフォーマル予測。", "",
        "VIX水準と将来リターンの関係を、重複観測・小標本・危機依存の観点から検証する研究です。",
        "後半ではVIX自体のh営業日先水準を予測し、区間幅と実際の被覆率を比較します。", "",
        "このREADMEの数値・判定は `build_key_results.py` が結果JSONから生成します。",
        "コード変更後は同梱スナップショットで再計算し、README・全数値表・図を同時に更新します。", "",
        "## 使用データ", "",
        "S&P500(^GSPC)、VIX(^VIX)、米10年国債利回り(^TNX)の日次終値。",
        "1990-01-02〜2023-12-29、8,565営業日。Yahoo Finance (yfinance 1.5.1)、2026-09-08取得。",
        "生データCSVを同梱し、既定ではネットワーク接続なしで再現します。",
        "S&P500の目的変数は126営業日先の価格リターン（配当を含まない、暦日6ヶ月ではない）。", "",
        "## 主な結果と解釈", "",
        f"- 日次フェーズANOVAは F={r01['anova_daily']['standard']['stat']:.4f}、"
        f"p={fmt_p(r01['anova_daily']['standard']['p'])}。ただし独立観測の前提が成立しません。",
        f"- ANOVA残差のDurbin-Watsonは {r01['durbin_watson_anova_model']:.4f}。"
        "日次の126営業日リターンは大幅に重複します。63日間引きでも重複は残り、"
        "そのKruskal-Wallis p=0.0450は探索的参考値です。",
        "- VIX_Zと金利水準の回帰では、HACラグ幅6通りすべてでVIX_Zは5%水準で非有意。"
        "水準のフェーズ比較とZスコアの線形回帰は別モデルであり、同じ仮説の再検定ではありません。",
        f"- 将来リターン期間が重なるPanicエピソードを統合すると {ep['n_episodes']}個。"
        f"{ep['n_exceeding_normal']}個でNormal平均を上回りました。独立性と成功確率0.5を仮定した"
        f"両側二項検定は p={ep['sign_test_two_sided_p']:.4f}。仮定は未検証で、確証的証拠とは扱いません。",
        f"- VIX_Z単体OLSの調整済みR²は {r04['model1_ols_adj_r2']*100:.2f}%。"
        f"フェーズANOVAのeta²は {r01['thinned_anova'][0]['eta_sq']*100:.2f}%で、別の量です。"
        "全標本内の説明力から売買収益やアウトオブサンプル予測力は判断できません。",
        f"- h=1点予測MAE: persistence {mae['persistence']:.3f} / linear {mae['linear']:.3f} / "
        f"GBM {mae['gbm']:.3f}。固定した設定ではGBMが単純モデルを下回ります。", "",
        f"## h=1予測区間の結果（GBM、N={h1['periods']['n_test']}）", "",
        "| 手法 | カバレッジ（目標90%） | 平均幅 | Winkler score |",
        "|---|---:|---:|---:|",
    ]
    for name, row in methods.items():
        lines.append(f"| {name} | {row['coverage']*100:.2f}% | {row['mean_width']:.3f} | {row['winkler_score']:.3f} |")
    lines += ["", "Winkler scoreは区間幅と外れ幅を同時に評価し、小さい方が良い指標です。",
              "カバレッジ単独の順位を予測区間の優劣として扱いません。", "",
              "## 手法差の不確実性", "",
              "2ホライズン × 2モデル × 4比較 × 2指標 = 32比較を1群としてBonferroni補正。",
              "moving block bootstrap 10,000回、ブロック長はh=1で21件、h=21で42件。",
              "各補正区間の信頼水準は99.84375%。時系列依存と非定常性に対する近似で、厳密な保証ではありません。", "",
              "| h | モデル | Winkler比較 | 平均差 | 補正CI | 判定 |", "|---:|---|---|---:|---|---|"]
    for tag, h in [("h1_primary", 1), ("h21_robustness", 21)]:
        for model, pairs in r05[tag]["bootstrap"].items():
            for pair, result in pairs.items():
                w = result["winkler_diff"]
                verdict = "差を確認（近似）" if w["significant"] else "明確な差なし"
                lines.append(f"| {h} | {model} | {pair} | {w['point_diff']:+.3f} | "
                             f"[{w['ci_family_low']:+.3f}, {w['ci_family_high']:+.3f}] | {verdict} |")
    lines += ["", "## 危機前後の記述（h=1、GBM）", "",
              "Panic突入を63営業日の空白で分ける記述用集計です。これらを独立標本とはみなしません。",
              "予測日基準の各21件の被覆率と、正確な対象日基準の45暦日窓を全数値表に併記します。", "",
              "| 手法 | 突入前（予測日基準） | 突入後 |", "|---|---:|---:|"]
    for name, row in h1["crisis_pre_post"]["average"].items():
        lines.append(f"| {name} | {row['pre_coverage']*100:.1f}% | {row['post_coverage']*100:.1f}% |")
    mde = r03["power_analysis"][-1]["mde_cohens_f_at_80pct"]
    lines += ["", "## 実装と限界", "",
              "- 特徴量とターゲットは市場営業日の元インデックスで作成。Treasury欠損除去前の"
              "`target_date`を保持し、再学習・残差の解放・較正・危機前後の集計に使用します。",
              "- 当日の終値を観測した後に予測を発行します。当日に確定した過去予測の残差を先に反映。"
              "較正末尾の未確定残差も対象日到来時に一度だけ履歴へ追加します。",
              "- FixedSplitとnaiveはceil((n+1)(1-alpha))番目の残差を使います。"
              "ローリング手法も同じ順序統計量ですが、順位が標本数を超える場合は最大残差で有限に打ち切ります。",
              "- ACIはalphaを[0.001,0.5]に制限し、直近500件の残差窓を使用。"
              "最大残差での打ち切り、更新モデル、時系列の非交換可能性を含むため、90%被覆の理論保証は主張しません。",
              f"- 観測効果量による事後検出力は使いません。Semiannual N=67の80% power MDEは "
              f"Cohen's f={mde:.4f}。独立・等分散ANOVAの計画上の近似で、実データの見逃し率ではありません。",
              "- 間引きの開始位置、VIX閾値、h=1/21、再学習252件、ACI gamma=0.01、"
              "ブロック長は固定した探索的設定。感度分析・別市場での一般化・売買バックテストは未実施。",
              "- 図のフェーズ平均CIはIID近似で、重複を補正していない参考表示です。"
              "コンフォーマル区間の図は対象日を横軸にして実際のターゲットVIXと比較します。", "",
              "## 再現方法", "", "Python 3.12、`requirements.txt`の固定バージョンで検証。", "", "```bash",
              "pip install -r requirements.txt",
              "python vix_analysis.py             # 全分析・h=1/21・結果JSON・全図・README・数値表",
              "python conformal_vix_forecast.py   # コンフォーマル部分だけ再計算",
              "python build_key_results.py        # 結果JSONからREADMEと数値表を再生成",
              "python -m pytest tests/ -v",
              "python -m flake8", "```", "",
              "`data_pipeline.py --source online --refresh-snapshot`で元の1990〜2023年データを再取得できます。",
              "対象期間の終端は固定されており、この指定で最新日まで拡張するわけではありません。", "",
              "## ドキュメントと図", "",
              "- [全数値表](docs/key_results.md)", "- [修正後の手法説明](docs/current_methods.md)",
              "- [以前の詳細レポート（歴史的記録）](docs/detailed_report.md)", "",
              "![フェーズ別リターン](figures/phase_returns_daily_vs_thinned.png)", "",
              "![危機前後の区間](figures/conformal_crisis_zoom.png)", "",
              "![カバレッジと区間幅](figures/conformal_efficiency.png)", "",
              "## 研究の位置づけとAI利用", "",
              "VIX水準を説明変数としており、実現分散の予測を差し引く分散リスクプレミアムを構成していません。",
              "Bollerslev, Tauchen & Zhou (2009), Expected Stock Returns and Variance Risk Premiaなどの"
              "既存研究に対する新規性や同じ予測力を主張するものではありません。",
              "重複リターンの扱いにはHodrick (1992)などの先行研究があり、本研究ではその比較は未実施です。", "",
              "初期分析と研究の方向づけは本人が行い、コード監査・再構築・テスト・文書化にClaude Codeを使用しました。",
              "その後、Codexでコードと文書の整合性、日付処理、統計推論を再確認し、修正後の結果を再計算しました。", ""]
    return "\n".join(lines)


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
    L.append("| Sampling | N | 最小群n | ANOVA p | Welch p | Kruskal p | eta² | 80% power MDE (f) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    power_by_name = {p["scenario"]: p for p in r03["power_analysis"]}
    for row in r01["thinned_anova"]:
        welch = "計算不可" if row.get("welch") is None else fmt_p(row["welch"]["p"])
        pw = power_by_name.get(row["sampling"], {}).get("mde_cohens_f_at_80pct")
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
    L.append(f"## 5. Leave-one-crisis-out（リターン期間非重複Panicエピソード = {r02['n_panic_episodes']}個 / "
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
    L.append("### エピソード単位の記述と条件付き参考検定\n")
    L.append(f"- 比較基準（Normal群平均）: {ep['normal_group_mean']:+.4f}")
    L.append(f"- {ep['n_episodes']}エピソード中 **{ep['n_exceeding_normal']}個** がNormal平均を上回った")
    L.append(f"- 独立性・成功確率0.5を仮定した両側二項検定の参考値: **p = {ep['sign_test_two_sided_p']:.4f}**")
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
            L.append("32比較全体にBonferroni補正した区間で判定。ブートストラップCIは近似であり、厳密保証ではない。\n")
            L.append("| モデル | 比較 | Winkler差 | 補正CI | 判定 | カバレッジ差 | 補正CI | 判定 |")
            L.append("|---|---|---:|---|---|---:|---|---|")
            for model_name, pairs in block["bootstrap"].items():
                for pair, d in pairs.items():
                    w, c = d["winkler_diff"], d["coverage_diff"]
                    wv = "差を確認" if w["significant"] else "**明確な差なし**"
                    cv = "差を確認" if c["significant"] else "**明確な差なし**"
                    L.append(
                        f"| {model_name} | {pair.replace('_vs_', ' − ')} | {w['point_diff']:+.3f} | "
                        f"[{w['ci_family_low']:+.3f}, {w['ci_family_high']:+.3f}] | {wv} | "
                        f"{c['point_diff']*100:+.2f}pt | "
                        f"[{c['ci_family_low']*100:+.2f}, {c['ci_family_high']*100:+.2f}] | {cv} |")
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
    (DOCS_DIR.parent / "README.md").write_text(render_readme(r01, r02, r03, r04, r05), encoding="utf-8")
    print(f"主要結果を {out.relative_to(DOCS_DIR.parent)} に生成しました（{len(L)}行）")


if __name__ == "__main__":
    main()
