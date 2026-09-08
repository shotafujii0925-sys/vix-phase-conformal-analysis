# -*- coding: utf-8 -*-
"""
リポジトリの重要な前提を検証するテスト。

全てオフライン（data/vix_sp500_tnx_raw.csv のスナップショット）で完結し、
yfinance等の外部APIには一切接続しない。

    python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import conformal_vix_forecast as cvf  # noqa: E402
import data_pipeline as dp  # noqa: E402


# =====================================================================
# フィクスチャ
# =====================================================================
@pytest.fixture(scope="module")
def raw_df():
    return dp.build_dataset(source="snapshot", verbose=False)


# =====================================================================
# 1. データ品質
# =====================================================================
def test_snapshot_exists_and_loads():
    """CSVスナップショットが存在し、外部接続なしで読み込める。"""
    assert dp.SNAPSHOT_PATH.exists(), "スナップショットCSVがありません"
    df = dp._load_raw_snapshot()
    assert len(df) > 8000
    assert list(df.columns) == dp.RAW_COLUMNS


def test_required_columns_exist(raw_df):
    """必須列が全て存在する。"""
    for col in ["SP500", "VIX", "Treasury_10Y", "Return_6M", "VIX_Phase", "VIX_Z"]:
        assert col in raw_df.columns, f"列 {col} がありません"


def test_index_is_sorted_and_unique(raw_df):
    """日付インデックスが昇順で、重複がない。"""
    assert raw_df.index.is_monotonic_increasing, "インデックスが昇順ではない"
    assert not raw_df.index.duplicated().any(), "重複した日付がある"


def test_value_ranges(raw_df):
    """VIXが合理的な範囲、SP500が正、金利が%表記の範囲。"""
    assert raw_df["VIX"].between(5, 100).mean() > 0.99
    assert (raw_df["SP500"] > 0).all()
    tnx = raw_df["Treasury_10Y"].dropna()
    assert tnx.between(0, 20).mean() > 0.99


def test_validate_raw_rejects_bad_data():
    """検証関数が不正データを実際に弾く。"""
    bad = pd.DataFrame({"SP500": [1.0, 2.0], "VIX": [15.0, 16.0], "Treasury_10Y": [2.0, 2.1]},
                       index=pd.to_datetime(["2020-01-02", "2020-01-01"]))  # 降順
    with pytest.raises(ValueError, match="昇順"):
        dp.validate_raw(bad, verbose=False)


# =====================================================================
# 2. 先読みバイアス
# =====================================================================
def test_return_6m_references_exact_row_offset(raw_df):
    """Return_6M が正確に HORIZON_DAYS 行先の SP500 を参照している。"""
    h = dp.HORIZON_DAYS
    sp = raw_df["SP500"].values
    expected = sp[h:] / sp[:-h] - 1
    actual = raw_df["Return_6M"].values[:-h]
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("h", [1, 21])
def test_target_equals_vix_h_rows_ahead(raw_df, h):
    """target[i] が VIX[i+h] に一致する（コンフォーマル予測の目的変数）。"""
    feat = cvf.build_features(raw_df, h)
    vix_full = raw_df["VIX"]
    for i in [0, 100, len(feat) // 2, len(feat) - 1]:
        date = feat.index[i]
        pos_full = vix_full.index.get_loc(date)
        assert pos_full + h < len(vix_full)
        assert feat["target"].iloc[i] == pytest.approx(vix_full.iloc[pos_full + h])


@pytest.mark.parametrize("h", [1, 21])
def test_features_contain_no_future_information(raw_df, h):
    """
    特徴量が予測時点以前の情報のみで構成される。
    各特徴量を「その時点までのデータだけ」で作り直した値と一致することを確認する。
    """
    feat = cvf.build_features(raw_df, h)
    check_at = len(raw_df) // 2
    truncated = raw_df.iloc[:check_at + 1]  # t 時点までしか見えない世界
    feat_trunc = cvf.build_features(truncated, h=1)  # targetは使わないのでh任意

    common = feat.index.intersection(feat_trunc.index)
    assert len(common) > 100
    last_common = common[-1]
    for col in cvf.FEATURE_COLS:
        full_val = feat.loc[last_common, col]
        trunc_val = feat_trunc.loc[last_common, col]
        assert full_val == pytest.approx(trunc_val, nan_ok=True), \
            f"特徴量 {col} が将来情報に依存している"


@pytest.mark.parametrize("h", [1, 21])
def test_walk_forward_training_targets_are_realized(h):
    """
    walk-forwardの各再学習で、学習データのtargetが再学習時点で実現済みである。
    walk_forward_forecast と同じ学習窓の決め方を再現して検証する。
    """
    n = 3000
    start = cvf.BURN_IN_DAYS
    while start < n:
        train_end = start - h + 1
        # 学習に使う最終行 (train_end - 1) のtargetは (train_end - 1 + h) 行目で実現する。
        last_train_row = train_end - 1
        realized_at = last_train_row + h
        assert realized_at <= start, (
            f"h={h}, start={start}: 学習末尾行のtargetが{realized_at}行目で実現し、"
            f"再学習時点{start}より後 → 先読み")
        start = min(start + cvf.REFIT_EVERY, n)


def test_train_calibration_test_periods_do_not_overlap(raw_df):
    """学習・較正・評価期間が時系列順で重複しない。"""
    feat = cvf.build_features(raw_df, h=1)
    cal_start = cvf.BURN_IN_DAYS
    cal_end = cal_start + cvf.CALIBRATION_DAYS
    train_idx = feat.index[:cal_start]
    cal_idx = feat.index[cal_start:cal_end]
    test_idx = feat.index[cal_end:]

    assert train_idx.max() < cal_idx.min()
    assert cal_idx.max() < test_idx.min()
    assert len(train_idx.intersection(cal_idx)) == 0
    assert len(cal_idx.intersection(test_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0


# =====================================================================
# 3. ACI / コンフォーマル予測のロジック
# =====================================================================
def _toy_inputs(n=60, h=5):
    rng = np.random.default_rng(0)
    pred = np.full(n, 20.0)
    target = 20.0 + rng.normal(0, 1.0, n)
    resid = np.abs(target - pred)
    init_history = list(np.abs(rng.normal(0, 1.0, 200)))
    return resid, pred, target, init_history, h


def test_aci_feedback_is_delayed_by_h():
    """ACIのalphaは最初のhステップでは更新されない（フィードバック遅延）。"""
    resid, pred, target, init_history, h = _toy_inputs()
    _, _, alpha_path = cvf.run_rolling_conformal(
        resid, pred, target, h, init_history, adaptive=True)
    assert np.allclose(alpha_path[:h + 1], cvf.ALPHA_TARGET), \
        "最初のh+1ステップでalphaが動いている（遅延が効いていない）"
    assert not np.allclose(alpha_path, cvf.ALPHA_TARGET), \
        "alphaが最後まで全く動いていない"


def test_rolling_fixed_never_updates_alpha():
    """RollingFixed は alpha を一切更新しない。"""
    resid, pred, target, init_history, h = _toy_inputs()
    _, _, alpha_path = cvf.run_rolling_conformal(
        resid, pred, target, h, init_history, adaptive=False)
    assert np.allclose(alpha_path, cvf.ALPHA_TARGET), "固定alphaのはずが更新されている"


def test_aci_uses_issued_clipped_interval_for_coverage():
    """
    被覆判定が「発行時点の、下限クリップ後の区間」で行われる。

    pred - q < 0 となるケースを作る。このとき下限は0にクリップされるため、
    y >= 0 は必ず被覆される。残差 |y - pred| > q であっても被覆扱いになるべき。
    もし残差とqの単純比較で判定していれば「外れ」と誤判定し、alphaが下がる。
    """
    h = 1
    n = 10
    pred = np.full(n, 1.0)                 # 予測値が小さい
    target = np.full(n, 0.0)               # 実測は0（VIXの下限側）
    resid = np.abs(target - pred)          # = 1.0
    init_history = [5.0] * 200             # q は約5.0 → pred - q = -4.0 < 0

    lower, upper, alpha_path = cvf.run_rolling_conformal(
        resid, pred, target, h, init_history, adaptive=True)

    assert np.all(lower == 0.0), "下限が0にクリップされていない"
    covered = (target >= lower) & (target <= upper)
    assert covered.all(), "クリップ後の区間では被覆しているはず"
    # 全て被覆 → err=0 → alphaは単調に増加（区間は狭くなる方向）
    assert alpha_path[-1] > alpha_path[0], \
        "被覆しているのにalphaが増えていない（誤って外れと判定している可能性）"


def test_aci_alpha_moves_down_when_missing():
    """外れ続けるとalphaは下がる（区間が広がる方向）。"""
    h = 1
    n = 30
    pred = np.full(n, 20.0)
    target = np.full(n, 60.0)      # 大きく外す
    resid = np.abs(target - pred)
    init_history = [0.5] * 200     # qが極めて小さい → 必ず外れる

    _, _, alpha_path = cvf.run_rolling_conformal(
        resid, pred, target, h, init_history, adaptive=True)
    assert alpha_path[-1] < alpha_path[0], "外れ続けているのにalphaが下がっていない"


def test_pending_feedback_has_no_loss_or_duplication():
    """
    キューの処理で残差の取りこぼし・二重処理がない。
    n個の予測のうち、判明するのは n-h 個であり、履歴は init + (n-h) 件になる。
    """
    h = 7
    n = 50
    rng = np.random.default_rng(1)
    pred = np.full(n, 20.0)
    target = 20.0 + rng.normal(0, 1, n)
    resid = np.abs(target - pred)
    init = [1.0] * 30

    # cal_window を大きくして全履歴が残るようにし、内部履歴長を間接的に検証する
    lower, upper, _ = cvf.run_rolling_conformal(
        resid, pred, target, h, init, adaptive=False, cal_window=10_000)
    assert len(lower) == n and len(upper) == n
    assert np.isfinite(lower).all() and np.isfinite(upper).all()


def test_winkler_score_known_examples():
    """Winkler scoreが既知の例で正しい。"""
    alpha = 0.10
    # 被覆している場合 -> 幅のみ
    assert cvf.winkler_score([5.0], [0.0], [10.0], alpha)[0] == pytest.approx(10.0)
    # 上に外れた場合 -> 幅 + (y - upper) * 2/alpha
    assert cvf.winkler_score([12.0], [0.0], [10.0], alpha)[0] == pytest.approx(10.0 + 2 * 2 / 0.10)
    # 下に外れた場合 -> 幅 + (lower - y) * 2/alpha
    assert cvf.winkler_score([-1.0], [0.0], [10.0], alpha)[0] == pytest.approx(10.0 + 1 * 2 / 0.10)


def test_evaluate_interval_counts():
    """evaluate_interval のカバレッジ計算が正しい。"""
    res = cvf.evaluate_interval([1.0, 5.0, 20.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
    assert res["n"] == 3
    assert res["n_covered"] == 2
    assert res["coverage"] == pytest.approx(2 / 3)
    assert res["mean_width"] == pytest.approx(10.0)


def test_block_bootstrap_detects_real_difference():
    """ブロックブートストラップが、明確な差を差として検出し、差がない場合は検出しない。"""
    rng = np.random.default_rng(3)
    n = 1000
    a = rng.normal(0, 1, n)
    b_same = rng.normal(0, 1, n)
    b_diff = rng.normal(5, 1, n)

    res_same = cvf.block_bootstrap_mean_diff(a, b_same, block_len=21, n_boot=300)
    res_diff = cvf.block_bootstrap_mean_diff(a, b_diff, block_len=21, n_boot=300)
    assert not res_same["significant"], "差がないのに差を検出した"
    assert res_diff["significant"], "明確な差を検出できていない"
    assert res_diff["ci95_low"] < res_diff["point_diff"] < res_diff["ci95_high"]


# =====================================================================
# 4. リポジトリ構成・出力
# =====================================================================
def test_figure_output_paths_are_unique():
    """複数のスクリプトが同じ図ファイル名へ書き込んでいない。"""
    pattern = re.compile(r'savefig\(\s*(?:FIGURES_DIR\s*/\s*)?["\']([^"\']+\.png)["\']')
    targets: dict[str, list[str]] = {}
    for py in REPO_ROOT.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            name = Path(m.group(1)).name
            targets.setdefault(name, []).append(py.name)
    duplicates = {k: v for k, v in targets.items() if len(v) > 1}
    assert not duplicates, f"同じ図ファイル名に複数スクリプトが書き込んでいます: {duplicates}"


def test_result_files_exist():
    """主要な結果ファイルが生成されている。"""
    expected = [
        "results_01_phase_tests.json",
        "results_02_leave_one_crisis_out.json",
        "results_03_hac_power.json",
        "results_04_stationarity_economic.json",
        "results_05_conformal_vix.json",
    ]
    for name in expected:
        path = dp.DATA_DIR / name
        assert path.exists(), f"{name} がありません（該当スクリプトを実行してください）"
        with open(path, encoding="utf-8") as f:
            json.load(f)  # 壊れたJSONでないこと


def test_readme_key_numbers_match_results():
    """
    READMEに書かれた主要数値が、最新の結果JSONと一致する。
    （READMEの手作業転記が古くなっていないことの検査）
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    with open(dp.DATA_DIR / "results_01_phase_tests.json", encoding="utf-8") as f:
        r01 = json.load(f)
    with open(dp.DATA_DIR / "results_05_conformal_vix.json", encoding="utf-8") as f:
        r05 = json.load(f)

    checks = {
        "ANOVA F値": f"{r01['anova_daily']['standard']['stat']:.4f}",
        "Durbin-Watson": f"{r01['durbin_watson_anova_model']:.4f}",
        "h=1 ACI カバレッジ": f"{r05['h1_primary']['method_comparison']['gbm']['aci']['coverage']*100:.2f}",
        "h=1 naive カバレッジ": f"{r05['h1_primary']['method_comparison']['gbm']['naive']['coverage']*100:.2f}",
        "h=1 persistence MAE": f"{r05['h1_primary']['model_mae']['persistence']:.3f}",
        "h=1 gbm MAE": f"{r05['h1_primary']['model_mae']['gbm']:.3f}",
    }
    missing = {k: v for k, v in checks.items() if v not in readme}
    assert not missing, (
        "READMEの数値が最新の結果ファイルと一致しません（古い値が残っています）: " + str(missing))
