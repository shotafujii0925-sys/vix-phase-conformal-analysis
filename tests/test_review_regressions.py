"""Calendar, calibration and inference regression tests."""
import numpy as np
import pandas as pd
import pytest

import conformal_vix_forecast as cvf
import data_pipeline as dp


@pytest.mark.parametrize("h", [1, 21])
def test_target_dates_survive_missing_treasury_rows(h):
    raw = dp.build_dataset(verbose=False)
    feat = cvf.build_features(raw, h)
    expected = pd.Series(raw.index, index=raw.index).shift(-h).loc[feat.index]
    pd.testing.assert_series_equal(feat.target_date, expected, check_names=False)
    assert np.allclose(feat.target, raw.loc[feat.target_date, "VIX"].values)
    assert (feat.target_date.values[:-h] != feat.index[h:].values).any()


def test_feedback_releases_on_actual_date_across_missing_days():
    issue = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    target = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    y = np.array([60., 60., 60.])
    _, _, path = cvf.run_rolling_conformal(
        y - 20, np.full(3, 20.), y, 21, [0.5] * 200,
        True, issue_dates=issue, target_dates=target)
    assert path == pytest.approx([0.1, 0.091, 0.082])


def test_calibration_pending_is_released_once_without_alpha_feedback(monkeypatch):
    captured = []
    original = cvf.conformal_quantile

    def record(scores, alpha, **kwargs):
        captured.append(list(scores))
        return original(scores, alpha, **kwargs)

    monkeypatch.setattr(cvf, "conformal_quantile", record)
    cvf.run_rolling_conformal([1., 2., 3.], [20.] * 3, [21., 22., 23.], 2, [0.5] * 10,
                              True, init_pending=[{"date": 1, "resid": 9.}])
    assert len(captured[0]) == 10
    assert captured[1].count(9.) == 1
    assert captured[2].count(9.) == 1
    assert len(captured[2]) == 12  # initial + pending calibration + first resolved test


def test_conformal_quantile_is_exact_order_statistic():
    assert cvf.conformal_quantile(np.arange(500), .1) == 450
    assert np.isinf(cvf.conformal_quantile([1., 2.], .1))
    assert cvf.conformal_quantile([1., 2.], .1, cap_at_max=True) == 2.
    with pytest.raises(ValueError):
        cvf.conformal_quantile([], .1)


def test_return_support_does_not_overlap_after_merging():
    raw = dp.build_dataset(verbose=False)
    data = dp.analysis_frame(raw, ["Return_6M"])
    episodes = dp.return_episodes(data)
    assert len(episodes) == 8
    for (_, previous_end), (start, _) in zip(episodes, episodes[1:]):
        assert data.loc[previous_end, "Return_6M_Target_Date"] < start


def test_family_corrected_intervals_include_individual_intervals():
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, 300))
    r = cvf.block_bootstrap_mean_diff(a, b, 21, n_boot=1000, family_size=32)
    assert r["ci_family_low"] <= r["ci95_low"] <= r["ci95_high"] <= r["ci_family_high"]
    assert r["significant"] == (r["ci_family_low"] > 0 or r["ci_family_high"] < 0)


def test_readme_is_generated_from_all_current_result_files():
    from build_key_results import load, render_readme

    results = [load(name) for name in (
        "results_01_phase_tests.json", "results_02_leave_one_crisis_out.json",
        "results_03_hac_power.json", "results_04_stationarity_economic.json", "results_05_conformal_vix.json",
    )]
    assert (dp.REPO_ROOT / "README.md").read_text(encoding="utf-8") == render_readme(*results)
