# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repo is a finished job-hunting portfolio piece (statistics + time series conformal prediction), not an actively growing project. **Do not add new tickers, models, or evaluation methods, and do not expand scope on your own initiative** — that dilutes the "AI × DS × finance" story it's meant to tell. Prefer small, targeted fixes (bug fixes, doc corrections, dependency bumps) over redesigns. If a change would alter a reported number, rerun the affected pipeline and update `docs/key_results.md` / `README.md` together (see "Generated artifacts flow one direction" below) — never edit numbers by hand.

`FINAL_FIX_REPORT.md` and `docs/detailed_report.md` are historical audit/fix records, not living documentation — they describe a one-time remediation pass and are not kept in sync with the current code. They're candidates for future cleanup (deletion or a move into `archived/`); don't treat them as a source of truth for current behavior, and don't feel obligated to update them when you change code.

## Commands

```bash
pip install -r requirements.txt

# Full pipeline (sections 1-6 + h=1 conformal prediction; uses the offline snapshot)
python vix_analysis.py

# h=21 robustness pass, bootstrap CIs, and all figures (not produced by vix_analysis.py)
python conformal_vix_forecast.py

# Regenerate docs/key_results.md from data/results_*.json (run after any analysis script changes)
python build_key_results.py

# Tests (fully offline, no network calls)
python -m pytest tests/ -v
python -m pytest tests/test_repo.py::test_aci_feedback_is_delayed_by_h -v   # single test

# Lint (config in setup.cfg: max-line-length=135, E402 ignored for matplotlib.use("Agg") ordering)
flake8
```

Individual analysis stages can also be run standalone: `python analysis_01_phase_tests.py` through `analysis_05_figure.py`. They write to `data/results_0N_*.json` / `figures/`, matching what `vix_analysis.py` and `build_key_results.py` consume.

## Architecture

**`data_pipeline.py` is the single source of truth for data.** Every analysis script imports `build_dataset()` from it — never fetch or read data independently. `build_dataset()` defaults to `source="snapshot"`, reading `data/vix_sp500_tnx_raw.csv` (committed, dated 2026-09-08) so all results are reproducible offline. Pass `source="online"` to refetch from yfinance, and `refresh_snapshot=True` to overwrite the snapshot with that pull. Derived columns (`Return_6M`, `VIX_Phase`, `VIX_Z`, `Treasury_10Y_Diff1`, etc.) are always recomputed from the raw columns, never read from the snapshot as-is.

The repo has two analytical layers that share `data_pipeline.py` but answer different questions:

1. **VIX-phase / future-return significance testing** (`analysis_01` through `analysis_05`, orchestrated end-to-end by `vix_analysis.py`): tests whether VIX regime predicts S&P 500 forward returns, correcting for autocorrelation from overlapping return windows (HAC, Durbin-Watson, thinned resampling, leave-one-crisis-out).
2. **Time series conformal prediction of VIX itself** (`conformal_vix_forecast.py`): forecasts future VIX levels and builds prediction intervals via four methods compared under identical conditions — ACI (rolling window + adaptive alpha), RollingFixed (rolling window + fixed alpha), FixedSplit (frozen calibration quantile), NaiveBaseline (model-free). The A-vs-B / B-vs-C / A-vs-C contrasts isolate "does adapting alpha help" from "does refreshing the calibration window help." `vix_analysis.py` calls into this module's `main()` (passing `df_raw` to avoid a second data load) for the h=1 pass only; the h=21 robustness pass and all figures require running `conformal_vix_forecast.py` directly.

**No-lookahead invariant**: in the walk-forward loop, a model refit at position `start` may only train on rows whose target has already resolved, i.e. rows up to `start - h + 1` (not `start`) — a row's target for horizon `h` isn't known until `h` steps later. Train/calibration/test are strict, non-overlapping, time-ordered slices; tests in `tests/test_repo.py` assert this directly (`test_walk_forward_training_targets_are_realized`, `test_train_calibration_test_periods_do_not_overlap`, `test_features_contain_no_future_information`).

**Generated artifacts flow one direction**: analysis scripts write `data/results_0N_*.json` and files under `figures/` → `build_key_results.py` reads those JSONs and writes `docs/key_results.md` → `README.md`'s headline numbers must match the JSON. `tests/test_repo.py::test_readme_key_numbers_match_results` enforces this — if you change an analysis script's output, rerun `build_key_results.py` and update `README.md` accordingly, don't hand-edit numbers in the README.

**Windows path length**: this repo has previously hit `FileNotFoundError` from matplotlib/git when the absolute path (repo root + nested output filename) exceeded Windows' 260-char `MAX_PATH`. Keep the repo near a short path and keep new output filenames short; `conformal_vix_forecast.py`'s `_safe_savefig()` checks path length explicitly and raises a clear error instead of the raw OS error.
