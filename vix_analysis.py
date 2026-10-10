# -*- coding: utf-8 -*-
"""Run the offline analysis pipeline and regenerate all published results."""
import json
from pathlib import Path
import runpy

from build_key_results import main as build_report
from conformal_vix_forecast import DATA_DIR, RESULTS, main as conformal_main
from data_pipeline import build_dataset


def main():
    root = Path(__file__).resolve().parent
    for stage in (
        "analysis_01_phase_tests.py", "analysis_02_leave_one_crisis_out.py",
        "analysis_03_hac_power.py", "analysis_04_stationarity_economic.py", "analysis_05_figure.py",
    ):
        runpy.run_path(str(root / stage), run_name="__main__")
    df = build_dataset()
    conformal_main(h=1, tag="h1_primary", make_plots=True, df_raw=df)
    conformal_main(h=21, tag="h21_robustness", make_plots=False, df_raw=df)
    with open(DATA_DIR / "results_05_conformal_vix.json", "w", encoding="utf-8") as handle:
        json.dump(RESULTS, handle, ensure_ascii=False, indent=2, allow_nan=False)
    build_report()


if __name__ == "__main__":
    main()
