"""Evaluation tests across all 18 synthetic clean and faulty datasets."""

import json
import os
import pytest
from backend.diagnostics.engine import run_diagnostic_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")


@pytest.mark.parametrize("subfolder,exp_id", [
    ("resistance", "resistance_4wire"),
    ("oscilloscope", "waveform"),
    ("electromagnetism", "helmholtz"),
    ("optics", "laser_profile"),
    ("faraday", "faraday"),
    ("absorption", "beer_lambert"),
])
def test_evaluation_on_seeded_datasets(subfolder, exp_id):
    meta_path = os.path.join(EXAMPLES_DIR, subfolder, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    for filename, spec in meta["datasets"].items():
        csv_path = os.path.join(EXAMPLES_DIR, subfolder, filename)
        with open(csv_path, "r") as f:
            raw_csv = f.read()

        report = run_diagnostic_pipeline(raw_csv, exp_id)
        assert "causes" in report
        assert len(report["causes"]) > 0

        top_cause = report["causes"][0]
        expected_diag = spec["expected_diagnosis"]

        if spec["injected_fault"] is None:
            # Clean dataset should have normal / no fault as top cause
            assert expected_diag in top_cause["name"], f"Clean {filename} misdiagnosed as {top_cause['name']}"
        else:
            # Faulty dataset should identify the fault
            assert expected_diag in top_cause["name"], f"Failed on {filename}: got {top_cause['name']}, expected {expected_diag}"

            # Next test should be recommended
            rec_test = spec["recommended_next_test"]
            if rec_test:
                test_ids = [t["id"] for t in report["next_tests"]]
                assert rec_test in test_ids, f"Expected {rec_test} in {test_ids} for {filename}"
