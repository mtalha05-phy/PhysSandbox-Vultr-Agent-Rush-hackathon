"""Core diagnostic engine orchestrating validation, physics, and fault inference."""

import io
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.physics import EXPERIMENT_METADATA, run_physics_analysis
from backend.diagnostics.fault_models import evaluate_diagnostic_report


def validate_csv_content(
    raw_content: str,
    experiment_id: str,
    max_rows: int = 20000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Parse, sanitize, and validate CSV content against the experiment schema.

    Enforces:
    - Comment stripping to prevent prompt injection attacks.
    - Required column schema checks.
    - NaN, Inf, and empty dataset rejection.
    - Monotonicity checks for time-series.
    """
    if not raw_content or not raw_content.strip():
        raise ValueError("Uploaded file is empty.")

    meta = EXPERIMENT_METADATA.get(experiment_id)
    if not meta:
        raise ValueError(f"Unknown experiment: {experiment_id}")

    expected_cols = meta["expected_columns"]

    lines = raw_content.splitlines()
    data_lines = []
    dropped_comments = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            dropped_comments += 1
            continue
        data_lines.append(stripped)

    if not data_lines:
        raise ValueError("No tabular data rows found after filtering comments.")

    if len(data_lines) - 1 > max_rows:
        raise ValueError(f"CSV exceeds maximum row limit ({max_rows} rows).")

    # Parse header
    header = [c.strip() for c in data_lines[0].split(",")]
    for col in expected_cols:
        if col not in header:
            raise ValueError(f"Missing required column '{col}'. Expected columns: {expected_cols}")

    col_indices = {col: header.index(col) for col in expected_cols}
    parsed_cols: Dict[str, List[float]] = {col: [] for col in expected_cols}

    for row_idx, row_str in enumerate(data_lines[1:], start=2):
        parts = [p.strip() for p in row_str.split(",")]
        if len(parts) < len(header):
            raise ValueError(f"Line {row_idx}: incomplete columns (got {len(parts)}, expected {len(header)}).")

        for col, idx in col_indices.items():
            val_str = parts[idx]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f"Line {row_idx}, column '{col}': non-numeric value '{val_str}'.")

            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Line {row_idx}, column '{col}': invalid NaN or Infinite value.")
            parsed_cols[col].append(val)

    n_rows = len(parsed_cols[expected_cols[0]])
    if n_rows < 3:
        raise ValueError(f"Dataset has insufficient data rows ({n_rows} < 3).")

    # Convert to NumPy arrays
    data_arrays = {col: np.array(vals, dtype=np.float64) for col, vals in parsed_cols.items()}

    # Check time series constraints where applicable
    if "time_s" in data_arrays:
        times = data_arrays["time_s"]
        dt = np.diff(times)
        if np.any(dt <= 0):
            raise ValueError("Time series data must have strictly increasing timestamps (no duplicates or reverse order).")

    # Column summary profiling
    columns_summary = []
    for col in expected_cols:
        arr = data_arrays[col]
        columns_summary.append({
            "name": col,
            "count": int(len(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })

    summary = {
        "rows": n_rows,
        "columns": columns_summary,
        "comment_lines_dropped": dropped_comments,
        "note": "Free text and comments in the file are never exposed to the AI model.",
    }

    return data_arrays, summary


def run_diagnostic_pipeline(
    raw_csv: str,
    experiment_id: str,
    expected_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute complete deterministic physics and evidence-based diagnostic pipeline."""
    if expected_params is None:
        expected_params = EXPERIMENT_METADATA.get(experiment_id, {}).get("default_expected", {})

    # 1. Validation & Schema Enforcement
    data_arrays, summary = validate_csv_content(raw_csv, experiment_id)

    # 2. Deterministic Physics Fitting & Residual Analysis
    physics_results = run_physics_analysis(experiment_id, data_arrays, expected_params)

    values = physics_results["values"]
    evidence = physics_results["evidence"]
    plot = physics_results["plot"]

    # 3. Bayesian Fault Diagnosis & Next-Test Recommendation
    causes, next_tests, status, explanation = evaluate_diagnostic_report(experiment_id, evidence)

    return {
        "experiment": experiment_id,
        "expected": expected_params,
        "summary": summary,
        "values": values,
        "evidence": evidence,
        "plot": plot,
        "causes": causes,
        "status": status,
        "next_tests": next_tests,
        "done": [],
        "test_run": None,
        "explanation": explanation,
        "explained_by": "template",
    }
