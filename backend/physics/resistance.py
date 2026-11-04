"""Deterministic physics module for Low-Resistance & Thermal EMF investigation.

Mathematical Model:
    V_measured = R * I + V_0
    Where:
        R   = actual resistance in Ohms
        I   = applied current in Amperes
        V_0 = constant thermal EMF / contact offset in Volts

When current reversal (+I, -I) is available:
    V(+I) =  R*I + V_0
    V(-I) = -R*I + V_0
    R_est   = (V(+I) - V(-I)) / (2*I)
    V_0_est = (V(+I) + V(-I)) / 2
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def analyze_resistance_data(
    currents: np.ndarray,
    voltages: np.ndarray,
    expected_r: float = 0.010,
) -> Dict[str, Any]:
    """Perform deterministic regression, parameter uncertainties, and residual checks.

    Args:
        currents: 1D array of applied currents in Amperes.
        voltages: 1D array of measured voltages in Volts.
        expected_r: Nominal/expected resistance in Ohms.

    Returns:
        Structured dictionary of values, evidence, and plot coordinates.
    """
    n = len(currents)
    if n < 3:
        raise ValueError(f"Insufficient data points for linear regression: {n} < 3.")

    # 1. Linear regression V = R * I + V_0 using least squares
    A = np.vstack([currents, np.ones_like(currents)]).T
    (r_fit, v0_fit), residuals_sum, rank, s = np.linalg.lstsq(A, voltages, rcond=None)

    y_pred = r_fit * currents + v0_fit
    residuals = voltages - y_pred

    # Residual sum of squares & degrees of freedom
    ss_res = float(np.sum(residuals**2))
    dof = max(1, n - 2)
    s_err = np.sqrt(ss_res / dof)

    # Covariance matrix for parameter uncertainties
    try:
        inv_ata = np.linalg.inv(A.T @ A)
        cov = s_err**2 * inv_ata
        sigma_r = float(np.sqrt(max(0.0, cov[0, 0])))
        sigma_v0 = float(np.sqrt(max(0.0, cov[1, 1])))
    except np.linalg.LinAlgError:
        sigma_r = 0.0
        sigma_v0 = 0.0

    # R-squared goodness of fit
    ss_tot = float(np.sum((voltages - np.mean(voltages))**2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    # Check for current reversal (+I and -I pairs)
    pos_mask = currents > 1e-9
    neg_mask = currents < -1e-9
    has_reversal = bool(np.any(pos_mask) and np.any(neg_mask))
    offset_survives_reversal = False
    r_reversal_est = None

    if has_reversal:
        # Match positive and negative magnitudes where available
        pos_i = currents[pos_mask]
        pos_v = voltages[pos_mask]
        neg_i = currents[neg_mask]
        neg_v = voltages[neg_mask]

        # Calculate mean offset across positive/negative points
        mean_v_pos = np.mean(pos_v)
        mean_v_neg = np.mean(neg_v)
        mean_i_pos = np.mean(pos_i)
        mean_i_neg = np.mean(np.abs(neg_i))

        if mean_i_pos > 0 and mean_i_neg > 0:
            v0_rev = (mean_v_pos + mean_v_neg) / 2.0
            r_rev = (mean_v_pos - mean_v_neg) / (mean_i_pos + mean_i_neg)
            r_reversal_est = float(r_rev)
            if abs(v0_rev) > 2.0 * sigma_v0:
                offset_survives_reversal = True

    # Check for non-linearity (curvature via quadratic fit: V = c0 + c1*I + c2*I^2)
    A_quad = np.vstack([currents**2, currents, np.ones_like(currents)]).T
    (c2, c1, c0), quad_res, _, _ = np.linalg.lstsq(A_quad, voltages, rcond=None)
    quad_residuals = voltages - (c2 * currents**2 + c1 * currents + c0)
    ss_quad = float(np.sum(quad_residuals**2))
    # F-test or relative variance improvement
    curvature_improvement = (ss_res - ss_quad) / ss_res if ss_res > 1e-18 else 0.0
    is_nonlinear = bool(curvature_improvement > 0.40 and abs(c2) > 1e-4)

    # Apparent resistance at lowest non-zero current
    min_idx = np.argmin(np.abs(currents))
    i_min = currents[min_idx]
    v_min = voltages[min_idx]
    r_apparent_lowest = float(v_min / i_min) if abs(i_min) > 1e-12 else float(r_fit)

    # Offset significance (Z-score)
    if sigma_v0 > 1e-12:
        offset_sigma = float(abs(v0_fit) / sigma_v0)
    elif abs(v0_fit) > 1e-9:
        offset_sigma = 999.0  # Perfect fit with non-zero offset
    else:
        offset_sigma = 0.0
    slope_error_pct = float(((r_fit - expected_r) / expected_r) * 100.0) if expected_r != 0 else 0.0
    apparent_r_error_pct = float(((r_apparent_lowest - expected_r) / expected_r) * 100.0) if expected_r != 0 else 0.0

    # Theoretical expected curve
    expected_curve = (expected_r * currents).tolist()

    values = {
        "resistance_fit_ohm": {"value": float(r_fit), "uncertainty": sigma_r},
        "offset_V": {"value": float(v0_fit), "uncertainty": sigma_v0},
        "apparent_resistance_at_lowest_current_ohm": {"value": r_apparent_lowest},
        "expected_resistance_ohm": {"value": float(expected_r)},
        "residual_rms_V": {"value": float(s_err)},
        "r_squared": {"value": float(r_squared)},
        "max_gap_from_expected_V": {"value": float(np.max(np.abs(voltages - (expected_r * currents))))},
    }
    if r_reversal_est is not None:
        values["resistance_reversal_ohm"] = {"value": r_reversal_est}

    evidence = {
        "offset_sigma": offset_sigma,
        "slope_error_pct": slope_error_pct,
        "apparent_r_low_error_pct": apparent_r_error_pct,
        "has_current_reversal": has_reversal,
        "offset_survives_reversal": offset_survives_reversal,
        "is_nonlinear": is_nonlinear,
        "curvature_improvement": float(curvature_improvement),
        "r_squared": float(r_squared),
    }

    # Sort data points for continuous plotting
    sort_order = np.argsort(currents)
    plot = {
        "x": currents[sort_order].tolist(),
        "y": voltages[sort_order].tolist(),
        "fit": y_pred[sort_order].tolist(),
        "expected": [expected_curve[i] for i in sort_order],
        "x_label": "current (A)",
        "y_label": "voltage (V)",
    }

    return {
        "values": values,
        "evidence": evidence,
        "plot": plot,
    }
