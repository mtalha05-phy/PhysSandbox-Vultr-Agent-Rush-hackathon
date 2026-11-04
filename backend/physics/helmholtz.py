"""Deterministic physics module for Helmholtz Coil Magnetic-Field Calibration.

Mathematical Model:
    B_theoretical = (4/5)^(3/2) * mu_0 * N * I / R = k_theoretical * I
    Where:
        mu_0 = 4 * pi * 1e-7 T*m/A (vacuum permeability)
        N    = number of turns per coil
        R    = radius of each coil in meters (and coil separation)
        I    = applied current in Amperes
        B    = center magnetic flux density (converted to microtesla µT)

    In real experiments with sensor offsets or ambient Earth magnetic field:
        B_measured = k_measured * I + B_0
"""

from typing import Any, Dict, List, Optional
import numpy as np

MU_0 = 4.0 * np.pi * 1e-7  # T*m/A


def calculate_theoretical_k(radius_m: float = 0.15, turns: int = 100) -> float:
    """Calculate theoretical calibration constant k in µT/A at the coil center."""
    if radius_m <= 0:
        raise ValueError(f"Coil radius must be positive, got {radius_m}.")
    if turns <= 0:
        raise ValueError(f"Number of turns must be positive, got {turns}.")
    k_tesla_per_amp = ((4.0 / 5.0) ** 1.5) * MU_0 * turns / radius_m
    return float(k_tesla_per_amp * 1e6)  # Convert T/A to µT/A


def analyze_helmholtz_data(
    currents: np.ndarray,
    measured_b_ut: np.ndarray,
    radius_m: float = 0.15,
    turns: int = 100,
) -> Dict[str, Any]:
    """Perform deterministic regression against Helmholtz theoretical prediction.

    Args:
        currents: 1D array of currents in Amperes.
        measured_b_ut: 1D array of measured magnetic field in microtesla (µT).
        radius_m: Coil radius in meters.
        turns: Number of turns per coil.

    Returns:
        Structured dictionary of values, evidence, and plot coordinates.
    """
    n = len(currents)
    if n < 3:
        raise ValueError(f"Insufficient points for magnetic calibration: {n} < 3.")

    k_theo = calculate_theoretical_k(radius_m=radius_m, turns=turns)

    # Linear fit: B = k_meas * I + B_0
    A = np.vstack([currents, np.ones_like(currents)]).T
    (k_meas, b0_fit), ss_res_arr, rank, s = np.linalg.lstsq(A, measured_b_ut, rcond=None)

    y_pred = k_meas * currents + b0_fit
    residuals = measured_b_ut - y_pred

    ss_res = float(np.sum(residuals**2))
    dof = max(1, n - 2)
    s_err = np.sqrt(ss_res / dof)

    try:
        inv_ata = np.linalg.inv(A.T @ A)
        cov = s_err**2 * inv_ata
        sigma_k = float(np.sqrt(max(0.0, cov[0, 0])))
        sigma_b0 = float(np.sqrt(max(0.0, cov[1, 1])))
    except np.linalg.LinAlgError:
        sigma_k = 0.0
        sigma_b0 = 0.0

    ss_tot = float(np.sum((measured_b_ut - np.mean(measured_b_ut))**2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    percentage_error = float(abs((k_meas - k_theo) / k_theo) * 100.0)
    if sigma_b0 > 1e-12:
        offset_sigma = float(abs(b0_fit) / sigma_b0)
    elif abs(b0_fit) > 1e-6:
        offset_sigma = 999.0  # Perfect fit with non-zero offset
    else:
        offset_sigma = 0.0

    expected_b = (k_theo * currents).tolist()

    has_gain_error = bool(percentage_error > 5.0 and r_squared > 0.95)
    has_magnetic_offset = bool(offset_sigma > 3.0 and abs(b0_fit) > 5.0)

    values = {
        "k_measured_uT_per_A": {"value": float(k_meas), "uncertainty": sigma_k},
        "k_theoretical_uT_per_A": {"value": float(k_theo)},
        "offset_B0_uT": {"value": float(b0_fit), "uncertainty": sigma_b0},
        "calibration_error_pct": {"value": percentage_error},
        "residual_rms_uT": {"value": float(s_err)},
        "r_squared": {"value": float(r_squared)},
        "coil_radius_m": {"value": float(radius_m)},
        "coil_turns": {"value": int(turns)},
    }

    evidence = {
        "calibration_error_pct": percentage_error,
        "offset_uT": float(b0_fit),
        "offset_sigma": offset_sigma,
        "has_gain_error": has_gain_error,
        "has_magnetic_offset": has_magnetic_offset,
        "r_squared": float(r_squared),
    }

    sort_order = np.argsort(currents)
    plot = {
        "x": currents[sort_order].tolist(),
        "y": measured_b_ut[sort_order].tolist(),
        "fit": y_pred[sort_order].tolist(),
        "expected": [expected_b[i] for i in sort_order],
        "x_label": "current (A)",
        "y_label": "magnetic field (µT)",
    }

    return {
        "values": values,
        "evidence": evidence,
        "plot": plot,
    }
