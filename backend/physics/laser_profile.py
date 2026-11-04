"""Deterministic physics module for Laser Beam Profile & Alignment.

Mathematical Model:
    V(x) = V_dark + V_peak * exp(-2 * ((x - x_0) / w)^2)
    Where:
        x      = transverse detector position in mm
        V_dark = dark / background ambient voltage in mV
        V_peak = peak detector voltage above background in mV
        x_0    = beam center position in mm
        w      = beam radius at the 1/e^2 intensity level in mm
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def fit_gaussian_beam(
    positions: np.ndarray,
    voltages: np.ndarray,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a 1D Gaussian beam profile using pure NumPy Levenberg-Marquardt.

    Returns:
        (params, cov, ss_res, r_squared)
        where params = [v_dark, v_peak, x0, w]
    """
    n = len(positions)
    # 1. Initial heuristic parameter guesses
    v_dark = float(np.percentile(voltages, 10))
    v_peak = max(1e-3, float(np.max(voltages) - v_dark))
    x0 = float(positions[np.argmax(voltages)])

    # Estimate waist w from half-max or 1/e^2 points
    half_mask = voltages >= (v_dark + 0.5 * v_peak)
    if np.sum(half_mask) >= 2:
        fwhm = float(np.max(positions[half_mask]) - np.min(positions[half_mask]))
        w = max(0.1, fwhm / 1.1774)  # w = FWHM / sqrt(2*ln(2))
    else:
        w = max(0.2, (np.max(positions) - np.min(positions)) / 4.0)

    p = np.array([v_dark, v_peak, x0, w], dtype=np.float64)
    lam = 1e-2

    for _ in range(max_iter):
        vd, vp, xc, bw = p
        bw = max(1e-4, bw)
        u = (positions - xc) / bw
        u2 = u**2
        exp_term = np.exp(np.clip(-2.0 * u2, -50.0, 0.0))
        y_model = vd + vp * exp_term
        residuals = voltages - y_model
        current_ss = np.sum(residuals**2)

        # Jacobian matrix: d(model)/dp
        # col 0: d/d(vd) = 1
        # col 1: d/d(vp) = exp_term
        # col 2: d/d(xc) = vp * exp_term * (4 * u / bw)
        # col 3: d/d(bw) = vp * exp_term * (4 * u2 / bw)
        J = np.empty((n, 4), dtype=np.float64)
        J[:, 0] = 1.0
        J[:, 1] = exp_term
        J[:, 2] = vp * exp_term * (4.0 * u / bw)
        J[:, 3] = vp * exp_term * (4.0 * u2 / bw)

        JtJ = J.T @ J
        diag_JtJ = np.diag(np.diag(JtJ))
        try:
            delta_p = np.linalg.solve(JtJ + lam * diag_JtJ, J.T @ residuals)
        except np.linalg.LinAlgError:
            break

        p_new = p + delta_p
        # Keep physical constraints
        p_new[1] = max(1e-4, p_new[1])
        p_new[3] = max(1e-4, p_new[3])

        vd_n, vp_n, xc_n, bw_n = p_new
        u_n = (positions - xc_n) / bw_n
        y_n = vd_n + vp_n * np.exp(np.clip(-2.0 * u_n**2, -50.0, 0.0))
        new_ss = np.sum((voltages - y_n)**2)

        if new_ss < current_ss:
            p = p_new
            lam = max(1e-7, lam * 0.4)
            if abs(current_ss - new_ss) < 1e-9 * current_ss:
                break
        else:
            lam = min(1e4, lam * 4.0)

    # Final statistics
    vd, vp, xc, bw = p
    u = (positions - xc) / bw
    exp_term = np.exp(np.clip(-2.0 * u**2, -50.0, 0.0))
    y_final = vd + vp * exp_term
    res = voltages - y_final
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((voltages - np.mean(voltages))**2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    dof = max(1, n - 4)
    s_err = np.sqrt(ss_res / dof)

    # Covariance
    J = np.empty((n, 4), dtype=np.float64)
    J[:, 0] = 1.0
    J[:, 1] = exp_term
    J[:, 2] = vp * exp_term * (4.0 * u / bw)
    J[:, 3] = vp * exp_term * (4.0 * u**2 / bw)
    try:
        cov = s_err**2 * np.linalg.pinv(J.T @ J)
    except Exception:
        cov = np.zeros((4, 4))

    return p, cov, s_err, r_squared


def analyze_laser_profile_data(
    positions_mm: np.ndarray,
    voltages_mv: np.ndarray,
    expected_center_mm: float = 0.0,
    expected_peak_mv: float = 500.0,
) -> Dict[str, Any]:
    """Perform deterministic Gaussian profile fit and alignment diagnosis.

    Args:
        positions_mm: 1D array of transverse detector positions in mm.
        voltages_mv: 1D array of photodetector output voltages in mV.
        expected_center_mm: Target alignment position (default 0.0 mm).
        expected_peak_mv: Nominal peak voltage in mV.

    Returns:
        Structured dictionary of values, evidence, and plot coordinates.
    """
    n = len(positions_mm)
    if n < 5:
        raise ValueError(f"Insufficient points for beam profile fit: {n} < 5.")

    p, cov, s_err, r_squared = fit_gaussian_beam(positions_mm, voltages_mv)
    v_dark, v_peak, x0, w = p

    sigma_v_dark = float(np.sqrt(max(0.0, cov[0, 0])))
    sigma_v_peak = float(np.sqrt(max(0.0, cov[1, 1])))
    sigma_x0 = float(np.sqrt(max(0.0, cov[2, 2])))
    sigma_w = float(np.sqrt(max(0.0, cov[3, 3])))

    # Compare with expected alignment
    center_offset_mm = float(abs(x0 - expected_center_mm))
    peak_ratio = float(v_peak / expected_peak_mv) if expected_peak_mv > 0 else 1.0

    is_misaligned = bool(center_offset_mm > 0.40)
    is_low_signal = bool(peak_ratio < 0.60)
    is_clean = bool(center_offset_mm <= 0.25 and peak_ratio >= 0.75 and r_squared > 0.95)

    # Generate smooth fitted and expected curves for plotting
    sort_order = np.argsort(positions_mm)
    sorted_x = positions_mm[sort_order]
    fitted_y = v_dark + v_peak * np.exp(-2.0 * ((sorted_x - x0) / w)**2)
    expected_y = v_dark + expected_peak_mv * np.exp(-2.0 * ((sorted_x - expected_center_mm) / w)**2)

    values = {
        "beam_center_x0_mm": {"value": float(x0), "uncertainty": sigma_x0},
        "beam_waist_radius_w_mm": {"value": float(w), "uncertainty": sigma_w},
        "peak_voltage_mV": {"value": float(v_peak), "uncertainty": sigma_v_peak},
        "dark_voltage_mV": {"value": float(v_dark), "uncertainty": sigma_v_dark},
        "alignment_offset_mm": {"value": center_offset_mm},
        "peak_signal_ratio": {"value": peak_ratio},
        "residual_rms_mV": {"value": float(s_err)},
        "r_squared": {"value": float(r_squared)},
        "expected_center_mm": {"value": float(expected_center_mm)},
    }

    evidence = {
        "alignment_offset_mm": center_offset_mm,
        "peak_signal_ratio": peak_ratio,
        "is_misaligned": is_misaligned,
        "is_low_signal": is_low_signal,
        "is_clean": is_clean,
        "r_squared": float(r_squared),
        "dark_voltage_mV": float(v_dark),
    }

    plot = {
        "x": sorted_x.tolist(),
        "y": voltages_mv[sort_order].tolist(),
        "fit": fitted_y.tolist(),
        "expected": expected_y.tolist(),
        "x_label": "position (mm)",
        "y_label": "detector voltage (mV)",
    }

    return {
        "values": values,
        "evidence": evidence,
        "plot": plot,
    }
