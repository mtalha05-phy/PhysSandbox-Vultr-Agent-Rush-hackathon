"""Deterministic physics module for Beer-Lambert Law of Optical Absorption.

Mathematical Model:
    I(x) = I_0 * exp(-alpha * x) + I_background
    Where:
        x            = sample thickness / optical path length in cm
        I(x)         = transmitted optical intensity (relative units)
        I_0          = incident optical intensity at x = 0
        alpha        = absorption coefficient in cm^(-1)
        I_background = ambient / stray light background intensity

Logarithmic Representation (ideal zero background):
    ln(I_0 / I(x)) = alpha * x
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def fit_beer_lambert(
    thicknesses: np.ndarray,
    intensities: np.ndarray,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fit exponential absorption model I(x) = I_0 * exp(-alpha * x) + I_bg.

    Returns:
        (params, cov, ss_res, r_squared)
        where params = [I_0, alpha, I_bg]
    """
    n = len(thicknesses)
    # Heuristic initial values
    i_min = float(np.min(intensities))
    i_max = float(np.max(intensities))

    i_bg = max(0.0, i_min * 0.5) if i_min > 0 else 0.0
    i_0 = max(1e-3, i_max - i_bg)
    # Estimate alpha from span
    dx = float(np.max(thicknesses) - np.min(thicknesses))
    alpha = max(0.01, float(np.log(max(1.1, i_max / max(1e-4, i_min))) / max(1e-3, dx)))

    p = np.array([i_0, alpha, i_bg], dtype=np.float64)
    lam = 1e-2

    for _ in range(max_iter):
        i0, a, bg = p
        a = max(1e-4, a)
        exp_term = np.exp(np.clip(-a * thicknesses, -50.0, 0.0))
        y_model = i0 * exp_term + bg
        res = intensities - y_model
        current_ss = np.sum(res**2)

        # Jacobian:
        # col 0: d/d(i0) = exp_term
        # col 1: d/d(a)  = -i0 * thicknesses * exp_term
        # col 2: d/d(bg) = 1
        J = np.empty((n, 3), dtype=np.float64)
        J[:, 0] = exp_term
        J[:, 1] = -i0 * thicknesses * exp_term
        J[:, 2] = 1.0

        JtJ = J.T @ J
        diag_JtJ = np.diag(np.diag(JtJ))
        try:
            delta_p = np.linalg.solve(JtJ + lam * diag_JtJ, J.T @ res)
        except np.linalg.LinAlgError:
            break

        p_new = p + delta_p
        p_new[0] = max(1e-4, p_new[0])
        p_new[1] = max(1e-5, p_new[1])
        p_new[2] = max(0.0, p_new[2])

        y_new = p_new[0] * np.exp(np.clip(-p_new[1] * thicknesses, -50.0, 0.0)) + p_new[2]
        new_ss = np.sum((intensities - y_new)**2)

        if new_ss < current_ss:
            p = p_new
            lam = max(1e-7, lam * 0.4)
            if abs(current_ss - new_ss) < 1e-9 * current_ss:
                break
        else:
            lam = min(1e4, lam * 4.0)

    # Final stats
    i0, a, bg = p
    exp_term = np.exp(np.clip(-a * thicknesses, -50.0, 0.0))
    y_final = i0 * exp_term + bg
    residuals = intensities - y_final
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((intensities - np.mean(intensities))**2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    dof = max(1, n - 3)
    s_err = np.sqrt(ss_res / dof)

    J = np.empty((n, 3), dtype=np.float64)
    J[:, 0] = exp_term
    J[:, 1] = -i0 * thicknesses * exp_term
    J[:, 2] = 1.0
    try:
        cov = s_err**2 * np.linalg.pinv(J.T @ J)
    except Exception:
        cov = np.zeros((3, 3))

    return p, cov, s_err, r_squared


def analyze_beer_lambert_data(
    thicknesses_cm: np.ndarray,
    transmitted_intensities: np.ndarray,
    expected_alpha: float = 0.50,
    expected_i0: float = 1.0,
) -> Dict[str, Any]:
    """Perform deterministic optical absorption fitting and background diagnosis.

    Args:
        thicknesses_cm: 1D array of sample thicknesses in cm.
        transmitted_intensities: 1D array of measured transmitted intensities.
        expected_alpha: Nominal absorption coefficient in cm^(-1).
        expected_i0: Nominal incident intensity.

    Returns:
        Structured dictionary of values, evidence, and plot coordinates.
    """
    n = len(thicknesses_cm)
    if n < 4:
        raise ValueError(f"Insufficient points for absorption fit: {n} < 4.")

    p, cov, s_err, r_squared = fit_beer_lambert(thicknesses_cm, transmitted_intensities)
    i0, alpha, bg = p

    sigma_i0 = float(np.sqrt(max(0.0, cov[0, 0])))
    sigma_alpha = float(np.sqrt(max(0.0, cov[1, 1])))
    sigma_bg = float(np.sqrt(max(0.0, cov[2, 2])))

    alpha_error_pct = float(((alpha - expected_alpha) / expected_alpha) * 100.0) if expected_alpha > 0 else 0.0
    bg_ratio = float(bg / i0) if i0 > 0 else 0.0

    has_high_absorption = bool(alpha_error_pct > 35.0)
    has_background_light = bool(bg_ratio > 0.05 and bg > 3.0 * sigma_bg)
    is_clean = bool(abs(alpha_error_pct) <= 15.0 and bg_ratio < 0.03 and r_squared > 0.96)

    sort_order = np.argsort(thicknesses_cm)
    sorted_x = thicknesses_cm[sort_order]
    y_fit = i0 * np.exp(-alpha * sorted_x) + bg
    y_expected = expected_i0 * np.exp(-expected_alpha * sorted_x)

    values = {
        "absorption_coefficient_alpha_cm_inv": {"value": float(alpha), "uncertainty": sigma_alpha},
        "incident_intensity_I0": {"value": float(i0), "uncertainty": sigma_i0},
        "background_intensity": {"value": float(bg), "uncertainty": sigma_bg},
        "alpha_error_pct": {"value": alpha_error_pct},
        "residual_rms": {"value": float(s_err)},
        "r_squared": {"value": float(r_squared)},
        "expected_alpha_cm_inv": {"value": float(expected_alpha)},
    }

    evidence = {
        "alpha_cm_inv": float(alpha),
        "alpha_error_pct": alpha_error_pct,
        "background_intensity": float(bg),
        "background_ratio": bg_ratio,
        "has_high_absorption": has_high_absorption,
        "has_background_light": has_background_light,
        "is_clean": is_clean,
        "r_squared": float(r_squared),
    }

    plot = {
        "x": sorted_x.tolist(),
        "y": transmitted_intensities[sort_order].tolist(),
        "fit": y_fit.tolist(),
        "expected": y_expected.tolist(),
        "x_label": "sample thickness (cm)",
        "y_label": "transmitted intensity",
    }

    return {
        "values": values,
        "evidence": evidence,
        "plot": plot,
    }
