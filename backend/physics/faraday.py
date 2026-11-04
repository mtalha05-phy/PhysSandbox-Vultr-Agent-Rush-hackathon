"""Deterministic physics module for Faraday's Law of Electromagnetic Induction.

Mathematical Model:
    EMF_predicted = -N * d(Phi_B) / dt
    Where:
        N     = number of turns in the coil
        Phi_B = magnetic flux per turn in Webers (Wb)
        t     = time in seconds

Discrete Numerical Differentiation:
    dPhi/dt[i] = (Phi[i+1] - Phi[i-1]) / (t[i+1] - t[i-1])  (central difference)

Linear Comparison Model:
    V_measured = gamma * EMF_predicted + V_offset
    Where:
        gamma    = amplitude response ratio (ideal = 1.0)
        V_offset = constant instrumentation/baseline offset in Volts (ideal = 0.0)
"""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_discrete_flux_derivative(
    times: np.ndarray,
    flux: np.ndarray,
    smooth_window: int = 3,
) -> np.ndarray:
    """Compute numerical time derivative of magnetic flux with optional smoothing."""
    n = len(times)
    # Apply moving average smoothing to damp high-frequency discretization noise
    if smooth_window > 1 and n >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        # Pad with edge values to preserve length
        pad_len = smooth_window // 2
        flux_padded = np.pad(flux, pad_len, mode="edge")
        flux_smooth = np.convolve(flux_padded, kernel, mode="valid")
        if len(flux_smooth) > n:
            flux_smooth = flux_smooth[:n]
    else:
        flux_smooth = flux

    dphi_dt = np.gradient(flux_smooth, times)
    return dphi_dt


def analyze_faraday_data(
    times: np.ndarray,
    flux_wb: np.ndarray,
    measured_voltage_v: np.ndarray,
    turns: int = 200,
) -> Dict[str, Any]:
    """Perform deterministic induction analysis and compare measured vs predicted EMF.

    Args:
        times: 1D array of timestamps in seconds.
        flux_wb: 1D array of magnetic flux per turn in Webers (Wb).
        measured_voltage_v: 1D array of induced coil voltage in Volts.
        turns: Coil turn count N (default 200).

    Returns:
        Structured dictionary of values, evidence, and plot coordinates.
    """
    n = len(times)
    if n < 5:
        raise ValueError(f"Insufficient induction time points: {n} < 5.")

    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Timestamps must be strictly increasing.")

    # 1. Compute predicted EMF: epsilon = -N * d(Phi_B)/dt
    dphi_dt = compute_discrete_flux_derivative(times, flux_wb)
    emf_pred = -float(turns) * dphi_dt

    # 2. Regress measured voltage on predicted EMF: V_meas = gamma * EMF_pred + V_offset
    A = np.vstack([emf_pred, np.ones_like(emf_pred)]).T
    (gamma, v_offset), ss_res_arr, rank, s = np.linalg.lstsq(A, measured_voltage_v, rcond=None)

    y_fit = gamma * emf_pred + v_offset
    residuals = measured_voltage_v - y_fit

    ss_res = float(np.sum(residuals**2))
    dof = max(1, n - 2)
    s_err = np.sqrt(ss_res / dof)

    try:
        inv_ata = np.linalg.inv(A.T @ A)
        cov = s_err**2 * inv_ata
        sigma_gamma = float(np.sqrt(max(0.0, cov[0, 0])))
        sigma_v_offset = float(np.sqrt(max(0.0, cov[1, 1])))
    except np.linalg.LinAlgError:
        sigma_gamma = 0.0
        sigma_v_offset = 0.0

    ss_tot = float(np.sum((measured_voltage_v - np.mean(measured_voltage_v))**2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    offset_sigma = float(abs(v_offset) / sigma_v_offset) if sigma_v_offset > 1e-12 else 0.0
    amplitude_error_pct = float(abs(gamma - 1.0) * 100.0)

    # Fault classification
    has_voltage_offset = bool(offset_sigma > 3.0 and abs(v_offset) > 0.015)
    has_reduced_signal = bool(gamma < 0.70 and r_squared > 0.75)
    is_normal_induction = bool(0.85 <= gamma <= 1.15 and not has_voltage_offset and r_squared > 0.85)

    values = {
        "amplitude_ratio_gamma": {"value": float(gamma), "uncertainty": sigma_gamma},
        "voltage_offset_V": {"value": float(v_offset), "uncertainty": sigma_v_offset},
        "amplitude_error_pct": {"value": amplitude_error_pct},
        "residual_rms_V": {"value": float(s_err)},
        "r_squared": {"value": float(r_squared)},
        "coil_turns": {"value": int(turns)},
    }

    evidence = {
        "amplitude_ratio": float(gamma),
        "voltage_offset_V": float(v_offset),
        "offset_sigma": offset_sigma,
        "has_voltage_offset": has_voltage_offset,
        "has_reduced_signal": has_reduced_signal,
        "is_normal_induction": is_normal_induction,
        "r_squared": float(r_squared),
    }

    # Downsample plot points if series is very dense
    step = max(1, n // 500)
    plot = {
        "x": times[::step].tolist(),
        "y": measured_voltage_v[::step].tolist(),
        "fit": y_fit[::step].tolist(),
        "expected": emf_pred[::step].tolist(),
        "x_label": "time (s)",
        "y_label": "induced voltage (V)",
    }

    return {
        "values": values,
        "evidence": evidence,
        "plot": plot,
    }
