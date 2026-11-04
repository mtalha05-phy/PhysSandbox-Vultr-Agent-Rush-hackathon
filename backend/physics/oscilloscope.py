"""Deterministic physics module for Oscilloscope Signal & Noise Analysis.

Implements FFT spectral analysis, fundamental frequency estimation,
power-line mains hum detection (50 Hz / 60 Hz and harmonics),
RMS noise calculation, and signal-to-noise ratio (SNR).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def analyze_oscilloscope_data(
    timestamps: np.ndarray,
    voltages: np.ndarray,
    expected_signal_hz: float = 1000.0,
    mains_hz: float = 50.0,
) -> Dict[str, Any]:
    """Perform deterministic FFT and spectral interference analysis.

    Args:
        timestamps: 1D array of sampling times in seconds.
        voltages: 1D array of measured voltages in Volts.
        expected_signal_hz: Expected fundamental signal frequency.
        mains_hz: Power-line frequency (default 50 Hz).

    Returns:
        Structured dictionary of values, evidence, and plot coordinates.
    """
    n = len(timestamps)
    if n < 16:
        raise ValueError(f"Insufficient waveform points: {n} < 16.")

    # Validate timestamps and compute sample rate
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        raise ValueError("Timestamps must be strictly monotonically increasing.")
    dt_mean = float(np.mean(dt))
    sample_rate_hz = 1.0 / dt_mean

    # Remove DC offset for spectral evaluation
    v_dc = float(np.mean(voltages))
    v_ac = voltages - v_dc

    # Compute single-sided FFT amplitude spectrum with Hanning window
    window = np.hanning(n)
    scale_factor = 2.0 / np.sum(window)
    fft_vals = np.fft.rfft(v_ac * window)
    freqs = np.fft.rfftfreq(n, d=dt_mean)
    amps = np.abs(fft_vals) * scale_factor

    # Frequency resolution
    delta_f = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # 1. Detect fundamental signal peak
    # Search above 10 Hz to ignore DC leakage
    valid_idx = np.where(freqs >= 10.0)[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(len(freqs))

    peak_idx = valid_idx[np.argmax(amps[valid_idx])]
    dom_freq = float(freqs[peak_idx])
    sig_amp = float(amps[peak_idx])

    # 2. Noise floor calculation (robust median of bins away from main signal and DC)
    away_mask = np.abs(freqs - dom_freq) > (5.0 * delta_f)
    away_mask &= freqs > 15.0
    noise_floor = float(np.median(amps[away_mask])) if np.any(away_mask) else 1e-6
    noise_floor = max(noise_floor, 1e-9)

    # 3. Detect power-line mains harmonics (1x, 2x, 3x, 4x, 5x)
    harmonics_dict = {}
    mains_total_amp_sq = 0.0
    harmonic_detected_count = 0
    odd_harmonics_present = False

    for multiplier in [1, 2, 3, 4, 5]:
        target_f = mains_hz * multiplier
        # Look within a small window around the harmonic frequency
        band = np.where(np.abs(freqs - target_f) <= max(2.5 * delta_f, 3.0))[0]
        if len(band) > 0:
            h_amp = float(np.max(amps[band]))
        else:
            h_amp = 0.0
        harmonics_dict[f"{int(target_f)}_Hz"] = h_amp
        mains_total_amp_sq += h_amp**2

        # Check if significant above noise
        if h_amp > 3.0 * noise_floor:
            harmonic_detected_count += 1
            if multiplier in [1, 3, 5]:
                odd_harmonics_present = True

    mains_total_amplitude = float(np.sqrt(mains_total_amp_sq))
    mains_snr = float(mains_total_amplitude / noise_floor) if noise_floor > 0 else 0.0
    mains_to_signal_ratio = float(mains_total_amplitude / sig_amp) if sig_amp > 1e-9 else 999.0

    # Check whether the signal matches expected frequency
    signal_at_expected_freq = bool(abs(dom_freq - expected_signal_hz) <= max(3.0 * delta_f, 15.0))

    # Waveform RMS
    waveform_rms = float(np.std(voltages))
    snr_db = float(20.0 * np.log10(sig_amp / (noise_floor * np.sqrt(2)))) if noise_floor > 0 else 60.0

    values = {
        "sample_rate_Hz": {"value": sample_rate_hz},
        "dominant_frequency_Hz": {"value": dom_freq},
        "signal_amplitude_V": {"value": sig_amp},
        "mains_harmonics_V": {"value": harmonics_dict},
        "mains_total_amplitude_V": {"value": mains_total_amplitude},
        "noise_floor_V": {"value": noise_floor},
        "waveform_rms_V": {"value": waveform_rms},
        "snr_dB": {"value": snr_db},
        "dc_offset_V": {"value": v_dc},
    }

    mains_50hz_amp = harmonics_dict.get("50_Hz", 0.0)
    has_mains_50hz = bool(mains_50hz_amp > 3.5 * noise_floor and mains_50hz_amp > 0.005 and (mains_50hz_amp / max(1e-6, sig_amp)) > 0.02)
    has_harmonics = bool(
        (harmonics_dict.get("100_Hz", 0.0) > 3.0 * noise_floor and harmonics_dict.get("100_Hz", 0.0) > 0.005) or
        (harmonics_dict.get("150_Hz", 0.0) > 3.0 * noise_floor and harmonics_dict.get("150_Hz", 0.0) > 0.005)
    )

    evidence = {
        "mains_to_signal_ratio": mains_to_signal_ratio,
        "mains_snr": mains_snr,
        "odd_harmonics_present": odd_harmonics_present,
        "harmonic_detected_count": harmonic_detected_count,
        "signal_at_expected_freq": signal_at_expected_freq,
        "has_mains_50hz": has_mains_50hz,
        "has_mains_harmonics": has_harmonics,
    }

    # Restrict plot to frequencies up to 2000 Hz or Nyquist
    max_plot_freq = min(float(freqs[-1]), max(expected_signal_hz * 2.5, 1200.0))
    plot_mask = freqs <= max_plot_freq

    # Subsample if too dense
    p_freqs = freqs[plot_mask]
    p_amps = amps[plot_mask]
    if len(p_freqs) > 1000:
        step = len(p_freqs) // 800
        p_freqs = p_freqs[::step]
        p_amps = p_amps[::step]

    plot = {
        "x": [round(float(f), 3) for f in p_freqs],
        "y": [round(float(a), 7) for a in p_amps],
        "x_label": "frequency (Hz)",
        "y_label": "amplitude (V)",
    }

    return {
        "values": values,
        "evidence": evidence,
        "plot": plot,
    }
