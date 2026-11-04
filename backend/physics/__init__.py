"""Unified Physics Registry for PhysSandbox.

Exposes deterministic mathematical analysis for all 6 experiments:
1. resistance_4wire: Low-resistance measurement & thermal EMF
2. waveform: Oscilloscope trace & mains hum noise
3. helmholtz: Helmholtz coil magnetic-field calibration
4. laser_profile: Laser beam profile & photodetector alignment
5. faraday: Faraday's law of electromagnetic induction
6. beer_lambert: Beer-Lambert law of optical absorption
"""

from typing import Any, Dict
import numpy as np

from backend.physics.resistance import analyze_resistance_data
from backend.physics.oscilloscope import analyze_oscilloscope_data
from backend.physics.helmholtz import analyze_helmholtz_data
from backend.physics.laser_profile import analyze_laser_profile_data
from backend.physics.faraday import analyze_faraday_data
from backend.physics.beer_lambert import analyze_beer_lambert_data


EXPERIMENT_METADATA = {
    "resistance_4wire": {
        "name": "Low-resistance measurement",
        "description": "Ohm's law verification and thermal EMF offset investigation",
        "expected_columns": ["current_A", "voltage_V"],
        "default_expected": {"resistance_ohm": 0.010},
    },
    "waveform": {
        "name": "Oscilloscope signal analysis",
        "description": "Time-domain waveform and FFT spectral noise detection",
        "expected_columns": ["time_s", "voltage_V"],
        "default_expected": {"signal_hz": 1000.0, "mains_hz": 50.0},
    },
    "helmholtz": {
        "name": "Helmholtz coil magnetic field",
        "description": "Theoretical coil center field comparison and calibration",
        "expected_columns": ["current_A", "measured_B_uT"],
        "default_expected": {"radius_m": 0.15, "turns": 100},
    },
    "laser_profile": {
        "name": "Laser beam profile & alignment",
        "description": "Gaussian intensity beam fitting and photodetector transverse alignment",
        "expected_columns": ["position_mm", "detector_voltage_mV"],
        "default_expected": {"expected_center_mm": 0.0, "expected_peak_mV": 500.0},
    },
    "faraday": {
        "name": "Faraday induction law",
        "description": "Electromagnetic induction and discrete flux derivative analysis",
        "expected_columns": ["time_s", "magnetic_flux_Wb", "induced_voltage_V"],
        "default_expected": {"turns": 200},
    },
    "beer_lambert": {
        "name": "Beer-Lambert optical absorption",
        "description": "Exponential light intensity attenuation vs thickness",
        "expected_columns": ["thickness_cm", "transmitted_intensity"],
        "default_expected": {"expected_alpha": 0.50, "expected_i0": 1.0},
    },
}


def run_physics_analysis(
    experiment_id: str,
    data_dict: Dict[str, np.ndarray],
    expected_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch data to the deterministic physics engine."""
    if experiment_id not in EXPERIMENT_METADATA:
        raise ValueError(f"Unknown experiment: {experiment_id}. Supported: {list(EXPERIMENT_METADATA.keys())}")

    if experiment_id == "resistance_4wire":
        currents = data_dict["current_A"]
        voltages = data_dict["voltage_V"]
        expected_r = float(expected_params.get("resistance_ohm", 0.010))
        return analyze_resistance_data(currents, voltages, expected_r=expected_r)

    elif experiment_id == "waveform":
        times = data_dict["time_s"]
        voltages = data_dict["voltage_V"]
        sig_hz = float(expected_params.get("signal_hz", 1000.0))
        mains_hz = float(expected_params.get("mains_hz", 50.0))
        return analyze_oscilloscope_data(times, voltages, expected_signal_hz=sig_hz, mains_hz=mains_hz)

    elif experiment_id == "helmholtz":
        currents = data_dict["current_A"]
        b_field = data_dict["measured_B_uT"]
        radius = float(expected_params.get("radius_m", 0.15))
        turns = int(expected_params.get("turns", 100))
        return analyze_helmholtz_data(currents, b_field, radius_m=radius, turns=turns)

    elif experiment_id == "laser_profile":
        positions = data_dict["position_mm"]
        voltages = data_dict["detector_voltage_mV"]
        center = float(expected_params.get("expected_center_mm", 0.0))
        peak = float(expected_params.get("expected_peak_mV", 500.0))
        return analyze_laser_profile_data(positions, voltages, expected_center_mm=center, expected_peak_mv=peak)

    elif experiment_id == "faraday":
        times = data_dict["time_s"]
        flux = data_dict["magnetic_flux_Wb"]
        voltages = data_dict["induced_voltage_V"]
        turns = int(expected_params.get("turns", 200))
        return analyze_faraday_data(times, flux, voltages, turns=turns)

    elif experiment_id == "beer_lambert":
        thicknesses = data_dict["thickness_cm"]
        intensities = data_dict["transmitted_intensity"]
        alpha = float(expected_params.get("expected_alpha", 0.50))
        i0 = float(expected_params.get("expected_i0", 1.0))
        return analyze_beer_lambert_data(thicknesses, intensities, expected_alpha=alpha, expected_i0=i0)

    raise ValueError(f"Unhandled experiment: {experiment_id}")
