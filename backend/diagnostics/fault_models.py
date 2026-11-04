"""Fault models, Bayesian hypothesis scoring, and Value-of-Information test ranking.

Supports all 6 PhysSandbox experiments.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def score_hypotheses(
    priors: Dict[str, float],
    likelihoods: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Compute normalized posterior probabilities from priors and likelihoods."""
    unnorm = {}
    for cid, prior in priors.items():
        like = likelihoods.get(cid, 0.05)
        unnorm[cid] = prior * max(1e-5, like)

    total = sum(unnorm.values())
    if total <= 0:
        total = 1.0

    posteriors = {cid: val / total for cid, val in unnorm.items()}
    return posteriors


def evaluate_resistance_faults(evidence: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    offset_sigma = evidence.get("offset_sigma", 0.0)
    slope_error_pct = evidence.get("slope_error_pct", 0.0)
    is_nonlinear = evidence.get("is_nonlinear", False)
    has_reversal = evidence.get("has_current_reversal", False)
    survives_rev = evidence.get("offset_survives_reversal", False)

    priors = {
        "thermoelectric_offset": 0.35,
        "meter_zero_offset": 0.25,
        "nonlinear_contact": 0.15,
        "calibration_gain_error": 0.10,
        "no_fault": 0.10,
        "unknown_cause": 0.05,
    }

    likes = {}
    if is_nonlinear:
        likes["nonlinear_contact"] = 0.90
        likes["no_fault"] = 0.01
        likes["thermoelectric_offset"] = 0.15
        likes["meter_zero_offset"] = 0.10
        likes["calibration_gain_error"] = 0.10
    elif offset_sigma > 4.0:
        if has_reversal:
            if survives_rev:
                likes["thermoelectric_offset"] = 0.85
                likes["meter_zero_offset"] = 0.50
            else:
                likes["thermoelectric_offset"] = 0.10
                likes["meter_zero_offset"] = 0.10
                likes["no_fault"] = 0.70
        else:
            likes["thermoelectric_offset"] = 0.70
            likes["meter_zero_offset"] = 0.40
        likes["no_fault"] = 0.02
        likes["nonlinear_contact"] = 0.05
        likes["calibration_gain_error"] = 0.05
    elif abs(slope_error_pct) > 5.0:
        likes["calibration_gain_error"] = 0.80
        likes["no_fault"] = 0.05
        likes["thermoelectric_offset"] = 0.10
    else:
        likes["no_fault"] = 0.85
        likes["thermoelectric_offset"] = 0.05
        likes["meter_zero_offset"] = 0.05
        likes["nonlinear_contact"] = 0.03

    posteriors = score_hypotheses(priors, likes)

    cause_defs = [
        {"id": "thermoelectric_offset", "name": "Thermoelectric (thermal EMF) offset at a contact", "typical_size": "0.2 µV/°C up to ~1000 µV/°C at oxidized junctions"},
        {"id": "meter_zero_offset", "name": "Voltmeter zero offset", "typical_size": "Microvolts baseline shift on instrument"},
        {"id": "nonlinear_contact", "name": "Nonlinear contact resistance / heating effect", "typical_size": "Curvature observed at higher currents"},
        {"id": "calibration_gain_error", "name": "Calibration (gain) error in meter or current source", "typical_size": "Consistent percentage slope discrepancy"},
        {"id": "no_fault", "name": "No fault found (Ohmic behavior confirmed)", "typical_size": None},
        {"id": "unknown_cause", "name": "Unknown cause", "typical_size": None},
    ]

    causes = []
    for c in cause_defs:
        causes.append({**c, "probability": posteriors.get(c["id"], 0.0)})
    causes.sort(key=lambda x: x["probability"], reverse=True)

    next_tests = [
        {
            "id": "current_reversal",
            "name": "Reverse the current and repeat the sweep (±I)",
            "how": "Swap current leads or supply negative currents. Thermal EMF maintains polarity while Ohmic drop reverses.",
            "minutes": 4,
            "info_gain_bits": 1.25,
            "bits_per_minute": 0.31,
            "safety": "Switch current source to zero before swapping leads.",
            "predicted_outcomes": {"thermoelectric_offset": "offset_stays", "no_fault": "no_offset", "nonlinear_contact": "asymmetry"},
        },
        {
            "id": "zero_current_reading",
            "name": "Read voltmeter at zero current (on sample vs shorted at meter)",
            "how": "Set I = 0 A. Record voltage across sample, then short voltmeter leads directly.",
            "minutes": 3,
            "info_gain_bits": 1.15,
            "bits_per_minute": 0.38,
            "safety": None,
            "predicted_outcomes": {"thermoelectric_offset": "offset_only_on_sample", "meter_zero_offset": "offset_both", "no_fault": "zero_both"},
        },
        {
            "id": "narrow_current_sweep",
            "name": "Repeat sweep across narrower current range",
            "how": "Measure at low currents (e.g. 1 mA to 20 mA) to determine whether nonlinearity is caused by Joulean heating.",
            "minutes": 4,
            "info_gain_bits": 0.90,
            "bits_per_minute": 0.22,
            "safety": None,
            "predicted_outcomes": {"nonlinear_contact": "becomes_linear", "no_fault": "linear"},
        },
    ]

    top = causes[0]
    status = "confirmed" if (top["probability"] > 0.75 and top["id"] != "no_fault") else ("clear" if top["id"] == "no_fault" and top["probability"] > 0.70 else "investigating")

    if top["id"] == "thermoelectric_offset":
        exp = f"Possible thermal EMF offset detected. The fitted intercept ({evidence.get('offset_V', 0)*1e6:.1f} µV) differs from zero by {offset_sigma:.1f}σ. Current reversal is recommended for confirmation."
    elif top["id"] == "nonlinear_contact":
        exp = "Nonlinear V-I relationship detected. Contact resistance, thermal heating, or instrument compliance limits may be affecting the measurement."
    elif top["id"] == "no_fault":
        exp = "Healthy linear Ohmic response. Measured resistance is consistent with expected value within experimental uncertainty."
    else:
        exp = f"Suspected {top['name'].lower()} with {round(top['probability']*100)}% probability. Recommend follow-up test to isolate cause."

    return causes, next_tests, status, exp


def evaluate_waveform_faults(evidence: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    mains_snr = evidence.get("mains_snr", 0.0)
    mains_ratio = evidence.get("mains_to_signal_ratio", 0.0)
    odd_harmonics = evidence.get("odd_harmonics_present", False)
    has_harmonics = evidence.get("has_mains_harmonics", False)
    correct_freq = evidence.get("signal_at_expected_freq", True)

    priors = {
        "ground_loop": 0.40,
        "aliasing": 0.25,
        "harmonic_interference": 0.20,
        "no_fault": 0.10,
        "unknown_cause": 0.05,
    }

    likes = {}
    if not correct_freq:
        likes["aliasing"] = 0.85
        likes["ground_loop"] = 0.10
        likes["no_fault"] = 0.02
    elif mains_snr > 10.0 or mains_ratio > 0.10:
        if odd_harmonics or has_harmonics:
            likes["ground_loop"] = 0.75
            likes["harmonic_interference"] = 0.60
        else:
            likes["ground_loop"] = 0.60
            likes["harmonic_interference"] = 0.30
        likes["no_fault"] = 0.01
        likes["aliasing"] = 0.05
    else:
        likes["no_fault"] = 0.85
        likes["ground_loop"] = 0.04
        likes["aliasing"] = 0.02
        likes["harmonic_interference"] = 0.02

    posteriors = score_hypotheses(priors, likes)

    cause_defs = [
        {"id": "ground_loop", "name": "Ground loop picking up mains hum", "typical_size": "Mains frequency (50/60 Hz) and odd harmonics"},
        {"id": "aliasing", "name": "Aliasing (sample rate too low for the signal)", "typical_size": "Apparent signal shifted to lower frequency fold"},
        {"id": "harmonic_interference", "name": "Power-supply or inductive harmonic interference", "typical_size": "Strong 100 Hz / 150 Hz components"},
        {"id": "no_fault", "name": "No fault found (Clean spectral trace)", "typical_size": None},
        {"id": "unknown_cause", "name": "Unknown cause", "typical_size": None},
    ]

    causes = [{**c, "probability": posteriors.get(c["id"], 0.0)} for c in cause_defs]
    causes.sort(key=lambda x: x["probability"], reverse=True)

    next_tests = [
        {
            "id": "single_point_ground",
            "name": "Connect instruments to a single-point grounding configuration",
            "how": "Power all instrumentation from a single power strip; remove ground loops across BNC shields.",
            "minutes": 5,
            "info_gain_bits": 1.45,
            "bits_per_minute": 0.29,
            "safety": "Never lift protective safety earth from power plugs.",
            "predicted_outcomes": {"ground_loop": "hum_eliminated", "harmonic_interference": "hum_remains", "no_fault": "clean"},
        },
        {
            "id": "raise_sample_rate",
            "name": "Re-record trace at 10x higher sampling frequency",
            "how": "Increase oscilloscope timebase sampling rate to verify Nyquist criterion.",
            "minutes": 2,
            "info_gain_bits": 0.93,
            "bits_per_minute": 0.46,
            "safety": None,
            "predicted_outcomes": {"aliasing": "frequency_shifts", "ground_loop": "frequency_constant", "no_fault": "constant"},
        },
        {
            "id": "twist_and_move_leads",
            "name": "Twist signal leads and isolate from power cables",
            "how": "Shorten lead loop area to minimize inductive pickup from nearby mains transformers.",
            "minutes": 4,
            "info_gain_bits": 1.10,
            "bits_per_minute": 0.27,
            "safety": None,
            "predicted_outcomes": {"harmonic_interference": "harmonics_drop", "ground_loop": "constant", "no_fault": "clean"},
        },
    ]

    top = causes[0]
    status = "confirmed" if (top["probability"] > 0.75 and top["id"] != "no_fault") else ("clear" if top["id"] == "no_fault" and top["probability"] > 0.70 else "investigating")

    if top["id"] == "ground_loop":
        exp = "Periodic power-line spectral component detected. While consistent with ground loops, electromagnetic coupling or power-supply ripple are plausible alternatives requiring single-point grounding verification."
    elif top["id"] == "aliasing":
        exp = "Dominant frequency deviates from expectation; potential aliasing caused by insufficient sampling rate relative to signal bandwidth."
    elif top["id"] == "no_fault":
        exp = "Clean waveform. Signal spectrum is dominated by fundamental frequency with low residual noise."
    else:
        exp = f"Suspected {top['name'].lower()} with {round(top['probability']*100)}% probability."

    return causes, next_tests, status, exp


def evaluate_helmholtz_faults(evidence: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    cal_err = evidence.get("calibration_error_pct", 0.0)
    offset_ut = evidence.get("offset_uT", 0.0)
    offset_sigma = evidence.get("offset_sigma", 0.0)
    has_gain = evidence.get("has_gain_error", False)
    has_offset = evidence.get("has_magnetic_offset", False)

    priors = {
        "ambient_magnetic_offset": 0.40,
        "sensor_zero_offset": 0.25,
        "sensor_gain_error": 0.20,
        "no_fault": 0.10,
        "unknown_cause": 0.05,
    }

    likes = {}
    if has_gain and cal_err > 8.0:
        likes["sensor_gain_error"] = 0.85
        likes["ambient_magnetic_offset"] = 0.20
        likes["no_fault"] = 0.02
    elif has_offset or abs(offset_ut) > 10.0:
        likes["ambient_magnetic_offset"] = 0.75
        likes["sensor_zero_offset"] = 0.60
        likes["sensor_gain_error"] = 0.15
        likes["no_fault"] = 0.02
    else:
        likes["no_fault"] = 0.85
        likes["ambient_magnetic_offset"] = 0.05
        likes["sensor_zero_offset"] = 0.05
        likes["sensor_gain_error"] = 0.05

    posteriors = score_hypotheses(priors, likes)

    cause_defs = [
        {"id": "ambient_magnetic_offset", "name": "Ambient Earth magnetic field or nearby ferromagnetic materials", "typical_size": "30 µT to 60 µT background field"},
        {"id": "sensor_zero_offset", "name": "Hall/fluxgate magnetometer sensor zero-offset", "typical_size": "Constant null-offset in sensor bridge"},
        {"id": "sensor_gain_error", "name": "Sensor gain / calibration scaling error", "typical_size": "Systematic slope percentage deviation"},
        {"id": "no_fault", "name": "No fault found (Calibrated within theoretical Helmholtz expectation)", "typical_size": None},
        {"id": "unknown_cause", "name": "Unknown cause", "typical_size": None},
    ]

    causes = [{**c, "probability": posteriors.get(c["id"], 0.0)} for c in cause_defs]
    causes.sort(key=lambda x: x["probability"], reverse=True)

    next_tests = [
        {
            "id": "zero_current_ambient",
            "name": "Measure background field at zero current (I = 0 A)",
            "how": "Switch off power supply and record magnetometer reading. Reverse sensor orientation by 180° to separate ambient Earth field from sensor electronic offset.",
            "minutes": 3,
            "info_gain_bits": 1.30,
            "bits_per_minute": 0.43,
            "safety": None,
            "predicted_outcomes": {"ambient_magnetic_offset": "reading_inverts_on_rotation", "sensor_zero_offset": "reading_constant_on_rotation", "no_fault": "zero"},
        },
        {
            "id": "calibrated_magnetometer_check",
            "name": "Cross-check against a calibrated reference magnetometer",
            "how": "Insert a certified reference probe at the coil geometric center and repeat calibration sweep.",
            "minutes": 6,
            "info_gain_bits": 1.35,
            "bits_per_minute": 0.22,
            "safety": None,
            "predicted_outcomes": {"sensor_gain_error": "slope_matches_reference", "no_fault": "identical"},
        },
    ]

    top = causes[0]
    status = "confirmed" if (top["probability"] > 0.75 and top["id"] != "no_fault") else ("clear" if top["id"] == "no_fault" and top["probability"] > 0.70 else "investigating")

    if top["id"] == "ambient_magnetic_offset":
        exp = f"Magnetic field offset detected (B0 = {offset_ut:.1f} µT). This is consistent with ambient Earth magnetic field or local magnetic objects. Zero-current measurement with probe rotation is recommended."
    elif top["id"] == "sensor_gain_error":
        exp = f"Calibration slope differs by {cal_err:.1f}% from the theoretical Helmholtz constant. Sensor sensitivity gain error or coil geometry discrepancies suspected."
    elif top["id"] == "no_fault":
        exp = "Coil calibration matches theoretical Helmholtz center field formula within experimental limits."
    else:
        exp = f"Suspected {top['name'].lower()} with {round(top['probability']*100)}% probability."

    return causes, next_tests, status, exp


def evaluate_laser_profile_faults(evidence: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    offset_mm = evidence.get("alignment_offset_mm", 0.0)
    peak_ratio = evidence.get("peak_signal_ratio", 1.0)
    is_misaligned = evidence.get("is_misaligned", False)
    is_low = evidence.get("is_low_signal", False)
    dark_v = evidence.get("dark_voltage_mV", 0.0)

    priors = {
        "transverse_misalignment": 0.40,
        "reduced_laser_power": 0.25,
        "high_dark_voltage": 0.15,
        "no_fault": 0.15,
        "unknown_cause": 0.05,
    }

    likes = {}
    if is_misaligned and offset_mm > 0.5:
        likes["transverse_misalignment"] = 0.85
        likes["reduced_laser_power"] = 0.20
        likes["no_fault"] = 0.01
    elif is_low:
        likes["reduced_laser_power"] = 0.75
        likes["transverse_misalignment"] = 0.40
        likes["no_fault"] = 0.02
    elif dark_v > 80.0:
        likes["high_dark_voltage"] = 0.80
        likes["no_fault"] = 0.05
    else:
        likes["no_fault"] = 0.85
        likes["transverse_misalignment"] = 0.05
        likes["reduced_laser_power"] = 0.05

    posteriors = score_hypotheses(priors, likes)

    cause_defs = [
        {"id": "transverse_misalignment", "name": "Photodetector transverse misalignment", "typical_size": "Fitted beam center shifted from nominal target axis"},
        {"id": "reduced_laser_power", "name": "Reduced laser output or optical window contamination", "typical_size": "Peak voltage significantly depressed across profile"},
        {"id": "high_dark_voltage", "name": "Ambient background illumination or photodetector dark current", "typical_size": "Elevated baseline voltage at large beam offsets"},
        {"id": "no_fault", "name": "No fault found (Beam aligned and nominal power confirmed)", "typical_size": None},
        {"id": "unknown_cause", "name": "Unknown cause", "typical_size": None},
    ]

    causes = [{**c, "probability": posteriors.get(c["id"], 0.0)} for c in cause_defs]
    causes.sort(key=lambda x: x["probability"], reverse=True)

    next_tests = [
        {
            "id": "adjust_transverse_stage",
            "name": "Adjust translation stage toward fitted beam center",
            "how": "Translate detector stage by the computed offset towards the fitted Gaussian peak to restore optical alignment.",
            "minutes": 4,
            "info_gain_bits": 1.40,
            "bits_per_minute": 0.35,
            "safety": "Wear appropriate laser safety goggles during optical alignment.",
            "predicted_outcomes": {"transverse_misalignment": "peak_centered", "reduced_laser_power": "peak_remains_low", "no_fault": "aligned"},
        },
        {
            "id": "measure_laser_optical_power",
            "name": "Measure laser output with calibrated optical power meter",
            "how": "Place thermal/semiconductor power sensor directly in front of laser aperture to distinguish emitter degradation from optical receiver issues.",
            "minutes": 5,
            "info_gain_bits": 1.20,
            "bits_per_minute": 0.24,
            "safety": None,
            "predicted_outcomes": {"reduced_laser_power": "power_low", "transverse_misalignment": "power_nominal", "no_fault": "power_nominal"},
        },
    ]

    top = causes[0]
    status = "confirmed" if (top["probability"] > 0.75 and top["id"] != "no_fault") else ("clear" if top["id"] == "no_fault" and top["probability"] > 0.70 else "investigating")

    if top["id"] == "transverse_misalignment":
        exp = f"Gaussian beam profile center is displaced by {offset_mm:.2f} mm from the target position. Stage alignment toward fitted center is recommended."
    elif top["id"] == "reduced_laser_power":
        exp = f"Measured optical peak is {round((1.0 - peak_ratio)*100)}% below expected power. Low signal can stem from laser diode aging, dirty optics, or clipping; optical power meter check recommended."
    elif top["id"] == "no_fault":
        exp = "Proper Gaussian profile alignment verified. Center position and peak intensity match specifications."
    else:
        exp = f"Suspected {top['name'].lower()} with {round(top['probability']*100)}% probability."

    return causes, next_tests, status, exp


def evaluate_faraday_faults(evidence: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    gamma = evidence.get("amplitude_ratio", 1.0)
    v_off = evidence.get("voltage_offset_V", 0.0)
    has_offset = evidence.get("has_voltage_offset", False)
    has_reduced = evidence.get("has_reduced_signal", False)

    priors = {
        "amplifier_dc_offset": 0.35,
        "coil_turn_mismatch": 0.30,
        "signal_attenuation": 0.20,
        "no_fault": 0.10,
        "unknown_cause": 0.05,
    }

    likes = {}
    if has_offset:
        likes["amplifier_dc_offset"] = 0.85
        likes["coil_turn_mismatch"] = 0.15
        likes["no_fault"] = 0.02
    elif has_reduced or gamma < 0.70:
        likes["coil_turn_mismatch"] = 0.75
        likes["signal_attenuation"] = 0.60
        likes["amplifier_dc_offset"] = 0.10
        likes["no_fault"] = 0.02
    else:
        likes["no_fault"] = 0.85
        likes["amplifier_dc_offset"] = 0.05
        likes["coil_turn_mismatch"] = 0.05

    posteriors = score_hypotheses(priors, likes)

    cause_defs = [
        {"id": "amplifier_dc_offset", "name": "Amplifier or instrumentation DC offset", "typical_size": "Nonzero voltage baseline during static flux conditions"},
        {"id": "coil_turn_mismatch", "name": "Coil turn count or area specification mismatch", "typical_size": "Systematic amplitude reduction by ratio of turn counts"},
        {"id": "signal_attenuation", "name": "Input impedance loading or low-pass filter attenuation", "typical_size": "Amplitude reduction without baseline shift"},
        {"id": "no_fault", "name": "Normal induction (Measured EMF follows -N dPhi/dt)", "typical_size": None},
        {"id": "unknown_cause", "name": "Unknown cause", "typical_size": None},
    ]

    causes = [{**c, "probability": posteriors.get(c["id"], 0.0)} for c in cause_defs]
    causes.sort(key=lambda x: x["probability"], reverse=True)

    next_tests = [
        {
            "id": "static_flux_baseline",
            "name": "Measure coil voltage baseline with static flux (dPhi/dt = 0)",
            "how": "Hold magnetic flux constant and record voltage to quantify electronics DC zero offset.",
            "minutes": 3,
            "info_gain_bits": 1.25,
            "bits_per_minute": 0.42,
            "safety": None,
            "predicted_outcomes": {"amplifier_dc_offset": "baseline_nonzero", "coil_turn_mismatch": "zero_baseline", "no_fault": "zero_baseline"},
        },
        {
            "id": "verify_coil_geometry",
            "name": "Verify coil turn count and test with calibrated sinusoidal flux",
            "how": "Apply known sinusoidal field to independently verify turn count N and probe loading.",
            "minutes": 6,
            "info_gain_bits": 1.30,
            "bits_per_minute": 0.22,
            "safety": None,
            "predicted_outcomes": {"coil_turn_mismatch": "confirmed_ratio", "signal_attenuation": "frequency_dependent", "no_fault": "unity_ratio"},
        },
    ]

    top = causes[0]
    status = "confirmed" if (top["probability"] > 0.75 and top["id"] != "no_fault") else ("clear" if top["id"] == "no_fault" and top["probability"] > 0.70 else "investigating")

    if top["id"] == "amplifier_dc_offset":
        exp = f"Induced voltage exhibits a constant DC offset (V0 = {v_off*1e3:.1f} mV) differing from the negative flux derivative. Baseline test under static flux recommended."
    elif top["id"] == "coil_turn_mismatch":
        exp = f"Measured voltage amplitude is {round((1.0 - gamma)*100)}% lower than theoretical prediction (-N dPhi/dt). Verify coil turns and probe impedance."
    elif top["id"] == "no_fault":
        exp = "Faraday induction verified. Induced voltage accurately tracks the time rate of change of magnetic flux."
    else:
        exp = f"Suspected {top['name'].lower()} with {round(top['probability']*100)}% probability."

    return causes, next_tests, status, exp


def evaluate_beer_lambert_faults(evidence: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    alpha_err = evidence.get("alpha_error_pct", 0.0)
    bg_ratio = evidence.get("background_ratio", 0.0)
    has_high = evidence.get("has_high_absorption", False)
    has_bg = evidence.get("has_background_light", False)

    priors = {
        "background_light_leakage": 0.40,
        "increased_medium_absorption": 0.30,
        "detector_saturation": 0.15,
        "no_fault": 0.10,
        "unknown_cause": 0.05,
    }

    likes = {}
    if has_bg or bg_ratio > 0.05:
        likes["background_light_leakage"] = 0.85
        likes["increased_medium_absorption"] = 0.15
        likes["no_fault"] = 0.02
    elif has_high or alpha_err > 35.0:
        likes["increased_medium_absorption"] = 0.80
        likes["background_light_leakage"] = 0.15
        likes["no_fault"] = 0.02
    else:
        likes["no_fault"] = 0.85
        likes["background_light_leakage"] = 0.05
        likes["increased_medium_absorption"] = 0.05

    posteriors = score_hypotheses(priors, likes)

    cause_defs = [
        {"id": "background_light_leakage", "name": "Stray ambient light or photodetector dark current offset", "typical_size": "Intensity curves plateau at larger sample thicknesses"},
        {"id": "increased_medium_absorption", "name": "Higher sample concentration or elevated absorption coefficient", "typical_size": "Steeper exponential attenuation slope"},
        {"id": "detector_saturation", "name": "Photodetector saturation or non-exponential scattering", "typical_size": "Clipping at low thickness or non-linear residuals"},
        {"id": "no_fault", "name": "Normal optical absorption (Beer-Lambert exponential decay confirmed)", "typical_size": None},
        {"id": "unknown_cause", "name": "Unknown cause", "typical_size": None},
    ]

    causes = [{**c, "probability": posteriors.get(c["id"], 0.0)} for c in cause_defs]
    causes.sort(key=lambda x: x["probability"], reverse=True)

    next_tests = [
        {
            "id": "blocked_source_dark_measurement",
            "name": "Block optical source and record detector background reading",
            "how": "Shutter the light source completely at the sample compartment and record the residual intensity to isolate ambient background.",
            "minutes": 2,
            "info_gain_bits": 1.35,
            "bits_per_minute": 0.68,
            "safety": None,
            "predicted_outcomes": {"background_light_leakage": "reading_nonzero", "increased_medium_absorption": "reading_zero", "no_fault": "reading_zero"},
        },
        {
            "id": "multi_thickness_sweep",
            "name": "Measure additional thickness increments to test log-linearity",
            "how": "Record at intermediate sample thicknesses to confirm pure single-exponential attenuation versus multi-scattering.",
            "minutes": 5,
            "info_gain_bits": 1.10,
            "bits_per_minute": 0.22,
            "safety": None,
            "predicted_outcomes": {"increased_medium_absorption": "remains_exponential", "detector_saturation": "non_linear_log", "no_fault": "linear_log"},
        },
    ]

    top = causes[0]
    status = "confirmed" if (top["probability"] > 0.75 and top["id"] != "no_fault") else ("clear" if top["id"] == "no_fault" and top["probability"] > 0.70 else "investigating")

    if top["id"] == "background_light_leakage":
        exp = f"Transmitted intensity fails to decay toward zero at large thicknesses (background ratio {round(bg_ratio*100)}%). Stray ambient light leakage or sensor dark offset suspected; blocked-source test recommended."
    elif top["id"] == "increased_medium_absorption":
        exp = f"Absorption coefficient is {alpha_err:.1f}% higher than nominal expectation. Verify sample concentration or solvent purity."
    elif top["id"] == "no_fault":
        exp = "Optical transmission strictly follows the Beer-Lambert exponential law with negligible background."
    else:
        exp = f"Suspected {top['name'].lower()} with {round(top['probability']*100)}% probability."

    return causes, next_tests, status, exp


def evaluate_diagnostic_report(
    experiment_id: str,
    evidence: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    """Unified entry point for hypothesis evaluation across all 6 experiments."""
    if experiment_id == "resistance_4wire":
        return evaluate_resistance_faults(evidence)
    elif experiment_id == "waveform":
        return evaluate_waveform_faults(evidence)
    elif experiment_id == "helmholtz":
        return evaluate_helmholtz_faults(evidence)
    elif experiment_id == "laser_profile":
        return evaluate_laser_profile_faults(evidence)
    elif experiment_id == "faraday":
        return evaluate_faraday_faults(evidence)
    elif experiment_id == "beer_lambert":
        return evaluate_beer_lambert_faults(evidence)
    else:
        raise ValueError(f"Unknown experiment_id for diagnostic evaluation: {experiment_id}")
