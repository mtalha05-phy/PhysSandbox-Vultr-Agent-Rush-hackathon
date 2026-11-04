"""Automated unit tests for all 6 physics calculation modules."""

import numpy as np
import pytest

from backend.physics.resistance import analyze_resistance_data
from backend.physics.oscilloscope import analyze_oscilloscope_data
from backend.physics.helmholtz import analyze_helmholtz_data, calculate_theoretical_k
from backend.physics.laser_profile import analyze_laser_profile_data
from backend.physics.faraday import analyze_faraday_data
from backend.physics.beer_lambert import analyze_beer_lambert_data


def test_resistance_clean():
    currents = np.linspace(0.01, 0.10, 10)
    voltages = 0.010 * currents
    res = analyze_resistance_data(currents, voltages, expected_r=0.010)
    assert abs(res["values"]["resistance_fit_ohm"]["value"] - 0.010) < 1e-6
    assert abs(res["values"]["offset_V"]["value"]) < 1e-7
    assert res["evidence"]["r_squared"] > 0.999
    assert not res["evidence"]["is_nonlinear"]


def test_resistance_thermal_offset():
    currents = np.linspace(0.01, 0.10, 10)
    v0_true = 50e-6  # 50 µV offset
    voltages = 0.010 * currents + v0_true
    res = analyze_resistance_data(currents, voltages, expected_r=0.010)
    assert abs(res["values"]["resistance_fit_ohm"]["value"] - 0.010) < 1e-6
    assert abs(res["values"]["offset_V"]["value"] - v0_true) < 1e-7
    assert res["evidence"]["offset_sigma"] > 5.0


def test_oscilloscope_clean_and_50hz():
    t = np.linspace(0, 0.1, 2000, endpoint=False)
    # 1 kHz clean
    v_clean = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    res_clean = analyze_oscilloscope_data(t, v_clean, expected_signal_hz=1000.0)
    assert abs(res_clean["values"]["dominant_frequency_Hz"]["value"] - 1000.0) < 20.0
    assert not res_clean["evidence"]["has_mains_50hz"]

    # 50 Hz contaminated
    v_noisy = 0.5 * np.sin(2 * np.pi * 1000.0 * t) + 0.15 * np.sin(2 * np.pi * 50.0 * t)
    res_noisy = analyze_oscilloscope_data(t, v_noisy, expected_signal_hz=1000.0)
    assert res_noisy["evidence"]["has_mains_50hz"]
    assert res_noisy["values"]["mains_harmonics_V"]["value"]["50_Hz"] > 0.10


def test_helmholtz_calibration():
    i = np.linspace(0.1, 1.0, 10)
    k_theo = calculate_theoretical_k(radius_m=0.15, turns=100)
    assert 590.0 < k_theo < 610.0  # Approx 599.47 µT/A

    # Clean
    b_clean = k_theo * i
    res_clean = analyze_helmholtz_data(i, b_clean, radius_m=0.15, turns=100)
    assert res_clean["evidence"]["calibration_error_pct"] < 0.1
    assert not res_clean["evidence"]["has_magnetic_offset"]

    # Offset (Earth magnetic field +45 µT)
    b_offset = k_theo * i + 45.0
    res_offset = analyze_helmholtz_data(i, b_offset, radius_m=0.15, turns=100)
    assert abs(res_offset["values"]["offset_B0_uT"]["value"] - 45.0) < 0.1
    assert res_offset["evidence"]["has_magnetic_offset"]

    # Gain error (15% low)
    b_gain = 0.85 * k_theo * i
    res_gain = analyze_helmholtz_data(i, b_gain, radius_m=0.15, turns=100)
    assert res_gain["evidence"]["has_gain_error"]
    assert abs(res_gain["evidence"]["calibration_error_pct"] - 15.0) < 1.0


def test_laser_beam_profile():
    x = np.linspace(-3.0, 3.0, 25)
    # Centered at x0 = 0.0, waist = 1.0, peak = 500 mV, dark = 10 mV
    v_clean = 10.0 + 500.0 * np.exp(-2 * ((x - 0.0) / 1.0)**2)
    res_clean = analyze_laser_profile_data(x, v_clean, expected_center_mm=0.0, expected_peak_mv=500.0)
    assert abs(res_clean["values"]["beam_center_x0_mm"]["value"]) < 0.05
    assert abs(res_clean["values"]["beam_waist_radius_w_mm"]["value"] - 1.0) < 0.05
    assert res_clean["evidence"]["is_clean"]

    # Misaligned: x0 = 0.8 mm
    v_mis = 10.0 + 500.0 * np.exp(-2 * ((x - 0.8) / 1.0)**2)
    res_mis = analyze_laser_profile_data(x, v_mis, expected_center_mm=0.0, expected_peak_mv=500.0)
    assert res_mis["evidence"]["is_misaligned"]
    assert abs(res_mis["values"]["beam_center_x0_mm"]["value"] - 0.8) < 0.08


def test_faraday_induction():
    t = np.linspace(0, 0.2, 100)
    f = 10.0
    phi0 = 1.0e-4
    flux = phi0 * np.sin(2 * np.pi * f * t)
    # Analytical dPhi/dt = phi0 * 2*pi*f * cos(2*pi*f*t)
    # EMF = -N * dPhi/dt with N = 200
    emf = -200.0 * (phi0 * 2 * np.pi * f * np.cos(2 * np.pi * f * t))

    # Clean
    res_clean = analyze_faraday_data(t, flux, emf, turns=200)
    assert abs(res_clean["values"]["amplitude_ratio_gamma"]["value"] - 1.0) < 0.05
    assert res_clean["evidence"]["is_normal_induction"]

    # Baseline Offset
    v_offset = emf + 0.20
    res_offset = analyze_faraday_data(t, flux, v_offset, turns=200)
    assert res_offset["evidence"]["has_voltage_offset"]
    assert abs(res_offset["values"]["voltage_offset_V"]["value"] - 0.20) < 0.02


def test_beer_lambert_absorption():
    x = np.linspace(0.0, 5.0, 11)
    alpha = 0.50
    i0 = 1.0

    # Clean exponential decay
    i_clean = i0 * np.exp(-alpha * x)
    res_clean = analyze_beer_lambert_data(x, i_clean, expected_alpha=0.50, expected_i0=1.0)
    assert abs(res_clean["values"]["absorption_coefficient_alpha_cm_inv"]["value"] - 0.50) < 0.02
    assert res_clean["evidence"]["is_clean"]

    # Background stray light
    i_bg = i0 * np.exp(-alpha * x) + 0.15
    res_bg = analyze_beer_lambert_data(x, i_bg, expected_alpha=0.50, expected_i0=1.0)
    assert res_bg["evidence"]["has_background_light"]
    assert res_bg["values"]["background_intensity"]["value"] > 0.10
