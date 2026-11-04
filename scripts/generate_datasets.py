"""Reproducible synthetic dataset generator for PhysSandbox.

Generates clean and seeded-fault experimental datasets for all 6 experiments:
1. Resistance & Thermal EMF
2. Oscilloscope Signal Analysis
3. Helmholtz Coil Magnetic Field
4. Laser Beam Profile & Alignment
5. Faraday Induction Law
6. Beer-Lambert Optical Absorption

All datasets are saved as CSV files with separate metadata.json files.
"""

import json
import os
import numpy as np

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "examples")


def generate_all_datasets(seed: int = 42):
    np.random.seed(seed)

    # ==========================================
    # 1. RESISTANCE EXPERIMENTS
    # ==========================================
    res_dir = os.path.join(OUTPUT_BASE, "resistance")
    os.makedirs(res_dir, exist_ok=True)

    # Nominal: R = 10 mΩ = 0.010 Ω
    currents = np.linspace(0.01, 0.10, 10)  # 10 mA to 100 mA
    r_true = 0.010

    # Clean
    v_clean = r_true * currents + np.random.normal(0, 1.2e-6, len(currents))
    np.savetxt(
        os.path.join(res_dir, "resistance_clean.csv"),
        np.column_stack([currents, v_clean]),
        delimiter=",",
        header="current_A,voltage_V",
        comments="",
        fmt=["%.4f", "%.9f"],
    )

    # Thermal Offset: V0 = 40 µV
    v_offset = r_true * currents + 40.0e-6 + np.random.normal(0, 1.2e-6, len(currents))
    np.savetxt(
        os.path.join(res_dir, "resistance_thermal_offset.csv"),
        np.column_stack([currents, v_offset]),
        delimiter=",",
        header="current_A,voltage_V",
        comments="",
        fmt=["%.4f", "%.9f"],
    )

    # Nonlinear: Heating / quadratic contact resistance V = R*I + beta*I^2
    v_nonlinear = r_true * currents + 0.08 * (currents**2) + np.random.normal(0, 1.5e-6, len(currents))
    np.savetxt(
        os.path.join(res_dir, "resistance_nonlinear.csv"),
        np.column_stack([currents, v_nonlinear]),
        delimiter=",",
        header="current_A,voltage_V",
        comments="",
        fmt=["%.4f", "%.9f"],
    )

    res_meta = {
        "experiment_id": "resistance_4wire",
        "description": "Low-resistance 4-wire measurement and thermal EMF investigation",
        "origin": "synthetic",
        "random_seed": seed,
        "theoretical_model": "V = R * I + V0",
        "nominal_parameters": {"resistance_ohm": 0.010},
        "datasets": {
            "resistance_clean.csv": {
                "injected_fault": None,
                "expected_diagnosis": "No fault found",
                "recommended_next_test": None,
            },
            "resistance_thermal_offset.csv": {
                "injected_fault": "thermal_emf_offset_40uV",
                "expected_diagnosis": "Thermoelectric (thermal EMF) offset at a contact",
                "recommended_next_test": "current_reversal",
            },
            "resistance_nonlinear.csv": {
                "injected_fault": "nonlinear_contact_heating",
                "expected_diagnosis": "Nonlinear contact resistance / heating effect",
                "recommended_next_test": "narrow_current_sweep",
            },
        },
    }
    with open(os.path.join(res_dir, "metadata.json"), "w") as f:
        json.dump(res_meta, f, indent=2)

    # ==========================================
    # 2. OSCILLOSCOPE EXPERIMENTS
    # ==========================================
    scope_dir = os.path.join(OUTPUT_BASE, "oscilloscope")
    os.makedirs(scope_dir, exist_ok=True)

    # fs = 20 kHz, duration = 0.1 s, 2000 samples
    t = np.linspace(0, 0.1, 2000, endpoint=False)
    sig_f = 1000.0  # 1 kHz fundamental
    sig_amp = 0.50  # 500 mV

    # Clean
    v_sc_clean = sig_amp * np.sin(2 * np.pi * sig_f * t) + np.random.normal(0, 0.015, len(t))
    np.savetxt(
        os.path.join(scope_dir, "oscilloscope_clean.csv"),
        np.column_stack([t, v_sc_clean]),
        delimiter=",",
        header="time_s,voltage_V",
        comments="",
        fmt=["%.7f", "%.6f"],
    )

    # 50 Hz interference (ground loop)
    v_sc_50hz = (
        sig_amp * np.sin(2 * np.pi * sig_f * t)
        + 0.12 * np.sin(2 * np.pi * 50.0 * t)
        + np.random.normal(0, 0.015, len(t))
    )
    np.savetxt(
        os.path.join(scope_dir, "oscilloscope_50hz_noise.csv"),
        np.column_stack([t, v_sc_50hz]),
        delimiter=",",
        header="time_s,voltage_V",
        comments="",
        fmt=["%.7f", "%.6f"],
    )

    # Harmonic interference: 50 Hz, 100 Hz, 150 Hz
    v_sc_harm = (
        sig_amp * np.sin(2 * np.pi * sig_f * t)
        + 0.10 * np.sin(2 * np.pi * 50.0 * t)
        + 0.04 * np.sin(2 * np.pi * 100.0 * t + 0.5)
        + 0.06 * np.sin(2 * np.pi * 150.0 * t + 1.2)
        + np.random.normal(0, 0.015, len(t))
    )
    np.savetxt(
        os.path.join(scope_dir, "oscilloscope_harmonic_noise.csv"),
        np.column_stack([t, v_sc_harm]),
        delimiter=",",
        header="time_s,voltage_V",
        comments="",
        fmt=["%.7f", "%.6f"],
    )

    scope_meta = {
        "experiment_id": "waveform",
        "description": "Oscilloscope signal analysis and noise detection",
        "origin": "synthetic",
        "random_seed": seed,
        "theoretical_model": "V(t) = A*sin(2*pi*f0*t) + B*sin(2*pi*fh*t) + n(t)",
        "nominal_parameters": {"signal_hz": 1000.0, "mains_hz": 50.0},
        "datasets": {
            "oscilloscope_clean.csv": {
                "injected_fault": None,
                "expected_diagnosis": "No fault found",
                "recommended_next_test": None,
            },
            "oscilloscope_50hz_noise.csv": {
                "injected_fault": "mains_50hz_interference",
                "expected_diagnosis": "Ground loop picking up mains hum",
                "recommended_next_test": "single_point_ground",
            },
            "oscilloscope_harmonic_noise.csv": {
                "injected_fault": "mains_harmonics_50_100_150hz",
                "expected_diagnosis": "Ground loop picking up mains hum",
                "recommended_next_test": "single_point_ground",
            },
        },
    }
    with open(os.path.join(scope_dir, "metadata.json"), "w") as f:
        json.dump(scope_meta, f, indent=2)

    # ==========================================
    # 3. HELMHOLTZ COIL EXPERIMENTS
    # ==========================================
    mag_dir = os.path.join(OUTPUT_BASE, "electromagnetism")
    os.makedirs(mag_dir, exist_ok=True)

    # R = 0.15 m, N = 100, k_theo = 599.47 µT/A
    mu0 = 4.0 * np.pi * 1e-7
    k_theo = ((4.0 / 5.0) ** 1.5) * mu0 * 100 / 0.15 * 1e6
    i_coils = np.linspace(0.1, 1.0, 10)  # 0.1 A to 1.0 A

    # Clean
    b_clean = k_theo * i_coils + np.random.normal(0, 2.5, len(i_coils))
    np.savetxt(
        os.path.join(mag_dir, "magnetic_clean.csv"),
        np.column_stack([i_coils, b_clean]),
        delimiter=",",
        header="current_A,measured_B_uT",
        comments="",
        fmt=["%.3f", "%.2f"],
    )

    # Magnetic offset: B0 = +45 µT (Earth's magnetic field component)
    b_offset = k_theo * i_coils + 45.0 + np.random.normal(0, 2.5, len(i_coils))
    np.savetxt(
        os.path.join(mag_dir, "magnetic_offset.csv"),
        np.column_stack([i_coils, b_offset]),
        delimiter=",",
        header="current_A,measured_B_uT",
        comments="",
        fmt=["%.3f", "%.2f"],
    )

    # Gain error: Sensor calibration is 18% low (e.g. 0.82 * k_theo)
    b_gain = (0.82 * k_theo) * i_coils + np.random.normal(0, 2.5, len(i_coils))
    np.savetxt(
        os.path.join(mag_dir, "magnetic_gain.csv"),
        np.column_stack([i_coils, b_gain]),
        delimiter=",",
        header="current_A,measured_B_uT",
        comments="",
        fmt=["%.3f", "%.2f"],
    )

    mag_meta = {
        "experiment_id": "helmholtz",
        "description": "Helmholtz coil magnetic-field calibration",
        "origin": "synthetic",
        "random_seed": seed,
        "theoretical_model": "B = (4/5)^(3/2) * mu0 * N * I / R = k * I",
        "nominal_parameters": {"radius_m": 0.15, "turns": 100, "k_theoretical_uT_per_A": round(k_theo, 2)},
        "datasets": {
            "magnetic_clean.csv": {
                "injected_fault": None,
                "expected_diagnosis": "No fault found",
                "recommended_next_test": None,
            },
            "magnetic_offset.csv": {
                "injected_fault": "ambient_earth_field_45uT",
                "expected_diagnosis": "Ambient Earth magnetic field or nearby ferromagnetic materials",
                "recommended_next_test": "zero_current_ambient",
            },
            "magnetic_gain.csv": {
                "injected_fault": "sensor_gain_calibration_error_18pct",
                "expected_diagnosis": "Sensor gain / calibration scaling error",
                "recommended_next_test": "calibrated_magnetometer_check",
            },
        },
    }
    with open(os.path.join(mag_dir, "metadata.json"), "w") as f:
        json.dump(mag_meta, f, indent=2)

    # ==========================================
    # 4. LASER BEAM PROFILE EXPERIMENTS
    # ==========================================
    optics_dir = os.path.join(OUTPUT_BASE, "optics")
    os.makedirs(optics_dir, exist_ok=True)

    # Transverse scan: -3.0 mm to +3.0 mm, 25 points
    x_pos = np.linspace(-3.0, 3.0, 25)
    w_nom = 1.0  # 1.0 mm waist radius
    v_peak_nom = 600.0  # 600 mV
    v_dark_nom = 15.0  # 15 mV dark voltage

    # Clean (centered at x0 = 0.0 mm)
    v_opt_clean = v_dark_nom + v_peak_nom * np.exp(-2 * ((x_pos - 0.0) / w_nom)**2) + np.random.normal(0, 4.0, len(x_pos))
    np.savetxt(
        os.path.join(optics_dir, "optics_clean.csv"),
        np.column_stack([x_pos, v_opt_clean]),
        delimiter=",",
        header="position_mm,detector_voltage_mV",
        comments="",
        fmt=["%.2f", "%.2f"],
    )

    # Misaligned: Beam centered at x0 = +0.85 mm
    v_opt_mis = v_dark_nom + v_peak_nom * np.exp(-2 * ((x_pos - 0.85) / w_nom)**2) + np.random.normal(0, 4.0, len(x_pos))
    np.savetxt(
        os.path.join(optics_dir, "optics_misaligned.csv"),
        np.column_stack([x_pos, v_opt_mis]),
        delimiter=",",
        header="position_mm,detector_voltage_mV",
        comments="",
        fmt=["%.2f", "%.2f"],
    )

    # Low Signal: Laser power attenuated, peak = 180 mV
    v_opt_low = v_dark_nom + 180.0 * np.exp(-2 * ((x_pos - 0.0) / w_nom)**2) + np.random.normal(0, 4.0, len(x_pos))
    np.savetxt(
        os.path.join(optics_dir, "optics_low_signal.csv"),
        np.column_stack([x_pos, v_opt_low]),
        delimiter=",",
        header="position_mm,detector_voltage_mV",
        comments="",
        fmt=["%.2f", "%.2f"],
    )

    optics_meta = {
        "experiment_id": "laser_profile",
        "description": "Laser beam profile and photodetector alignment",
        "origin": "synthetic",
        "random_seed": seed,
        "theoretical_model": "V(x) = V_dark + V_peak * exp(-2 * ((x - x0) / w)^2)",
        "nominal_parameters": {"expected_center_mm": 0.0, "expected_peak_mV": 600.0, "waist_mm": 1.0},
        "datasets": {
            "optics_clean.csv": {
                "injected_fault": None,
                "expected_diagnosis": "No fault found",
                "recommended_next_test": None,
            },
            "optics_misaligned.csv": {
                "injected_fault": "transverse_stage_misalignment_0.85mm",
                "expected_diagnosis": "Photodetector transverse misalignment",
                "recommended_next_test": "adjust_transverse_stage",
            },
            "optics_low_signal.csv": {
                "injected_fault": "laser_optical_power_attenuation_70pct",
                "expected_diagnosis": "Reduced laser output or optical window contamination",
                "recommended_next_test": "measure_laser_optical_power",
            },
        },
    }
    with open(os.path.join(optics_dir, "metadata.json"), "w") as f:
        json.dump(optics_meta, f, indent=2)

    # ==========================================
    # 5. FARADAY LAW EXPERIMENTS
    # ==========================================
    faraday_dir = os.path.join(OUTPUT_BASE, "faraday")
    os.makedirs(faraday_dir, exist_ok=True)

    # Sinusoidal flux Phi(t) = Phi0 * sin(2*pi*f*t), N = 200 turns, f = 10 Hz
    t_f = np.linspace(0, 0.2, 100)
    phi0 = 1.0e-4  # 100 µWb
    f_ind = 10.0
    flux_t = phi0 * np.sin(2 * np.pi * f_ind * t_f)
    # Theoretical EMF = -N * dPhi/dt = -200 * (phi0 * 2*pi*f * cos(...)) = -1.2566 * cos(...)
    emf_true = -200.0 * (phi0 * 2 * np.pi * f_ind * np.cos(2 * np.pi * f_ind * t_f))

    # Clean
    v_far_clean = emf_true + np.random.normal(0, 0.03, len(t_f))
    np.savetxt(
        os.path.join(faraday_dir, "faraday_clean.csv"),
        np.column_stack([t_f, flux_t, v_far_clean]),
        delimiter=",",
        header="time_s,magnetic_flux_Wb,induced_voltage_V",
        comments="",
        fmt=["%.4f", "%.7e", "%.4f"],
    )

    # Offset: Amplifier DC offset = +0.15 V
    v_far_offset = emf_true + 0.15 + np.random.normal(0, 0.03, len(t_f))
    np.savetxt(
        os.path.join(faraday_dir, "faraday_offset.csv"),
        np.column_stack([t_f, flux_t, v_far_offset]),
        delimiter=",",
        header="time_s,magnetic_flux_Wb,induced_voltage_V",
        comments="",
        fmt=["%.4f", "%.7e", "%.4f"],
    )

    # Reduced response: Effective turn count is 100 instead of 200 (50% amplitude)
    v_far_red = 0.50 * emf_true + np.random.normal(0, 0.03, len(t_f))
    np.savetxt(
        os.path.join(faraday_dir, "faraday_reduced_signal.csv"),
        np.column_stack([t_f, flux_t, v_far_red]),
        delimiter=",",
        header="time_s,magnetic_flux_Wb,induced_voltage_V",
        comments="",
        fmt=["%.4f", "%.7e", "%.4f"],
    )

    faraday_meta = {
        "experiment_id": "faraday",
        "description": "Faraday's law of electromagnetic induction",
        "origin": "synthetic",
        "random_seed": seed,
        "theoretical_model": "EMF = -N * d(Phi_B) / dt",
        "nominal_parameters": {"turns": 200, "frequency_hz": 10.0, "peak_flux_Wb": phi0},
        "datasets": {
            "faraday_clean.csv": {
                "injected_fault": None,
                "expected_diagnosis": "Normal induction",
                "recommended_next_test": None,
            },
            "faraday_offset.csv": {
                "injected_fault": "amplifier_dc_offset_150mV",
                "expected_diagnosis": "Amplifier or instrumentation DC offset",
                "recommended_next_test": "static_flux_baseline",
            },
            "faraday_reduced_signal.csv": {
                "injected_fault": "coil_turn_count_half_nominal",
                "expected_diagnosis": "Coil turn count or area specification mismatch",
                "recommended_next_test": "verify_coil_geometry",
            },
        },
    }
    with open(os.path.join(faraday_dir, "metadata.json"), "w") as f:
        json.dump(faraday_meta, f, indent=2)

    # ==========================================
    # 6. BEER-LAMBERT EXPERIMENTS
    # ==========================================
    abs_dir = os.path.join(OUTPUT_BASE, "absorption")
    os.makedirs(abs_dir, exist_ok=True)

    # Path lengths: 0.0 cm to 5.0 cm, 11 points
    x_thick = np.linspace(0.0, 5.0, 11)
    alpha_nom = 0.50  # 0.50 cm^-1
    i0_nom = 1.0  # relative intensity 1.0

    # Clean
    i_abs_clean = i0_nom * np.exp(-alpha_nom * x_thick) + np.random.normal(0, 0.008, len(x_thick))
    np.savetxt(
        os.path.join(abs_dir, "absorption_clean.csv"),
        np.column_stack([x_thick, i_abs_clean]),
        delimiter=",",
        header="thickness_cm,transmitted_intensity",
        comments="",
        fmt=["%.2f", "%.5f"],
    )

    # High absorption: alpha = 1.10 cm^-1 (more than 2x higher)
    i_abs_high = i0_nom * np.exp(-1.10 * x_thick) + np.random.normal(0, 0.008, len(x_thick))
    np.savetxt(
        os.path.join(abs_dir, "absorption_high.csv"),
        np.column_stack([x_thick, i_abs_high]),
        delimiter=",",
        header="thickness_cm,transmitted_intensity",
        comments="",
        fmt=["%.2f", "%.5f"],
    )

    # Background light: I_bg = 0.12 (plateaus at large thickness)
    i_abs_bg = i0_nom * np.exp(-alpha_nom * x_thick) + 0.12 + np.random.normal(0, 0.008, len(x_thick))
    np.savetxt(
        os.path.join(abs_dir, "absorption_background.csv"),
        np.column_stack([x_thick, i_abs_bg]),
        delimiter=",",
        header="thickness_cm,transmitted_intensity",
        comments="",
        fmt=["%.2f", "%.5f"],
    )

    abs_meta = {
        "experiment_id": "beer_lambert",
        "description": "Beer-Lambert law of optical absorption",
        "origin": "synthetic",
        "random_seed": seed,
        "theoretical_model": "I(x) = I_0 * exp(-alpha * x) + I_background",
        "nominal_parameters": {"expected_alpha": 0.50, "expected_i0": 1.0},
        "datasets": {
            "absorption_clean.csv": {
                "injected_fault": None,
                "expected_diagnosis": "Normal optical absorption",
                "recommended_next_test": None,
            },
            "absorption_high.csv": {
                "injected_fault": "elevated_absorption_coefficient_1.1cm_inv",
                "expected_diagnosis": "Higher sample concentration or elevated absorption coefficient",
                "recommended_next_test": "multi_thickness_sweep",
            },
            "absorption_background.csv": {
                "injected_fault": "stray_ambient_light_leakage_12pct",
                "expected_diagnosis": "Stray ambient light or photodetector dark current offset",
                "recommended_next_test": "blocked_source_dark_measurement",
            },
        },
    }
    with open(os.path.join(abs_dir, "metadata.json"), "w") as f:
        json.dump(abs_meta, f, indent=2)

    print("Successfully generated all 18 reproducible synthetic datasets and metadata.")


if __name__ == "__main__":
    generate_all_datasets()
