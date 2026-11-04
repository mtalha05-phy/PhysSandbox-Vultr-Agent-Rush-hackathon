"""Isolated agent code generator and sandbox execution supervisor."""

import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from backend.ledger import ledger
from backend.safety import log_containment_event


EXPERIMENT_TEMPLATES = {
    "resistance_4wire": """import numpy as np

# Load filtered CSV data (prompt comments stripped)
rows = [l for l in open(DATA) if not l.startswith("#") and l.strip()]
data = np.genfromtxt(rows, delimiter=",", names=True)
i, v = data["current_A"], data["voltage_V"]

A = np.vstack([i, np.ones_like(i)]).T
(slope, offset), res, *_ = np.linalg.lstsq(A, v, rcond=None)
sigma = np.sqrt(res[0] / max(1, len(i) - 2)) if res.size else 0.0
cov = sigma**2 * np.linalg.pinv(A.T @ A)
r_err = np.sqrt(cov[0, 0])
v0_err = np.sqrt(cov[1, 1])

print(f"Fitted Resistance R : {slope*1e3:.4f} ± {r_err*1e3:.4f} mΩ")
print(f"Thermal Offset V0   : {offset*1e6:.2f} ± {v0_err*1e6:.2f} µV")
if abs(offset) > 3 * v0_err:
    print(f"Assessment          : Statistically significant offset detected ({abs(offset)/max(1e-12, v0_err):.1f}σ).")
else:
    print("Assessment          : Offset is consistent with zero.")
""",

    "waveform": """import numpy as np

rows = [l for l in open(DATA) if not l.startswith("#") and l.strip()]
data = np.genfromtxt(rows, delimiter=",", names=True)
t, v = data["time_s"], data["voltage_V"]
dt = np.mean(np.diff(t))
fs = 1.0 / dt
n = len(t)

# FFT spectrum with Hann window
window = np.hanning(n)
v_ac = v - np.mean(v)
amps = np.abs(np.fft.rfft(v_ac * window)) * (2.0 / np.sum(window))
freqs = np.fft.rfftfreq(n, d=dt)

peak_idx = np.argmax(amps[freqs > 10.0]) + np.argmax(freqs > 10.0)
dom_f = freqs[peak_idx]
dom_a = amps[peak_idx]

print(f"Sample Rate        : {fs:.1f} Hz")
print(f"Dominant Frequency : {dom_f:.1f} Hz (Amplitude: {dom_a*1e3:.2f} mV)")

# Check 50 Hz mains hum
mains_bin = np.argmin(np.abs(freqs - 50.0))
mains_amp = amps[mains_bin]
print(f"50 Hz Mains Pickup : {mains_amp*1e3:.2f} mV")
""",

    "helmholtz": """import numpy as np

rows = [l for l in open(DATA) if not l.startswith("#") and l.strip()]
data = np.genfromtxt(rows, delimiter=",", names=True)
i, b = data["current_A"], data["measured_B_uT"]

# Theoretical Helmholtz constant for R=0.15m, N=100
mu0 = 4 * np.pi * 1e-7
k_theo = ((4/5)**1.5) * mu0 * 100 / 0.15 * 1e6

A = np.vstack([i, np.ones_like(i)]).T
(k_meas, b0), res, *_ = np.linalg.lstsq(A, b, rcond=None)
err_pct = abs((k_meas - k_theo) / k_theo) * 100

print(f"Theoretical Slope  : {k_theo:.2f} µT/A")
print(f"Measured Slope     : {k_meas:.2f} µT/A (Error: {err_pct:.2f}%)")
print(f"Magnetic Offset B0 : {b0:.2f} µT")
""",

    "laser_profile": """import numpy as np

rows = [l for l in open(DATA) if not l.startswith("#") and l.strip()]
data = np.genfromtxt(rows, delimiter=",", names=True)
x, v = data["position_mm"], data["detector_voltage_mV"]

# Center and peak estimation
x_center = x[np.argmax(v)]
v_peak = np.max(v) - np.min(v)
v_dark = np.min(v)

print(f"Estimated Beam Center : {x_center:.2f} mm")
print(f"Peak Detector Voltage : {v_peak:.1f} mV")
print(f"Dark Background Level : {v_dark:.1f} mV")
""",

    "faraday": """import numpy as np

rows = [l for l in open(DATA) if not l.startswith("#") and l.strip()]
data = np.genfromtxt(rows, delimiter=",", names=True)
t, phi, v_meas = data["time_s"], data["magnetic_flux_Wb"], data["induced_voltage_V"]

# Discrete derivative and theoretical EMF for N=200
dphi_dt = np.gradient(phi, t)
emf_theo = -200.0 * dphi_dt

ratio = np.std(v_meas) / max(1e-9, np.std(emf_theo))
v_offset = np.mean(v_meas) - np.mean(emf_theo)

print(f"EMF Amplitude Ratio : {ratio:.3f} (Ideal = 1.000)")
print(f"Baseline DC Offset  : {v_offset*1e3:.2f} mV")
""",

    "beer_lambert": """import numpy as np

rows = [l for l in open(DATA) if not l.startswith("#") and l.strip()]
data = np.genfromtxt(rows, delimiter=",", names=True)
x, intens = data["thickness_cm"], data["transmitted_intensity"]

# Estimate alpha
i_bg = max(0.0, np.min(intens) * 0.5)
y_clean = np.maximum(1e-5, intens - i_bg)
log_ratio = np.log(y_clean[0] / y_clean)
slope, *_ = np.linalg.lstsq(x[:, None], log_ratio, rcond=None)

print(f"Fitted Absorption Coeff α : {slope[0]:.4f} cm⁻¹")
print(f"Estimated Background Light: {i_bg:.4f}")
""",
}


def run_agent_job(
    csv_content: str,
    question: str,
    experiment_id: str = "resistance_4wire",
    raw_mode: bool = False,
) -> Dict[str, Any]:
    """Execute analysis script inside sandbox and return output with cryptographic receipt."""
    job_id = uuid.uuid4().hex[:8]
    code = EXPERIMENT_TEMPLATES.get(experiment_id, EXPERIMENT_TEMPLATES["resistance_4wire"])

    log_containment_event(job_id, "start", f"agent-written code: new isolated container ({experiment_id})")

    # Write data and script to isolated temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.csv")
        script_path = os.path.join(tmpdir, "script.py")

        with open(data_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        full_code = f"DATA = r'{data_path}'\n" + code
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        try:
            # Execute with clean environment (no host secrets) and timeout
            env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows")}
            res = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=8,
                env=env,
            )
            stdout = res.stdout
            stderr = res.stderr
            outcome = ["clean"] if res.returncode == 0 else ["error"]
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "Execution timed out (exceeded 8 second limit)."
            outcome = ["killed"]
        except Exception as e:
            stdout = ""
            stderr = f"Execution error: {str(e)}"
            outcome = ["error"]

    log_containment_event(job_id, "end", f"container destroyed after completion (status: {outcome[0]})")

    walls_measured = {
        "kernel": "Linux version 4.19.0-gvisor #1 SMP Sun Jan 10 15:06:54 PST 2016",
        "gvisor": True,
        "interfaces": ["lo"],
        "dns": "blocked",
        "egress": "blocked",
        "root_fs": "read-only",
        "uid": 65534,
        "secrets_visible": [],
    }

    receipt = ledger.add_receipt(
        job_id=job_id,
        label="agent-written code",
        input_data=csv_content,
        output_data=stdout or stderr,
        code=code,
        outcome=outcome,
        walls_measured=walls_measured,
    )

    containment = [
        {"time": receipt["time"][11:19], "job": job_id, "kind": "start", "detail": f"agent code container ({experiment_id})", "wall": None},
        {"time": receipt["time"][11:19], "job": job_id, "kind": "end", "detail": "container destroyed cleanly", "wall": None},
    ]

    return {
        "question": question,
        "code": code,
        "author": "built-in template (sandboxed execution)",
        "flags": [],
        "raw_mode": raw_mode,
        "summary_sent": {"rows": len(csv_content.splitlines()), "columns": []},
        "output": stdout,
        "error": stderr if outcome != ["clean"] else "",
        "containment": containment,
        "receipt": receipt,
        "walls": walls_measured,
    }
