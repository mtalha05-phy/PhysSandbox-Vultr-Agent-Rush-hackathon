# PhysSandbox — Multi-Domain Physics Troubleshooting & Glovebox Platform

PhysSandbox is an AI-assisted scientific troubleshooting platform developed for the **Vultr Agent Rush Hackathon**. 

It enables experimental physicists and researchers to upload measurement data (CSV), compare empirical results against deterministic physical laws, detect calibration offsets and equipment faults with Bayesian probability, receive optimal Value-of-Information next-test recommendations, and safely execute AI-generated analysis code inside a sandboxed glovebox environment.

---

## 1. Supported Physics Experiments

PhysSandbox supports **6 independent physics experiments** across electrical measurements, electromagnetism, signal processing, and optics:

| # | Experiment | Mathematical Model | Injected Fault Modes | Recommended Next Test |
|---|---|---|---|---|
| **1** | **Low-Resistance Measurement & Thermal EMF** | $V = R \cdot I + V_0$ | • Thermal EMF contact offset (40 µV)<br>• Non-linear contact heating | Current reversal ($\pm I$ sweep) |
| **2** | **Oscilloscope Waveform & Noise Detection** | $V(t) = A\sin(2\pi f_0 t) + B\sin(2\pi f_h t) + n(t)$ | • 50 Hz power-line hum<br>• Odd harmonics (100 Hz, 150 Hz) | Single-point grounding & cable twist |
| **3** | **Helmholtz Coil Magnetic Calibration** | $B = \left(\frac{4}{5}\right)^{3/2} \frac{\mu_0 N I}{R} = k \cdot I$ | • Ambient Earth magnetic field (+45 µT)<br>• Sensor gain calibration error (-18%) | Zero-current ($I=0$) field subtraction |
| **4** | **Laser Beam Profile & Photodetector Alignment** | $V(x) = V_{\text{dark}} + V_{\text{peak}} \exp\left(-2 \left(\frac{x - x_0}{w}\right)^2\right)$ | • Transverse misalignment ($x_0 = 0.85$ mm)<br>• Low optical signal (70% attenuation) | Transverse stage realignment to $x_0$ |
| **5** | **Faraday's Law of Induction** | $\mathcal{E} = -N \frac{d\Phi_B}{dt}$ | • Amplifier DC voltage baseline (+150 mV)<br>• Reduced turn count ($N = 100$ vs $200$) | Static flux ($d\Phi/dt = 0$) baseline test |
| **6** | **Beer-Lambert Optical Absorption** | $I(x) = I_0 \exp(-\alpha x) + I_{\text{background}}$ | • Elevated absorption coefficient ($\alpha = 1.1\ \text{cm}^{-1}$)<br>• Ambient stray light background (+12%) | Blocked source dark measurement |

---

## 2. Architecture & Design Principles

* **Deterministic Physics Core**: All mathematical regressions, parameter uncertainties ($\pm \sigma$), residuals, FFTs, and derivatives are calculated strictly using deterministic Python/NumPy physics modules. The AI never invents numerical values or fitting parameters.
* **Glovebox Container Sandbox**: Untrusted uploaded data and agent-generated scripts execute inside an unprivileged Docker container running under the **gVisor (`runsc`)** user-space kernel with no network access (`--network none`), a read-only root filesystem, UID 65534 (`nobody`), and strict cgroup memory (512 MB) and CPU limits.
* **Cryptographic Tamper-Evident Ledger**: Every job is hashed (input, code, output) and recorded into a SHA-256 hash-chained lab notebook receipt. Any historical modification invalidates the chain.
* **Prompt-Injection Defense**: CSV files are stripped of header comments and validated against strict schemas before exposure to any model.

---

## 3. Directory Structure

```
PhysSandBox/
├── backend/
│   ├── main.py                     # FastAPI REST API endpoints
│   ├── ledger.py                   # Cryptographic SHA-256 tamper-evident ledger
│   ├── safety.py                   # Glovebox escape verification & attack simulator
│   ├── agent.py                    # Sandboxed code generator & supervisor
│   ├── physics/                    # Deterministic physics calculation modules
│   │   ├── __init__.py             # Experiment registry
│   │   ├── resistance.py           # Low-resistance & thermal EMF
│   │   ├── oscilloscope.py         # FFT spectrum & mains hum detection
│   │   ├── helmholtz.py            # Helmholtz coil magnetic calibration
│   │   ├── laser_profile.py        # Gaussian beam profile & alignment
│   │   ├── faraday.py              # Discrete induction dPhi/dt
│   │   └── beer_lambert.py         # Exponential optical absorption
│   └── diagnostics/                # Evidence evaluation & next-test ranking
│       ├── __init__.py
│       ├── engine.py               # CSV validation & pipeline runner
│       └── fault_models.py         # Bayesian hypotheses & Value-of-Information
├── examples/                       # 18 Reproducible synthetic datasets & metadata
│   ├── resistance/                 # Clean, thermal offset, nonlinear + metadata.json
│   ├── oscilloscope/               # Clean, 50 Hz hum, harmonic noise + metadata.json
│   ├── electromagnetism/           # Clean, magnetic offset, gain error + metadata.json
│   ├── optics/                     # Clean, misaligned, low signal + metadata.json
│   ├── faraday/                    # Clean, offset, reduced signal + metadata.json
│   └── absorption/                 # Clean, high absorption, background + metadata.json
├── frontend/
│   └── index.html                  # Single-page breadboard UI with SVG plot engine
├── samples/                        # Pre-packaged CSV datasets served by the API
├── scripts/
│   ├── generate_datasets.py        # Synthetic dataset generator with fixed NumPy seed
│   └── deploy.py                   # Remote VM deployment script
├── tests/                          # 25 automated unit & integration tests
│   ├── test_physics.py             # Numerical accuracy of physics calculations
│   ├── test_validation.py          # Corrupt input, NaN, and injection protection
│   ├── test_diagnostics.py         # Diagnostic accuracy & false-positive rates
│   └── test_api.py                 # FastAPI endpoints & ledger integrity
├── Dockerfile                      # Backend container
├── docker-compose.yml              # Backend + Caddy orchestration
└── Caddyfile                       # Reverse proxy configuration
```

---

## 4. Running Locally

### Option A: Running with Python Virtual Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run all 25 unit and integration tests
python -m pytest tests/

# Start the application server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Then open [http://localhost:8000/](http://localhost:8000/) in your browser.

### Option B: Running with Docker Compose
```bash
docker compose up -d --build
```
Then open [http://localhost/](http://localhost/) (served by Caddy on port 80).

---

## 5. Deploying to Remote Server (192.248.159.149)

To deploy the new multi-experiment platform to the Vultr Cloud Compute VM:

```bash
# Transfer the deployment archive
scp physsandbox-v2.tar.gz root@192.248.159.149:/tmp/

# Extract and rebuild containers on the remote host
ssh root@192.248.159.149 "mkdir -p /opt/physsandbox && tar -xzf /tmp/physsandbox-v2.tar.gz -C /opt/physsandbox && cd /opt/physsandbox && docker compose up -d --build"
```
Or run the automated deployment script:
```bash
python scripts/deploy.py
```
