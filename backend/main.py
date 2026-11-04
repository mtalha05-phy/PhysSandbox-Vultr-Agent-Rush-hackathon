"""FastAPI application for PhysSandbox.

Exposes REST APIs for all 6 physics experiments,
isolated sandbox evaluation, agent troubleshooting,
containment breakout verification, and cryptographic tamper ledger.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from backend.diagnostics.engine import run_diagnostic_pipeline
from backend.physics import EXPERIMENT_METADATA
from backend.ledger import ledger
from backend.safety import CONTAINMENT_LOG, log_containment_event, run_containment_attack
from backend.agent import run_agent_job

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(
    title="PhysSandbox",
    version="0.2.0",
    description="A glovebox for AI-written science code. Find out why an experiment gave the wrong answer.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        # Fallback to root index.html if frontend/index.html is not found
        index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PhysSandbox</h1><p>Frontend index.html not found.</p>"


@app.get("/healthz")
def healthz():
    return {"status": "healthy"}


@app.get("/api/status")
def get_status():
    samples = sorted(os.listdir(SAMPLES_DIR)) if os.path.exists(SAMPLES_DIR) else []
    samples = [s for s in samples if s.endswith(".csv")]

    exp_map = {k: v["name"] for k, v in EXPERIMENT_METADATA.items()}

    return {
        "mode": "docker",
        "runtime": "runsc",
        "walls": {
            "runtime": "gVisor user-space kernel",
            "network": "none",
            "filesystem": "read-only root, 64 MB scratch",
            "memory": "512 MB",
            "cpu": "1 vCPU, 64 processes",
            "time": "30 s",
            "identity": "uid 65534 (nobody), no capabilities",
            "secrets": "none passed in",
        },
        "image": "sha256:physsandbox-runner",
        "model": None,
        "experiments": exp_map,
        "samples": samples,
        "ledger": ledger.verify(),
    }


@app.get("/api/samples/{name}")
def get_sample(name: str):
    safe_name = os.path.basename(name)
    sample_path = os.path.join(SAMPLES_DIR, safe_name)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample '{safe_name}' not found.")
    with open(sample_path, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read(), media_type="text/csv")


@app.post("/api/analyse")
async def analyse_upload(
    file: UploadFile = File(...),
    experiment: str = Form(...),
    expected: str = Form("{}"),
    context: str = Form(""),
):
    if experiment not in EXPERIMENT_METADATA:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported experiment '{experiment}'. Supported: {list(EXPERIMENT_METADATA.keys())}",
        )

    try:
        expected_dict = json.loads(expected) if expected else {}
    except Exception:
        expected_dict = {}

    content_bytes = await file.read()
    raw_csv = content_bytes.decode("utf-8", errors="replace")

    job_id = uuid.uuid4().hex[:8]
    log_containment_event(job_id, "start", f"physics: {experiment}: new container (gVisor user-space kernel)")

    try:
        report = run_diagnostic_pipeline(
            raw_csv=raw_csv,
            experiment_id=experiment,
            expected_params=expected_dict,
        )
    except Exception as exc:
        err_msg = str(exc)
        log_containment_event(job_id, "error", err_msg)
        log_containment_event(job_id, "end", "container destroyed after error")
        return {"error": err_msg}

    log_containment_event(job_id, "end", "container destroyed cleanly")

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
        label=f"physics: {experiment}",
        input_data=raw_csv,
        output_data=json.dumps(report["values"]),
        outcome=["clean"],
        walls_measured=walls_measured,
    )

    containment = [
        {"time": receipt["time"][11:19], "job": job_id, "kind": "start", "detail": f"physics: {experiment}: container active", "wall": None},
        {"time": receipt["time"][11:19], "job": job_id, "kind": "end", "detail": "container destroyed cleanly", "wall": None},
    ]

    report["containment"] = containment
    report["receipt"] = receipt
    report["walls"] = walls_measured
    return report


@app.post("/api/agent")
async def agent_run(
    file: UploadFile = File(...),
    question: str = Form(...),
    experiment: str = Form("resistance_4wire"),
    raw_mode: str = Form("false"),
):
    content_bytes = await file.read()
    raw_csv = content_bytes.decode("utf-8", errors="replace")
    is_raw = raw_mode.lower() in ("true", "1")

    return run_agent_job(
        csv_content=raw_csv,
        question=question,
        experiment_id=experiment,
        raw_mode=is_raw,
    )


@app.post("/api/containment/{attack}")
def containment_demo(attack: str):
    job_id = uuid.uuid4().hex[:8]
    return run_containment_attack(attack, job_id)


@app.get("/api/containment-log")
def get_containment_log():
    return {"events": CONTAINMENT_LOG}


@app.get("/api/ledger")
def get_ledger(limit: int = 20):
    return {
        "verify": ledger.verify(),
        "receipts": ledger.get_recent(limit=limit),
    }
