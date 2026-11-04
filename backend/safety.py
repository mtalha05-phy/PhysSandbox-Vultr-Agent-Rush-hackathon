"""Glovebox safety containment and breakout test simulator."""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CONTAINMENT_LOG: List[Dict[str, Any]] = [
    {"time": "00:00:01", "job": "boot", "kind": "info", "detail": "PhysSandbox Glovebox initialized with gVisor / Docker containment", "wall": None}
]


def log_containment_event(job_id: str, kind: str, detail: str, wall: Optional[str] = None):
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
    CONTAINMENT_LOG.append({
        "time": now_str,
        "job": job_id,
        "kind": kind,
        "detail": detail,
        "wall": wall,
    })


def run_containment_attack(attack_type: str, job_id: str) -> Dict[str, Any]:
    """Execute and safely contain deliberate security breakout attempts."""
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

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

    if attack_type == "egress":
        detail = "Outbound socket connect(8.8.8.8:53) rejected: Network is unreachable (--network none)"
        wall = "network"
        containment = [
            {"time": now_str, "job": job_id, "kind": "start", "detail": "egress probe: attempt outbound TCP socket", "wall": None},
            {"time": now_str, "job": job_id, "kind": "contain", "detail": detail, "wall": wall},
            {"time": now_str, "job": job_id, "kind": "end", "detail": "container terminated cleanly", "wall": None},
        ]
        log_containment_event(job_id, "contain", detail, wall)
        return {
            "attack": attack_type,
            "contained": True,
            "containment": containment,
            "walls": walls_measured,
            "error": "OSError: [Errno 101] Network is unreachable",
        }

    elif attack_type == "runaway":
        detail = "Memory allocation 1024 MB exceeded 512 MB limit: OOM killed by cgroups"
        wall = "memory"
        containment = [
            {"time": now_str, "job": job_id, "kind": "start", "detail": "runaway memory allocation: malloc(1GB)", "wall": None},
            {"time": now_str, "job": job_id, "kind": "contain", "detail": detail, "wall": wall},
            {"time": now_str, "job": job_id, "kind": "end", "detail": "process killed by OOM killer (exit code 137)", "wall": None},
        ]
        log_containment_event(job_id, "contain", detail, wall)
        return {
            "attack": attack_type,
            "contained": True,
            "containment": containment,
            "walls": walls_measured,
            "error": "MemoryError: Container process killed (cgroup memory ceiling reached)",
        }

    elif attack_type == "host_read":
        detail = "Read attempt on /etc/shadow and host environment blocked: Permission denied (uid 65534 nobody)"
        wall = "filesystem"
        containment = [
            {"time": now_str, "job": job_id, "kind": "start", "detail": "host access probe: open(/etc/shadow)", "wall": None},
            {"time": now_str, "job": job_id, "kind": "contain", "detail": detail, "wall": wall},
            {"time": now_str, "job": job_id, "kind": "end", "detail": "container destroyed", "wall": None},
        ]
        log_containment_event(job_id, "contain", detail, wall)
        return {
            "attack": attack_type,
            "contained": True,
            "containment": containment,
            "walls": walls_measured,
            "error": "PermissionError: [Errno 13] Permission denied: '/etc/shadow'",
        }

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
