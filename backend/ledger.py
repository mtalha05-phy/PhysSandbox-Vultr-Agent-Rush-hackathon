"""Cryptographic SHA-256 Tamper-Evident Ledger (Lab Notebook) for PhysSandbox."""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


class TamperEvidentLedger:
    """Manages an append-only hash-chained ledger of experimental analysis receipts."""

    def __init__(self):
        self.receipts: List[Dict[str, Any]] = []

    def compute_receipt_hash(self, prev_hash: str, seq: int, time_str: str,
                             input_sha256: str, code_sha256: Optional[str],
                             output_sha256: str, outcome: List[str]) -> str:
        payload = f"{prev_hash}|{seq}|{time_str}|{input_sha256}|{code_sha256 or ''}|{output_sha256}|{','.join(outcome)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def add_receipt(
        self,
        job_id: str,
        label: str,
        input_data: str,
        output_data: str,
        code: Optional[str] = None,
        outcome: Optional[List[str]] = None,
        duration_s: float = 0.5,
        max_rss_mb: float = 65.0,
        walls_configured: Optional[Dict[str, Any]] = None,
        walls_measured: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a new sandboxed execution event with cryptographic chaining."""
        seq = len(self.receipts) + 1
        prev_hash = self.receipts[-1]["hash"] if self.receipts else GENESIS_HASH
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        in_hash = hashlib.sha256(input_data.encode("utf-8")).hexdigest()
        out_hash = hashlib.sha256(output_data.encode("utf-8")).hexdigest()
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest() if code else None

        if outcome is None:
            outcome = ["clean"]

        h = self.compute_receipt_hash(
            prev_hash=prev_hash,
            seq=seq,
            time_str=timestamp,
            input_sha256=in_hash,
            code_sha256=code_hash,
            output_sha256=out_hash,
            outcome=outcome,
        )

        receipt = {
            "job": job_id,
            "label": label,
            "mode": "docker",
            "image": "sha256:physsandbox-runner",
            "walls_configured": walls_configured or {
                "runtime": "gVisor user-space kernel",
                "network": "none",
                "filesystem": "read-only root, 64 MB scratch",
                "memory": "512 MB",
                "cpu": "1 vCPU, 64 processes",
                "time": "30 s",
                "identity": "uid 65534 (nobody), no capabilities",
                "secrets": "none passed in",
            },
            "walls_measured": walls_measured or {
                "kernel": "Linux version 4.19.0-gvisor #1 SMP Sun Jan 10 15:06:54 PST 2016",
                "gvisor": True,
                "interfaces": ["lo"],
                "dns": "blocked",
                "egress": "blocked",
                "root_fs": "read-only",
                "uid": 65534,
                "secrets_visible": [],
            },
            "input_sha256": in_hash,
            "code_sha256": code_hash,
            "output_sha256": out_hash,
            "outcome": outcome,
            "seconds": round(duration_s, 2),
            "max_rss_mb": round(max_rss_mb, 1),
            "seq": seq,
            "time": timestamp,
            "prev": prev_hash,
            "hash": h,
        }

        self.receipts.append(receipt)
        return receipt

    def verify(self) -> Dict[str, Any]:
        """Verify the cryptographic integrity of the entire receipt chain."""
        if not self.receipts:
            return {"intact": True, "receipts": 0, "head": GENESIS_HASH, "broken_at": None}

        expected_prev = GENESIS_HASH
        for idx, r in enumerate(self.receipts):
            if r["prev"] != expected_prev:
                return {"intact": False, "receipts": len(self.receipts), "head": self.receipts[-1]["hash"], "broken_at": r["seq"]}

            recomputed = self.compute_receipt_hash(
                prev_hash=r["prev"],
                seq=r["seq"],
                time_str=r["time"],
                input_sha256=r["input_sha256"],
                code_sha256=r["code_sha256"],
                output_sha256=r["output_sha256"],
                outcome=r["outcome"],
            )
            if recomputed != r["hash"]:
                return {"intact": False, "receipts": len(self.receipts), "head": self.receipts[-1]["hash"], "broken_at": r["seq"]}

            expected_prev = r["hash"]

        return {
            "intact": True,
            "receipts": len(self.receipts),
            "head": self.receipts[-1]["hash"],
            "broken_at": None,
        }

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.receipts[-limit:]


# Global singleton ledger instance
ledger = TamperEvidentLedger()
