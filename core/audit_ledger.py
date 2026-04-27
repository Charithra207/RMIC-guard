"""Append-only audit ledger with per-entry Ed25519 signatures."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

__all__ = ["AuditEntry", "AuditLedger"]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AuditEntry:
    timestamp: str
    agent_id: str
    input_hash: str
    ids_score: float
    drift_type: str | None
    drift_velocity: float | None
    decision: str
    contract_hash: str
    recovery_attempted: bool
    role_distance: float | None = None
    semantic_grounding: float | None = None
    trajectory_curvature: float | None = None
    failure_reason: str | None = None
    false_positive: bool = False
    false_negative: bool = False


class AuditLedger:
    """
    Append-only JSONL ledger. Each line is a JSON object:
    { "entry": {...}, "signature": "<base64>" }
    """

    def __init__(self, path: str | Path, private_key: Ed25519PrivateKey | None = None) -> None:
        self.path = Path(path)
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._public_key

    def public_key_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def append(self, entry: AuditEntry) -> None:
        payload = json.dumps(asdict(entry), sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = self._private_key.sign(payload)
        line = json.dumps(
            {
                "entry": asdict(entry),
                "signature": base64.b64encode(sig).decode("ascii"),
            },
            separators=(",", ":"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def verify_line(line: str, public_key: Ed25519PublicKey) -> bool:
        obj = json.loads(line)
        entry = obj["entry"]
        sig = base64.b64decode(obj["signature"])
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            public_key.verify(sig, payload)
        except Exception:
            return False
        return True


def hash_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 text (for input_hash)."""
    return sha256(text.encode("utf-8")).hexdigest()


def entry_from_mapping(m: Mapping[str, Any]) -> AuditEntry:
    return AuditEntry(
        timestamp=str(m["timestamp"]),
        agent_id=str(m["agent_id"]),
        input_hash=str(m["input_hash"]),
        ids_score=float(m["ids_score"]),
        role_distance=None if m.get("role_distance") is None else float(m["role_distance"]),
        semantic_grounding=(
            None if m.get("semantic_grounding") is None else float(m["semantic_grounding"])
        ),
        trajectory_curvature=(
            None if m.get("trajectory_curvature") is None else float(m["trajectory_curvature"])
        ),
        drift_type=m.get("drift_type"),
        drift_velocity=None if m.get("drift_velocity") is None else float(m["drift_velocity"]),
        decision=str(m["decision"]),
        failure_reason=(None if m.get("failure_reason") is None else str(m.get("failure_reason"))),
        false_positive=bool(m.get("false_positive", False)),
        false_negative=bool(m.get("false_negative", False)),
        contract_hash=str(m["contract_hash"]),
        recovery_attempted=bool(m["recovery_attempted"]),
    )
