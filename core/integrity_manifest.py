"""Manifest helpers for reproducible experiment inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_prompt_manifest(prompts_dir: Path) -> dict:
    """SHA-256 of each prompt file + master hash of all files combined."""
    file_hashes = {}
    for f in sorted(prompts_dir.glob("*.json")):
        content = f.read_bytes()
        file_hashes[f.name] = hashlib.sha256(content).hexdigest()
    master = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()
    return {"files": file_hashes, "master_hash": master}


def compute_contract_manifest(contracts_dir: Path) -> dict:
    """SHA-256 of each contract file's contract_hash field."""
    contract_hashes = {}
    for f in sorted(contracts_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        contract_hashes[f.name] = data.get("contract_hash", "UNSEALED")
    master = hashlib.sha256(json.dumps(contract_hashes, sort_keys=True).encode()).hexdigest()
    return {"contracts": contract_hashes, "master_hash": master}


def verify_manifest(current: dict, stored_hash: str) -> bool:
    """Verify current manifest matches stored master hash."""
    return current.get("master_hash") == stored_hash
