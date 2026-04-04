"""Seal RMIC contracts with real embeddings when possible, else stdlib-only fallback."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CONTRACT_RELPATHS = (
    "contracts/financial_agent.json",
    "contracts/support_agent.json",
    "contracts/healthcare_research_agent.json",
)


def compute_contract_hash(data: dict) -> str:
    """SHA-256 over stable JSON of all fields except contract_hash (matches core.contract_loader)."""
    payload = {k: v for k, v in sorted(data.items()) if k != "contract_hash"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _seal_fallback(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("contract file must contain a JSON object")

    data["anchor_embedding"] = []
    if data.get("created_at") in (None, ""):
        data["created_at"] = datetime.now(timezone.utc).isoformat()
    data["contract_hash"] = compute_contract_hash(data)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    agent_id = str(data.get("agent_id", path.stem))
    h = data["contract_hash"]
    print(f"Sealed (fallback mode — no embedding): {agent_id}: {h[:16]}...")


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = [root / rel for rel in CONTRACT_RELPATHS]

    try:
        from core.contract_loader import seal_contract_file

        for p in paths:
            seal_contract_file(p)
    except (ImportError, OSError, Exception):
        for p in paths:
            _seal_fallback(p)

    print("All 3 contracts sealed. Now run: python setup.py")


if __name__ == "__main__":
    main()
