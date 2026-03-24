"""Load, cryptographically seal, and verify RMIC identity contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "DataScope",
    "ParameterConstraint",
    "RMICContract",
    "canonical_contract_dict_for_hash",
    "compute_contract_hash",
    "load_contract",
    "seal_contract_file",
]


def canonical_contract_dict_for_hash(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serialisable dict for hashing (excludes contract_hash only)."""
    return {k: v for k, v in sorted(data.items()) if k != "contract_hash"}


def compute_contract_hash(data: Mapping[str, Any]) -> str:
    """SHA-256 over stable JSON of all fields except contract_hash."""
    payload = canonical_contract_dict_for_hash(dict(data))
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class DataScope:
    """Immutable data-scope slice from the contract JSON."""

    accessible: tuple[str, ...]
    prohibited: tuple[str, ...]
    pii_categories: tuple[str, ...]

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> DataScope:
        return DataScope(
            accessible=tuple(m.get("accessible") or ()),
            prohibited=tuple(m.get("prohibited") or ()),
            pii_categories=tuple(m.get("pii_categories") or ()),
        )


@dataclass(frozen=True)
class ParameterConstraint:
    """Single parameter bound as loaded from the contract."""

    name: str
    max: float | int | None
    min: float | int | None
    value_type: str

    @staticmethod
    def from_entry(name: str, spec: Mapping[str, Any]) -> ParameterConstraint:
        return ParameterConstraint(
            name=name,
            max=spec.get("max"),
            min=spec.get("min"),
            value_type=str(spec.get("type", "float")),
        )


@dataclass(frozen=True)
class RMICContract:
    """Frozen runtime identity contract. No field may change after construction."""

    agent_id: str
    role_name: str
    sector: str
    role_description: str
    semantic_anchors: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    data_scope: DataScope
    parameter_constraints: tuple[ParameterConstraint, ...]
    ids_warn_threshold: float
    ids_block_threshold: float
    drift_velocity_threshold: float
    recovery_policy: str
    compliance_tags: tuple[str, ...]
    contract_version: str
    created_at: str | None
    contract_hash: str
    anchor_embedding: tuple[float, ...]

    def constraints_by_name(self) -> dict[str, ParameterConstraint]:
        return {c.name: c for c in self.parameter_constraints}


def _as_tuple_str(v: Any) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def _contract_from_dict(data: dict[str, Any], *, require_hash_match: bool) -> RMICContract:
    stored_hash = data.get("contract_hash")
    if require_hash_match:
        if not stored_hash:
            raise ValueError("contract_hash missing; seal the contract before load with verify")
        computed = compute_contract_hash(data)
        if computed != stored_hash:
            raise ValueError("contract_hash mismatch — contract was tampered with or corrupted")

    ds_raw = data.get("data_scope") or {}
    if not isinstance(ds_raw, dict):
        raise TypeError("data_scope must be an object")

    pc_raw = data.get("parameter_constraints") or {}
    if not isinstance(pc_raw, dict):
        raise TypeError("parameter_constraints must be an object")

    constraints: list[ParameterConstraint] = []
    for name in sorted(pc_raw.keys()):
        spec = pc_raw[name]
        if not isinstance(spec, dict):
            raise TypeError(f"parameter_constraints.{name} must be an object")
        constraints.append(ParameterConstraint.from_entry(name, spec))

    anchors = _as_tuple_str(data.get("semantic_anchors"))
    if not anchors:
        raise ValueError("semantic_anchors must contain at least one sentence")

    emb = data.get("anchor_embedding")
    if emb is None:
        anchor_embedding: tuple[float, ...] = ()
    else:
        if not isinstance(emb, list):
            raise TypeError("anchor_embedding must be a list of floats (seal the contract first)")
        anchor_embedding = tuple(float(x) for x in emb)

    return RMICContract(
        agent_id=str(data["agent_id"]),
        role_name=str(data["role_name"]),
        sector=str(data["sector"]),
        role_description=str(data.get("role_description", "")),
        semantic_anchors=anchors,
        allowed_actions=_as_tuple_str(data.get("allowed_actions")),
        forbidden_actions=_as_tuple_str(data.get("forbidden_actions")),
        data_scope=DataScope.from_mapping(ds_raw),
        parameter_constraints=tuple(constraints),
        ids_warn_threshold=float(data.get("ids_warn_threshold", 0.35)),
        ids_block_threshold=float(data.get("ids_block_threshold", 0.60)),
        drift_velocity_threshold=float(data.get("drift_velocity_threshold", 0.05)),
        recovery_policy=str(data.get("recovery_policy", "re-anchor")),
        compliance_tags=_as_tuple_str(data.get("compliance_tags")),
        contract_version=str(data.get("contract_version", "1.0.0")),
        created_at=data.get("created_at"),
        contract_hash=str(stored_hash or compute_contract_hash(data)),
        anchor_embedding=anchor_embedding,
    )


def load_contract(path: str | Path, *, verify_hash: bool = True) -> RMICContract:
    """Load a contract JSON file and optionally verify SHA-256 integrity."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("contract file must contain a JSON object")
    return _contract_from_dict(raw, require_hash_match=verify_hash)


def seal_contract_file(
    path: str | Path,
    *,
    write_back: bool = True,
    model_name: str | None = None,
) -> RMICContract:
    """
    Compute anchor_embedding (once) and contract_hash, optionally persist to disk.

    anchor_embedding is the L2-normalised mean embedding of semantic_anchors.
    """
    from core.embedder import anchor_centroid_from_anchors

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("contract file must contain a JSON object")

    anchors = _as_tuple_str(data.get("semantic_anchors"))
    if not anchors:
        raise ValueError("semantic_anchors must contain at least one sentence")

    centroid = anchor_centroid_from_anchors(list(anchors), model_name=model_name)
    data["anchor_embedding"] = [float(x) for x in centroid.tolist()]

    if data.get("created_at") in (None, ""):
        data["created_at"] = datetime.now(timezone.utc).isoformat()

    data["contract_hash"] = compute_contract_hash(data)

    if write_back:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return _contract_from_dict(data, require_hash_match=True)
