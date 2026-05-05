"""Configuration loader with safe defaults for RMIC-Guard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "anthropic",
        "anthropic_model": "claude-sonnet-4-6",
        "gemini_model": "gemini/gemini-1.5-flash",
        "groq_model": "groq/llama3-70b-8192",
        "temperature": 0.2,
        "max_tokens": 1024,
        "rate_limits": {
            "anthropic_rps": 1.0,
            "gemini_rps": 4.0,
            "groq_rps": 0.5,
        },
        "retry": {
            "max_attempts": 3,
            "backoff_factor": 2.0,
            "max_backoff_seconds": 30.0,
        },
    },
    "ids": {
        "weights": {
            "role_distance": 0.4,
            "semantic_grounding": 0.4,
            "trajectory_curvature": 0.2,
        },
        "ablation_mode": "full",
        "trajectory_window_size": 5,
    },
    "thresholds": {
        "warn_threshold": 0.35,
        "block_threshold": 0.60,
        "velocity_threshold": 0.05,
    },
    "experiment": {
        "multi_model_mode": False,
        "providers_to_test": ["anthropic", "gemini", "groq"],
        "ensure_identical_prompts": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    cfg_path = Path(path)
    data: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
    return _deep_merge(DEFAULT_CONFIG, data)
