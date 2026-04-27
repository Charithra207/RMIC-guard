"""Basic threshold tuning helpers for IDS decisions."""

from __future__ import annotations

from typing import Any


def tune_thresholds(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """
    Basic grid search tuning.

    dataset item format:
      {"ids_score": float, "is_drift": bool}
    """
    if not dataset:
        return {
            "warn_threshold": 0.35,
            "block_threshold": 0.60,
            "velocity_threshold": 0.05,
        }

    candidates_warn = [0.20, 0.25, 0.30, 0.35, 0.40]
    candidates_block = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best = (float("-inf"), 0.35, 0.60)

    for warn in candidates_warn:
        for block in candidates_block:
            if warn >= block:
                continue
            tp = fp = fn = tn = 0
            for row in dataset:
                score = float(row.get("ids_score", 0.0))
                is_drift = bool(row.get("is_drift", False))
                pred_block = score >= block
                if is_drift and pred_block:
                    tp += 1
                elif is_drift and not pred_block:
                    fn += 1
                elif (not is_drift) and pred_block:
                    fp += 1
                else:
                    tn += 1
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            objective = tpr - 0.5 * fpr
            if objective > best[0]:
                best = (objective, warn, block)

    return {
        "warn_threshold": float(best[1]),
        "block_threshold": float(best[2]),
        "velocity_threshold": 0.05,
    }

