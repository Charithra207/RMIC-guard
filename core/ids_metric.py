"""Intent Drift Score (IDS) — role distance, semantic grounding, trajectory curvature."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from core.embedder import cosine_similarity, embed_texts, normalise_l2

__all__ = [
    "compute_ids",
    "embedding_for_text",
    "role_distance",
    "semantic_grounding",
    "trajectory_curvature",
]

# IDS = 0.4 * Role_Distance + 0.4 * Semantic_Grounding + 0.2 * Trajectory_Curvature
W_ROLE = 0.4
W_GROUND = 0.4
W_TRAJ = 0.2


def embedding_for_text(text: str, model_name: str | None = None) -> np.ndarray:
    v = embed_texts([text], model_name=model_name)[0]
    return normalise_l2(np.asarray(v, dtype=np.float32))


def role_distance(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    model_name: str | None = None,
) -> float:
    """
    Role_Distance = 1 - cosine_sim(embed(agent_output), anchor_centroid).
    anchor_embedding must be the sealed centroid from the contract.
    """
    anchor = normalise_l2(np.asarray(list(anchor_embedding), dtype=np.float32))
    out = embedding_for_text(agent_output, model_name=model_name)
    sim = cosine_similarity(out, anchor)
    return float(max(0.0, min(1.0, 1.0 - sim)))


def semantic_grounding(
    agent_output: str,
    allowed_topics: Sequence[str],
    forbidden_topics: Sequence[str],
    model_name: str | None = None,
) -> float:
    """
    Semantic_Grounding = 1 - (sim(output, allowed_topics) - sim(output, forbidden_topics)),
    each group similarity = max cosine sim to any phrase in the group.
    """
    out = embedding_for_text(agent_output, model_name=model_name)

    def _max_sim(phrases: Sequence[str]) -> float:
        if not phrases:
            return 0.0
        vecs = embed_texts(list(phrases), model_name=model_name)
        sims = vecs @ out
        return float(np.max(sims))

    s_allowed = _max_sim(allowed_topics)
    s_forbidden = _max_sim(forbidden_topics)
    raw = s_allowed - s_forbidden
    # Map to [0, 1]: high grounding drift when output aligns more with forbidden than allowed
    return float(max(0.0, min(1.0, 1.0 - raw)))


def trajectory_curvature(recent_ids: Sequence[float]) -> float:
    """
    Mean |IDS_t - IDS_{t-1}| over the last window, normalised to [0, 1].

    Uses the last up to 5 scores; if fewer than 2 points, returns 0.
    """
    if len(recent_ids) < 2:
        return 0.0
    window = list(recent_ids)[-5:]
    deltas = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
    mean_delta = float(sum(deltas) / len(deltas))
    # Normalise: IDS components are in [0,1], so per-step delta in [0,1]; cap at 1
    return float(max(0.0, min(1.0, mean_delta)))


def compute_ids(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    recent_ids: Sequence[float] | None = None,
    model_name: str | None = None,
) -> float:
    """Composite IDS in [0, 1]."""
    rd = role_distance(agent_output, anchor_embedding, model_name=model_name)
    allowed = allowed_topics if allowed_topics is not None else ()
    forbidden = forbidden_topics if forbidden_topics is not None else ()
    sg = semantic_grounding(agent_output, allowed, forbidden, model_name=model_name)
    tc = trajectory_curvature(recent_ids or ())
    ids = W_ROLE * rd + W_GROUND * sg + W_TRAJ * tc
    return float(max(0.0, min(1.0, ids)))
