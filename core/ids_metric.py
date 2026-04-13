"""Intent Drift Score (IDS) with parallel statistical drift signals."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import softmax

from core.embedder import cosine_similarity, embed_texts, normalise_l2

__all__ = [
    "compute_ids",
    "compute_ids_components",
    "embedding_for_text",
    "role_distance",
    "semantic_grounding",
    "trajectory_curvature",
    "mahalanobis_drift",
    "kl_divergence_drift",
    "jensen_shannon_drift",
    "wasserstein_drift",
    "hellinger_drift",
    "tool_frequency_drift",
]

# Base IDS = 0.4*Role + 0.4*Grounding + 0.2*Trajectory
W_ROLE = 0.4
W_GROUND = 0.4
W_TRAJ = 0.2

EPS = 1e-8


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


def _topic_embeddings(topics: Sequence[str], model_name: str | None = None) -> np.ndarray | None:
    if not topics:
        return None
    vecs = embed_texts(list(topics), model_name=model_name)
    return np.asarray(vecs, dtype=np.float32)


def _topic_distribution(
    query_embedding: np.ndarray,
    topic_embeddings: np.ndarray,
    *,
    temperature: float = 0.15,
) -> np.ndarray:
    # Use cosine-like similarities and convert to probabilities via temperature-scaled softmax.
    q = normalise_l2(np.asarray(query_embedding, dtype=np.float32))
    t = np.asarray(topic_embeddings, dtype=np.float32)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), EPS)
    logits = (t @ q) / max(temperature, EPS)
    probs = softmax(logits)
    return np.asarray(probs, dtype=np.float64)


def mahalanobis_drift(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    model_name: str | None = None,
) -> float:
    """
    Mahalanobis drift between output embedding and anchor centroid.
    Topic embeddings provide an empirical covariance estimate.
    """
    anchor = normalise_l2(np.asarray(list(anchor_embedding), dtype=np.float32))
    out = embedding_for_text(agent_output, model_name=model_name)

    topics = tuple(allowed_topics or ()) + tuple(forbidden_topics or ())
    topic_vecs = _topic_embeddings(topics, model_name=model_name)
    if topic_vecs is None or topic_vecs.shape[0] < 2:
        # Fallback to Euclidean distance if covariance cannot be estimated.
        dist = float(np.linalg.norm(out - anchor))
        return float(dist / (1.0 + dist))

    cov = np.cov(topic_vecs, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    dim = cov.shape[0]
    cov = cov + (1e-3 * np.eye(dim, dtype=np.float64))
    inv_cov = np.linalg.pinv(cov)
    delta = (out - anchor).astype(np.float64)
    dist_sq = float(delta.T @ inv_cov @ delta)
    dist = float(np.sqrt(max(0.0, dist_sq)))
    return float(dist / (1.0 + dist))


def kl_divergence_drift(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    model_name: str | None = None,
) -> float:
    """
    KL divergence between output topic distribution and anchor-induced topic distribution.
    Returns value normalised to [0, 1].
    """
    topics = tuple(allowed_topics or ()) + tuple(forbidden_topics or ())
    topic_vecs = _topic_embeddings(topics, model_name=model_name)
    if topic_vecs is None or topic_vecs.shape[0] == 0:
        return 0.0

    out = embedding_for_text(agent_output, model_name=model_name)
    anchor = normalise_l2(np.asarray(list(anchor_embedding), dtype=np.float32))
    p = _topic_distribution(out, topic_vecs) + EPS
    q = _topic_distribution(anchor, topic_vecs) + EPS
    p = p / np.sum(p)
    q = q / np.sum(q)

    kl = float(np.sum(p * np.log(p / q)))
    return float(1.0 - np.exp(-max(0.0, kl)))


def jensen_shannon_drift(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    model_name: str | None = None,
) -> float:
    """
    Jensen-Shannon divergence between output and anchor topic distributions.
    Returns value in [0, 1].
    """
    topics = tuple(allowed_topics or ()) + tuple(forbidden_topics or ())
    topic_vecs = _topic_embeddings(topics, model_name=model_name)
    if topic_vecs is None or topic_vecs.shape[0] == 0:
        return 0.0

    out = embedding_for_text(agent_output, model_name=model_name)
    anchor = normalise_l2(np.asarray(list(anchor_embedding), dtype=np.float32))
    p = _topic_distribution(out, topic_vecs) + EPS
    q = _topic_distribution(anchor, topic_vecs) + EPS
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    js_max = np.log(2.0)
    return float(max(0.0, min(1.0, float(js / js_max))))


def wasserstein_drift(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    model_name: str | None = None,
) -> float:
    topics = tuple(allowed_topics or ()) + tuple(forbidden_topics or ())
    topic_vecs = _topic_embeddings(topics, model_name=model_name)
    if topic_vecs is None or topic_vecs.shape[0] == 0:
        return 0.0
    out = embedding_for_text(agent_output, model_name=model_name)
    anchor = normalise_l2(np.asarray(list(anchor_embedding), dtype=np.float32))
    p = _topic_distribution(out, topic_vecs)
    q = _topic_distribution(anchor, topic_vecs)
    wd = float(wasserstein_distance(p, q))
    max_expected_wasserstein = 2.0
    return float(max(0.0, min(1.0, wd / max_expected_wasserstein)))


def hellinger_drift(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    model_name: str | None = None,
) -> float:
    topics = tuple(allowed_topics or ()) + tuple(forbidden_topics or ())
    topic_vecs = _topic_embeddings(topics, model_name=model_name)
    if topic_vecs is None or topic_vecs.shape[0] == 0:
        return 0.0
    out = embedding_for_text(agent_output, model_name=model_name)
    anchor = normalise_l2(np.asarray(list(anchor_embedding), dtype=np.float32))
    p = _topic_distribution(out, topic_vecs) + EPS
    q = _topic_distribution(anchor, topic_vecs) + EPS
    p = p / np.sum(p)
    q = q / np.sum(q)
    h = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    return float(max(0.0, min(1.0, float(h))))


def tool_frequency_drift(
    tool_call_history: list[str],
    allowed_actions: tuple[str, ...],
    window_size: int = 10,
) -> float:
    if len(tool_call_history) < 2:
        return 0.0
    window = tool_call_history[-max(1, window_size):]
    non_allowed = sum(1 for name in window if name not in allowed_actions)
    return float(max(0.0, min(1.0, non_allowed / len(window))))


def compute_ids_components(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    recent_ids: Sequence[float] | None = None,
    tool_call_history: list[str] | None = None,
    allowed_actions: tuple[str, ...] | None = None,
    model_name: str | None = None,
) -> dict[str, float]:
    """Return base IDS and three independent statistical scores in [0, 1]."""
    allowed = tuple(allowed_topics or ())
    forbidden = tuple(forbidden_topics or ())

    rd = role_distance(agent_output, anchor_embedding, model_name=model_name)
    sg = semantic_grounding(agent_output, allowed, forbidden, model_name=model_name)
    tc = trajectory_curvature(recent_ids or ())
    base_ids = W_ROLE * rd + W_GROUND * sg + W_TRAJ * tc

    mahal = mahalanobis_drift(
        agent_output,
        anchor_embedding,
        allowed_topics=allowed,
        forbidden_topics=forbidden,
        model_name=model_name,
    )
    kl = kl_divergence_drift(
        agent_output,
        anchor_embedding,
        allowed_topics=allowed,
        forbidden_topics=forbidden,
        model_name=model_name,
    )
    js = jensen_shannon_drift(
        agent_output,
        anchor_embedding,
        allowed_topics=allowed,
        forbidden_topics=forbidden,
        model_name=model_name,
    )
    wasserstein = wasserstein_drift(
        agent_output,
        anchor_embedding,
        allowed_topics=allowed,
        forbidden_topics=forbidden,
        model_name=model_name,
    )
    hellinger = hellinger_drift(
        agent_output,
        anchor_embedding,
        allowed_topics=allowed,
        forbidden_topics=forbidden,
        model_name=model_name,
    )
    tf = tool_frequency_drift(
        tool_call_history=list(tool_call_history or []),
        allowed_actions=tuple(allowed_actions or ()),
    )
    return {
        "role_distance": float(max(0.0, min(1.0, rd))),
        "semantic_grounding": float(max(0.0, min(1.0, sg))),
        "trajectory_curvature": float(max(0.0, min(1.0, tc))),
        "base_ids": float(max(0.0, min(1.0, base_ids))),
        "mahalanobis": float(max(0.0, min(1.0, mahal))),
        "kl_divergence": float(max(0.0, min(1.0, kl))),
        "js_divergence": float(max(0.0, min(1.0, js))),
        "wasserstein": float(max(0.0, min(1.0, wasserstein))),
        "hellinger": float(max(0.0, min(1.0, hellinger))),
        "tool_frequency": float(max(0.0, min(1.0, tf))),
    }


def compute_ids(
    agent_output: str,
    anchor_embedding: Sequence[float] | np.ndarray,
    *,
    allowed_topics: Sequence[str] | None = None,
    forbidden_topics: Sequence[str] | None = None,
    recent_ids: Sequence[float] | None = None,
    model_name: str | None = None,
) -> float:
    """Base IDS in [0, 1] (original cosine-based formula only)."""
    components = compute_ids_components(
        agent_output,
        anchor_embedding,
        allowed_topics=allowed_topics,
        forbidden_topics=forbidden_topics,
        recent_ids=recent_ids,
        model_name=model_name,
    )
    return components["base_ids"]
