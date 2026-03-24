"""Local sentence embeddings for IDS and sealed anchor centroid."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = [
    "DEFAULT_MODEL_NAME",
    "anchor_centroid_from_anchors",
    "cosine_similarity",
    "embed_texts",
    "get_model",
    "normalise_l2",
]

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_model(model_name: str | None = None) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer as ST

    name = model_name or DEFAULT_MODEL_NAME
    return ST(name)


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_model(model_name)
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def normalise_l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return (v / n).astype(np.float32)


def anchor_centroid_from_anchors(
    semantic_anchors: list[str],
    model_name: str | None = None,
) -> np.ndarray:
    """Mean of anchor embeddings, then L2-normalised (single centroid vector)."""
    e = embed_texts(semantic_anchors, model_name=model_name)
    if e.size == 0:
        raise ValueError("semantic_anchors is empty")
    centroid = np.mean(e, axis=0).astype(np.float32)
    return normalise_l2(centroid)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalised vectors equals dot product."""
    return float(np.dot(np.asarray(a, dtype=np.float32).ravel(), np.asarray(b, dtype=np.float32).ravel()))
