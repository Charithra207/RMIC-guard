"""Local sentence embeddings for IDS and sealed anchor centroid."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_MODEL_NAME",
    "anchor_centroid_from_anchors",
    "cosine_similarity",
    "embed_texts",
    "get_model",
    "normalise_l2",
]

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # fastembed model name


@lru_cache(maxsize=4)
def get_model(model_name: str | None = None) -> Any:
    """Return a fastembed TextEmbedding model, or SentenceTransformer if fastembed is unavailable."""
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name or DEFAULT_MODEL_NAME)
    except (ImportError, OSError):
        from sentence_transformers import SentenceTransformer as ST

        return ST(model_name or "all-MiniLM-L6-v2")


def _embed_with_sentence_transformers(texts: list[str], model_name: str | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_st = SentenceTransformer(model_name or "all-MiniLM-L6-v2")
    emb = model_st.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name or DEFAULT_MODEL_NAME)
        embeddings = list(model.embed(texts))
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32)
    except (ImportError, OSError):
        return _embed_with_sentence_transformers(texts, model_name)


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
