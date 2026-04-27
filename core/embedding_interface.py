"""Embedding abstraction for local and OpenAI embeddings."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from core.embedder import embed_texts, normalise_l2


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


class LocalEmbeddingProvider:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    def embed(self, text: str) -> np.ndarray:
        vec = embed_texts([text], model_name=self.model_name)[0]
        return normalise_l2(np.asarray(vec, dtype=np.float32))


class OpenAIEmbeddingProvider:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        from openai import OpenAI

        key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key)
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        out = self.client.embeddings.create(model=self.model, input=text)
        vec = np.asarray(out.data[0].embedding, dtype=np.float32)
        return normalise_l2(vec)

