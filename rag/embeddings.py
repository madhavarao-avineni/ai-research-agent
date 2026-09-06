"""
rag/embeddings.py
=================
Pluggable embeddings behind a tiny interface so the provider can be swapped.

Default: local sentence-transformers (`all-MiniLM-L6-v2`) — no API key.
Optional: OpenAI embeddings when EMBEDDINGS_PROVIDER=openai and a key is set.

`get_embedder()` returns an object with `.embed(list[str]) -> list[list[float]]`
and a `.name` for logging. Falls back to a deterministic hashing embedder if no
backend is available so the RAG pipeline still runs in tests/offline.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from config.settings import get_settings
from observability.logging_config import get_logger

log = get_logger("rag.embeddings")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """sentence-transformers backend."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self.name = f"local:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class OpenAIEmbedder:
    """OpenAI embeddings backend."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # lazy

        self.name = f"openai:{model}"
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


class HashingEmbedder:
    """
    Deterministic, dependency-free fallback embedder (bag-of-hashed-tokens).
    Not semantically strong, but keeps the pipeline functional offline and in
    tests. Vectors are L2-normalized so cosine similarity behaves.
    """

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing:{dim}"
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self._dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder() -> Embedder:
    """Select an embedder based on settings, with graceful fallback."""
    settings = get_settings()
    if settings.use_openai_embeddings:
        try:
            return OpenAIEmbedder(settings.openai_api_key)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenAI embedder unavailable (%s); trying local.", exc)
    try:
        return LocalEmbedder(settings.embeddings_model)
    except Exception as exc:  # noqa: BLE001
        log.warning("Local embedder unavailable (%s); using hashing fallback.", exc)
        return HashingEmbedder()
