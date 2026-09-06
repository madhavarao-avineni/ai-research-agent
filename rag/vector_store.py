"""
rag/vector_store.py
===================
Vector store behind an abstract interface so Chroma can be swapped later
(pgvector, FAISS, etc.) without touching agents.

`VectorStore` is the interface. `ChromaVectorStore` is the default backend.
`InMemoryVectorStore` is a dependency-free fallback used when chromadb isn't
installed (keeps tests/offline runs working). Both compute cosine similarity
over embeddings produced by rag.embeddings.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol

from observability.logging_config import get_logger
from rag.embeddings import Embedder, get_embedder

log = get_logger("rag.vector_store")


class VectorStore(Protocol):
    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None: ...
    def query(self, text: str, n_results: int = 5) -> list[dict]: ...
    def count(self) -> int: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Simple, dependency-free cosine-similarity store."""

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self._embedder = embedder or get_embedder()
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self._vecs: list[list[float]] = []
        log.info("Using InMemoryVectorStore (embedder=%s)", self._embedder.name)

    def add(self, ids, documents, metadatas) -> None:
        vecs = self._embedder.embed(documents)
        self._ids += ids
        self._docs += documents
        self._metas += metadatas
        self._vecs += vecs

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        if not self._vecs:
            return []
        qv = self._embedder.embed([text])[0]
        scored = [
            (i, _cosine(qv, v)) for i, v in enumerate(self._vecs)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        out = []
        for idx, score in scored[:n_results]:
            out.append(
                {
                    "id": self._ids[idx],
                    "document": self._docs[idx],
                    "metadata": self._metas[idx],
                    "score": round(float(score), 4),
                }
            )
        return out

    def count(self) -> int:
        return len(self._ids)


class ChromaVectorStore:
    """Chroma backend (persistent). Uses our embedder for consistency."""

    def __init__(self, persist_dir: str, collection: str = "research", embedder: Optional[Embedder] = None) -> None:
        import chromadb  # lazy

        self._embedder = embedder or get_embedder()
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        log.info("Using ChromaVectorStore at %s (embedder=%s)", persist_dir, self._embedder.name)

    def add(self, ids, documents, metadatas) -> None:
        embeddings = self._embedder.embed(documents)
        self._col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        qv = self._embedder.embed([text])[0]
        res = self._col.query(query_embeddings=[qv], n_results=n_results)
        out = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": round(1.0 - float(dists[i]), 4) if i < len(dists) else 0.0,
                }
            )
        return out

    def count(self) -> int:
        return self._col.count()


def get_vector_store(persist_dir: str, embedder: Optional[Embedder] = None) -> VectorStore:
    """Return Chroma if available, else the in-memory fallback."""
    try:
        return ChromaVectorStore(persist_dir, embedder=embedder)
    except Exception as exc:  # noqa: BLE001
        log.warning("Chroma unavailable (%s); using in-memory store.", exc)
        return InMemoryVectorStore(embedder=embedder)
