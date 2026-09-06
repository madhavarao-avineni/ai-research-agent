"""
rag/retriever.py
================
RAG pipeline: chunk -> embed -> store -> semantic retrieve.

    Documents -> Chunking -> Embeddings -> Vector DB -> Semantic Retrieval -> LLM

`RagRetriever` wraps a `VectorStore`. `index_sources()` chunks RetrievedSource
content and upserts it with provenance metadata (so retrieved chunks trace back
to their source_id). `search()` returns semantically-relevant chunks with
scores and the originating source metadata preserved.
"""

from __future__ import annotations

from typing import Optional

from config.settings import get_settings
from models.state import RetrievedSource
from observability.logging_config import get_logger
from rag.vector_store import VectorStore, get_vector_store

log = get_logger("rag.retriever")


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Word-based chunking with overlap. Simple, deterministic, dependency-free."""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


class RagRetriever:
    def __init__(self, store: Optional[VectorStore] = None) -> None:
        settings = get_settings()
        self._store = store or get_vector_store(settings.chroma_persist_dir)

    def index_sources(self, sources: list[RetrievedSource]) -> int:
        """Chunk + embed + store sources. Returns number of chunks indexed."""
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for src in sources:
            chunks = chunk_text(src.content)
            for i, chunk in enumerate(chunks):
                ids.append(f"{src.source_id}__c{i}")
                docs.append(chunk)
                metas.append(
                    {
                        "source_id": src.source_id,
                        "title": src.title,
                        "url": src.url or "",
                        "source_type": src.source_type.value,
                        "publisher": src.publisher or "",
                        "publication_date": src.publication_date or "",
                    }
                )
        if not ids:
            log.info("index_sources: nothing to index.")
            return 0
        try:
            self._store.add(ids, docs, metas)
        except Exception as exc:  # noqa: BLE001
            log.error("index_sources failed: %s", type(exc).__name__)
            return 0
        log.info("Indexed %d chunks from %d sources.", len(ids), len(sources))
        return len(ids)

    def search(self, query: str, *, n_results: int = 5) -> list[dict]:
        """Semantic search over indexed chunks. Returns [] on failure."""
        try:
            return self._store.query(query, n_results=n_results)
        except Exception as exc:  # noqa: BLE001
            log.error("rag search failed: %s", type(exc).__name__)
            return []

    def count(self) -> int:
        try:
            return self._store.count()
        except Exception:  # noqa: BLE001
            return 0
