"""
tools/document_retrieval.py
==========================
Document Retrieval Tool — RAG-backed retrieval over indexed documents, plus a
hook for future PDF / internal-document ingestion.

Today it:
  * ingests a list of RetrievedSource (or raw text) into the RAG store,
  * ingests PDFs from disk (best-effort, via pypdf) for future integration,
  * semantically retrieves relevant chunks for a query, returning
    RetrievedSource objects (INTERNAL_DOCUMENT) with provenance metadata.

Designed so the RAG layer can later be swapped and new connectors (SharePoint,
Drive, SQL) added without changing the agent that calls this tool.
"""

from __future__ import annotations

from typing import Optional

from models.state import RetrievedSource, SourceType
from observability.logging_config import get_logger
from rag.retriever import RagRetriever
from tools.base import make_source

log = get_logger("tool.document_retrieval")


class DocumentRetrievalTool:
    def __init__(self, retriever: Optional[RagRetriever] = None) -> None:
        self._rag = retriever or RagRetriever()

    # --- ingestion -------------------------------------------------------
    def ingest_sources(self, sources: list[RetrievedSource]) -> int:
        return self._rag.index_sources(sources)

    def ingest_pdf(self, path: str, *, publisher: Optional[str] = None) -> int:
        """Best-effort PDF ingestion. Returns chunks indexed (0 on failure)."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            log.error("PDF ingest failed for %s: %s", path, type(exc).__name__)
            return 0
        if not text.strip():
            log.warning("PDF %s produced no extractable text (scanned?).", path)
            return 0
        src = make_source(
            title=path.split("/")[-1].split("\\")[-1],
            url=None,
            source_type=SourceType.INTERNAL_DOCUMENT,
            content=text,
            publisher=publisher,
            relevance_score=0.6,
        )
        return self._rag.index_sources([src])

    # --- retrieval -------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        n_results: int = 5,
        sub_question_id: Optional[str] = None,
    ) -> list[RetrievedSource]:
        """Semantic retrieval -> RetrievedSource objects (never raises)."""
        hits = self._rag.search(query, n_results=n_results)
        out: list[RetrievedSource] = []
        for h in hits:
            meta = h.get("metadata", {}) or {}
            out.append(
                make_source(
                    title=meta.get("title") or "(document)",
                    url=meta.get("url") or None,
                    source_type=SourceType.INTERNAL_DOCUMENT,
                    content=h.get("document", ""),
                    snippet=(h.get("document", "") or "")[:300],
                    publisher=meta.get("publisher") or None,
                    publication_date=meta.get("publication_date") or None,
                    relevance_score=float(h.get("score", 0.5)),
                    sub_question_id=sub_question_id,
                )
            )
        log.info("document_retrieval '%s' -> %d chunks", query[:60], len(out))
        return out
