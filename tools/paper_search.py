"""
tools/paper_search.py
=====================
Research Paper Tool — searches arXiv (public, no key required).

Returns `RetrievedSource` objects with abstract as content, plus real author,
date, and URL metadata. Degrades gracefully to [] on failure.
"""

from __future__ import annotations

from typing import Optional

from models.state import RetrievedSource, SourceType
from observability.logging_config import get_logger
from tools.base import make_source

log = get_logger("tool.paper_search")


def paper_search(
    query: str,
    *,
    max_results: int = 5,
    sub_question_id: Optional[str] = None,
) -> list[RetrievedSource]:
    """Search arXiv for papers. Returns [] on any failure (never raises)."""
    try:
        import arxiv
    except Exception as exc:  # noqa: BLE001
        log.warning("arxiv package unavailable: %s", exc)
        return []

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=2)
        results = list(client.results(search))
    except Exception as exc:  # noqa: BLE001
        log.error("paper_search API failure: %s", type(exc).__name__)
        return []

    sources: list[RetrievedSource] = []
    for r in results:
        try:
            authors = ", ".join(a.name for a in getattr(r, "authors", [])[:6])
            pub_date = None
            if getattr(r, "published", None):
                pub_date = r.published.date().isoformat()
            sources.append(
                make_source(
                    title=r.title,
                    url=r.entry_id,
                    source_type=SourceType.RESEARCH_PAPER,
                    content=(r.summary or "").strip(),
                    snippet=(r.summary or "")[:300],
                    author=authors or None,
                    publication_date=pub_date,
                    publisher="arXiv",
                    relevance_score=0.7,
                    sub_question_id=sub_question_id,
                )
            )
        except Exception as exc:  # noqa: BLE001 - skip a bad record, keep going
            log.debug("skipping malformed arxiv record: %s", exc)
            continue

    log.info("paper_search '%s' -> %d papers", query[:60], len(sources))
    return sources
