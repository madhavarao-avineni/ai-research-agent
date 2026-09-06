"""
tools/base.py
=============
Shared helpers for research tools. Every tool returns a list of
`RetrievedSource` objects and NEVER raises on API failure — it logs and returns
an empty list so a single unavailable source can't crash the workflow.

`credibility_heuristic` gives a rough 0–1 prior based on source type and
publisher, refined later by the Critical Analysis agent.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from models.state import RetrievedSource, SourceType

# Domains we treat as more authoritative (rough prior; not exhaustive).
_HIGH_TRUST = {
    "nature.com", "science.org", "arxiv.org", "acm.org", "ieee.org",
    "nber.org", "oecd.org", "worldbank.org", "imf.org", "gov", "edu",
    "mckinsey.com", "gartner.com", "forrester.com", "brookings.edu",
    "stackoverflow.blog", "github.blog", "reuters.com", "bloomberg.com",
    "wsj.com", "ft.com", "economist.com",
}

_TYPE_PRIOR = {
    SourceType.RESEARCH_PAPER: 0.75,
    SourceType.REPORT: 0.70,
    SourceType.NEWS: 0.55,
    SourceType.WEB: 0.45,
    SourceType.INTERNAL_DOCUMENT: 0.60,
}


def credibility_heuristic(source_type: SourceType, url: Optional[str]) -> float:
    """Rough credibility prior in [0, 1]."""
    score = _TYPE_PRIOR.get(source_type, 0.5)
    if url:
        host = (urlparse(url).netloc or "").lower()
        if any(host.endswith(d) or ("." + d) in host or host == d for d in _HIGH_TRUST):
            score = min(1.0, score + 0.2)
    return round(score, 3)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def make_source(
    *,
    title: str,
    url: Optional[str],
    source_type: SourceType,
    content: str,
    snippet: Optional[str] = None,
    author: Optional[str] = None,
    publication_date: Optional[str] = None,
    publisher: Optional[str] = None,
    relevance_score: float = 0.5,
    sub_question_id: Optional[str] = None,
) -> RetrievedSource:
    """Construct a RetrievedSource with a credibility prior filled in."""
    return RetrievedSource(
        title=title.strip() or "(untitled)",
        url=url,
        source_type=source_type,
        author=author,
        publication_date=publication_date,
        publisher=publisher,
        content=content or "",
        snippet=snippet,
        relevance_score=clamp01(relevance_score),
        credibility_score=credibility_heuristic(source_type, url),
        sub_question_ids=[sub_question_id] if sub_question_id else [],
    )
