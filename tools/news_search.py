"""
tools/news_search.py
====================
News Search Tool — recent articles.

Primary backend: NewsAPI (requires NEWSAPI_KEY).
Fallback: Google News RSS via feedparser (no key). This guarantees the tool
still returns something useful when no news API key is configured.

Returns `RetrievedSource` objects; [] on total failure (never raises).
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus

from config.settings import get_settings
from models.state import RetrievedSource, SourceType
from observability.logging_config import get_logger
from tools.base import make_source

log = get_logger("tool.news_search")


def _via_newsapi(query: str, max_results: int, sub_question_id: Optional[str]) -> list[RetrievedSource]:
    settings = get_settings()
    try:
        import requests

        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "pageSize": max_results,
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": settings.newsapi_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("NewsAPI failed (%s); will try RSS fallback.", type(exc).__name__)
        return []

    out: list[RetrievedSource] = []
    for a in data.get("articles", [])[:max_results]:
        out.append(
            make_source(
                title=a.get("title") or "(untitled)",
                url=a.get("url"),
                source_type=SourceType.NEWS,
                content=a.get("content") or a.get("description") or "",
                snippet=a.get("description"),
                author=a.get("author"),
                publication_date=(a.get("publishedAt") or "")[:10] or None,
                publisher=(a.get("source") or {}).get("name"),
                relevance_score=0.55,
                sub_question_id=sub_question_id,
            )
        )
    return out


def _via_rss(query: str, max_results: int, sub_question_id: Optional[str]) -> list[RetrievedSource]:
    try:
        import feedparser

        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
    except Exception as exc:  # noqa: BLE001
        log.error("news RSS fallback failed: %s", type(exc).__name__)
        return []

    out: list[RetrievedSource] = []
    for e in feed.entries[:max_results]:
        out.append(
            make_source(
                title=getattr(e, "title", "(untitled)"),
                url=getattr(e, "link", None),
                source_type=SourceType.NEWS,
                content=getattr(e, "summary", "") or "",
                snippet=getattr(e, "summary", "")[:300],
                publication_date=getattr(e, "published", None),
                publisher=getattr(getattr(e, "source", None), "title", None),
                relevance_score=0.5,
                sub_question_id=sub_question_id,
            )
        )
    return out


def news_search(
    query: str,
    *,
    max_results: int = 5,
    sub_question_id: Optional[str] = None,
) -> list[RetrievedSource]:
    """Search recent news. NewsAPI if keyed, else Google News RSS."""
    settings = get_settings()
    results: list[RetrievedSource] = []
    if settings.newsapi_key:
        results = _via_newsapi(query, max_results, sub_question_id)
    if not results:
        results = _via_rss(query, max_results, sub_question_id)
    log.info("news_search '%s' -> %d articles", query[:60], len(results))
    return results
