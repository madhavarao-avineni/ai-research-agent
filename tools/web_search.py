"""
tools/web_search.py
===================
Web Search Tool — current information from the internet via Tavily.

Independent of agents: given a query, returns `RetrievedSource` objects.
Fetches full page content (not just snippets) when possible. Degrades
gracefully: missing key or API failure -> logged warning + empty list.
"""

from __future__ import annotations

from typing import Optional

from config.settings import get_settings
from models.state import RetrievedSource, SourceType
from observability.logging_config import get_logger
from tools.base import clamp01, make_source

log = get_logger("tool.web_search")


def _fetch_full_text(url: str, timeout: int = 10) -> Optional[str]:
    """Best-effort fetch + extract readable text. Returns None on failure."""
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "research-assistant/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:20000] if text else None
    except Exception as exc:  # noqa: BLE001
        log.debug("full-text fetch failed for %s: %s", url, exc)
        return None


def web_search(
    query: str,
    *,
    max_results: int = 5,
    sub_question_id: Optional[str] = None,
    fetch_content: bool = True,
) -> list[RetrievedSource]:
    """Search the web. Returns [] on any failure (never raises)."""
    settings = get_settings()
    if not settings.tavily_api_key:
        log.warning("TAVILY_API_KEY not set — web_search disabled.")
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("web_search API failure: %s", type(exc).__name__)
        return []

    results = (resp or {}).get("results", []) if isinstance(resp, dict) else []
    sources: list[RetrievedSource] = []
    for r in results:
        url = r.get("url")
        content = r.get("raw_content") or r.get("content") or ""
        if fetch_content and url and len(content) < 500:
            full = _fetch_full_text(url)
            if full:
                content = full
        sources.append(
            make_source(
                title=r.get("title") or url or "(untitled)",
                url=url,
                source_type=SourceType.WEB,
                content=content,
                snippet=r.get("content"),
                publisher=(url or "").split("/")[2] if url and "//" in url else None,
                relevance_score=clamp01(r.get("score", 0.5)),
                sub_question_id=sub_question_id,
            )
        )
    log.info("web_search '%s' -> %d results", query[:60], len(sources))
    return sources
