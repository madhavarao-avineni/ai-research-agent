"""
config/settings.py
==================
Central configuration for the AI Research Assistant.

Loads all runtime configuration from environment variables (via a `.env` file
in development) into a single immutable `Settings` object. No secret is ever
hard-coded; missing secrets degrade gracefully with a logged warning rather
than crashing at import time.

Usage
-----
    from config.settings import get_settings
    settings = get_settings()
    if settings.tavily_api_key:
        ...  # web search available

Design notes
------------
* `get_settings()` is cached (`functools.lru_cache`) so the environment is read
  once per process.
* Feature availability is expressed via `has_*` helpers so agents/tools can
  branch on capability instead of poking at raw env vars.
* `missing_required()` returns a list of missing hard-requirements so the UI /
  CLI can show a friendly setup message instead of a stack trace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

try:
    # python-dotenv is optional at runtime; if absent we just read os.environ.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    # --- LLM ---
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    # --- Embeddings ---
    embeddings_provider: str = "local"  # local | openai
    embeddings_model: str = "all-MiniLM-L6-v2"

    # --- Tools ---
    tavily_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None

    # --- Vector store ---
    chroma_persist_dir: str = "./.chroma"

    # --- Research loop ---
    max_research_iterations: int = 2
    target_source_count: int = 20

    # --- Observability ---
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "ai-research-assistant"
    log_level: str = "INFO"

    # --- capability helpers -------------------------------------------------
    @property
    def has_llm(self) -> bool:
        return bool(self.openai_api_key or self.anthropic_api_key or self.openrouter_api_key)

    @property
    def has_web_search(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_news_api(self) -> bool:
        return bool(self.newsapi_key)

    @property
    def use_openai_embeddings(self) -> bool:
        return self.embeddings_provider.lower() == "openai" and bool(
            self.openai_api_key
        )

    def missing_required(self) -> list[str]:
        """
        Hard requirements for a *useful* run. The app still imports and the UI
        can render a setup screen; this just enumerates what's missing.
        """
        missing: list[str] = []
        if not self.has_llm:
            missing.append("OPENAI_API_KEY or ANTHROPIC_API_KEY or OPENROUTER_API_KEY")
        if not self.has_web_search:
            missing.append("TAVILY_API_KEY (web search)")
        return missing

    def safe_summary(self) -> dict[str, object]:
        """
        Config summary safe for logging — secrets reduced to booleans, never
        the raw values.
        """
        return {
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "embeddings_provider": self.embeddings_provider,
            "embeddings_model": self.embeddings_model,
            "chroma_persist_dir": self.chroma_persist_dir,
            "max_research_iterations": self.max_research_iterations,
            "target_source_count": self.target_source_count,
            "langchain_tracing_v2": self.langchain_tracing_v2,
            "log_level": self.log_level,
            "has_openai_key": bool(self.openai_api_key),
            "has_anthropic_key": bool(self.anthropic_api_key),
            "has_openrouter_key": bool(self.openrouter_api_key),
            "has_tavily_key": bool(self.tavily_api_key),
            "has_newsapi_key": bool(self.newsapi_key),
            "has_langsmith_key": bool(self.langchain_api_key),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the environment once and return a cached Settings snapshot."""

    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "local"),
        embeddings_model=os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        newsapi_key=os.getenv("NEWSAPI_KEY") or None,
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./.chroma"),
        max_research_iterations=_get_int("MAX_RESEARCH_ITERATIONS", 2),
        target_source_count=_get_int("TARGET_SOURCE_COUNT", 20),
        langchain_tracing_v2=_get_bool("LANGCHAIN_TRACING_V2", False),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY") or None,
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "ai-research-assistant"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    # If LangSmith tracing is requested, propagate the standard env vars so the
    # LangChain runtime picks them up automatically.
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

    return settings
