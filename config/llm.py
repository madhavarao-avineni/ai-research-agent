"""
config/llm.py
=============
Provider-agnostic LLM client for the AI Research Assistant.

Why a wrapper?
--------------
Agents should not care which provider is configured. They call
`get_llm().complete_json(system, user)` (or `.complete_text(...)`) and receive
a parsed dict (or string). This lets us:

* Swap OpenAI / Anthropic / OpenRouter / local models without touching agent code.
* Inject a fake client in tests (no network, no keys) via `set_llm()`.
* Centralize retries, timeouts, JSON-repair, and graceful failure.

Graceful degradation
---------------------
If no provider key is configured, `complete_json` raises `LLMUnavailable`,
which agents catch and convert into a safe fallback result (never a crash).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional, Protocol

from config.settings import get_settings
from observability.logging_config import get_logger

log = get_logger("llm")


class LLMUnavailable(RuntimeError):
    """Raised when no usable LLM provider is configured or all retries fail."""


# ---------------------------------------------------------------------------
# JSON extraction helper (robust to code fences and prose around JSON)
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Any:
    """
    Best-effort parse of a JSON object/array out of an LLM response.

    Handles ```json fences, leading/trailing prose, and trailing commas.
    Raises ValueError if nothing parseable is found.
    """
    if text is None:
        raise ValueError("empty LLM response")

    # 1) Try direct parse.
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2) Strip code fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            stripped = candidate

    # 3) Grab the largest {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            # Remove trailing commas before } or ]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError("no parseable JSON found in LLM response")


# ---------------------------------------------------------------------------
# Client protocol + implementations
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    def complete_text(self, system: str, user: str, **kwargs: Any) -> str: ...


class OpenAIClient:
    """Thin OpenAI Chat Completions wrapper. Imports lazily."""

    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._client = None

    def _ensure(self) -> None:
        if self._client is None:
            try:
                from openai import OpenAI  # lazy import
            except Exception as exc:  # pragma: no cover
                raise LLMUnavailable(f"openai package not installed: {exc}") from exc
            self._client = OpenAI(api_key=self._api_key)

    def complete_text(self, system: str, user: str, **kwargs: Any) -> str:
        self._ensure()
        temperature = kwargs.get("temperature", self._temperature)
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class OpenRouterClient:
    """OpenRouter API client via OpenAI SDK compatibility."""

    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._client = None

    def _ensure(self) -> None:
        if self._client is None:
            try:
                from openai import OpenAI  # lazy import
            except Exception as exc:  # pragma: no cover
                raise LLMUnavailable(f"openai package not installed: {exc}") from exc
            # OpenRouter is OpenAI-compatible; just swap the base URL
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    def complete_text(self, system: str, user: str, **kwargs: Any) -> str:
        self._ensure()
        temperature = kwargs.get("temperature", self._temperature)
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class FakeClient:
    """
    Test double. Provide a `responder(system, user) -> str` callable that
    returns a canned response. Used by unit tests to run agents offline.
    """

    def __init__(self, responder: Callable[[str, str], str]) -> None:
        self._responder = responder

    def complete_text(self, system: str, user: str, **kwargs: Any) -> str:
        return self._responder(system, user)


# ---------------------------------------------------------------------------
# High-level LLM facade with retries + JSON parsing
# ---------------------------------------------------------------------------


class LLM:
    """The object agents actually use."""

    def __init__(self, client: Optional[LLMClient], *, max_retries: int = 2) -> None:
        self._client = client
        self._max_retries = max_retries

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete_text(self, system: str, user: str, **kwargs: Any) -> str:
        if self._client is None:
            raise LLMUnavailable("no LLM provider configured")
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            try:
                return self._client.complete_text(system, user, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = min(2 ** attempt, 8)
                log.warning("LLM call failed (attempt %s): %s; retrying in %ss",
                            attempt, type(exc).__name__, wait)
                time.sleep(wait if attempt <= self._max_retries else 0)
        raise LLMUnavailable(f"LLM call failed after retries: {last_exc}")

    def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        """Call the model and parse a JSON object/array from its output."""
        raw = self.complete_text(system, user, **kwargs)
        return extract_json(raw)


# ---------------------------------------------------------------------------
# Module-level singleton with override for tests
# ---------------------------------------------------------------------------

_LLM_SINGLETON: Optional[LLM] = None


def get_llm() -> LLM:
    """Return the process LLM, building it from settings on first use."""
    global _LLM_SINGLETON
    if _LLM_SINGLETON is not None:
        return _LLM_SINGLETON

    settings = get_settings()
    client: Optional[LLMClient] = None

    # Priority: OpenRouter > OpenAI > Anthropic (fallback)
    if settings.openrouter_api_key:
        client = OpenRouterClient(
            settings.openrouter_api_key, settings.llm_model, settings.llm_temperature
        )
        log.info("LLM provider: OpenRouter (%s)", settings.llm_model)
    elif settings.openai_api_key:
        client = OpenAIClient(
            settings.openai_api_key, settings.llm_model, settings.llm_temperature
        )
        log.info("LLM provider: OpenAI (%s)", settings.llm_model)
    else:
        log.warning("No LLM provider configured — agents will use safe fallbacks.")

    _LLM_SINGLETON = LLM(client)
    return _LLM_SINGLETON


def set_llm(llm: LLM) -> None:
    """Override the singleton (used by tests to inject a FakeClient)."""
    global _LLM_SINGLETON
    _LLM_SINGLETON = llm
