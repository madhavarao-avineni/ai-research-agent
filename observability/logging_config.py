"""
observability/logging_config.py
===============================
Structured logging + per-agent tracing for the AI Research Assistant.

Provides:
  * `configure_logging()` — idempotent root-logger setup driven by settings.
  * `get_logger(name)`     — module-level logger accessor.
  * `SecretRedactingFilter` — scrubs anything that looks like an API key from
    log records so secrets never reach the logs.
  * `trace_agent(...)`     — context manager that times an agent, captures an
    `AgentTrace`, logs start/end/errors, and returns the trace so the node can
    push it onto `ResearchState.traces`.

The tracing here is provider-agnostic and always on. LangSmith tracing (if
enabled via env) layers on top automatically through the LangChain runtime —
this module never logs raw secrets regardless.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from config.settings import get_settings
from models.state import AgentTrace


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

# Patterns that look like credentials. We redact aggressively — false positives
# in logs are far cheaper than a leaked key.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),          # OpenAI-style
    re.compile(r"tvly-[A-Za-z0-9_\-]{8,}"),         # Tavily
    re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*)\S+"),  # key: value / key=value
    re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"),
]


def _redact(text: str) -> str:
    redacted = text
    for pat in _SECRET_PATTERNS:
        if pat.groups:
            redacted = pat.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pat.sub("[REDACTED]", redacted)
    return redacted


class SecretRedactingFilter(logging.Filter):
    """Redacts secrets from both the message and its args before emit."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            if isinstance(record.msg, str):
                record.msg = _redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: _redact(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                else:
                    record.args = tuple(
                        _redact(a) if isinstance(a, str) else a
                        for a in record.args
                    )
        except Exception:  # pragma: no cover - never let logging crash a run
            pass
        return True


# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure the root logger once. Safe to call repeatedly."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_level = (level or settings.log_level or "INFO").upper()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(SecretRedactingFilter())

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))
    # Avoid duplicate handlers if some library configured logging already.
    root.handlers = [handler]

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    configure_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Agent tracing
# ---------------------------------------------------------------------------


def _summarize(value: object, limit: int = 200) -> str:
    """Short, log-safe one-line summary of an arbitrary value."""
    text = str(value).replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return _redact(text)


@contextmanager
def trace_agent(
    agent_name: str,
    *,
    input_summary: str = "",
    research_iteration: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Iterator[AgentTrace]:
    """
    Context manager that wraps an agent execution with structured tracing.

    Yields a mutable `AgentTrace`. The node body should update
    `trace.output_summary`, `trace.tools_used`, and `trace.sources_retrieved`
    before exiting. On exit (success or error) the trace is finalized and
    logged, then it can be appended to `ResearchState["traces"]`.

    Example
    -------
        with trace_agent("planner", input_summary=question) as trace:
            plan = do_work()
            trace.output_summary = f"{len(plan.sub_questions)} sub-questions"
        state["traces"] = [trace]
    """

    log = logger or get_logger(f"agent.{agent_name}")
    start = time.perf_counter()
    trace = AgentTrace(
        agent_name=agent_name,
        start_time=datetime.now(timezone.utc).isoformat(),
        input_summary=_summarize(input_summary),
        research_iteration=research_iteration,
    )
    log.info("[>] %s started (iteration=%s) | in: %s",
             agent_name, research_iteration, trace.input_summary)
    try:
        yield trace
    except Exception as exc:  # noqa: BLE001 - we record then re-raise
        trace.error = _summarize(repr(exc))
        log.error("[X] %s failed: %s", agent_name, trace.error)
        raise
    finally:
        trace.end_time = datetime.now(timezone.utc).isoformat()
        elapsed = time.perf_counter() - start
        if trace.error is None:
            log.info(
                "[OK] %s done in %.2fs | tools=%s | sources=%s | out: %s",
                agent_name,
                elapsed,
                trace.tools_used or "-",
                trace.sources_retrieved,
                _summarize(trace.output_summary),
            )
