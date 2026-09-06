"""
agents/fact_checker.py
=====================
Agent 5 — Fact Checker (INDEPENDENT validation).

Re-evaluates important claims against the source evidence itself, rather than
trusting prior agents. Returns `list[ValidationResult]` with status,
confidence, and supporting/contradicting source_ids.

Claims flagged CONTRADICTED / INSUFFICIENT_EVIDENCE are surfaced to the report.
Graceful fallback: marks claims INSUFFICIENT_EVIDENCE (honest) if the LLM is
unavailable — it never rubber-stamps a claim as SUPPORTED without evidence.
"""

from __future__ import annotations

import json
from typing import Optional

from config.llm import LLMUnavailable, get_llm
from models.state import (
    Claim,
    ResearchState,
    RetrievedSource,
    ValidationResult,
    ValidationStatus,
)
from observability.logging_config import get_logger, trace_agent
from prompts.prompts import FACTCHECKER_SYSTEM, FACTCHECKER_USER

log = get_logger("agent.fact_checker")

MAX_CLAIMS = 12
MAX_SOURCES_IN_PROMPT = 24
CONTENT_EXCERPT_CHARS = 1000


def _select_claims(claims: list[Claim]) -> list[Claim]:
    """Prioritize the most important claims (bounded for prompt size)."""
    return sorted(claims, key=lambda c: c.importance, reverse=True)[:MAX_CLAIMS]


def _claims_block(claims: list[Claim]) -> str:
    return json.dumps(
        [{"claim": c.text, "cited_source_ids": c.supporting_source_ids} for c in claims],
        default=str,
    )


def _sources_block(sources: list[RetrievedSource]) -> str:
    lines = []
    for s in sources[:MAX_SOURCES_IN_PROMPT]:
        excerpt = (s.content or s.snippet or "")[:CONTENT_EXCERPT_CHARS]
        lines.append(f"[{s.source_id}] type={s.source_type.value} title={s.title!r}\n{excerpt}\n")
    return "\n".join(lines)


def _valid_ids(ids, valid: set[str]) -> list[str]:
    return [i for i in (ids or []) if isinstance(i, str) and i in valid]


def fact_check(question: str, claims: list[Claim], sources: list[RetrievedSource]) -> list[ValidationResult]:
    if not claims:
        return []
    selected = _select_claims(claims)
    valid = {s.source_id for s in sources}
    llm = get_llm()
    if not llm.available:
        log.warning("Fact checker fallback: marking claims INSUFFICIENT_EVIDENCE (no LLM).")
        return [
            ValidationResult(
                claim=c.text,
                status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                reason="Validation unavailable (no LLM configured).",
            )
            for c in selected
        ]

    try:
        data = llm.complete_json(
            FACTCHECKER_SYSTEM,
            FACTCHECKER_USER.format(
                question=question,
                claims_block=_claims_block(selected),
                sources_block=_sources_block(sources),
            ),
        )
        if not isinstance(data, dict):
            raise ValueError("fact checker returned non-object JSON")
    except (LLMUnavailable, ValueError) as exc:
        log.error("Fact checker failed (%s); marking INSUFFICIENT_EVIDENCE.", exc)
        return [
            ValidationResult(
                claim=c.text, status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0, reason="Validation error.",
            )
            for c in selected
        ]

    results: list[ValidationResult] = []
    for r in data.get("validation_results", []):
        if not isinstance(r, dict) or not r.get("claim"):
            continue
        status_raw = str(r.get("status", "INSUFFICIENT_EVIDENCE")).upper().strip()
        try:
            status = ValidationStatus(status_raw)
        except ValueError:
            status = ValidationStatus.INSUFFICIENT_EVIDENCE
        conf = float(r.get("confidence", 0.0) or 0.0)
        results.append(
            ValidationResult(
                claim=str(r["claim"]),
                status=status,
                confidence=max(0.0, min(1.0, conf)),
                supporting_sources=_valid_ids(r.get("supporting_sources"), valid),
                contradicting_sources=_valid_ids(r.get("contradicting_sources"), valid),
                reason=str(r.get("reason", "")),
            )
        )
    return results


def fact_checker_node(state: ResearchState) -> dict:
    question = state["question"]
    claims = state.get("key_claims", [])
    sources = state.get("sources", [])
    with trace_agent(
        "fact_checker",
        input_summary=f"{len(claims)} claims",
        research_iteration=state.get("research_iteration", 0),
    ) as trace:
        results = fact_check(question, claims, sources)
        flagged = [r for r in results if r.is_flagged]
        trace.output_summary = f"{len(results)} validated, {len(flagged)} flagged"
    return {
        "validation_results": results,
        "traces": [trace],
    }
