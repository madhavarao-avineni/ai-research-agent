"""
agents/quality.py
=================
Research-Quality Decision + follow-up query generation (Step 8/9 support).

`assess_quality` decides SUFFICIENT vs INSUFFICIENT from the current evidence
state using deterministic signals (source count, unresolved contradictions,
evidence gaps, flagged claims). When INSUFFICIENT and iterations remain, it
generates targeted follow-up queries (LLM if available, else heuristic) aimed
at gaps, contradictions, unverified claims and missing source types.
"""

from __future__ import annotations

import json
from typing import Optional

from config.llm import LLMUnavailable, get_llm
from config.settings import get_settings
from models.state import (
    EvidenceSufficiency,
    ResearchState,
    ValidationStatus,
)
from observability.logging_config import get_logger, trace_agent
from prompts.prompts import FOLLOWUP_SYSTEM, FOLLOWUP_USER

log = get_logger("agent.quality")


def _generate_followups(state: ResearchState) -> list[str]:
    analysis = state.get("analysis")
    gaps = [g.description for g in (analysis.evidence_gaps if analysis else [])]
    contradictions = [c.topic for c in (analysis.contradictions if analysis else [])]
    flagged = [
        v.claim for v in state.get("validation_results", [])
        if v.status in (ValidationStatus.CONTRADICTED, ValidationStatus.INSUFFICIENT_EVIDENCE)
    ]

    llm = get_llm()
    if llm.available and (gaps or contradictions or flagged):
        try:
            data = llm.complete_json(
                FOLLOWUP_SYSTEM,
                FOLLOWUP_USER.format(
                    question=state["question"],
                    gaps=json.dumps(gaps, default=str),
                    contradictions=json.dumps(contradictions, default=str),
                    flagged_claims=json.dumps(flagged, default=str),
                ),
            )
            queries = [str(q) for q in (data.get("followup_queries") or [])][:5]
            if queries:
                return queries
        except (LLMUnavailable, ValueError) as exc:
            log.warning("Follow-up LLM generation failed (%s); using heuristic.", exc)

    # Heuristic fallback: build queries from gaps / contradictions / claims.
    q = state["question"]
    heuristic = []
    for g in gaps[:2]:
        heuristic.append(f"{q} evidence on {g}")
    for c in contradictions[:2]:
        heuristic.append(f"{q} resolve disagreement about {c}")
    for cl in flagged[:2]:
        heuristic.append(f"verify: {cl}")
    return heuristic or [f"{q} recent authoritative evidence"]


def assess_quality(state: ResearchState) -> tuple[EvidenceSufficiency, list[str]]:
    """Return (sufficiency, followup_queries). Queries empty if SUFFICIENT."""
    settings = get_settings()
    target = settings.target_source_count
    sources = state.get("sources", [])
    analysis = state.get("analysis")
    validations = state.get("validation_results", [])

    n_sources = len(sources)
    n_gaps = len(analysis.evidence_gaps) if analysis else 0
    unresolved_contradictions = sum(
        1 for c in (analysis.contradictions if analysis else []) if not c.explanation
    )
    flagged = sum(
        1 for v in validations
        if v.status in (ValidationStatus.CONTRADICTED, ValidationStatus.INSUFFICIENT_EVIDENCE)
    )

    # Sufficiency heuristic: enough sources, no big gaps, contradictions mostly
    # explained, and not too many flagged claims.
    enough_sources = n_sources >= max(8, target // 2)
    ok_gaps = n_gaps == 0
    ok_contra = unresolved_contradictions == 0
    ok_flagged = flagged <= max(1, len(validations) // 4)

    sufficient = enough_sources and ok_gaps and ok_contra and ok_flagged
    log.info(
        "quality: sources=%d(enough=%s) gaps=%d contra_unresolved=%d flagged=%d -> %s",
        n_sources, enough_sources, n_gaps, unresolved_contradictions, flagged,
        "SUFFICIENT" if sufficient else "INSUFFICIENT",
    )

    if sufficient:
        return EvidenceSufficiency.SUFFICIENT, []
    return EvidenceSufficiency.INSUFFICIENT, _generate_followups(state)


def quality_node(state: ResearchState) -> dict:
    """
    LangGraph node: sets evidence_sufficiency, increments iteration when looping
    back, and stores follow-up queries for the retriever.
    """
    iteration = state.get("research_iteration", 0)
    max_iter = state.get("max_iterations", 2)
    with trace_agent("quality_check", input_summary=f"iter={iteration}", research_iteration=iteration) as trace:
        sufficiency, followups = assess_quality(state)
        will_loop = sufficiency == EvidenceSufficiency.INSUFFICIENT and iteration < max_iter
        trace.output_summary = (
            f"{sufficiency.value}; "
            + (f"looping with {len(followups)} follow-ups" if will_loop else "proceeding to report")
        )
    out: dict = {"evidence_sufficiency": sufficiency, "traces": [trace]}
    if will_loop:
        out["followup_queries"] = followups
        out["research_iteration"] = iteration + 1
    else:
        # Budget exhausted or sufficient: clear any stale follow-ups so the
        # router can never loop again (belt-and-suspenders against infinite loops).
        out["followup_queries"] = []
    return out


def quality_router(state: ResearchState) -> str:
    """
    Conditional-edge function: returns "retriever" to loop or "report_builder"
    to finish. Respects the max-iteration bound to prevent infinite loops.
    """
    sufficiency = state.get("evidence_sufficiency", EvidenceSufficiency.SUFFICIENT)
    iteration = state.get("research_iteration", 0)
    max_iter = state.get("max_iterations", 2)
    if sufficiency == EvidenceSufficiency.INSUFFICIENT and iteration <= max_iter and state.get("followup_queries"):
        return "retriever"
    return "report_builder"
