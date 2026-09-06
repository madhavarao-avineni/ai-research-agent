"""
agents/analyzer.py
==================
Agent 3 — Critical Analysis.

Analyzes retrieved sources to extract key claims (with provenance), assess
credibility & limitations, and identify agreements, contradictions, and
evidence gaps. Produces a structured `AnalysisResult` + `list[Claim]`.

Only source_ids present in the input are accepted in outputs (fabricated ids
are dropped). Graceful fallback returns an empty-but-valid analysis.
"""

from __future__ import annotations

from typing import Optional

from config.llm import LLMUnavailable, get_llm
from models.state import (
    Agreement,
    AnalysisResult,
    Claim,
    Contradiction,
    EvidenceGap,
    ResearchState,
    RetrievedSource,
    SourceQualityAssessment,
)
from observability.logging_config import get_logger, trace_agent
from prompts.prompts import ANALYZER_SYSTEM, ANALYZER_USER

log = get_logger("agent.analyzer")

MAX_SOURCES_IN_PROMPT = 24
CONTENT_EXCERPT_CHARS = 1200


def _sources_block(sources: list[RetrievedSource]) -> str:
    lines = []
    for s in sources[:MAX_SOURCES_IN_PROMPT]:
        excerpt = (s.content or s.snippet or "")[:CONTENT_EXCERPT_CHARS]
        lines.append(
            f"[{s.source_id}] type={s.source_type.value} "
            f"cred~{s.credibility_score} title={s.title!r}\n{excerpt}\n"
        )
    return "\n".join(lines)


def _valid_ids(ids, valid: set[str]) -> list[str]:
    return [i for i in (ids or []) if isinstance(i, str) and i in valid]


def analyze(question: str, sources: list[RetrievedSource]) -> tuple[AnalysisResult, list[Claim]]:
    valid = {s.source_id for s in sources}
    llm = get_llm()
    if not llm.available or not sources:
        log.warning("Analyzer fallback (llm_available=%s, n_sources=%d).", llm.available, len(sources))
        return AnalysisResult(), []

    try:
        data = llm.complete_json(
            ANALYZER_SYSTEM,
            ANALYZER_USER.format(question=question, sources_block=_sources_block(sources)),
        )
        if not isinstance(data, dict):
            raise ValueError("analyzer returned non-object JSON")
    except (LLMUnavailable, ValueError) as exc:
        log.error("Analyzer failed (%s); returning empty analysis.", exc)
        return AnalysisResult(), []

    claims: list[Claim] = []
    for c in data.get("key_claims", []):
        if isinstance(c, dict) and c.get("text"):
            claims.append(
                Claim(
                    text=str(c["text"]),
                    supporting_source_ids=_valid_ids(c.get("supporting_source_ids"), valid),
                    importance=float(c.get("importance", 0.5) or 0.5),
                )
            )

    agreements = [
        Agreement(statement=str(a.get("statement", "")), source_ids=_valid_ids(a.get("source_ids"), valid))
        for a in data.get("agreements", []) if isinstance(a, dict) and a.get("statement")
    ]

    contradictions = []
    for c in data.get("contradictions", []):
        if not isinstance(c, dict):
            continue
        contradictions.append(
            Contradiction(
                topic=str(c.get("topic", "")),
                position_a=str(c.get("position_a", "")),
                position_a_source_ids=_valid_ids(c.get("position_a_source_ids"), valid),
                position_b=str(c.get("position_b", "")),
                position_b_source_ids=_valid_ids(c.get("position_b_source_ids"), valid),
                explanation=(c.get("explanation") or None),
            )
        )

    gaps = [
        EvidenceGap(topic=str(g.get("topic", "")), description=str(g.get("description", "")))
        for g in data.get("evidence_gaps", []) if isinstance(g, dict) and (g.get("topic") or g.get("description"))
    ]

    quality = []
    for q in data.get("source_quality_assessment", []):
        if isinstance(q, dict) and q.get("source_id") in valid:
            quality.append(
                SourceQualityAssessment(
                    source_id=q["source_id"],
                    credibility_score=float(q.get("credibility_score", 0.5) or 0.5),
                    limitations=[str(x) for x in (q.get("limitations") or [])],
                    notes=(q.get("notes") or None),
                )
            )

    analysis = AnalysisResult(
        key_findings=[str(f) for f in data.get("key_findings", [])],
        agreements=agreements,
        contradictions=contradictions,
        evidence_gaps=gaps,
        source_quality_assessment=quality,
    )
    return analysis, claims


def analyzer_node(state: ResearchState) -> dict:
    question = state["question"]
    sources = state.get("sources", [])
    with trace_agent(
        "analyzer",
        input_summary=f"{len(sources)} sources",
        research_iteration=state.get("research_iteration", 0),
    ) as trace:
        analysis, claims = analyze(question, sources)
        trace.output_summary = (
            f"{len(claims)} claims, {len(analysis.agreements)} agreements, "
            f"{len(analysis.contradictions)} contradictions, {len(analysis.evidence_gaps)} gaps"
        )
    return {
        "analysis": analysis,
        "key_claims": claims,
        "agreements": analysis.agreements,
        "contradictions": analysis.contradictions,
        "evidence_gaps": analysis.evidence_gaps,
        "traces": [trace],
    }
