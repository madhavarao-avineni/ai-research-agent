"""
agents/insight_generator.py
==========================
Agent 4 — Insight Generation.

Produces higher-level insights from the analysis, each classified by epistemic
tier (SUPPORTED / TREND / HYPOTHESIS / SPECULATION) and referencing supporting
source_ids. HYPOTHESIS/SPECULATION are separated so the report never presents
them as fact.

Graceful fallback: returns [] insights (workflow continues).
"""

from __future__ import annotations

import json
from typing import Optional

from config.llm import LLMUnavailable, get_llm
from models.state import (
    AnalysisResult,
    Insight,
    InsightTier,
    ResearchState,
    RetrievedSource,
)
from observability.logging_config import get_logger, trace_agent
from prompts.prompts import INSIGHT_SYSTEM, INSIGHT_USER

log = get_logger("agent.insight")


def _analysis_block(analysis: AnalysisResult) -> str:
    return json.dumps(
        {
            "key_findings": analysis.key_findings,
            "agreements": [a.model_dump() for a in analysis.agreements],
            "contradictions": [c.model_dump() for c in analysis.contradictions],
            "evidence_gaps": [g.model_dump() for g in analysis.evidence_gaps],
        },
        default=str,
    )[:6000]


def _valid_ids(ids, valid: set[str]) -> list[str]:
    return [i for i in (ids or []) if isinstance(i, str) and i in valid]


def generate_insights(
    question: str, analysis: AnalysisResult, sources: list[RetrievedSource]
) -> list[Insight]:
    valid = {s.source_id for s in sources}
    llm = get_llm()
    if not llm.available:
        log.warning("Insight generator fallback (no LLM).")
        return []
    try:
        data = llm.complete_json(
            INSIGHT_SYSTEM,
            INSIGHT_USER.format(
                question=question,
                analysis_block=_analysis_block(analysis),
                source_ids=", ".join(sorted(valid)) or "(none)",
            ),
        )
        if not isinstance(data, dict):
            raise ValueError("insight generator returned non-object JSON")
    except (LLMUnavailable, ValueError) as exc:
        log.error("Insight generation failed (%s); returning [].", exc)
        return []

    insights: list[Insight] = []
    for item in data.get("insights", []):
        if not isinstance(item, dict) or not item.get("statement"):
            continue
        tier_raw = str(item.get("tier", "HYPOTHESIS")).upper().strip()
        try:
            tier = InsightTier(tier_raw)
        except ValueError:
            tier = InsightTier.HYPOTHESIS  # unknown tier -> be conservative
        insights.append(
            Insight(
                statement=str(item["statement"]),
                tier=tier,
                supporting_source_ids=_valid_ids(item.get("supporting_source_ids"), valid),
                reasoning=(item.get("reasoning") or None),
            )
        )
    return insights


def insight_node(state: ResearchState) -> dict:
    question = state["question"]
    analysis = state.get("analysis") or AnalysisResult()
    sources = state.get("sources", [])
    with trace_agent(
        "insight_generator",
        input_summary=f"{len(analysis.key_findings)} findings",
        research_iteration=state.get("research_iteration", 0),
    ) as trace:
        insights = generate_insights(question, analysis, sources)
        # Split hypotheses/speculation out for the report.
        hypotheses = [i for i in insights if i.tier in (InsightTier.HYPOTHESIS, InsightTier.SPECULATION)]
        supported_or_trend = [i for i in insights if i.tier in (InsightTier.SUPPORTED, InsightTier.TREND)]
        trace.output_summary = (
            f"{len(insights)} insights "
            f"({len(supported_or_trend)} supported/trend, {len(hypotheses)} hypothesis/spec)"
        )
    return {
        "insights": insights,
        "hypotheses": hypotheses,
        "traces": [trace],
    }
