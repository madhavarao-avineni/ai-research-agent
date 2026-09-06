"""
agents/report_builder.py
=======================
Agent 6 — Report Builder.

Assembles the final `FinalReport` and renders it to Markdown with all mandated
sections: Executive Summary, Methodology, Key Findings, Evidence,
Contradictions, Emerging Trends, Insights, Hypotheses, Risks & Limitations,
Conclusion, References.

Prose sections (summary/methodology/conclusion/etc.) are drafted by the LLM;
structured sections (contradictions, insights, references) are built
deterministically from state so citations are REAL and never fabricated.
The Markdown renderer works even with no LLM (uses findings/analysis directly).
"""

from __future__ import annotations

import json
from typing import Optional

from config.llm import LLMUnavailable, get_llm
from models.state import (
    AnalysisResult,
    FinalReport,
    Insight,
    InsightTier,
    ResearchState,
    RetrievedSource,
    ValidationResult,
    ValidationStatus,
)
from observability.logging_config import get_logger, trace_agent
from prompts.prompts import REPORT_SYSTEM, REPORT_USER

log = get_logger("agent.report_builder")


def _sources_block(sources: list[RetrievedSource]) -> str:
    return "\n".join(
        f"[{s.source_id}] {s.title} | {s.publisher or 'n/a'} | {s.publication_date or 'n/d'} | {s.url or 'n/a'}"
        for s in sources[:30]
    )


def _draft_prose(state: ResearchState) -> dict:
    """LLM-drafted prose sections; falls back to deterministic text."""
    question = state["question"]
    analysis: AnalysisResult = state.get("analysis") or AnalysisResult()
    insights = state.get("insights", [])
    validations = state.get("validation_results", [])
    sources = state.get("sources", [])

    fallback = {
        "executive_summary": (
            f"This report investigates: {question} Based on {len(sources)} sources analyzed, "
            f"{len(analysis.key_findings)} key findings and {len(analysis.contradictions)} "
            f"points of disagreement were identified."
        ),
        "methodology": (
            "The question was decomposed into focused sub-questions. Multiple source types "
            "(web, research papers, news, reports) were searched in parallel, deduplicated, and "
            "ranked by blended relevance and credibility. Evidence was critically analyzed for "
            "agreements, contradictions and gaps; claims were independently fact-checked."
        ),
        "key_findings": analysis.key_findings,
        "evidence": [a.statement for a in analysis.agreements],
        "emerging_trends": [i.statement for i in insights if i.tier == InsightTier.TREND],
        "risks_and_limitations": [g.description for g in analysis.evidence_gaps]
        or ["Evidence gaps and source limitations may affect confidence."],
        "conclusion": (
            f"Based on the available evidence, a definitive answer to '{question}' remains "
            f"nuanced; see key findings, contradictions and flagged claims for detail."
        ),
    }

    llm = get_llm()
    if not llm.available:
        return fallback
    try:
        data = llm.complete_json(
            REPORT_SYSTEM,
            REPORT_USER.format(
                question=question,
                objective=state.get("research_objective", ""),
                findings=json.dumps(analysis.key_findings, default=str),
                agreements=json.dumps([a.model_dump() for a in analysis.agreements], default=str),
                contradictions=json.dumps([c.model_dump() for c in analysis.contradictions], default=str),
                gaps=json.dumps([g.model_dump() for g in analysis.evidence_gaps], default=str),
                insights=json.dumps([i.model_dump() for i in insights], default=str),
                validations=json.dumps([v.model_dump() for v in validations], default=str),
                sources_block=_sources_block(sources),
            ),
        )
        if not isinstance(data, dict):
            raise ValueError("report builder returned non-object JSON")
        # Merge onto fallback so missing keys are still populated.
        for k, v in fallback.items():
            data.setdefault(k, v)
        return data
    except (LLMUnavailable, ValueError) as exc:
        log.error("Report prose drafting failed (%s); using deterministic text.", exc)
        return fallback


def _render_markdown(report: FinalReport, question: str) -> str:
    L: list[str] = []
    A = L.append
    A(f"# Research Report\n")
    A(f"**Research Question:** {question}\n")

    A("## Executive Summary\n")
    A(report.executive_summary + "\n")

    A("## Research Methodology\n")
    A(report.methodology + "\n")

    A("## Key Findings\n")
    if report.key_findings:
        for f in report.key_findings:
            A(f"- {f}")
    else:
        A("_No firm findings could be established from the available evidence._")
    A("")

    A("## Evidence\n")
    if report.evidence:
        for e in report.evidence:
            A(f"- {e}")
    else:
        A("_See references; evidence was limited._")
    A("")

    A("## Contradictions\n")
    if report.contradictions:
        for c in report.contradictions:
            A(f"**{c.topic}**")
            A(f"- Position A: {c.position_a}  \n  _Sources:_ {', '.join(c.position_a_source_ids) or 'n/a'}")
            A(f"- Position B: {c.position_b}  \n  _Sources:_ {', '.join(c.position_b_source_ids) or 'n/a'}")
            if c.explanation:
                A(f"- Explanation: {c.explanation}")
            A("")
    else:
        A("_No material contradictions detected._\n")

    A("## Emerging Trends\n")
    if report.emerging_trends:
        for t in report.emerging_trends:
            A(f"- {t}")
    else:
        A("_No clear cross-source trends identified._")
    A("")

    A("## Insights\n")
    supported = [i for i in report.insights if i.tier in (InsightTier.SUPPORTED, InsightTier.TREND)]
    if supported:
        for i in supported:
            A(f"- **[{i.tier.value}]** {i.statement}  \n  _Sources:_ {', '.join(i.supporting_source_ids) or 'n/a'}")
    else:
        A("_No evidence-backed insights met the bar._")
    A("")

    A("## Hypotheses\n")
    if report.hypotheses:
        A("_The following are NOT established facts — they are hypotheses/speculation flagged for further research:_\n")
        for h in report.hypotheses:
            A(f"- **[{h.tier.value}]** {h.statement}  \n  _Sources:_ {', '.join(h.supporting_source_ids) or 'n/a'}")
    else:
        A("_No hypotheses generated._")
    A("")

    A("## Risks and Limitations\n")
    for r in report.risks_and_limitations:
        A(f"- {r}")
    A("")

    A("## Conclusion\n")
    A(report.conclusion + "\n")

    A("## References\n")
    if report.references:
        for idx, s in enumerate(report.references, 1):
            bits = [f"**{s.title}**"]
            if s.publisher:
                bits.append(s.publisher)
            if s.author:
                bits.append(s.author)
            if s.publication_date:
                bits.append(s.publication_date)
            line = " — ".join(bits)
            url = f" <{s.url}>" if s.url else ""
            A(f"{idx}. [{s.source_id}] {line}.{url}")
    else:
        A("_No sources were available._")
    A("")
    return "\n".join(L)


def build_report(state: ResearchState) -> FinalReport:
    question = state["question"]
    analysis: AnalysisResult = state.get("analysis") or AnalysisResult()
    insights: list[Insight] = state.get("insights", [])
    validations: list[ValidationResult] = state.get("validation_results", [])
    sources: list[RetrievedSource] = state.get("sources", [])

    prose = _draft_prose(state)

    hypotheses = [i for i in insights if i.tier in (InsightTier.HYPOTHESIS, InsightTier.SPECULATION)]

    # Risks: merge prose risks with flagged claims for transparency.
    risks = list(prose.get("risks_and_limitations", []))
    flagged = [v for v in validations if v.status in (ValidationStatus.CONTRADICTED, ValidationStatus.INSUFFICIENT_EVIDENCE)]
    for v in flagged:
        risks.append(f"Claim flagged [{v.status.value}]: {v.claim} — {v.reason}")

    report = FinalReport(
        executive_summary=prose.get("executive_summary", ""),
        methodology=prose.get("methodology", ""),
        key_findings=list(prose.get("key_findings", []) or analysis.key_findings),
        evidence=list(prose.get("evidence", [])),
        contradictions=analysis.contradictions,
        emerging_trends=list(prose.get("emerging_trends", [])),
        insights=insights,
        hypotheses=hypotheses,
        risks_and_limitations=risks,
        conclusion=prose.get("conclusion", ""),
        references=sources,
    )
    report.markdown = _render_markdown(report, question)
    return report


def report_builder_node(state: ResearchState) -> dict:
    with trace_agent(
        "report_builder",
        input_summary=f"{len(state.get('sources', []))} sources",
        research_iteration=state.get("research_iteration", 0),
    ) as trace:
        report = build_report(state)
        trace.output_summary = f"{len(report.markdown)} chars, {len(report.references)} references"
    return {"final_report": report, "traces": [trace]}
