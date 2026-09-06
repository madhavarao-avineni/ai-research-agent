"""Research-loop decision and report generation tests."""

import json

from agents.quality import assess_quality, quality_node, quality_router
from agents.report_builder import build_report
from models.state import (
    Agreement,
    AnalysisResult,
    Contradiction,
    EvidenceGap,
    EvidenceSufficiency,
    Insight,
    InsightTier,
    SourceType,
    ValidationResult,
    ValidationStatus,
    new_research_state,
)
from tools.base import make_source


def _many_sources(n):
    return [make_source(title=str(i), url=f"https://s/{i}", source_type=SourceType.WEB, content="c") for i in range(n)]


def test_quality_insufficient_with_gaps(no_llm):
    st = new_research_state("Q")
    st.update({
        "sources": _many_sources(2),
        "analysis": AnalysisResult(evidence_gaps=[EvidenceGap(topic="t", description="d")]),
    })
    suff, followups = assess_quality(st)
    assert suff == EvidenceSufficiency.INSUFFICIENT
    assert followups  # heuristic follow-ups generated


def test_quality_sufficient(no_llm):
    st = new_research_state("Q")
    st.update({
        "sources": _many_sources(12),
        "analysis": AnalysisResult(key_findings=["f"]),
        "validation_results": [ValidationResult(claim="c", status=ValidationStatus.SUPPORTED, confidence=0.8)],
    })
    suff, followups = assess_quality(st)
    assert suff == EvidenceSufficiency.SUFFICIENT
    assert followups == []


def test_loop_bounded_by_max_iterations(no_llm):
    st = new_research_state("Q", max_iterations=2)
    st.update({
        "sources": _many_sources(1),
        "analysis": AnalysisResult(evidence_gaps=[EvidenceGap(topic="t", description="d")]),
        "research_iteration": 2,  # already at max
    })
    out = quality_node(st)
    # At max: does not schedule another loop.
    assert out.get("research_iteration", 2) == 2
    st.update(out)
    assert quality_router(st) == "report_builder"


def test_router_loops_when_budget_remains(no_llm):
    st = new_research_state("Q", max_iterations=2)
    st.update({
        "evidence_sufficiency": EvidenceSufficiency.INSUFFICIENT,
        "followup_queries": ["fq1"],
        "research_iteration": 1,
    })
    assert quality_router(st) == "retriever"


def test_report_has_all_sections(no_llm):
    st = new_research_state("Will GenAI reduce dev demand?")
    srcs = _many_sources(3)
    ids = [s.source_id for s in srcs]
    st.update({
        "sources": srcs,
        "analysis": AnalysisResult(
            key_findings=["speed up"],
            agreements=[Agreement(statement="speed", source_ids=ids[:2])],
            contradictions=[Contradiction(topic="jobs", position_a="fall",
                                          position_a_source_ids=[ids[0]], position_b="grow",
                                          position_b_source_ids=[ids[2]])],
            evidence_gaps=[EvidenceGap(topic="5yr", description="no long-term data")],
        ),
        "insights": [
            Insight(statement="supported", tier=InsightTier.SUPPORTED, supporting_source_ids=[ids[0]]),
            Insight(statement="hypo", tier=InsightTier.HYPOTHESIS, supporting_source_ids=[ids[1]]),
        ],
        "validation_results": [ValidationResult(claim="c", status=ValidationStatus.CONTRADICTED,
                                                 confidence=0.6, reason="r")],
    })
    report = build_report(st)
    md = report.markdown
    for section in [
        "## Executive Summary", "## Research Methodology", "## Key Findings",
        "## Evidence", "## Contradictions", "## Emerging Trends", "## Insights",
        "## Hypotheses", "## Risks and Limitations", "## Conclusion", "## References",
    ]:
        assert section in md, f"missing {section}"
    # Hypotheses separated from supported insights.
    assert "hypo" in md.split("## Hypotheses")[1]
    # Real citations only.
    for sid in ids:
        assert sid in md
