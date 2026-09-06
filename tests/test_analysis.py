"""Contradiction detection, claim validation, and provenance-filtering tests."""

import json

from agents.analyzer import analyze
from agents.fact_checker import fact_check
from agents.insight_generator import generate_insights
from models.state import (
    AnalysisResult,
    Claim,
    InsightTier,
    SourceType,
    ValidationStatus,
)
from tools.base import make_source


def _sources():
    return [
        make_source(title=f"S{i}", url=f"https://x/{i}",
                    source_type=SourceType.RESEARCH_PAPER, content="body")
        for i in range(3)
    ]


def test_contradiction_detection_keeps_both_sides(fake_llm_factory):
    srcs = _sources()
    ids = [s.source_id for s in srcs]
    payload = {
        "key_claims": [{"text": "claim", "supporting_source_ids": [ids[0]], "importance": 0.8}],
        "key_findings": ["f"],
        "agreements": [],
        "contradictions": [{
            "topic": "jobs", "position_a": "fall", "position_a_source_ids": [ids[0]],
            "position_b": "grow", "position_b_source_ids": [ids[2]], "explanation": None,
        }],
        "evidence_gaps": [],
        "source_quality_assessment": [],
    }
    fake_llm_factory(lambda s, u: json.dumps(payload))
    analysis, claims = analyze("Q?", srcs)
    assert len(analysis.contradictions) == 1
    c = analysis.contradictions[0]
    assert c.position_a_source_ids and c.position_b_source_ids


def test_analyzer_drops_fabricated_source_ids(fake_llm_factory):
    srcs = _sources()
    ids = [s.source_id for s in srcs]
    payload = {
        "key_claims": [{"text": "c", "supporting_source_ids": [ids[0], "src_FAKE"], "importance": 0.5}],
        "key_findings": [], "agreements": [], "contradictions": [],
        "evidence_gaps": [], "source_quality_assessment": [],
    }
    fake_llm_factory(lambda s, u: json.dumps(payload))
    _, claims = analyze("Q?", srcs)
    assert "src_FAKE" not in claims[0].supporting_source_ids


def test_insight_tiers_and_unknown_coercion(fake_llm_factory):
    srcs = _sources()
    ids = [s.source_id for s in srcs]
    payload = {"insights": [
        {"statement": "supported one", "tier": "SUPPORTED", "supporting_source_ids": [ids[0]]},
        {"statement": "weird tier", "tier": "MADE_UP", "supporting_source_ids": [ids[1]]},
    ]}
    fake_llm_factory(lambda s, u: json.dumps(payload))
    insights = generate_insights("Q?", AnalysisResult(), srcs)
    tiers = {i.statement: i.tier for i in insights}
    assert tiers["supported one"] == InsightTier.SUPPORTED
    assert tiers["weird tier"] == InsightTier.HYPOTHESIS  # conservative default


def test_fact_checker_flags_contradicted(fake_llm_factory):
    srcs = _sources()
    ids = [s.source_id for s in srcs]
    claims = [Claim(text="Net jobs fall", supporting_source_ids=[ids[0]], importance=0.9)]
    payload = {"validation_results": [{
        "claim": "Net jobs fall", "status": "CONTRADICTED", "confidence": 0.7,
        "supporting_sources": [], "contradicting_sources": [ids[2]], "reason": "disagree",
    }]}
    fake_llm_factory(lambda s, u: json.dumps(payload))
    results = fact_check("Q?", claims, srcs)
    assert results[0].status == ValidationStatus.CONTRADICTED
    assert results[0].is_flagged


def test_fact_checker_honest_without_llm(no_llm):
    srcs = _sources()
    claims = [Claim(text="X", importance=0.9)]
    results = fact_check("Q?", claims, srcs)
    # Never rubber-stamps: no SUPPORTED without evidence.
    assert all(r.status == ValidationStatus.INSUFFICIENT_EVIDENCE for r in results)
