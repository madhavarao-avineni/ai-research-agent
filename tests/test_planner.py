"""Planner output tests."""

import json

from agents.planner import plan_research


def test_planner_llm_path(fake_llm_factory):
    payload = {
        "main_question": "Q?",
        "research_objective": "obj",
        "sub_questions": [
            {"question": "a", "required_evidence_types": ["web", "research_paper"]},
            {"question": "b", "required_evidence_types": ["news"]},
            {"question": "c", "required_evidence_types": ["web"]},
            {"question": "d", "required_evidence_types": ["report"]},
        ],
        "research_strategy": ["s1"],
        "required_source_types": ["web", "research_paper", "news"],
        "ambiguities": ["x"],
    }
    fake_llm_factory(lambda s, u: json.dumps(payload))
    plan = plan_research("Q?")
    assert 3 <= len(plan.sub_questions) <= 7
    assert plan.research_objective == "obj"
    assert plan.required_source_types


def test_planner_trims_to_seven(fake_llm_factory):
    payload = {"sub_questions": [{"question": f"q{i}"} for i in range(12)]}
    fake_llm_factory(lambda s, u: json.dumps(payload))
    plan = plan_research("Q?")
    assert len(plan.sub_questions) == 7


def test_planner_fallback_without_llm(no_llm):
    plan = plan_research("Some question")
    assert len(plan.sub_questions) >= 3  # heuristic still valid


def test_planner_malformed_json_falls_back(fake_llm_factory):
    fake_llm_factory(lambda s, u: "not json")
    plan = plan_research("Q?")
    assert len(plan.sub_questions) >= 3
