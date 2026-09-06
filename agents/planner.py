"""
agents/planner.py
=================
Agent 1 — Research Planner.

Decomposes the user's research question into a structured `ResearchPlan`
(objective, 3–7 sub-questions, strategy, required source types, ambiguities).
Performs NO research.

Graceful fallback: if the LLM is unavailable or returns unparseable output, a
deterministic heuristic plan is produced so the workflow can still proceed.
"""

from __future__ import annotations

from typing import Any

from config.llm import LLMUnavailable, get_llm
from models.state import (
    ResearchPlan,
    ResearchState,
    SourceType,
    SubQuestion,
)
from observability.logging_config import get_logger, trace_agent
from prompts.prompts import PLANNER_SYSTEM, PLANNER_USER

log = get_logger("agent.planner")

_ALLOWED = {t.value for t in SourceType}


def _coerce_source_types(values: Any) -> list[SourceType]:
    out: list[SourceType] = []
    if isinstance(values, list):
        for v in values:
            if isinstance(v, str) and v in _ALLOWED:
                out.append(SourceType(v))
    return out or [SourceType.WEB, SourceType.RESEARCH_PAPER, SourceType.NEWS]


def _parse_plan(question: str, data: dict) -> ResearchPlan:
    """Validate/normalize raw LLM JSON into a ResearchPlan (bounded 3–7 SQs)."""
    raw_sqs = data.get("sub_questions") or []
    sub_questions: list[SubQuestion] = []
    for item in raw_sqs:
        if isinstance(item, str):
            sub_questions.append(SubQuestion(question=item))
        elif isinstance(item, dict) and item.get("question"):
            sub_questions.append(
                SubQuestion(
                    question=str(item["question"]),
                    rationale=item.get("rationale"),
                    required_evidence_types=_coerce_source_types(
                        item.get("required_evidence_types")
                    ),
                )
            )
    # Enforce 3–7 bound.
    if len(sub_questions) > 7:
        sub_questions = sub_questions[:7]

    return ResearchPlan(
        main_question=str(data.get("main_question") or question),
        research_objective=str(
            data.get("research_objective") or f"Answer: {question}"
        ),
        sub_questions=sub_questions,
        research_strategy=[str(s) for s in (data.get("research_strategy") or [])],
        required_source_types=_coerce_source_types(
            data.get("required_source_types")
        ),
        ambiguities=[str(a) for a in (data.get("ambiguities") or [])],
    )


def _fallback_plan(question: str) -> ResearchPlan:
    """Deterministic minimal plan when the LLM can't be used."""
    log.warning("Planner using heuristic fallback plan.")
    base_types = [SourceType.WEB, SourceType.RESEARCH_PAPER, SourceType.NEWS]
    sqs = [
        SubQuestion(
            question=f"What does current evidence say about: {question}?",
            rationale="Establish the baseline state of knowledge.",
            required_evidence_types=base_types,
        ),
        SubQuestion(
            question=f"What are the main arguments FOR the proposition in: {question}?",
            rationale="Capture supporting evidence and its strength.",
            required_evidence_types=base_types,
        ),
        SubQuestion(
            question=f"What are the main arguments AGAINST the proposition in: {question}?",
            rationale="Capture disconfirming evidence and counterpoints.",
            required_evidence_types=base_types,
        ),
    ]
    return ResearchPlan(
        main_question=question,
        research_objective=f"Answer the question based on multi-source evidence: {question}",
        sub_questions=sqs,
        research_strategy=[
            "Gather baseline evidence from multiple source types.",
            "Contrast supporting and opposing evidence.",
            "Assess credibility and resolve contradictions where possible.",
        ],
        required_source_types=base_types,
        ambiguities=["Timeframe, scope, and key term definitions may be underspecified."],
    )


def plan_research(question: str) -> ResearchPlan:
    """Core planner logic (LLM with heuristic fallback)."""
    llm = get_llm()
    if not llm.available:
        return _fallback_plan(question)
    try:
        data = llm.complete_json(
            PLANNER_SYSTEM, PLANNER_USER.format(question=question)
        )
        if not isinstance(data, dict):
            raise ValueError("planner returned non-object JSON")
        plan = _parse_plan(question, data)
        if len(plan.sub_questions) < 3:
            # Top up with heuristic sub-questions to satisfy the 3–7 rule.
            extra = _fallback_plan(question).sub_questions
            for sq in extra:
                if len(plan.sub_questions) >= 3:
                    break
                plan.sub_questions.append(sq)
        return plan
    except (LLMUnavailable, ValueError) as exc:
        log.error("Planner failed (%s); using fallback.", exc)
        return _fallback_plan(question)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def planner_node(state: ResearchState) -> dict:
    """LangGraph node wrapper: reads `question`, writes plan channels."""
    question = state["question"]
    with trace_agent(
        "planner",
        input_summary=question,
        research_iteration=state.get("research_iteration", 0),
    ) as trace:
        plan = plan_research(question)
        trace.output_summary = (
            f"{len(plan.sub_questions)} sub-questions, "
            f"{len(plan.required_source_types)} source types"
        )
    return {
        "research_plan": plan,
        "research_objective": plan.research_objective,
        "sub_questions": plan.sub_questions,
        "traces": [trace],
    }
