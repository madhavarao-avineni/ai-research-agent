"""
models/state.py
================
Shared, typed state for the AI-Powered Multi-Agent Research Assistant.

This module defines:
  1. Strongly-typed Pydantic models for every structured artifact that flows
     through the LangGraph workflow (sources, claims, analysis, insights,
     validation results, and the final report).
  2. `ResearchState` — the single TypedDict that LangGraph threads between
     nodes. Every agent reads from and writes to this object.

Design principles
-----------------
* **Source provenance is never lost.** Evidence, claims and insights all keep
  references (by `source_id`) back to the `RetrievedSource` that produced them,
  so any conclusion in the final report is traceable to its evidence.
* **Pydantic for structured artifacts, TypedDict for the graph channel.**
  LangGraph merges the top-level `ResearchState` dict; the rich nested objects
  are Pydantic models validated at agent boundaries.
* **Enums, not magic strings.** Statuses (SUPPORTED / CONTRADICTED / ...) and
  insight tiers (SUPPORTED / TREND / HYPOTHESIS / SPECULATION) are enums so the
  UI, report builder and tests share one vocabulary.

NOTE: This file is part of the *foundation* delivery. It contains the data
contracts only — no agent logic runs here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional
from uuid import uuid4

import operator
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ---------------------------------------------------------------------------
# Enumerations — shared vocabulary across agents, UI, and report
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """The kind of source a piece of evidence came from."""

    WEB = "web"
    RESEARCH_PAPER = "research_paper"
    NEWS = "news"
    REPORT = "report"
    INTERNAL_DOCUMENT = "internal_document"  # RAG / future PDF ingestion


class InsightTier(str, Enum):
    """
    Epistemic strength of an insight. Never present a HYPOTHESIS or
    SPECULATION as an established fact.
    """

    SUPPORTED = "SUPPORTED"        # Backed by concrete, corroborated evidence
    TREND = "TREND"                # Pattern emerging across multiple sources
    HYPOTHESIS = "HYPOTHESIS"      # Plausible, testable, not yet confirmed
    SPECULATION = "SPECULATION"    # Weakly grounded; explicitly uncertain


class ValidationStatus(str, Enum):
    """Fact-checker verdict for an individual claim."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceSufficiency(str, Enum):
    """Output of the research-quality decision node."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Planning artifacts
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A single focused research sub-question produced by the Planner."""

    id: str = Field(default_factory=lambda: f"sq_{uuid4().hex[:8]}")
    question: str
    rationale: Optional[str] = Field(
        default=None, description="Why this sub-question matters to the main goal."
    )
    required_evidence_types: list[SourceType] = Field(
        default_factory=list,
        description="Which source types are expected to answer this sub-question.",
    )


class ResearchPlan(BaseModel):
    """Structured output of the Research Planner Agent."""

    main_question: str
    research_objective: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    research_strategy: list[str] = Field(
        default_factory=list, description="Ordered strategy steps / approach."
    )
    required_source_types: list[SourceType] = Field(default_factory=list)
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Potential ambiguities in the question the researcher should note.",
    )


# ---------------------------------------------------------------------------
# Retrieval artifacts
# ---------------------------------------------------------------------------


class RetrievedSource(BaseModel):
    """
    A single retrieved source with full provenance metadata.

    `source_id` is the stable handle used by every downstream agent to
    reference this source (in claims, analysis, insights, validation).
    """

    source_id: str = Field(default_factory=lambda: f"src_{uuid4().hex[:10]}")
    title: str
    url: Optional[str] = None
    source_type: SourceType = SourceType.WEB
    author: Optional[str] = None
    publication_date: Optional[str] = None  # kept as string; upstream formats vary
    publisher: Optional[str] = None
    content: str = ""  # full retrieved text (not just a snippet)
    snippet: Optional[str] = None
    # Which sub-question(s) this source was retrieved for.
    sub_question_ids: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    credibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("url", mode="before")
    @classmethod
    def _coerce_url(cls, v: Any) -> Optional[str]:
        # Accept plain strings, HttpUrl, or None without hard failure.
        if v is None:
            return None
        return str(v)

    def dedupe_key(self) -> str:
        """Key used by the retriever to remove duplicates."""
        if self.url:
            return self.url.strip().lower().rstrip("/")
        return self.title.strip().lower()


# ---------------------------------------------------------------------------
# Analysis artifacts
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """A key claim extracted from a source, retaining provenance."""

    id: str = Field(default_factory=lambda: f"clm_{uuid4().hex[:8]}")
    text: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    sub_question_id: Optional[str] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class Agreement(BaseModel):
    """Where multiple sources support the same conclusion."""

    statement: str
    source_ids: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    """Where sources disagree. Both sides keep their source references."""

    topic: str
    position_a: str
    position_a_source_ids: list[str] = Field(default_factory=list)
    position_b: str
    position_b_source_ids: list[str] = Field(default_factory=list)
    explanation: Optional[str] = Field(
        default=None,
        description="Only populated when evidence actually supports an explanation.",
    )


class EvidenceGap(BaseModel):
    """A sub-question / topic where evidence is missing or insufficient."""

    topic: str
    description: str
    related_sub_question_id: Optional[str] = None


class SourceQualityAssessment(BaseModel):
    """Per-source credibility / limitation assessment."""

    source_id: str
    credibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: list[str] = Field(
        default_factory=list,
        description="e.g. small sample size, vendor bias, opinion piece, outdated.",
    )
    notes: Optional[str] = None


class AnalysisResult(BaseModel):
    """Structured output of the Critical Analysis Agent."""

    key_findings: list[str] = Field(default_factory=list)
    agreements: list[Agreement] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    source_quality_assessment: list[SourceQualityAssessment] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Insight artifacts
# ---------------------------------------------------------------------------


class Insight(BaseModel):
    """
    A higher-level insight. `tier` records epistemic strength; every insight
    must reference the sources that support it.
    """

    id: str = Field(default_factory=lambda: f"ins_{uuid4().hex[:8]}")
    statement: str
    tier: InsightTier
    supporting_source_ids: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Validation artifacts
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """
    Independent fact-checker verdict for a single claim. The fact checker must
    evaluate the evidence itself, not merely echo prior agents.
    """

    claim: str
    status: ValidationStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_sources: list[str] = Field(default_factory=list)
    contradicting_sources: list[str] = Field(default_factory=list)
    reason: str = ""

    @property
    def is_flagged(self) -> bool:
        """CONTRADICTED / INSUFFICIENT_EVIDENCE claims are flagged for report."""
        return self.status in (
            ValidationStatus.CONTRADICTED,
            ValidationStatus.INSUFFICIENT_EVIDENCE,
        )


# ---------------------------------------------------------------------------
# Reporting artifacts
# ---------------------------------------------------------------------------


class FinalReport(BaseModel):
    """The structured final report. `markdown` is the rendered deliverable."""

    executive_summary: str = ""
    methodology: str = ""
    key_findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    emerging_trends: list[str] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    hypotheses: list[Insight] = Field(default_factory=list)
    risks_and_limitations: list[str] = Field(default_factory=list)
    conclusion: str = ""
    references: list[RetrievedSource] = Field(default_factory=list)
    markdown: str = ""  # fully rendered report, ready for download


# ---------------------------------------------------------------------------
# Observability artifact
# ---------------------------------------------------------------------------


class AgentTrace(BaseModel):
    """One structured log record per agent execution (see observability/)."""

    agent_name: str
    start_time: str
    end_time: Optional[str] = None
    input_summary: str = ""
    output_summary: str = ""
    tools_used: list[str] = Field(default_factory=list)
    sources_retrieved: int = 0
    research_iteration: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# The shared LangGraph state
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """
    Single shared state object threaded through the LangGraph workflow.

    `total=False` so nodes can populate incrementally. List channels that may
    be appended to across the research loop use `operator.add` reducers so
    parallel / iterative writes accumulate instead of overwrite.
    """

    # --- Input ---
    question: str
    config: dict[str, Any]  # runtime config (max_iterations, model, etc.)

    # --- Planner output ---
    research_objective: str
    sub_questions: list[SubQuestion]
    research_plan: ResearchPlan

    # --- Retriever output (accumulate across iterations) ---
    sources: Annotated[list[RetrievedSource], operator.add]
    retrieved_documents: Annotated[list[RetrievedSource], operator.add]

    # --- Analysis output ---
    key_claims: list[Claim]
    analysis: AnalysisResult
    agreements: list[Agreement]
    contradictions: list[Contradiction]
    evidence_gaps: list[EvidenceGap]

    # --- Insight output ---
    insights: list[Insight]
    hypotheses: list[Insight]

    # --- Validation output ---
    validation_results: list[ValidationResult]

    # --- Quality-loop control ---
    research_iteration: int
    max_iterations: int
    evidence_sufficiency: EvidenceSufficiency
    followup_queries: list[str]  # generated when evidence is INSUFFICIENT

    # --- Report output ---
    final_report: FinalReport

    # --- Observability (accumulate) ---
    traces: Annotated[list[AgentTrace], operator.add]
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def new_research_state(
    question: str,
    *,
    max_iterations: int = 2,
    config: Optional[dict[str, Any]] = None,
) -> ResearchState:
    """Create an initial ResearchState with sane, empty defaults."""

    return ResearchState(
        question=question,
        config=config or {},
        sub_questions=[],
        sources=[],
        retrieved_documents=[],
        key_claims=[],
        agreements=[],
        contradictions=[],
        evidence_gaps=[],
        insights=[],
        hypotheses=[],
        validation_results=[],
        research_iteration=0,
        max_iterations=max_iterations,
        followup_queries=[],
        traces=[],
        errors=[],
    )
