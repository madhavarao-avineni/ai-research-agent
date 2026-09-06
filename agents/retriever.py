"""
agents/retriever.py
===================
Agent 2 — Contextual Retriever.

Takes sub-questions and searches multiple sources (web, papers, news, and
RAG-backed documents) in parallel, then:
  * removes duplicates (by URL/title),
  * ranks by a blended relevance+credibility score,
  * indexes retrieved content into the RAG store for later semantic recall,
  * preserves full source metadata / provenance.

On follow-up iterations it uses `followup_queries` (from the quality loop)
instead of the original sub-questions, to target gaps and contradictions.

Every tool call is isolated: a failure in one tool/source never aborts the run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from models.state import (
    ResearchState,
    RetrievedSource,
    SourceType,
    SubQuestion,
)
from observability.logging_config import get_logger, trace_agent
from tools.document_retrieval import DocumentRetrievalTool
from tools.news_search import news_search
from tools.paper_search import paper_search
from tools.web_search import web_search

log = get_logger("agent.retriever")


# Map source type -> callable tool for that type.
def _tool_for(source_type: SourceType):
    return {
        SourceType.WEB: web_search,
        SourceType.RESEARCH_PAPER: paper_search,
        SourceType.NEWS: news_search,
        SourceType.REPORT: web_search,  # reports discovered via web search
    }.get(source_type)


def deduplicate(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    """Remove duplicates by dedupe_key, merging sub_question_ids + keeping best."""
    seen: dict[str, RetrievedSource] = {}
    for s in sources:
        key = s.dedupe_key()
        if key not in seen:
            seen[key] = s
        else:
            kept = seen[key]
            # Merge which sub-questions this covers.
            for sqid in s.sub_question_ids:
                if sqid not in kept.sub_question_ids:
                    kept.sub_question_ids.append(sqid)
            # Keep the richer content and higher relevance.
            if len(s.content) > len(kept.content):
                kept.content = s.content
            kept.relevance_score = max(kept.relevance_score, s.relevance_score)
    return list(seen.values())


def rank(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    """Rank by blended score: 0.65*relevance + 0.35*credibility."""
    return sorted(
        sources,
        key=lambda s: 0.65 * s.relevance_score + 0.35 * s.credibility_score,
        reverse=True,
    )


def _search_one(source_type: SourceType, query: str, sqid: Optional[str], per_source: int) -> list[RetrievedSource]:
    tool = _tool_for(source_type)
    if tool is None:
        return []
    try:
        return tool(query, max_results=per_source, sub_question_id=sqid)
    except Exception as exc:  # noqa: BLE001 - isolate tool failures
        log.error("tool %s failed for '%s': %s", source_type.value, query[:40], type(exc).__name__)
        return []


def retrieve(
    *,
    queries: list[tuple[str, Optional[str], list[SourceType]]],
    per_source: int = 4,
    max_workers: int = 6,
    doc_tool: Optional[DocumentRetrievalTool] = None,
) -> list[RetrievedSource]:
    """
    Run searches in parallel. `queries` is a list of
    (query_text, sub_question_id, [source_types_to_search]).
    Returns deduped, ranked sources; also indexes them into RAG.
    """
    tasks: list[tuple[SourceType, str, Optional[str]]] = []
    for query, sqid, types in queries:
        for st in types:
            tasks.append((st, query, sqid))

    collected: list[RetrievedSource] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_search_one, st, q, sqid, per_source): (st, q)
                for (st, q, sqid) in tasks
            }
            for fut in as_completed(futures):
                try:
                    collected.extend(fut.result() or [])
                except Exception as exc:  # noqa: BLE001
                    log.error("search task error: %s", type(exc).__name__)

    deduped = deduplicate(collected)
    ranked = rank(deduped)

    # Index into RAG for semantic recall (best-effort).
    try:
        (doc_tool or DocumentRetrievalTool()).ingest_sources(ranked)
    except Exception as exc:  # noqa: BLE001
        log.warning("RAG indexing skipped: %s", type(exc).__name__)

    return ranked


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def retriever_node(state: ResearchState) -> dict:
    """
    LangGraph node. First iteration searches per sub-question; later iterations
    use followup_queries. Appends to `sources` / `retrieved_documents`.
    """
    iteration = state.get("research_iteration", 0)
    followups = state.get("followup_queries") or []
    plan = state.get("research_plan")
    default_types = (
        plan.required_source_types if plan and plan.required_source_types
        else [SourceType.WEB, SourceType.RESEARCH_PAPER, SourceType.NEWS]
    )

    with trace_agent("retriever", input_summary=f"iter={iteration}", research_iteration=iteration) as trace:
        queries: list[tuple[str, Optional[str], list[SourceType]]] = []
        if iteration > 0 and followups:
            # Follow-up round: broad search across default types.
            for q in followups:
                queries.append((q, None, default_types))
            trace.input_summary = f"iter={iteration}, {len(followups)} follow-up queries"
        else:
            for sq in state.get("sub_questions", []):
                types = sq.required_evidence_types or default_types
                queries.append((sq.question, sq.id, types))
            trace.input_summary = f"iter={iteration}, {len(queries)} sub-questions"

        sources = retrieve(queries=queries)
        trace.tools_used = ["web_search", "paper_search", "news_search", "document_retrieval"]
        trace.sources_retrieved = len(sources)
        trace.output_summary = f"{len(sources)} unique ranked sources"

    return {
        "sources": sources,
        "retrieved_documents": sources,
        "traces": [trace],
        # clear follow-ups now that they've been consumed
        "followup_queries": [],
    }
