"""
graph/research_graph.py
======================
LangGraph orchestration of the multi-agent research workflow.

    START -> planner -> retriever -> analyzer -> insight_generator
          -> fact_checker -> quality_check
                 ├── INSUFFICIENT (iterations remain) -> retriever   (loop)
                 └── SUFFICIENT / budget exhausted     -> report_builder -> END

The quality loop uses a conditional edge (`quality_router`) and is bounded by
`max_iterations` to prevent infinite loops.

`build_graph()` returns a compiled LangGraph app. `run_research()` is a
convenience runner that builds initial state and invokes the graph, returning
the final `ResearchState`.
"""

from __future__ import annotations

from typing import Optional

from agents.analyzer import analyzer_node
from agents.fact_checker import fact_checker_node
from agents.insight_generator import insight_node
from agents.planner import planner_node
from agents.quality import quality_node, quality_router
from agents.report_builder import report_builder_node
from agents.retriever import retriever_node
from config.settings import get_settings
from models.state import ResearchState, new_research_state
from observability.logging_config import configure_logging, get_logger

log = get_logger("graph")


def build_graph():
    """Build and compile the LangGraph workflow."""
    from langgraph.graph import END, START, StateGraph

    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("insight_generator", insight_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("quality_check", quality_node)
    workflow.add_node("report_builder", report_builder_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "insight_generator")
    workflow.add_edge("insight_generator", "fact_checker")
    workflow.add_edge("fact_checker", "quality_check")

    # Conditional edge: loop back to retriever OR proceed to report.
    workflow.add_conditional_edges(
        "quality_check",
        quality_router,
        {"retriever": "retriever", "report_builder": "report_builder"},
    )
    workflow.add_edge("report_builder", END)

    return workflow.compile()


def run_research(
    question: str,
    *,
    max_iterations: Optional[int] = None,
    recursion_limit: int = 50,
) -> ResearchState:
    """Build state, run the graph, and return the final ResearchState."""
    configure_logging()
    settings = get_settings()
    if max_iterations is None:
        max_iterations = settings.max_research_iterations

    state = new_research_state(question, max_iterations=max_iterations)
    if settings.missing_required():
        log.warning("Missing config: %s — agents will use fallbacks.", settings.missing_required())

    app = build_graph()
    log.info("Starting research: %r (max_iterations=%d)", question, max_iterations)
    final_state = app.invoke(state, config={"recursion_limit": recursion_limit})
    log.info("Research complete. Sources=%d, iterations=%d",
             len(final_state.get("sources", [])), final_state.get("research_iteration", 0))
    return final_state
