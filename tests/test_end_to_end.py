"""
End-to-end test: runs the full LangGraph workflow with a FakeClient LLM and
stubbed search tools (no network, no API keys). Verifies all agents run and a
cited report is produced.
"""

import json

import pytest

from config.llm import LLM, FakeClient, set_llm
from models.state import SourceType
from tools.base import make_source


@pytest.fixture
def stub_everything(monkeypatch):
    ids = {"list": []}

    def responder(system, user):
        if "Research Planner" in system:
            return json.dumps({
                "main_question": "q", "research_objective": "obj",
                "sub_questions": [
                    {"question": "sqA", "required_evidence_types": ["web", "research_paper"]},
                    {"question": "sqB", "required_evidence_types": ["web", "news"]},
                    {"question": "sqC", "required_evidence_types": ["web"]},
                ],
                "research_strategy": ["s"], "required_source_types": ["web", "research_paper", "news"],
                "ambiguities": [],
            })
        if "Critical Analysis" in system:
            return json.dumps({
                "key_claims": [{"text": "AI speeds codegen", "supporting_source_ids": ids["list"][:1], "importance": 0.9}],
                "key_findings": ["speed up"],
                "agreements": [{"statement": "speed", "source_ids": ids["list"][:2]}],
                "contradictions": [], "evidence_gaps": [], "source_quality_assessment": [],
            })
        if "Insight Generation" in system:
            return json.dumps({"insights": [
                {"statement": "validation matters", "tier": "SUPPORTED", "supporting_source_ids": ids["list"][:1]}]})
        if "Fact Checker" in system:
            return json.dumps({"validation_results": [
                {"claim": "AI speeds codegen", "status": "SUPPORTED", "confidence": 0.8,
                 "supporting_sources": ids["list"][:1], "contradicting_sources": [], "reason": "ok"}]})
        if "Report Builder" in system:
            return json.dumps({
                "executive_summary": "ES", "methodology": "M", "key_findings": ["speed up"],
                "evidence": ["e"], "emerging_trends": ["t"], "risks_and_limitations": ["r"], "conclusion": "C"})
        return "{}"

    set_llm(LLM(FakeClient(responder)))

    import agents.retriever as R

    def web(q, max_results=4, sub_question_id=None):
        out = [make_source(title=f"web{i}", url=f"https://web/{abs(hash((q, i))) % 99999}",
                           source_type=SourceType.WEB, content="web " + q,
                           relevance_score=0.6, sub_question_id=sub_question_id) for i in range(3)]
        return out

    def paper(q, max_results=4, sub_question_id=None):
        return [make_source(title=f"paper{i}", url=f"https://arxiv.org/{abs(hash((q, i))) % 99999}",
                            source_type=SourceType.RESEARCH_PAPER, content="paper " + q,
                            relevance_score=0.7, sub_question_id=sub_question_id) for i in range(2)]

    def news(q, max_results=4, sub_question_id=None):
        return [make_source(title=f"news{i}", url=f"https://reuters.com/{abs(hash((q, i))) % 99999}",
                            source_type=SourceType.NEWS, content="news " + q,
                            relevance_score=0.55, sub_question_id=sub_question_id) for i in range(2)]

    monkeypatch.setattr(R, "web_search", web)
    monkeypatch.setattr(R, "paper_search", paper)
    monkeypatch.setattr(R, "news_search", news)

    # In-memory RAG (no chroma) for DocumentRetrievalTool used inside retrieve().
    import tools.document_retrieval as D
    from rag.retriever import RagRetriever
    from rag.vector_store import InMemoryVectorStore
    from rag.embeddings import HashingEmbedder

    orig_init = D.DocumentRetrievalTool.__init__

    def patched(self, retriever=None):
        orig_init(self, retriever=retriever or RagRetriever(
            store=InMemoryVectorStore(embedder=HashingEmbedder(dim=64))))

    monkeypatch.setattr(D.DocumentRetrievalTool, "__init__", patched)

    # expose ids after retrieval: patch analyzer to read current sources
    yield ids
    set_llm(LLM(None))


def test_full_workflow_produces_cited_report(stub_everything):
    from graph.research_graph import build_graph
    from models.state import new_research_state

    ids = stub_everything
    app = build_graph()
    state = new_research_state("Will GenAI reduce developer demand?", max_iterations=1)

    # Run streaming so we can capture source ids for the fake analyzer/factchecker.
    final = dict(state)
    for event in app.stream(state, config={"recursion_limit": 50}):
        for node, out in event.items():
            if out:
                for k, v in out.items():
                    if k in ("sources", "retrieved_documents", "traces", "errors") and isinstance(v, list):
                        final[k] = final.get(k, []) + v
                    else:
                        final[k] = v
            if node == "retriever":
                ids["list"] = [s.source_id for s in final.get("sources", [])]

    assert final.get("final_report") is not None
    report = final["final_report"]
    assert report.markdown
    assert "## References" in report.markdown
    assert len(final.get("sources", [])) >= 3
    # All agents were traced.
    traced = {t.agent_name for t in final.get("traces", [])}
    for agent in ["planner", "retriever", "analyzer", "insight_generator", "fact_checker", "report_builder"]:
        assert agent in traced
