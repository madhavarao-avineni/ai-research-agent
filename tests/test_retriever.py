"""Search tool, duplicate removal, and ranking tests."""

from agents.retriever import deduplicate, rank, retrieve
from models.state import RetrievedSource, SourceType
from tools.base import make_source


def _src(title, url, st=SourceType.WEB, content="x", rel=0.5, sqid=None):
    return make_source(title=title, url=url, source_type=st, content=content,
                       relevance_score=rel, sub_question_id=sqid)


def test_deduplicate_merges_and_removes():
    s1 = _src("A", "https://ex.com/x", rel=0.4, sqid="sq1")
    s2 = _src("A dup", "https://EX.com/x/", content="longer content body", rel=0.9, sqid="sq2")
    s3 = _src("B", "https://ex.com/y", rel=0.5)
    out = deduplicate([s1, s2, s3])
    assert len(out) == 2
    merged = [s for s in out if s.dedupe_key() == "https://ex.com/x"][0]
    assert set(merged.sub_question_ids) == {"sq1", "sq2"}
    assert merged.relevance_score == 0.9  # kept max


def test_ranking_orders_by_blended_score():
    low = _src("low", "https://a", rel=0.2)
    high = _src("high", "https://b", rel=0.95)
    out = rank([low, high])
    assert out[0].title == "high"


def test_retrieve_isolates_tool_failure(monkeypatch):
    import agents.retriever as R

    def ok_web(q, max_results=4, sub_question_id=None):
        return [_src("web", f"https://w/{q}", rel=0.6, sqid=sub_question_id)]

    def boom(q, max_results=4, sub_question_id=None):
        raise RuntimeError("API down")

    monkeypatch.setattr(R, "web_search", ok_web)
    monkeypatch.setattr(R, "paper_search", boom)
    monkeypatch.setattr(R, "news_search", boom)

    # avoid RAG/chroma side effects
    class _NoIndex:
        def ingest_sources(self, s):
            return 0

    out = retrieve(
        queries=[("q1", "sq1", [SourceType.WEB, SourceType.RESEARCH_PAPER, SourceType.NEWS])],
        doc_tool=_NoIndex(),
    )
    # web succeeded, others failed but did not crash
    assert len(out) >= 1
    assert all(isinstance(s, RetrievedSource) for s in out)


def test_empty_search_results(monkeypatch):
    import agents.retriever as R

    def empty(q, max_results=4, sub_question_id=None):
        return []

    monkeypatch.setattr(R, "web_search", empty)
    monkeypatch.setattr(R, "paper_search", empty)
    monkeypatch.setattr(R, "news_search", empty)

    class _NoIndex:
        def ingest_sources(self, s):
            return 0

    out = retrieve(queries=[("q", None, [SourceType.WEB])], doc_tool=_NoIndex())
    assert out == []
