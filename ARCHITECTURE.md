# Architecture — AI-Powered Multi-Agent Research Assistant

## 1. Purpose

A team of specialized AI agents that, given **one complex research question**,
collaboratively plan, retrieve from multiple sources, critically analyze,
detect contradictions, generate insights, independently fact-check, decide
whether more research is needed, and produce a **cited final report**.

Orchestration: **LangGraph**. Implementation: **Python**. RAG: **Chroma**.

This is a proof of concept that demonstrably does more than a single LLM call:
multiple agents with distinct responsibilities pass a shared, typed state and
every conclusion is traceable to its supporting evidence.

---

## 2. Agent Team

| # | Agent | Responsibility | Key output (model) |
|---|-------|----------------|--------------------|
| 1 | **Research Planner** | Decompose question into 3–7 sub-questions, define objective, strategy, required evidence types, note ambiguities. No research here. | `ResearchPlan` |
| 2 | **Contextual Retriever** | Query web / papers / news / reports per sub-question. Dedupe, rank by relevance, capture provenance & credibility. | `list[RetrievedSource]` |
| 3 | **Critical Analysis** | Extract key claims, assess credibility & limitations, find agreements, contradictions, evidence gaps. | `AnalysisResult`, `list[Claim]` |
| 4 | **Insight Generation** | Produce higher-level insights, each tagged SUPPORTED / TREND / HYPOTHESIS / SPECULATION, each referencing sources. | `list[Insight]` |
| 5 | **Fact Checker** | Independently validate important claims against the evidence — must not just echo prior agents. | `list[ValidationResult]` |
| 6 | **Report Builder** | Assemble the structured, cited Markdown report. | `FinalReport` |

A **Research-Quality Decision** node (not an agent, a conditional edge) sits
between the fact checker and the report builder.

---

## 3. Workflow (LangGraph)

```
                         START
                           │
                           ▼
                      ┌─────────┐
                      │ planner │
                      └─────────┘
                           │
                           ▼
                    ┌─────────────┐   ◄────────────────────┐
                    │  retriever  │                        │
                    └─────────────┘                        │
                           │                                │
                           ▼                                │
                    ┌─────────────┐                         │
                    │  analyzer   │                         │
                    └─────────────┘                         │
                           │                                │
                           ▼                                │
                 ┌────────────────────┐                     │
                 │ insight_generator  │                     │
                 └────────────────────┘                     │
                           │                                │
                           ▼                                │
                    ┌──────────────┐                        │
                    │ fact_checker │                        │
                    └──────────────┘                        │
                           │                                │
                           ▼                                │
                  ┌──────────────────┐                      │
                  │  quality_check   │  (conditional edge)  │
                  └──────────────────┘                      │
                    │              │                        │
        INSUFFICIENT │              │ SUFFICIENT            │
        (iteration<max)             │                        │
                    └───────────────┼── back to retriever ──┘
                                    │
                                    ▼
                            ┌────────────────┐
                            │ report_builder │
                            └────────────────┘
                                    │
                                    ▼
                                   END
```

**Quality-check loop.** If evidence is `INSUFFICIENT` and
`research_iteration < max_iterations`, the graph loops back to `retriever`
with `followup_queries` targeting evidence gaps, contradictions, unverified
claims and missing source types. The loop is bounded (default 2 extra
iterations) to prevent runaway execution.

**Parallelism.** Independent sub-question retrieval is fanned out inside the
retriever node (thread pool over tools/sub-questions) so multi-source lookups
run concurrently while the graph topology stays simple.

---

## 4. Shared State

A single typed `ResearchState` (TypedDict) is threaded between nodes; rich
artifacts are Pydantic models validated at agent boundaries. See
[`models/state.py`](./models/state.py).

Key channels: `question`, `research_objective`, `sub_questions`,
`research_plan`, `sources`, `retrieved_documents`, `key_claims`, `analysis`,
`agreements`, `contradictions`, `evidence_gaps`, `insights`, `hypotheses`,
`validation_results`, `research_iteration`, `final_report`, plus observability
channels `traces` and `errors`.

**Provenance rule.** Every `Claim`, `Insight`, `Agreement`, `Contradiction`
and `ValidationResult` references sources by `source_id`. Evidence is never
merged in a way that loses the original source, so any statement in the final
report is traceable back to the `RetrievedSource` that supports it.

Accumulating channels (`sources`, `retrieved_documents`, `traces`, `errors`)
use `operator.add` reducers so writes across research-loop iterations
**append** rather than overwrite.

---

## 5. Tools (independent from agents)

| Tool | File | Backend | Notes |
|------|------|---------|-------|
| Web Search | `tools/web_search.py` | Tavily (key) | Current web info; graceful fallback |
| Research Papers | `tools/paper_search.py` | arXiv API | Publicly available papers |
| News Search | `tools/news_search.py` | NewsAPI → RSS fallback | Recent articles |
| Document Retrieval | `tools/document_retrieval.py` | RAG / Chroma | Future PDF & internal docs |

Agents **decide when** a tool is needed; search logic lives in the tools, not
in prompts. Each tool returns `RetrievedSource` objects with full metadata and
degrades gracefully (empty list + logged error) when its API is unavailable.

---

## 6. RAG Layer

```
Documents → Chunking → Embeddings → Chroma Vector DB → Semantic Retrieval → LLM
```

- `rag/embeddings.py` — pluggable embeddings (local sentence-transformers by
  default; OpenAI optional).
- `rag/vector_store.py` — thin Chroma wrapper behind an interface so the store
  can be swapped later (pgvector, FAISS, etc.).
- `rag/retriever.py` — chunk + upsert + semantic query.

The vector-store interface is deliberately abstract so Chroma can be replaced
without touching agents.

---

## 7. Observability & Security

- **Logging** (`observability/logging_config.py`): per-agent structured
  `AgentTrace` records — name, start/end, input/output summaries, tools used,
  sources retrieved, iteration, errors.
- **LangSmith**: enabled via `LANGCHAIN_TRACING_V2` env var (optional).
- **Security**: all secrets via environment variables; `.env.example` ships
  placeholders only; API keys are never logged.

---

## 8. Error Handling

Search API failure, LLM failure, empty results, duplicate sources, invalid
documents, rate limits, timeouts, and missing env vars are all handled
gracefully. A single unavailable source never crashes the run — it is logged
and the workflow continues with what it has.

---

## 9. Technology Stack

LangGraph · LangChain · Pydantic v2 · Chroma · sentence-transformers ·
Tavily · arXiv · Streamlit · LangSmith · pytest.
