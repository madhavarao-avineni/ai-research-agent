# AI-Powered Multi-Agent Research Assistant

> A team of specialized AI agents that takes **one complex research question**
> and autonomously plans, retrieves from multiple sources, analyzes evidence,
> detects contradictions, generates insights, independently fact-checks,
> decides whether more research is needed, and produces a **cited final report**.
>
> Built with **LangGraph** (orchestration), **Python**, and **Chroma** (RAG).

---

## 1. Project Objective

Automate end-to-end research. Instead of manually searching, reading,
comparing, validating and writing up findings, the user asks one question and a
collaborating team of agents does the work — with full source traceability.

## 2. Architecture

See **[ARCHITECTURE.md](./ARCHITECTURE.md)** for the full diagram and design.

```
User → Planner → Retriever → Analyzer → Insight Generator → Fact Checker
     → Quality Check ──(insufficient)──► back to Retriever
                     └─(sufficient)────► Report Builder → Final Report
```

## 3. Agent Responsibilities

| Agent | Does |
|-------|------|
| **Research Planner** | Decomposes the question into 3–7 sub-questions, defines objective & strategy |
| **Contextual Retriever** | Multi-source retrieval (web, papers, news, reports), dedupe, ranking, provenance |
| **Critical Analysis** | Key claims, credibility, agreements, contradictions, evidence gaps |
| **Insight Generation** | Insights tagged SUPPORTED / TREND / HYPOTHESIS / SPECULATION |
| **Fact Checker** | Independent claim validation (SUPPORTED … INSUFFICIENT_EVIDENCE) |
| **Report Builder** | Structured, cited Markdown report |

## 4. Technology Stack

LangGraph · LangChain · Pydantic v2 · Chroma · sentence-transformers ·
Tavily (web) · arXiv (papers) · NewsAPI/RSS (news) · Streamlit · LangSmith · pytest.

## 5. Installation

```bash
cd ai-research-assistant
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 6. Environment Variables

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM provider key |
| `LLM_MODEL` | e.g. `gpt-4o-mini` |
| `TAVILY_API_KEY` | Web search |
| `NEWSAPI_KEY` | News (optional; RSS fallback) |
| `EMBEDDINGS_PROVIDER` | `local` or `openai` |
| `CHROMA_PERSIST_DIR` | Chroma storage path |
| `MAX_RESEARCH_ITERATIONS` | Loop bound (default 2) |
| `LANGCHAIN_TRACING_V2` | `true` to enable LangSmith |

Secrets are read from the environment only — never hard-coded, never logged.

## 7. How to Run

```bash
# CLI (headless)
python main.py --question "Will generative AI significantly reduce the demand for software developers over the next five years?"

# UI
streamlit run ui/app.py
```

## 8. Example Research Question

> "Will generative AI significantly reduce the demand for software developers
> over the next five years?"

## 9. Example Output

A Markdown report with: Executive Summary · Methodology · Key Findings ·
Evidence · Contradictions · Emerging Trends · Insights · Hypotheses · Risks &
Limitations · Conclusion · References — downloadable from the UI.

## 10. Architecture Diagram

See [ARCHITECTURE.md §3](./ARCHITECTURE.md).

## 11. Future Enhancements

MCP tools · enterprise KB · PDF ingestion · SharePoint/Drive connectors ·
SQL research · persistent research memory · human approval gate · multiple LLM
providers · cost tracking · agent evaluation · confidence scoring · scheduled
research · Email/Teams delivery.

---

## Project Status

Foundation delivered (structure, shared state models, docs). Implementation
proceeds per **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)** after approval.


---

## Project Structure

```
ai-research-assistant/
├── agents/            planner, retriever, analyzer, insight_generator,
│                      fact_checker, quality, report_builder
├── tools/             base, web_search, paper_search, news_search,
│                      document_retrieval
├── rag/               embeddings, vector_store (Chroma + in-memory), retriever
├── graph/             research_graph  (LangGraph workflow + quality loop)
├── models/            state.py  (ResearchState + all Pydantic models)
├── prompts/           prompts.py (all agent prompts)
├── config/            settings.py, llm.py (provider-agnostic LLM)
├── observability/     logging_config.py (structured logs, secret redaction,
│                      AgentTrace, LangSmith hook)
├── ui/                app.py  (Streamlit)
├── tests/             planner, retriever, analysis, quality/report, e2e
├── main.py            CLI entry point
├── requirements.txt   .env.example   .gitignore
└── README.md  ARCHITECTURE.md  IMPLEMENTATION_PLAN.md
```

## Running the Tests

```bash
pytest -q            # all tests
pytest tests/test_end_to_end.py -q   # full workflow (mocked LLM/tools, no keys)
```

Tests use a `FakeClient` LLM and in-memory vector store, so they run **offline
with no API keys**. They cover: planner output & bounds, search tool behavior,
duplicate removal, source ranking, contradiction detection, claim validation,
research-loop bounding, report generation, empty search results, and API
failure isolation — plus one end-to-end run of the whole graph.

## Notes on Graceful Degradation

Every agent has a safe fallback: if no LLM key is set the planner uses a
heuristic plan, the fact checker honestly marks claims `INSUFFICIENT_EVIDENCE`
(never rubber-stamps), and the report is still rendered deterministically.
Search tools return `[]` on failure so one dead source never crashes a run.
The news tool falls back from NewsAPI to Google News RSS (no key). The vector
store falls back from Chroma to an in-memory cosine store, and embeddings fall
back from local sentence-transformers to a hashing embedder.
