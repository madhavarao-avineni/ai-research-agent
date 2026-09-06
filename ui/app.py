"""
ui/app.py
=========
Streamlit UI for the AI Research Assistant — Luminous Dark Slate Editorial Design.

Run:  streamlit run ui/app.py

Sections:
  * Research question input + Start button
  * Agent progress checklist (updates as each agent completes)
  * Sources table (title, type, publisher, date, url, relevance)
  * Contradictions (shown separately)
  * Insights (Findings / Trends / Hypotheses, clearly labelled)
  * Final report (rendered Markdown) + Download as .md
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run via `streamlit run ui/app.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from config.settings import get_settings
from graph.research_graph import build_graph
from models.state import (
    EvidenceSufficiency,
    InsightTier,
    ValidationStatus,
    new_research_state,
)
from observability.logging_config import configure_logging

configure_logging()

st.set_page_config(page_title="Luminous Research Portal", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# Brutalist SaaS Design System - Golden Yellow + Charcoal
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Anton:wght@400&family=Satoshi:wght@400;500;700&display=swap');

    :root {
        --charcoal: #171e19;
        --dark-gray: #272727;
        --yellow: #ffe17c;
        --sage: #b7c6c2;
        --white: #ffffff;
    }

    * {
        font-family: 'Satoshi', sans-serif;
    }

    body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        color: var(--charcoal) !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--white);
        background-image:
            linear-gradient(to right, rgba(183, 198, 194, 0.1) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(183, 198, 194, 0.1) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1 {
        font-family: 'Anton', sans-serif;
        font-size: 3.5rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: -0.02em;
        line-height: 0.9;
        color: var(--charcoal);
        margin-bottom: 0.5rem;
    }

    h2 {
        font-family: 'Anton', sans-serif;
        font-size: 2.5rem;
        font-weight: 400;
        text-transform: uppercase;
        color: var(--charcoal);
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--yellow);
        padding-bottom: 1rem;
        line-height: 0.9;
        letter-spacing: -0.01em;
    }

    h3 {
        font-family: 'Anton', sans-serif;
        font-size: 1.25rem;
        font-weight: 400;
        text-transform: uppercase;
        color: var(--charcoal);
        margin-top: 1.5rem;
        letter-spacing: -0.01em;
    }

    p, span, div, label, li {
        color: var(--charcoal) !important;
    }

    .caption-text {
        font-family: 'Satoshi', sans-serif;
        color: var(--charcoal) !important;
        font-size: 0.875rem;
        font-weight: 500;
        letter-spacing: 0.05em;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: var(--white) !important;
        border: 1px solid rgba(23, 30, 25, 0.2) !important;
        color: var(--charcoal) !important;
        border-radius: 0.5rem !important;
        font-size: 1rem !important;
        font-family: 'Satoshi', sans-serif !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--yellow) !important;
        background-color: var(--white) !important;
        box-shadow: 0 0 0 3px rgba(255, 225, 124, 0.1) !important;
    }

    [data-testid="stSlider"] {
        padding: 1.5rem 0;
    }

    button[data-testid="stBaseButton-primary"] {
        font-family: 'Anton', sans-serif !important;
        background-color: var(--yellow) !important;
        color: var(--charcoal) !important;
        border-radius: 0.5rem !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 0.875rem 2rem !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--charcoal) !important;
        color: var(--yellow) !important;
        transform: scale(1.05) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Anton', sans-serif !important;
        color: var(--yellow) !important;
        font-size: 2.5rem !important;
        font-weight: 400 !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Satoshi', sans-serif !important;
        color: var(--charcoal) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 2px solid var(--yellow);
    }

    [data-testid="stTabs"] [role="tab"] {
        font-family: 'Satoshi', sans-serif !important;
        color: var(--charcoal) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--yellow) !important;
        border-bottom: 3px solid var(--yellow) !important;
    }

    [data-testid="stDataFrame"] {
        background-color: var(--white) !important;
        border: 1px solid rgba(23, 30, 25, 0.15) !important;
        border-radius: 0.5rem !important;
    }

    [data-testid="stDataFrame"] th {
        background-color: rgba(255, 225, 124, 0.15) !important;
        color: var(--charcoal) !important;
        font-weight: 600 !important;
        font-family: 'Satoshi', sans-serif !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stDataFrame"] td {
        color: var(--charcoal) !important;
        border-color: rgba(23, 30, 25, 0.1) !important;
        font-family: 'Satoshi', sans-serif !important;
    }

    .stSuccess {
        background-color: rgba(255, 225, 124, 0.15) !important;
        border-left: 4px solid var(--yellow) !important;
        color: var(--charcoal) !important;
    }

    .stInfo {
        background-color: rgba(23, 30, 25, 0.05) !important;
        border-left: 4px solid var(--charcoal) !important;
        color: var(--charcoal) !important;
    }

    .stWarning {
        background-color: rgba(255, 225, 124, 0.1) !important;
        border-left: 4px solid var(--yellow) !important;
        color: var(--charcoal) !important;
    }

    .stError {
        background-color: rgba(255, 107, 107, 0.1) !important;
        border-left: 4px solid #ff6b6b !important;
        color: var(--charcoal) !important;
    }

    [data-testid="stDivider"] {
        background-color: var(--yellow) !important;
        opacity: 0.3;
    }

    .header-section {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 2px solid var(--yellow);
        margin-bottom: 2rem;
    }

    .subtitle {
        font-family: 'Satoshi', sans-serif;
        font-size: 1.125rem;
        color: rgba(23, 30, 25, 0.7);
        font-weight: 400;
        letter-spacing: 0.03em;
    }

    .progress-item {
        padding: 0.875rem 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid var(--yellow);
        background-color: rgba(255, 225, 124, 0.08);
        border-radius: 0.5rem;
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--charcoal);
        font-family: 'Satoshi', sans-serif;
    }

    code {
        background-color: var(--charcoal) !important;
        color: var(--yellow) !important;
        border: 1px solid rgba(255, 225, 124, 0.2) !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 0.25rem !important;
        font-size: 0.85rem !important;
    }

    pre {
        background-color: var(--charcoal) !important;
        border: 1px solid rgba(255, 225, 124, 0.2) !important;
        border-radius: 0.5rem !important;
        padding: 1.5rem !important;
        overflow-x: auto !important;
        color: var(--yellow) !important;
    }

    a {
        color: var(--yellow) !important;
        text-decoration: none;
        font-weight: 500;
    }

    a:hover {
        color: var(--charcoal) !important;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Header Section - Brutalist Style
st.markdown('<div class="header-section">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1>LUMINOUS<span style="color: #ffe17c;">.</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Multi-Agent Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

settings = get_settings()
missing = settings.missing_required()
if missing:
    st.warning(
        "⚠️ Some capabilities are limited: "
        + ", ".join(missing)
        + ". The system uses safe fallbacks."
    )

AGENT_STEPS = [
    ("planner", "[>] Research Planning"),
    ("retriever", "[>] Source Retrieval"),
    ("analyzer", "[>] Critical Analysis"),
    ("insight_generator", "[>] Insight Generation"),
    ("fact_checker", "[>] Fact Checking"),
    ("report_builder", "[>] Report Generation"),
]

# Sidebar Settings
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown('<p class="caption-text">LLM Provider</p>', unsafe_allow_html=True)
    st.caption(f"**{settings.llm_model}**")
    st.markdown('<p class="caption-text">Search Capability</p>', unsafe_allow_html=True)
    st.caption(f"{'✓ Enabled' if settings.has_web_search else '✗ Disabled'}")
    max_iter = st.slider("Max Research Iterations", 0, 3, settings.max_research_iterations)
    st.divider()
    st.markdown('<p class="caption-text">About</p>', unsafe_allow_html=True)
    st.caption("Luminous Dark Slate Editorial Design • Multi-Agent Research Pipeline")

# Research Question Input
st.markdown("### 🔬 Research Question")
question = st.text_area(
    "Enter your research query",
    value="Will generative AI significantly reduce the demand for software developers over the next five years?",
    height=120,
    label_visibility="collapsed",
    placeholder="What would you like to research?",
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start = st.button("🔍 Begin Research", type="primary", use_container_width=True)
with col2:
    st.info(f"Iterations: {max_iter}", icon="⚡")

if start and question.strip():
    st.divider()
    st.markdown("### 🎯 Agent Pipeline")

    progress_area = st.container()
    status_placeholders = {key: st.empty() for key, _ in AGENT_STEPS}

    completed: set[str] = set()

    def mark_done(name: str) -> None:
        for key, label in AGENT_STEPS:
            if key == name:
                completed.add(key)
        for key, label in AGENT_STEPS:
            icon = "[OK]" if key in completed else "[ ]"
            status_placeholders[key].markdown(f'<div class="progress-item">{icon} {label}</div>', unsafe_allow_html=True)

    app = build_graph()
    state = new_research_state(question.strip(), max_iterations=max_iter)

    final_state = None
    try:
        # Stream node-by-node so we can update the progress checklist live.
        for event in app.stream(state, config={"recursion_limit": 50}):
            for node_name, node_output in event.items():
                mark_done(node_name)
                if node_name == "report_builder":
                    final_state = {**state, **node_output}
                # keep merging outputs into a local view for final display
                if node_output:
                    for k, v in node_output.items():
                        if k in ("sources", "retrieved_documents", "traces", "errors") and isinstance(v, list):
                            state[k] = state.get(k, []) + v
                        else:
                            state[k] = v
        final_state = state
    except Exception as exc:  # noqa: BLE001
        st.error(f"🚫 Research failed: {type(exc).__name__}\n\n{str(exc)}", icon="❌")

    if final_state:
        report = final_state.get("final_report")
        sources = final_state.get("sources", [])
        contradictions = final_state.get("contradictions", [])
        insights = final_state.get("insights", [])

        st.divider()

        # Results Summary Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Sources Analyzed", len(sources))
        with metric_col2:
            st.metric("Iterations", final_state.get('research_iteration', 0) + 1)
        with metric_col3:
            evidence_status = final_state.get('evidence_sufficiency', EvidenceSufficiency.SUFFICIENT).value if final_state.get('evidence_sufficiency') else 'n/a'
            st.metric("Evidence Status", evidence_status.upper())

        st.success(f"✅ Research complete — {len(insights)} insights generated")

        tab_report, tab_sources, tab_contra, tab_insights = st.tabs(
            ["📄 Report", f"📚 Sources ({len(sources)})", f"⚔️ Contradictions ({len(contradictions)})", f"💡 Insights ({len(insights)})"]
        )

        with tab_sources:
            if sources:
                rows = [
                    {
                        "Title": s.title[:50] + "..." if len(s.title) > 50 else s.title,
                        "Type": s.source_type.value,
                        "Relevance": round(s.relevance_score, 2),
                        "Credibility": round(s.credibility_score, 2),
                    }
                    for s in sources
                ]
                st.dataframe(rows, use_container_width=True, height=400)
            else:
                st.info("📭 No sources retrieved. Check API configuration.")

        with tab_contra:
            if contradictions:
                for idx, c in enumerate(contradictions, 1):
                    st.markdown(f"### Contradiction {idx}: {c.topic}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Position A**  \n{c.position_a}")
                        st.caption(f"Sources: {', '.join(c.position_a_source_ids) or 'n/a'}")
                    with col_b:
                        st.markdown(f"**Position B**  \n{c.position_b}")
                        st.caption(f"Sources: {', '.join(c.position_b_source_ids) or 'n/a'}")
                    if c.explanation:
                        st.markdown(f"**Analysis:** {c.explanation}")
                    st.divider()
            else:
                st.info("✓ No material contradictions detected.")

        with tab_insights:
            findings = [i for i in insights if i.tier == InsightTier.SUPPORTED]
            trends = [i for i in insights if i.tier == InsightTier.TREND]
            hypotheses = [i for i in insights if i.tier in (InsightTier.HYPOTHESIS, InsightTier.SPECULATION)]

            st.markdown("#### ✓ Findings (Evidence-Backed)")
            if findings:
                for i in findings:
                    st.markdown(f"**{i.statement}**")
                    st.caption(f"Sources: {', '.join(i.supporting_source_ids) or 'n/a'}")
                    st.divider()
            else:
                st.info("None met the evidence threshold.")

            st.markdown("#### 📈 Emerging Trends")
            if trends:
                for i in trends:
                    st.markdown(f"**{i.statement}**")
                    st.caption(f"Sources: {', '.join(i.supporting_source_ids) or 'n/a'}")
                    st.divider()
            else:
                st.info("No cross-source trends identified.")

            st.markdown("#### 🔬 Hypotheses & Speculation")
            if hypotheses:
                for i in hypotheses:
                    st.markdown(f"**[{i.tier.value.upper()}]** {i.statement}")
                    st.caption(f"Sources: {', '.join(i.supporting_source_ids) or 'n/a'}")
                    st.divider()
            else:
                st.info("No hypotheses generated.")

        with tab_report:
            if report and report.markdown:
                st.markdown(report.markdown)
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "⬇️ Download as Markdown",
                        data=report.markdown,
                        file_name="research_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col2:
                    st.info("Share this report with your team", icon="📤")
            else:
                st.warning("📄 No report was generated.", icon="⚠️")

elif start:
    st.error("🚫 Please enter a research question to begin.", icon="❌")
