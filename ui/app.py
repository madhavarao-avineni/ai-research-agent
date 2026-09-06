"""
ui/app.py
=========
Streamlit UI for the AI Research Assistant — Softly Theme (Warm Pastels, Grain Texture).

Run:  streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys

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

st.set_page_config(page_title="Luminous Research", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# Softly Theme - Warm Pastels with Grain Texture
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Reenie+Beanie&display=swap');

    :root {
        --bg-primary: #FDFCF8;
        --bg-sage: #E8EFE8;
        --bg-lavender: #EFEDF4;
        --accent-coral: #FFB7B2;
        --text-dark: #292524;
        --text-muted: #78716C;
        --white: #FFFFFF;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body {
        background-color: var(--bg-primary);
        color: var(--text-dark);
        font-family: 'Outfit', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
        color: var(--text-dark);
    }

    [data-testid="stMainBlockContainer"] {
        padding: 2rem 1rem;
        max-width: 100%;
    }

    .block-container {
        max-width: 1000px;
        margin: 0 auto;
    }

    /* Grain overlay */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><filter id="noise"><feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="4" /></filter><rect width="100" height="100" fill="%23000" filter="url(%23noise)"/></svg>');
        background-size: 100px 100px;
        opacity: 0.35;
        pointer-events: none;
        z-index: 1;
        mix-blend-mode: overlay;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: var(--text-dark);
        font-weight: 600;
        line-height: 1.2;
        margin-top: 0;
    }

    h1 {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        text-transform: none;
    }

    h2 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 1.5rem;
    }

    h3 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    p, span, div, label, li {
        color: var(--text-dark) !important;
        font-family: 'Outfit', sans-serif;
    }

    .accent-text {
        font-family: 'Reenie Beanie', cursive;
        color: var(--accent-coral);
        font-size: 1.1em;
        font-weight: 400;
    }

    /* Containers & Cards */
    .card {
        background-color: var(--white);
        border-radius: 2rem;
        padding: 2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: none;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: #F5F3F0 !important;
        border: 1px solid rgba(41, 37, 36, 0.1) !important;
        color: var(--text-dark) !important;
        border-radius: 1rem !important;
        font-size: 1rem !important;
        font-family: 'Outfit', sans-serif !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: var(--text-muted) !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-coral) !important;
        background-color: var(--white) !important;
        box-shadow: 0 0 0 3px rgba(255, 183, 178, 0.15) !important;
    }

    /* Buttons */
    button[data-testid="stBaseButton-primary"] {
        background-color: var(--accent-coral) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 1rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.875rem 1.5rem !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(255, 183, 178, 0.2) !important;
    }

    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #FF9D98 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 183, 178, 0.3) !important;
    }

    button[data-testid="stBaseButton-secondary"] {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border: 1px solid rgba(41, 37, 36, 0.15) !important;
        border-radius: 1rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: var(--bg-sage) !important;
        border-color: var(--accent-coral) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        color: var(--accent-coral) !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-muted) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em;
    }

    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 2px solid rgba(41, 37, 36, 0.1) !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--accent-coral) !important;
        border-bottom: 2px solid var(--accent-coral) !important;
    }

    /* Data Frames */
    [data-testid="stDataFrame"] {
        background-color: var(--white) !important;
        border: 1px solid rgba(41, 37, 36, 0.1) !important;
        border-radius: 1rem !important;
    }

    [data-testid="stDataFrame"] th {
        background-color: rgba(232, 239, 232, 0.5) !important;
        color: var(--text-dark) !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    [data-testid="stDataFrame"] td {
        color: var(--text-dark) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Messages */
    .stSuccess {
        background-color: rgba(232, 239, 232, 0.3) !important;
        border-left: 4px solid #86B383 !important;
        color: var(--text-dark) !important;
        border-radius: 1rem !important;
    }

    .stInfo {
        background-color: rgba(239, 237, 244, 0.3) !important;
        border-left: 4px solid #B8A3D9 !important;
        color: var(--text-dark) !important;
        border-radius: 1rem !important;
    }

    .stWarning {
        background-color: rgba(255, 183, 178, 0.2) !important;
        border-left: 4px solid var(--accent-coral) !important;
        color: var(--text-dark) !important;
        border-radius: 1rem !important;
    }

    .stError {
        background-color: rgba(255, 107, 107, 0.15) !important;
        border-left: 4px solid #ff6b6b !important;
        color: var(--text-dark) !important;
        border-radius: 1rem !important;
    }

    /* Divider */
    [data-testid="stDivider"] {
        background-color: rgba(41, 37, 36, 0.1) !important;
        opacity: 1 !important;
    }

    /* Code */
    code {
        background-color: #F5F3F0 !important;
        color: var(--accent-coral) !important;
        border: 1px solid rgba(255, 183, 178, 0.2) !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 0.5rem !important;
        font-size: 0.85rem !important;
        font-family: 'Courier New', monospace !important;
    }

    pre {
        background-color: #F5F3F0 !important;
        border: 1px solid rgba(41, 37, 36, 0.1) !important;
        border-radius: 1rem !important;
        padding: 1.5rem !important;
        overflow-x: auto !important;
        color: var(--text-dark) !important;
    }

    pre code {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        color: var(--text-dark) !important;
    }

    /* Links */
    a {
        color: var(--accent-coral) !important;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease !important;
    }

    a:hover {
        color: #FF9D98 !important;
        text-decoration: underline;
    }

    /* Progress Items */
    .progress-item {
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid var(--accent-coral);
        background-color: rgba(255, 183, 178, 0.08);
        border-radius: 1rem;
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--text-dark);
        font-family: 'Outfit', sans-serif;
        animation: slideIn 0.3s ease forwards;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Header */
    .header-section {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
    }

    .header-section h1 {
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 1.125rem;
        color: var(--text-muted);
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Slider */
    [data-testid="stSlider"] {
        padding: 1.5rem 0;
    }

    [data-testid="stSlider"] [role="slider"] {
        accent-color: var(--accent-coral) !important;
    }

    /* Responsive */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem;
        }

        h2 {
            font-size: 1.5rem;
        }

        .card {
            padding: 1.5rem;
            border-radius: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-section">
    <h1>Luminous <span class="accent-text">Research</span></h1>
    <p class="subtitle">AI-powered multi-agent intelligence platform</p>
</div>
""", unsafe_allow_html=True)

settings = get_settings()
missing = settings.missing_required()
if missing:
    st.warning(
        "⚠️ Some capabilities are limited: "
        + ", ".join(missing)
        + ". The system uses safe fallbacks."
    )

AGENT_STEPS = [
    ("planner", "🎯 Research Planning"),
    ("retriever", "🔍 Source Retrieval"),
    ("analyzer", "🔬 Critical Analysis"),
    ("insight_generator", "💡 Insight Generation"),
    ("fact_checker", "✓ Fact Checking"),
    ("report_builder", "📄 Report Generation"),
]

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**LLM:** {settings.llm_model}")
    st.markdown(f"**Search:** {'✓ Enabled' if settings.has_web_search else '✗ Disabled'}")
    max_iter = st.slider("Max iterations", 0, 3, settings.max_research_iterations)

# Research Input
st.markdown("### 🔬 What would you like to research?")
question = st.text_area(
    "Research question",
    value="Will generative AI significantly reduce the demand for software developers over the next five years?",
    height=120,
    label_visibility="collapsed",
    placeholder="Ask anything...",
)

col1, col2, _ = st.columns([1, 1, 2])
with col1:
    start = st.button("Begin research", type="primary", use_container_width=True)
with col2:
    st.info(f"{max_iter} iterations", icon="⚡")

if start and question.strip():
    st.divider()
    st.markdown("### Research Pipeline")

    status_placeholders = {key: st.empty() for key, _ in AGENT_STEPS}
    completed: set[str] = set()

    def mark_done(name: str) -> None:
        for key, _ in AGENT_STEPS:
            if key == name:
                completed.add(key)
        for key, label in AGENT_STEPS:
            icon = "✓" if key in completed else "○"
            status_placeholders[key].markdown(f'<div class="progress-item">{icon} {label}</div>', unsafe_allow_html=True)

    app = build_graph()
    state = new_research_state(question.strip(), max_iterations=max_iter)

    final_state = None
    try:
        for event in app.stream(state, config={"recursion_limit": 50}):
            for node_name, node_output in event.items():
                mark_done(node_name)
                if node_name == "report_builder":
                    final_state = {**state, **node_output}
                if node_output:
                    for k, v in node_output.items():
                        if k in ("sources", "retrieved_documents", "traces", "errors") and isinstance(v, list):
                            state[k] = state.get(k, []) + v
                        else:
                            state[k] = v
        final_state = state
    except Exception as exc:
        st.error(f"Research failed: {type(exc).__name__}\n\n{str(exc)}", icon="❌")

    if final_state:
        report = final_state.get("final_report")
        sources = final_state.get("sources", [])
        contradictions = final_state.get("contradictions", [])
        insights = final_state.get("insights", [])

        st.divider()

        # Summary Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        evidence_status = final_state.get('evidence_sufficiency', EvidenceSufficiency.SUFFICIENT).value if final_state.get('evidence_sufficiency') else 'pending'

        with metric_col1:
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 2.5rem; color: #FFB7B2; font-weight: 700;'>{len(sources)}</div><div style='font-size: 0.875rem; color: #78716C;'>Sources analyzed</div></div>", unsafe_allow_html=True)
        with metric_col2:
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 2.5rem; color: #FFB7B2; font-weight: 700;'>{final_state.get('research_iteration', 0) + 1}</div><div style='font-size: 0.875rem; color: #78716C;'>Iterations</div></div>", unsafe_allow_html=True)
        with metric_col3:
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 2.5rem; color: #FFB7B2; font-weight: 700;'>{evidence_status.upper()}</div><div style='font-size: 0.875rem; color: #78716C;'>Evidence status</div></div>", unsafe_allow_html=True)

        st.success(f"✅ Research complete — {len(insights)} insights generated")

        # Tabs
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
                st.info("No sources retrieved. Check API configuration.")

        with tab_contra:
            if contradictions:
                for idx, c in enumerate(contradictions, 1):
                    st.markdown(f"### Contradiction {idx}: {c.topic}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Position A**\n{c.position_a}")
                        st.caption(f"Sources: {', '.join(c.position_a_source_ids) or 'n/a'}")
                    with col_b:
                        st.markdown(f"**Position B**\n{c.position_b}")
                        st.caption(f"Sources: {', '.join(c.position_b_source_ids) or 'n/a'}")
                    if c.explanation:
                        st.markdown(f"**Analysis:** {c.explanation}")
                    st.divider()
            else:
                st.info("No material contradictions detected.")

        with tab_insights:
            findings = [i for i in insights if i.tier == InsightTier.SUPPORTED]
            trends = [i for i in insights if i.tier == InsightTier.TREND]
            hypotheses = [i for i in insights if i.tier in (InsightTier.HYPOTHESIS, InsightTier.SPECULATION)]

            st.markdown("#### ✓ Findings (Evidence-backed)")
            if findings:
                for i in findings:
                    st.markdown(f"**{i.statement}**")
                    st.caption(f"Sources: {', '.join(i.supporting_source_ids) or 'n/a'}")
                    st.divider()
            else:
                st.info("None met the evidence threshold.")

            st.markdown("#### 📈 Emerging trends")
            if trends:
                for i in trends:
                    st.markdown(f"**{i.statement}**")
                    st.caption(f"Sources: {', '.join(i.supporting_source_ids) or 'n/a'}")
                    st.divider()
            else:
                st.info("No cross-source trends identified.")

            st.markdown("#### 🔬 Hypotheses & speculation")
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
                        file_name=f"research_{len(sources)}_sources.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
            else:
                st.info("Report unavailable.")
