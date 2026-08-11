"""
Streamlit UI for the Agentic Research Assistant.
Shows: query input, routing decision, step-by-step tool trace, final answer.
Streaming: uses app.stream() so each tool step appears as it fires,
not all at once after the full run completes.
"""

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from src.graph import build_graph
from src.state import AgentState

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 780px; }

.page-header { text-align: center; margin-bottom: 2.5rem; }
.page-header h1 {
    font-size: 1.6rem; font-weight: 600;
    letter-spacing: -0.02em; color: #E6EDF3; margin-bottom: 0.3rem;
}
.page-header p {
    font-size: 0.85rem; color: #7D8590;
    font-family: 'JetBrains Mono', monospace;
}

.stTextArea textarea {
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    caret-color: #2DD4BF;
}
.stTextArea textarea:focus {
    border-color: #2DD4BF !important;
    box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.1) !important;
}

.stButton > button {
    background: #2DD4BF !important; color: #0D1117 !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.15s ease !important; width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:disabled { opacity: 0.4 !important; }

.routing-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #161B22; border: 1px solid #30363D;
    border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 1.2rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
}
.routing-badge .label { color: #7D8590; }
.routing-badge .tool  { color: #2DD4BF; font-weight: 500; }
.routing-badge .reason { color: #8B949E; margin-left: 0.3rem; }

.trace-container { border-left: 2px solid #21262D; padding-left: 1.2rem; margin-bottom: 1.5rem; }

.trace-step { margin-bottom: 1rem; }
.step-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem; }
.step-num { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #30363D; min-width: 1.5rem; }
.tool-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 500; padding: 0.15rem 0.5rem; border-radius: 4px; }
.tool-retrieval    { background: #1C2D3A; color: #58A6FF; border: 1px solid #1F4060; }
.tool-web_search   { background: #1E2D22; color: #3FB950; border: 1px solid #244D29; }
.tool-calculator   { background: #2D1F3A; color: #D2A8FF; border: 1px solid #4D2D6B; }
.tool-direct_answer{ background: #2D2208; color: #E3B341; border: 1px solid #5C3D0A; }
.step-desc { font-size: 0.82rem; color: #8B949E; padding-left: 2rem; font-family: 'JetBrains Mono', monospace; }

.evaluate-note {
    font-size: 0.78rem; color: #7D8590;
    font-family: 'JetBrains Mono', monospace;
    padding: 0.4rem 0.8rem;
    border-left: 2px solid #2DD4BF22;
    margin: 0.3rem 0 0.8rem 1.8rem;
    background: #0D1117;
}

.section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7D8590; margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
}

.answer-box {
    background: #161B22; border: 1px solid #2DD4BF33;
    border-radius: 8px; padding: 1.2rem 1.4rem;
    font-size: 0.95rem; line-height: 1.7;
    color: #E6EDF3; white-space: pre-wrap;
}
.answer-box.insufficient { border-color: #E3B34133; color: #C9A227; }

.trace-divider { border: none; border-top: 1px solid #21262D; margin: 1.5rem 0; }

.corpus-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.corpus-pill {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: #7D8590; background: #161B22;
    border: 1px solid #21262D; border-radius: 4px; padding: 0.15rem 0.5rem;
}

.error-box {
    background: #2D1515; border: 1px solid #8B1A1A;
    border-radius: 8px; padding: 1rem 1.2rem;
    font-size: 0.85rem; color: #FF7B72;
    font-family: 'JetBrains Mono', monospace;
}

/* Pulsing dot for active tool */
.thinking-dot {
    display: inline-block; width: 6px; height: 6px;
    background: #2DD4BF; border-radius: 50%;
    margin-left: 0.4rem;
    animation: pulse 1s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TOOL_META = {
    "retrieval":    {"icon": "📄", "label": "retrieval",    "desc": "searching ArXiv corpus (BM25 + FAISS + rerank thresholding)", "css": "tool-retrieval"},
    "web_search":   {"icon": "🌐", "label": "web_search",   "desc": "searching the live web via Tavily API",                      "css": "tool-web_search"},
    "calculator":   {"icon": "🧮", "label": "calculator",   "desc": "extracting and evaluating math expression via numexpr",       "css": "tool-calculator"},
    "direct_answer":{"icon": "💬", "label": "direct_answer","desc": "answering from general LLM knowledge — no tool needed",      "css": "tool-direct_answer"},
}

CORPUS_PAPERS = ["Attention Is All You Need", "ResNet", "BERT", "DDPM", "GPT-3", "+ 40 Ingested ArXiv Papers (~5,500 Chunks)"]

# ── Graph — built once, reused across all queries ─────────────────────────────
@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


# ── HTML helpers ──────────────────────────────────────────────────────────────
def routing_badge_html(tool: str, reason: str, pending: bool = False) -> str:
    dot = '<span class="thinking-dot"></span>' if pending else ""
    tool_display = tool if not pending else "routing..."
    return (
        f'<div class="routing-badge">'
        f'<span class="label">router →</span>'
        f'<span class="tool">{tool_display}{dot}</span>'
        f'{"<span class=\"reason\">// " + reason + "</span>" if reason else ""}'
        f'</div>'
    )


def trace_html(steps: list[dict], active: bool = False) -> str:
    """
    Builds the full trace HTML from accumulated steps.
    steps: list of {"tool": str} dicts — tool name only, not full ToolCallRecord.
    active: if True, adds a pulsing dot to the last step (currently running).
    """
    html = '<div class="trace-container">'
    for i, step in enumerate(steps):
        tool = step["tool"]
        meta = TOOL_META.get(tool, {"icon": "🔧", "label": tool, "desc": f"ran {tool}", "css": "tool-direct_answer"})
        is_last = i == len(steps) - 1
        dot = '<span class="thinking-dot"></span>' if (active and is_last) else ""
        num = f"0{i+1}" if i < 9 else str(i+1)

        html += f"""
        <div class="trace-step">
            <div class="step-header">
                <span class="step-num">{num}</span>
                <span class="tool-tag {meta['css']}">{meta['icon']} {meta['label']}{dot}</span>
            </div>
            <div class="step-desc">{meta['desc']}</div>
        </div>"""

        if not is_last:
            html += '<div class="evaluate-note">↳ evaluate — evidence insufficient, fetching more...</div>'

    html += '</div>'
    return html


def answer_html(text: str, sufficient: bool) -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    cls = "answer-box" if sufficient else "answer-box insufficient"
    return f'<div class="{cls}">{safe}</div>'


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🔬 Agentic Research Assistant</h1>
    <p>LangGraph · Groq · BM25 + FAISS + Cross-Encoder · Tavily · numexpr</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Local corpus</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="corpus-pills">' +
    "".join(f'<span class="corpus-pill">{p}</span>' for p in CORPUS_PAPERS) +
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

query = st.text_area(
    label="query",
    label_visibility="collapsed",
    placeholder="Ask anything — about the papers, current ML research, or a calculation...",
    height=100,
    key="query_input",
)

run_clicked = st.button("Run", disabled=not bool(query and query.strip()))

# ── Streaming run ─────────────────────────────────────────────────────────────
if run_clicked and query.strip():
    st.markdown('<hr class="trace-divider">', unsafe_allow_html=True)

    app = get_graph()
    initial_state: AgentState = {
        "query": query.strip(),
        "next_tool": None,
        "routing_reason": None,
        "tool_outputs": [],
        "final_answer": None,
        "missing_info": "",
        "_grade_sufficient": True,
    }

    # Placeholders — each gets updated in place as stream events arrive
    routing_ph  = st.empty()
    trace_label_ph = st.empty()
    trace_ph    = st.empty()
    divider_ph  = st.empty()
    answer_label_ph = st.empty()
    answer_ph   = st.empty()
    caveat_ph   = st.empty()

    # Show pending routing badge immediately
    routing_ph.markdown(routing_badge_html("", "", pending=True), unsafe_allow_html=True)

    # State accumulated across stream events
    routing_tool   = ""
    routing_reason = ""
    steps          = []          # list of {"tool": str}
    final_answer   = None
    grade_sufficient = True
    missing_info   = ""

    try:
        # app.stream() yields one dict per node completion: {node_name: node_output}
        for event in app.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():

                # ── Router fired ──────────────────────────────────────────
                if node_name == "router":
                    routing_tool   = node_output.get("next_tool", "unknown")
                    routing_reason = node_output.get("routing_reason", "")
                    routing_ph.markdown(
                        routing_badge_html(routing_tool, routing_reason, pending=False),
                        unsafe_allow_html=True,
                    )

                # ── Tool node fired ───────────────────────────────────────
                elif node_name in TOOL_META:
                    new_records = node_output.get("tool_outputs", [])
                    for rec in new_records:
                        steps.append({"tool": rec["tool"]})

                    trace_label_ph.markdown(
                        '<div class="section-label">Reasoning trace</div>',
                        unsafe_allow_html=True,
                    )
                    trace_ph.markdown(
                        trace_html(steps, active=True),
                        unsafe_allow_html=True,
                    )

                # ── Evaluate fired ────────────────────────────────────────
                elif node_name == "evaluate":
                    grade_sufficient = node_output.get("_grade_sufficient", True)
                    missing_info     = node_output.get("missing_info", "")
                    trace_ph.markdown(
                        trace_html(steps, active=False),
                        unsafe_allow_html=True,
                    )

                # ── Synthesize fired ──────────────────────────────────────
                elif node_name == "synthesize":
                    final_answer = node_output.get("final_answer", "")
                    trace_ph.markdown(
                        trace_html(steps, active=False),
                        unsafe_allow_html=True,
                    )
                    divider_ph.markdown('<hr class="trace-divider">', unsafe_allow_html=True)
                    answer_label_ph.markdown(
                        '<div class="section-label">Answer</div>',
                        unsafe_allow_html=True,
                    )
                    answer_ph.markdown(
                        answer_html(final_answer, grade_sufficient),
                        unsafe_allow_html=True,
                    )
                    if not grade_sufficient and missing_info:
                        caveat_ph.markdown(
                            f'<div class="evaluate-note" style="margin-top:0.8rem;">'
                            f'⚠ could not find: {missing_info}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    except Exception as e:
        st.markdown(
            f'<div class="error-box">Agent error: {e}</div>',
            unsafe_allow_html=True,
        )