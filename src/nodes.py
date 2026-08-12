"""
src/nodes.py
Defines all node functions for the LangGraph agent execution graph.
Handles router dispatching, individual tool execution, dynamic evaluation, and answer synthesis.
"""

import re
from typing import Optional, Literal
from pydantic import BaseModel, Field

from src.state import AgentState
from src.llm import get_llm, invoke_with_retry
from src.tools.retrieval_tool import retrieval_tool
from src.tools.web_search_tool import web_search_tool
from src.tools.calculator_tool import calculator_tool

MAX_EVIDENCE_STEPS = 4  # hard cap on total tool calls per query


COMPUTATION_SIGNAL = re.compile(
    r"\b(compute|calculate|how many times|how much faster|how many gb|"
    r"times (bigger|larger|smaller|faster)|as a (percentage|fraction|ratio|multiple))\b",
    re.IGNORECASE,
)


# ── TOOL NODES ────────────────────────────────────────────────────────────────

def retrieval_node(state: AgentState) -> dict:
    """Executes hybrid vector/BM25 retrieval over local ArXiv ML paper corpus."""
    query = state["query"]
    output = retrieval_tool(query) # string since, retrieval_tool returns a string of concatenated evidence
    record = {"tool": "retrieval", "output": output}
    return {"tool_outputs": [record]}


def web_search_node(state: AgentState) -> dict:
    """Executes live web search via Tavily API."""
    query = state["query"]
    output = web_search_tool(query)
    record = {"tool": "web_search", "output": output}
    return {"tool_outputs": [record]}


def calculator_node(state: AgentState) -> dict:
    """Extracts numbers from query + gathered evidence and evaluates math via numexpr."""
    query = state["query"]
    evidence = "\n".join(r["output"] for r in state["tool_outputs"])
    full_context = f"Query: {query}\nEvidence:\n{evidence}"
    output = calculator_tool(full_context)
    record = {"tool": "calculator", "output": output}
    return {"tool_outputs": [record]}


def direct_answer_node(state: AgentState) -> dict:
    """Uses general LLM parametric knowledge directly without tools."""
    llm = get_llm(temperature=0.0)
    messages = [
        {"role": "system", "content": "Answer the following question clearly using your pre-trained knowledge."},
        {"role": "user", "content": state["query"]}
    ]
    response = invoke_with_retry(llm, messages)
    record = {"tool": "direct_answer", "output": response.content}
    return {"tool_outputs": [record]}


# ── EVALUATOR NODE & HELPERS ──────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    sufficient: bool = Field(
        description="True ONLY if the evidence fully answers every part of "
        "the question, with nothing left to look up."
    )
    
    missing: str = Field(default="", description="If insufficient, ONE short sentence naming what's missing.")

    next_tool: Optional[Literal["retrieval", "web_search", "calculator"]] = Field(
        default=None, description="If insufficient, which tool addresses what's missing."
    )


EVALUATOR_PROMPT = """You are evaluating whether gathered evidence is sufficient to answer a question.

Key Rules that you have to follow:
- Mark sufficient if the evidence already contains a direct, complete answer to exactly what was asked.

- For multi-part questions, ALL parts must be covered to mark sufficient.

- Mark insufficient ONLY if a specific fact the question needs is genuinely absent from the
evidence — not because more context could theoretically be added.

- Also mark insufficient if the evidence is a direct_answer that explicitly admits it doesn't
know, is speculating, or is limited by an outdated knowledge cutoff — treat that the same as a
missing fact, and route to web_search for current information.

- If insufficient, name the ONE tool that addresses what's missing. Do NOT pick a tool that has
ALREADY been tried and did not surface this specific missing piece.

Examples:

Question: "What activation function does the paper use in its hidden layers?"
Evidence: [retrieval] "...the paper uses ReLU activation in all hidden layers, chosen over
sigmoid for faster convergence..."
-> sufficient: true. The evidence directly names the activation function asked for. Nothing
further is needed even though more detail COULD be added.

Question: "If Model A scores 82.1 and Model B scores 88.4 on the same benchmark, express Model
B's result relative to Model A as one combined figure."
Evidence: [retrieval] "Model A: 82.1. Model B: 88.4."
-> sufficient: false, next_tool: "calculator". Both raw numbers are present, but the question
asks for a single derived figure combining them, which has not been produced yet — the evidence
has the inputs, not the answer.

Question: "What dataset size did the paper use for pretraining, and has a larger-scale version
been released since 2025?"
Evidence: [retrieval] "...the model was pretrained on a 40GB text corpus..."
-> sufficient: false, next_tool: "web_search". Only the first half of this two-part question is
covered. Nothing in the evidence addresses newer releases, and local paper retrieval can't
answer that — this needs a web search.
"""


def _has_retrieved_numbers(tool_outputs: list) -> bool:
    """Returns True if retrieval or web_search has already run and produced output."""
    return any(r["tool"] in ("retrieval", "web_search") for r in tool_outputs)


def _looks_sufficient_without_llm(state: AgentState) -> bool:
    """Deterministic fast-path for single-part retrieval queries."""
    tried_tools = [r["tool"] for r in state["tool_outputs"]]
    
    if tried_tools != ["retrieval"]:
        return False
    
    retrieval_output = state["tool_outputs"][0]["output"]
    if len(retrieval_output) < 500:
        return False
    
    query = state["query"].lower()

    explicit_tool_signals = [
        "use web search", "search the web", "look it up online",
        "find online", "search online", "check online"
    ]
    if any(signal in query for signal in explicit_tool_signals):
        return False
    
    multi_part_signals = [
        " and ", " also ", "as well as", "additionally",
        "compare", "versus", "vs", "both", "difference between",
        "how many", "how much", "compute", "calculate",
        "recent", "latest", "current", "2024", "2025", "2026",
    ]
    if any(signal in query for signal in multi_part_signals):
        return False
    
    single_part_signals = [
        "summarize", "summary", "explain", "what is", "what are",
        "describe", "overview", "tell me about", "how does",
        "what does", "define", "definition",
    ]
    if any(signal in query for signal in single_part_signals):
        return True
    
    return False


# --- [PRODUCTION FIX: DYNAMIC CORPUS RECOGNITION] ---
def _is_query_about_local_corpus(query: str) -> bool:
    """
    Dynamically checks if a query mentions any ingested paper or general paper research terms.
    Avoids hardcoded static keyword lists.
    """
    query_lower = query.lower()

    # 1. Generic research document terms
    generic_paper_signals = [
        "paper", "author", "arxiv", "section", "table", "figure", "equation", "dataset", "benchmark"
    ]
    if any(sig in query_lower for sig in generic_paper_signals):
        return True

    # 2. Dynamic check against actual ingested paper titles in chunks.json
    try:
        import json
        import os
        chunks_path = "data/chunks.json"
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            papers = set(c["paper"].lower() for c in chunks if c.get("paper"))
            for p in papers:
                if p in query_lower:
                    return True
    except Exception:
        pass

    return False
# ----------------------------------------------------


def evaluate_node(state: AgentState) -> dict:
    tried_tools = [r["tool"] for r in state["tool_outputs"]]

    if len(tried_tools) >= MAX_EVIDENCE_STEPS:  # Guardrail
        return {
            "next_tool": None,
            "missing_info": "Reached max evidence-gathering steps.",
            "_grade_sufficient": False,
        }

    if _looks_sufficient_without_llm(state):
        return {
            "next_tool": None,
            "missing_info": "",
            "_grade_sufficient": True,
        }

    if (
        COMPUTATION_SIGNAL.search(state["query"])
        and "calculator" not in tried_tools
        and _has_retrieved_numbers(state["tool_outputs"])
    ):
        return {
            "next_tool": "calculator",
            "missing_info": "Question requires a computed value; calculator not yet run.",
            "_grade_sufficient": False,
        }

    evidence = "\n\n".join(
        f"[{r['tool']}]\n{r['output']}" for r in state["tool_outputs"]
    ) or "(no evidence gathered yet)"

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(EvaluationResult)

    messages = [
        {"role": "system", "content": EVALUATOR_PROMPT},
        {"role": "user", "content": (
            f"Question: {state['query']}\n"
            f"Tools already tried: {tried_tools}\n\n"
            f"Evidence gathered so far:\n{evidence}"
        )},
    ]

    try:
        result: EvaluationResult = invoke_with_retry(structured_llm, messages)
        next_tool = result.next_tool

        # --- [PRODUCTION FIX: WEB SEARCH FALLBACK PROTECTION] ---
        # If web_search ran and fetched content, prevent falling back to local paper retrieval
        # unless the query explicitly asks about a specific paper or paper author.
        if "web_search" in tried_tools and next_tool == "retrieval":
            if not _is_query_about_local_corpus(state["query"]):
                next_tool = None  # Prevent falling back to local ML paper search for non-paper web queries
        # ---------------------------------------------------------

        if not result.sufficient and (next_tool is None or next_tool in tried_tools):
            # --- [PRODUCTION FIX: TOOL-AWARE SAFETY NET] ---
            # If web_search was already tried and query isn't about local corpus, don't fall back to retrieval
            if "web_search" in tried_tools and not _is_query_about_local_corpus(state["query"]):
                next_tool = None  # Route to synthesize best effort
            else:
                priority = ["web_search", "retrieval"]
                remaining = [t for t in priority if t not in tried_tools]
                next_tool = remaining[0] if remaining else None
            # -------------------------------------------------

        return {
            "next_tool": next_tool,
            "missing_info": result.missing,
            "_grade_sufficient": result.sufficient,
        }
    except Exception as e:
        return {
            "next_tool": None,
            "missing_info": "",
            "_grade_sufficient": True,
        }


def evaluate_route_decision(state: AgentState) -> str:
    return state["next_tool"] if state["next_tool"] is not None else "synthesize"


# ── SYNTHESIZE NODE ───────────────────────────────────────────────────────────

SYNTHESIZE_PROMPT = """You are a precise, objective AI research assistant.
Synthesize a clear, direct answer to the user's question using ONLY the provided evidence.

Strict Rules:
1. Do NOT hallucinate facts outside the provided evidence.
2. If evidence is partial or insufficient, answer what is supported by evidence and explicitly note what missing info could not be found.
3. Keep the tone academic, grounded, and concise.
"""


def synthesize_node(state: AgentState) -> dict:
    """Synthesizes final answer from gathered tool evidence with strict grounding guardrails."""
    evidence = "\n\n".join(
        f"[{r['tool']}]\n{r['output']}" for r in state["tool_outputs"]
    ) or "(No evidence gathered)"

    llm = get_llm(temperature=0.0)
    messages = [
        {"role": "system", "content": SYNTHESIZE_PROMPT},
        {"role": "user", "content": (
            f"Question: {state['query']}\n\n"
            f"Evidence Gathered:\n{evidence}\n\n"
            f"Missing Info Warning (if any): {state.get('missing_info', '')}"
        )}
    ]

    response = invoke_with_retry(llm, messages)
    return {"final_answer": response.content}