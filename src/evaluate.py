"""
Single evaluator node. Decides whether gathered evidence is sufficient to
answer a question, and if not, which tool to call next.

A regex pre-check handles ONE specific decision deterministically (does the
question require a calculated value AND are numbers already in evidence)
because that judgment proved unreliable via prompting alone across two
different model sizes. Everything else stays an LLM judgment call.
"""

import re
from typing import Optional, Literal
from pydantic import BaseModel, Field
from src.state import AgentState
from src.llm import get_llm, invoke_with_retry

MAX_EVIDENCE_STEPS = 4  # hard cap on total tool calls per query

COMPUTATION_SIGNAL = re.compile(
    r"\b(compute|calculate|how many times|how much faster|how many gb|"
    r"times (bigger|larger|smaller|faster)|as a (percentage|fraction|ratio|multiple))\b",
    re.IGNORECASE,
) # this creates a compiled regex object that matches phrases indicating a computation is required in the question. Pattern -> Compile Once -> Reuse Many times
# r stands for raw string, which treats backslashes as literal characters.
# \b is a word boundary, so it matches whole words only. The pattern looks for phrases like "compute", "calculate", "how many times", 
# "how much faster", "how many gb", "times bigger/larger/smaller/faster", and "as a percentage/fraction/ratio/multiple". 
# The re.IGNORECASE flag makes the matching case-insensitive.


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
- Mark sufficient if the evidence already contains a direct, complete answer to exactly what was
asked.

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
    """Returns True if retrieval or web_search has already run and produced
    output — meaning the calculator has actual numbers to work with.
    Guards against forcing calculator before any evidence exists."""
    return any(r["tool"] in ("retrieval", "web_search") for r in tool_outputs)
# tool_outputs is a list of dictionaries, where each dictionary represents the output of a tool that has been run. 
# Each dictionary contains at least a "tool" key indicating which tool produced that output. 
# The function checks if any of the outputs were produced by either the "retrieval" or "web_search" tools. 
# If at least one such output exists, it returns True, indicating that there are numbers available for the calculator to use. 
# Otherwise, it returns False.

def _looks_sufficient_without_llm(state: AgentState) -> bool:
    """
    Deterministic fast-path: if retrieval ran and returned substantial content
    AND the query is a single-part question (no 'and', no multi-part signals),
    mark sufficient immediately without an LLM call.
    This prevents the evaluator from over-escalating simple retrieval questions.
    """
    tried_tools = [r["tool"] for r in state["tool_outputs"]]
    
    # Only apply to single-tool retrieval so far
    if tried_tools != ["retrieval"]:
        return False
    
    # Check retrieval actually returned content
    retrieval_output = state["tool_outputs"][0]["output"]
    if len(retrieval_output) < 500:  # thin result — let LLM evaluate
        return False
    
    query = state["query"].lower()

    # Never override explicit user instructions to use a specific tool
    explicit_tool_signals = [
    "use web search", "search the web", "look it up online",
    "find online", "search online", "check online"
    ]
    if any(signal in query for signal in explicit_tool_signals):
        return False
    
    # Multi-part signals — let LLM handle these
    multi_part_signals = [
        " and ", " also ", "as well as", "additionally",
        "compare", "versus", "vs", "both", "difference between",
        "how many", "how much", "compute", "calculate",
        "recent", "latest", "current", "2024", "2025", "2026",
    ]
    if any(signal in query for signal in multi_part_signals):
        return False
    
    # Single-part summarization/explanation queries — retrieval is enough
    single_part_signals = [
        "summarize", "summary", "explain", "what is", "what are",
        "describe", "overview", "tell me about", "how does",
        "what does", "define", "definition",
    ]
    if any(signal in query for signal in single_part_signals):
        return True
    
    return False


def evaluate_node(state: AgentState) -> dict:
    tried_tools = [r["tool"] for r in state["tool_outputs"]]

    if len(tried_tools) >= MAX_EVIDENCE_STEPS: # Guardrail
        return {
            "next_tool": None,
            "missing_info": "Reached max evidence-gathering steps.",
            "_grade_sufficient": False,
        }

    
    # Deterministic fast-path for simple single-part retrieval questions.
    # Avoids LLM over-escalation on summarize/explain queries.
    if _looks_sufficient_without_llm(state):
        return {
            "next_tool": None,
            "missing_info": "",
            "_grade_sufficient": True,
        }

    # Deterministic check: force calculator only when BOTH conditions hold:
    # (1) query has a computation signal, AND (2) retrieval/web_search has
    # already run so the calculator has actual numbers to work with.
    # Without condition (2), calculator would receive raw query words with
    # no numbers to extract — producing garbage or hallucinated expressions.
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

        if not result.sufficient and (next_tool is None or next_tool in tried_tools):
            # Safety net: LLM failed to suggest a valid next tool.
            # Pick the first untried info-gathering tool from priority list.
            # calculator excluded — it is not an information source and should
            # never be called blindly without numbers already in evidence.
            priority = ["web_search", "retrieval"]
            remaining = [t for t in priority if t not in tried_tools]
            next_tool = remaining[0] if remaining else None

        return {
            "next_tool": next_tool,
            "missing_info": result.missing,
            "_grade_sufficient": result.sufficient,
        }
    except Exception as e:
        # Fail open: route to synthesize with whatever evidence exists.
        # missing_info left empty — there is no genuine missing piece,
        # just an evaluator failure. Avoids injecting a confusing caveat.
        # this is Fail Open because if the evaluation fails, we still want to proceed 
        # with the synthesis step rather than halting the process. 
        # The missing_info is left empty to avoid misleading the user into thinking that there is a specific piece of information that is missing, 
        # when in fact the failure was due to an error in the evaluation process itself.
        return {
            "next_tool": None,
            "missing_info": "",
            "_grade_sufficient": True,
        }


def evaluate_route_decision(state: AgentState) -> str:
    return state["next_tool"] if state["next_tool"] is not None else "synthesize"