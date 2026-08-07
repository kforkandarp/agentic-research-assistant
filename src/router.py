from typing import Literal
from pydantic import BaseModel, Field
from src.llm import get_fast_llm, invoke_with_retry 
from src.state import AgentState


class RouterDecision(BaseModel):
    tool: Literal["retrieval", "web_search", "calculator", "direct_answer"] = Field(
        description="Which tool should handle this query first"
    )
    reason: str = Field(description="One short sentence explaining why")


def _build_router_prompt() -> str:
    """Reads the paper list from chunks.json at import time so the prompt
    stays in sync with the actual corpus automatically."""
    try:
        import json
        with open("data/chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        papers = sorted(set(c["paper"] for c in chunks))
        paper_list = "\n  ".join(f"- {p}" for p in papers)
    except Exception:
        paper_list = "(could not load corpus — treat all paper questions as retrieval)"

    return f"""You are a routing agent for a research assistant over ArXiv ML papers.
Given a user's question, decide which ONE tool should handle it first:

- "retrieval": question is about specific facts, numbers, or content from these papers
  in the local corpus:
  {paper_list}
  If the question explicitly mentions a paper or model NOT in this list, use "web_search".
  If the question asks about something "recent", "current", "latest", or from a specific
  year that implies it post-dates these papers, use "web_search".
- "web_search": question needs current/recent information, or is about anything outside
  the papers listed above
- "calculator": the user's PRIMARY TASK is performing a numerical computation with known
  numbers (e.g. "what is 15% of 2340", "how many times bigger is X than Y").
  Do NOT choose calculator for questions that EXPLAIN or NAME a formula/concept —
  route those to "direct_answer" or "retrieval" instead.
- "direct_answer": question is general ML knowledge the LLM already knows, no tool needed.
  Also use direct_answer for questions that are clearly unanswerable by any tool — future
  predictions, hypotheticals, or questions explicitly framed as unanswerable. Do not send
  these to web_search hoping to find an answer that cannot exist.
"""

ROUTER_SYSTEM_PROMPT = _build_router_prompt()


def router_node(state: AgentState) -> dict:
    llm = get_fast_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RouterDecision)

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": state["query"]},
    ]

    try:
        decision: RouterDecision = invoke_with_retry(structured_llm, messages)
        tool = decision.tool
        reason = decision.reason
    except Exception as e:
        tool = "direct_answer"
        reason = f"Structured routing failed ({e}); defaulting to direct answer."

    return {"next_tool": tool, "routing_reason": reason}


if __name__ == "__main__":
    test_cases = [
        "What is the BLEU score of the base Transformer model?",
        "What is the current SOTA on GLUE this month?",
        "What's 15% of 2340?",
        "What is overfitting in machine learning?",
    ]

    for query in test_cases:
        state: AgentState = {
            "query": query, "next_tool": None, "routing_reason": None,
            "tool_outputs": [], "final_answer": None,
            "missing_info": "", "_grade_sufficient": True,
        }
        update = router_node(state)
        print(f"Query: {query}")
        print(f"  -> Tool: {update['next_tool']} | Reason: {update['routing_reason']}\n")