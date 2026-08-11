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
    try:
        import json
        with open("data/chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        papers = sorted(set(c["paper"] for c in chunks if c.get("paper")))
        paper_list = "\n  ".join(f"- {p}" for p in papers[:20]) # List top ingested papers
    except Exception:
        paper_list = "(local paper corpus loaded)"

    return f"""You are a routing agent for a research assistant over ArXiv ML papers.
Given a user's question, decide which ONE tool should handle it first:

- "retrieval": The question asks about specific facts, architecture details, hyperparameters, or metrics from an ML paper, OR explicitly mentions a paper name (e.g., Transformer, ResNet, BERT, DDPM, GPT-3, LoRA). ALWAYS use retrieval when a paper name or core paper concept is mentioned.
- "web_search": The question asks about recent events, current state-of-the-art leaderboards, software versions, news, or work from 2025/2026.
- "calculator": The question is a PURE mathematical computation problem with raw numbers already given in the prompt (e.g., "what's 15% of 2340", "compute 2^16 / 8"). Do NOT use calculator if numbers need to be looked up from a paper first.
- "direct_answer": The question asks for general, high-level ML definitions or explanations (e.g., "what is overfitting", "difference between precision and recall", "what is gradient descent") that do NOT depend on a specific paper. Also use for general geography/knowledge.
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