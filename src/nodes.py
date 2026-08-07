"""
Tool-executor nodes. Each one calls its underlying tool function and
writes ONLY the keys it changes back into state.
"""

from pydantic import BaseModel, Field
from src.state import AgentState
from src.tools.retrieval_tool import retrieval_tool
from src.tools.web_search_tool import web_search_tool
from src.tools.calculator_tool import calculator_tool
from src.llm import get_fast_llm, get_llm, invoke_with_retry


def retrieval_node(state: AgentState) -> dict:
    output = retrieval_tool(state["query"])
    return {"tool_outputs": [{"tool": "retrieval", "output": output}]}


def web_search_node(state: AgentState) -> dict:
    output = web_search_tool(state["query"])
    return {"tool_outputs": [{"tool": "web_search", "output": output}]}


class ExtractedExpression(BaseModel):
    expression: str = Field(
        description="Pure arithmetic expression using only numbers, "
        "+ - * / ** and parentheses. No words, no units, no punctuation."
    )


CALCULATOR_EXTRACTION_PROMPT = """Extract a pure arithmetic expression from the question,
using numbers from the retrieved evidence if provided.
Output ONLY numbers and math operators — no words, no units.

Example: "What's 15% of 2340?" -> "15/100 * 2340"
Example: "How many times bigger is 41.0 than 28.4?" -> "41.0/28.4"
Example question with evidence: "How much faster is Model B?"
  Evidence: "Model A: 82.1 BLEU. Model B: 88.4 BLEU."
  -> "88.4/82.1"
"""


def calculator_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ExtractedExpression)

    # Include any prior tool evidence so the extractor can work from
    # actual retrieved numbers, not just the words of the original query.
    prior_evidence = "\n\n".join(
        f"[{r['tool']}]\n{r['output']}" for r in state["tool_outputs"]
        if r["tool"] != "calculator"
    )
    user_content = (
        f"Question: {state['query']}\n\n"
        f"Retrieved evidence (use these numbers for the expression):\n{prior_evidence}"
        if prior_evidence
        else state["query"]
    )

    messages = [
        {"role": "system", "content": CALCULATOR_EXTRACTION_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        extracted: ExtractedExpression = invoke_with_retry(structured_llm, messages)
        expression = extracted.expression
    except Exception as e:
        return {"tool_outputs": [{
            "tool": "calculator",
            "output": f"Could not extract a valid expression: {e}",
        }]}

    result = calculator_tool(expression)
    return {"tool_outputs": [{
        "tool": "calculator",
        "output": f"Expression evaluated: {expression} = {result}",
    }]}


def direct_answer_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.0)
    messages = [{"role": "user", "content": state["query"]}]
    response = invoke_with_retry(llm, messages)
    return {"tool_outputs": [{"tool": "direct_answer", "output": response.content}]}


def synthesize_node(state: AgentState) -> dict:


    if len(state["tool_outputs"]) == 1 and state["tool_outputs"][0]["tool"] == "direct_answer": # state ka tool output is list of dict
        return {"final_answer": state["tool_outputs"][0]["output"]}

    
    evidence = "\n\n".join(
        f"[{record['tool']}]\n{record['output']}" for record in state["tool_outputs"]
    )

    caveat = "" 
    if not state.get("_grade_sufficient", True):  # even after 4 iterations of grader or evaluator node, our evidence is not sufficient 
            # to answer the question, so we need to tell the LLM to explicitly state that it cannot answer the question based on the evidence provided.
            # therefore we are just changing the System prompt here, to control LLM from hallucination

        caveat = (
            f"\n\nIMPORTANT: after review, this specific piece was still not "
            f"found in the evidence: {state.get('missing_info', 'unspecified')}. "
            f"You MUST explicitly state that this part is not available from "
            f"the gathered sources. Do NOT infer, estimate, or fill it in from "
            f"general knowledge, even with hedging language."
        )

    llm = get_llm(temperature=0.0)
    messages = [   # we edit system prompt here from CAVEAT
        {"role": "system", "content": (
    "Answer the user's question clearly and concisely using the tool "
"output(s) below as your primary source. You may synthesize, "
"paraphrase, and organize information from the evidence into a "
"coherent answer — that is your job. Do not add facts that are "
"completely absent from the evidence, but do construct a full "
"answer from what is there. Only say something is unavailable "
"if it is genuinely missing from all evidence, not just because "
"it isn't stated word-for-word. If the user requests a specific word count "
    "but the evidence doesn't support it, give a shorter honest answer "
    "rather than padding with general knowledge." + caveat
)},
        {"role": "user", "content": (
            f"Question: {state['query']}\n\nTool output(s):\n{evidence}"
        )},
    ]
    response = invoke_with_retry(llm, messages)
    return {"final_answer": response.content}