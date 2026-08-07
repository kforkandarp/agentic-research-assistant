from typing import TypedDict, Literal, Optional, Annotated
import operator


ToolName = Literal["retrieval", "web_search", "calculator", "direct_answer"]


class ToolCallRecord(TypedDict):
    """One tool invocation's result — tagged with WHICH tool produced it,
    not just a raw string, so the synthesizer and future UI can tell them apart."""
    tool: str
    output: str


class AgentState(TypedDict):
    """The shared object passed between every node in the graph."""

    query: str                                  # the user's original question
    next_tool: Optional[ToolName]                # which tool to call next (router or evaluator sets this)
    routing_reason: Optional[str]                # WHY the router picked its tool

    # Annotated + operator.add: LangGraph APPENDS returned lists here
    # instead of overwriting, so multiple tool calls accumulate.
    tool_outputs: Annotated[list[ToolCallRecord], operator.add]

    final_answer: Optional[str]                  # the synthesized answer, once ready
    missing_info: str                    # what evaluate found missing; empty string if nothing missing
    _grade_sufficient: bool              # internal: was evidence judged sufficient? always set by evaluate_node