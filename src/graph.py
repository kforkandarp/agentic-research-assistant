"""
Full graph: START -> router -> tool -> evaluate -> (loop back to a tool,
OR synthesize) -> synthesize -> END.
direct_answer routes through evaluate too — a self-admitted "I don't know"
can escalate to web_search instead of silently stopping.
"""

from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.router import router_node
from src.evaluate import evaluate_node, evaluate_route_decision
from src.nodes import (
    retrieval_node,
    web_search_node,
    calculator_node,
    direct_answer_node,
    synthesize_node,
)


def route_decision(state: AgentState) -> str:
    return state["next_tool"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router", route_decision,
        {
            "retrieval": "retrieval", "web_search": "web_search",
            "calculator": "calculator", "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieval", "evaluate")
    graph.add_edge("web_search", "evaluate")
    graph.add_edge("calculator", "evaluate")
    graph.add_edge("direct_answer", "evaluate")

    graph.add_conditional_edges(
        "evaluate", evaluate_route_decision,
        {
            "retrieval": "retrieval", "web_search": "web_search",
            "calculator": "calculator", "synthesize": "synthesize",
        },
    )

    graph.add_edge("synthesize", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    with open("graph_diagram.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("[saved] graph_diagram.png")

    # test_queries = [
    #     "What is GPT-3's parameter count according to the paper, and how "
    #     "many times larger is that compared to GPT-2's parameter count?",
    # ]

    # for query in test_queries:
    #     initial_state: AgentState = {
    #         "query": query, "next_tool": None, "routing_reason": None,
    #         "tool_outputs": [], "final_answer": None,
    #         "missing_info": "", "_grade_sufficient": True,
    #     }
    #     result = app.invoke(initial_state)
    #     print(f"Query: {query}")
    #     print(f"  Tools used (in order): {[r['tool'] for r in result['tool_outputs']]}")
    #     print(f"  Final answer: {result['final_answer']}\n")