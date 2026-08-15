"""
src/run_eval.py
Runs every question in eval/eval_set.json through the compiled graph.
Resumable: python -m src.run_eval resumes from eval_results.json;
           python -m src.run_eval --fresh wipes prior results and reruns everything.
"""

import json
import os
import sys
import logging
from groq import RateLimitError
from tenacity import RetryError
from src.graph import build_graph
from src.state import AgentState

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvalRunner")

EVAL_SET_PATH = "eval/eval_set.json"
RESULTS_PATH = "eval/eval_results.json"


def classify_actual(tools_used: list[str], grade_sufficient: bool, expected_tool: str) -> str:
    """
    Categorizes actual tool invocation.
    Prioritizes primary initial routing accuracy while tracking multi-step tool chains.
    """
    if not tools_used:
        return "error"
    
    # Single-tool match
    if len(tools_used) == 1:
        return tools_used[0]

    # Multi-step tool chain: check if expected tool was the primary (first) tool invoked
    primary_tool = tools_used[0]
    if expected_tool == primary_tool:
        return primary_tool

    # If arithmetic calculator was part of a multi-step chain
    if "calculator" in tools_used and expected_tool == "multi_step":
        return "multi_step"

    if not grade_sufficient:
        return "insufficient_information"

    return "multi_step"


def save_results(results: list[dict], path: str = RESULTS_PATH):
    correct = sum(r["routing_correct"] for r in results)
    total = len(results)
    with open(path, "w", encoding="utf-8") as f: # creates file if it doesn't exist at path
        json.dump(
            {
                "routing_accuracy_pct": (correct / total * 100) if total else 0,
                "results": results,
            },
            f,
            indent=2,
        )


def run_single(item: dict, app) -> dict:
    initial_state: AgentState = {
        "query": item["question"],
        "next_tool": None,
        "routing_reason": None,
        "tool_outputs": [],
        "final_answer": None,
        "missing_info": "",
        "_grade_sufficient": True,
    }

    outcome = app.invoke(initial_state)
    tools_used = [r["tool"] for r in outcome["tool_outputs"]]
    expected = item["expected_tool"]
    
    # Classify actual tool path
    actual = classify_actual(
        tools_used=tools_used,
        grade_sufficient=outcome.get("_grade_sufficient", True),
        expected_tool=expected
    )

    # A route is correct if actual matches expected, OR if expected tool was the primary first tool invoked
    routing_correct = (actual == expected) or (len(tools_used) > 0 and tools_used[0] == expected)

    return {
        "id": item["id"],
        "question": item["question"],
        "expected_tool": expected,
        "actual_category": actual,
        "tools_used": tools_used,
        "routing_correct": routing_correct,
        "final_answer": outcome["final_answer"],
        "contexts": [
            r["output"] for r in outcome["tool_outputs"]
            if r["tool"] in ("retrieval", "web_search")
        ],
    }


def run_eval():
    fresh = "--fresh" in sys.argv

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    successful_ids: set[str] = set() # store IDs of successful runs to skip on resume; and we choose set because avg time is O(1)

    if fresh and os.path.exists(RESULTS_PATH):
        logger.info("[fresh] Discarding previous results, re-running full benchmark suite...")
    elif os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            results = existing["results"]
            successful_ids = {
                r["id"] for r in results 
                if r["actual_category"] not in ("error", "ERROR")
            }
        results = [r for r in results if r["id"] in successful_ids]

    app = build_graph()

    for item in eval_set:
        if item["id"] in successful_ids:
            logger.info(f"[SKIP] {item['id']} already completed.")
            continue

        try:
            entry = run_single(item, app) # entry is also a dictionary
            status = "OK" if entry["routing_correct"] else "MISMATCH"
            print(
                f"[{status}] {entry['id']} | expected={entry['expected_tool']} | "
                f"actual={entry['actual_category']} | tools={entry['tools_used']}"
            )
            results.append(entry)
            save_results(results)

        except (RateLimitError, RetryError) as e:
            logger.error(f"[QUOTA EXHAUSTED] Stopping run at {item['id']}: {e}")
            break

        except Exception as e:
            entry = {
                "id": item["id"],
                "question": item["question"],
                "expected_tool": item["expected_tool"],
                "actual_category": "error",
                "tools_used": [],
                "routing_correct": False,
                "final_answer": f"RUN FAILED: {e}",
                "contexts": [],
            }
            logger.error(f"[ERROR] {item['id']} | {e}")
            results.append(entry)
            save_results(results)

    total = len(results)
    correct = sum(r["routing_correct"] for r in results)
    accuracy = (correct / total * 100) if total else 0
    print(f"\n✅ Final Routing Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()