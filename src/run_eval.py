"""
Runs every question in eval/eval_set.json through the compiled graph.
Resumable: python -m src.run_eval resumes from eval_results.json;
           python -m src.run_eval --fresh wipes prior results and reruns everything.
Holdout:   python -m src.run_eval --holdout runs the one-shot holdout set.
           Never resumable by design — re-running the holdout overfits to it.
"""

import json
import os
import sys
from groq import RateLimitError
from tenacity import RetryError
from src.graph import build_graph
from src.state import AgentState

EVAL_SET_PATH = "eval/eval_set.json"
RESULTS_PATH = "eval/eval_results.json"
HOLDOUT_SET_PATH = "eval/holdout_set.json"
HOLDOUT_RESULTS_PATH = "eval/holdout_results.json"


def classify_actual(tools_used: list[str], grade_sufficient: bool) -> str:
    if not tools_used:
        # Should never happen in normal graph execution — every path runs
        # at least one tool. Guard exists to catch future graph changes
        # that might create a zero-tool path silently.
        return "error"
    if not grade_sufficient:
        return "insufficient_information"
    if len(tools_used) > 1:
        return "multi_step"
    return tools_used[0]


def save_results(results: list[dict], path: str = RESULTS_PATH):
    correct = sum(r["routing_correct"] for r in results)
    total = len(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "routing_accuracy_pct": correct / total * 100 if total else 0,
                "results": results,
            },
            f,
            indent=2,
        )

# THIS HANDLES ONLY ONE QUESTION and IT JUST RETURNS THE RESULT ENTRY. It does not save to disk or print anything.
def run_single(item: dict, app) -> dict: # app is the compiled graph, Instead of rebuilding the graph 
    # 100 times, you build it once and pass the same graph into every call
    """Runs one eval question through the graph and returns a result entry.
    Shared by run_eval and run_holdout to avoid duplicating try/except logic."""
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
    actual = classify_actual(tools_used, outcome.get("_grade_sufficient", True))
    expected = item["expected_tool"]

    return {
        "id": item["id"],
        "question": item["question"],
        "expected_tool": expected,
        "actual_category": actual,
        "tools_used": tools_used,
        "routing_correct": actual == expected,
        "final_answer": outcome["final_answer"],
        "contexts": [
            r["output"] for r in outcome["tool_outputs"]
            if r["tool"] in ("retrieval", "web_search")
        ],
    }


def run_eval(): # evals the whole dataset now, run_single runs for one question
    fresh = "--fresh" in sys.argv

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f) # list of question dictionaries

    results = []
    # successful_ids: only questions that completed without ERROR.
    # ERROR entries are intentionally excluded so they re-run on resume.
    successful_ids: set[str] = set()
    # Checking membership inside a Python set is much faster than a list. This is exactly what sets are designed for.

    if fresh and os.path.exists(RESULTS_PATH):
        print("[fresh] discarding previous results, re-running everything")
    elif os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            results = existing["results"]
            successful_ids = {r["id"] for r in results if r["actual_category"] != "error" and r["actual_category"] != "ERROR"}
        results = [r for r in results if r["id"] in successful_ids]

    app = build_graph()

    for item in eval_set:
        if item["id"] in successful_ids:
            print(f"[SKIP] {item['id']} already completed")
            continue

        try:
            entry = run_single(item, app)
            status = "OK" if entry["routing_correct"] else "MISMATCH"
            print(f"[{status}] {entry['id']} | expected={entry['expected_tool']} | actual={entry['actual_category']} | tools={entry['tools_used']}")
            results.append(entry)
            save_results(results)

        except (RateLimitError, RetryError) as e:
            print(f"[QUOTA EXHAUSTED] Stopping run at {item['id']}: {e}")
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
            print(f"[ERROR] {item['id']} | {e}")
            results.append(entry)
            save_results(results)

    total = len(results)
    correct = sum(r["routing_correct"] for r in results)
    print(f"\nRouting accuracy: {correct}/{total} ({correct / total * 100 if total else 0:.1f}%)")
    print(f"Full results saved to {RESULTS_PATH}")


def run_holdout():
    """
    Runs the holdout set exactly once. Never resumable by design —
    the holdout is a one-shot validity check, not an iterative tuning loop.
    If holdout_results.json already exists, stop immediately.
    Re-running it would mean tuning against it, defeating its purpose.
    """
    if os.path.exists(HOLDOUT_RESULTS_PATH):
        print("[HOLDOUT] Results already exist at eval/holdout_results.json")
        print("[HOLDOUT] Delete the file manually only if you have a genuine reason to re-run.")
        print("[HOLDOUT] Re-running after prompt changes = overfitting to the holdout.")
        return

    with open(HOLDOUT_SET_PATH, "r", encoding="utf-8") as f:
        holdout_set = json.load(f)

    app = build_graph()
    results = []

    for item in holdout_set:
        try:
            entry = run_single(item, app)
            status = "OK" if entry["routing_correct"] else "MISMATCH"
            print(f"[{status}] {entry['id']} | expected={entry['expected_tool']} | actual={entry['actual_category']} | tools={entry['tools_used']}")
            results.append(entry)

        except (RateLimitError, RetryError) as e:
            print(f"[QUOTA EXHAUSTED] Stopping holdout at {item['id']}: {e}")
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
            print(f"[ERROR] {item['id']} | {e}")
            results.append(entry)

    total = len(results)
    correct = sum(r["routing_correct"] for r in results)
    accuracy = correct / total * 100 if total else 0

    # Save only at the end — holdout has no incremental save.
    # A partial holdout run (quota exhaustion) saves what it has so far.
    save_results(results, path=HOLDOUT_RESULTS_PATH)

    print(f"\nHoldout accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Results saved to {HOLDOUT_RESULTS_PATH}")


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        run_holdout()
    else:
        run_eval()