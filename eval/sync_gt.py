"""
sync_gt.py — run this after every --fresh eval run.
Copies updated agent results (final_answer, tools_used, actual_category,
routing_correct, contexts) from eval_results.json into eval_results_with_gt.json,
preserving the ground_truth strings that run_eval.py never writes.

Usage: python sync_gt.py
"""

import json

RESULTS_PATH = "eval/eval_results.json"
GT_PATH = "eval/eval_results_with_gt.json"

def sync():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        new_results = json.load(f)

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Index new results by id for fast lookup
    new_by_id = {r["id"]: r for r in new_results["results"]} # id becomes the key, entire record new_results["results"] becomes the value. 

    # Fields that come from the agent run — overwrite these
    agent_fields = [
        "final_answer", "tools_used", "actual_category",
        "routing_correct", "contexts"
    ]

    updated = 0
    for entry in gt_data["results"]:
        qid = entry["id"]
        if qid in new_by_id:
            for field in agent_fields:
                if field in new_by_id[qid]:
                    entry[field] = new_by_id[qid][field]
            updated += 1

    # Update accuracy too
    gt_data["routing_accuracy_pct"] = new_results["routing_accuracy_pct"]

    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    print(f"Synced {updated} entries from eval_results.json → eval_results_with_gt.json")
    print(f"New routing accuracy: {gt_data['routing_accuracy_pct']:.1f}%")
    print("ground_truth strings preserved.")

if __name__ == "__main__":
    sync()