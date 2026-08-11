"""
eval/sync_gt.py
Merges fresh 50-question benchmark outputs from eval_results.json into eval_results_with_gt.json.
Preserves existing ground_truth strings and appends any newly added questions (q19-q50).
"""

import json

RESULTS_PATH = "eval/eval_results.json"
GT_PATH = "eval/eval_results_with_gt.json"

def sync():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    try:
        with open(GT_PATH, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        gt_data = {"results": []}

    # Index existing ground_truth values by question ID
    gt_by_id = {r["id"]: r.get("ground_truth", "") for r in gt_data.get("results", [])}

    merged_results = []
    for entry in new_data["results"]:
        qid = entry["id"]
        # Preserve ground_truth if it previously existed, else default to empty string
        entry["ground_truth"] = gt_by_id.get(qid, "")
        merged_results.append(entry)

    updated_data = {
        "routing_accuracy_pct": new_data.get("routing_accuracy_pct", 0.0),
        "note": "Copy of eval_results.json with ground_truth added for RAGAS. Never overwritten by run_eval.py.",
        "results": merged_results
    }

    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Synced {len(merged_results)} entries into eval_results_with_gt.json!")
    print(f"Routing accuracy: {updated_data['routing_accuracy_pct']:.1f}%")

if __name__ == "__main__":
    sync()