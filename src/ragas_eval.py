"""
RAGAS evaluation for the Agentic Research Assistant.

Metrics and their scope:
  - answer_relevancy:   13 questions — every entry where agent produced a real answer.
                        Does not need contexts or ground_truth.
  - faithfulness:       6 questions  — entries where retrieval/web_search ran and
                        returned contexts. Checks synthesizer grounding. (and hallucinations)
  - context_precision:  3 questions  — pure retrieval entries with ground_truth only.
                        Multi-step excluded (mixed retrieval+web_search context blob
                        makes scores uninterpretable).

Metric NOT used:
  - context_recall: Corpus is intentionally fixed at 5 papers. Low recall reflects
                    corpus scope not retriever failure — not an actionable metric here.

Reads from eval/eval_results_with_gt.json — static file, never overwritten by run_eval.py.

Run modes:
  python -m src.ragas_eval                  # pilot: representative sample, all 3 metrics
  python -m src.ragas_eval --all            # full run: all eligible per metric
"""

import json
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from src.llm import get_llm

RESULTS_PATH = "eval/eval_results_with_gt.json"
RAGAS_OUTPUT_PATH = "eval/ragas_results.csv"
PILOT_SIZE = 4  # per metric group — keeps total judge calls manageable


def load_results() -> list[dict]:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["results"]


def get_answer_relevancy_entries(results: list[dict], use_all: bool) -> list[dict]:
    """
    answer_relevancy: needs question + answer only.
    Eligible: any entry where agent produced a real answer.
    Excluded: insufficient_information (honest I-dont-know, no claim to score)
              and error entries.
    """
    eligible = [
        r for r in results
        if r["actual_category"].lower() not in ("insufficient_information", "error")
        and r.get("final_answer")
    ]
    return eligible if use_all else eligible[:PILOT_SIZE]


def get_faithfulness_entries(results: list[dict], use_all: bool) -> list[dict]:
    """
    faithfulness: needs question + answer + contexts.
    Eligible: entries where retrieval or web_search ran and saved contexts.
    Excluded: calculator-only, direct_answer-only (no retrieved context to ground against),
              insufficient_information (no meaningful answer to check).
    """
    eligible = [
        r for r in results
        if r["actual_category"].lower() not in ("insufficient_information", "error")
        and r.get("contexts")
        and r.get("final_answer")
    ]
    return eligible if use_all else eligible[:PILOT_SIZE]


def get_context_precision_entries(results: list[dict], use_all: bool) -> list[dict]:
    """
    context_precision: needs question + answer + contexts + ground_truth.
    Eligible: pure retrieval entries with ground_truth only.
    Multi-step excluded: their contexts field is a concatenation of retrieval
    output and web_search output. context_precision cannot distinguish which
    chunks came from which tool — scores would conflate retriever quality with
    Tavily quality, making them uninterpretable.
    """
    eligible = [
        r for r in results
        if r["actual_category"] == "retrieval"
        and r.get("contexts")
        and r.get("ground_truth")
        and r.get("final_answer")
    ]
    return eligible if use_all else eligible[:PILOT_SIZE]


def run_metric(entries: list[dict], metrics: list, dataset_dict: dict,
               judge_llm, judge_embeddings, label: str) -> dict:
    """Runs a single RAGAS evaluate() call and returns scores as a dict."""
    if not entries: # if our evaluation dataset is empty, RAGAS will throw an error. Skip and return empty dict. this is early exit
        print(f"[SKIP] {label} — no eligible entries.")
        return {}
    # creating ids to prvnt traceability since ragas schema only takes question/ans/contexts/ground_truth, not ids.
    ids = [r["id"] for r in entries]
    print(f"\n[{label}] scoring {len(ids)} questions: {ids}")

    dataset = Dataset.from_dict(dataset_dict)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scores = {} # this dict acts as a collector for scores of each metric.
    try:
        for m in metrics:
            val = result[m.name]
            # ragas 0.2.x returns a list of per-question scores, not an aggregate float
            score = sum(val) / len(val) if isinstance(val, list) else float(val)
            scores[m.name] = score
        for name, score in scores.items():
            print(f"  {name}: {score:.4f}")
    except Exception as e:
        print(f"  Raw: {result}")

    # Attach question IDs to per-question rows
    # since we also need per-question scores for debugging and analysis
    try:
        df = result.to_pandas()
        df.insert(0, "question_id", ids) # pandas inserts ids list that we extracted earlier as a new column in df dataframe
        scores["_df"] = df
    except Exception:
        pass

    return scores


def run_ragas():
    use_all = "--all" in sys.argv  # true or false
    mode = "all eligible entries" if use_all else f"pilot ({PILOT_SIZE} per metric)"
    print(f"=== RAGAS Evaluation — {mode} ===")

    results = load_results()

    judge_llm = LangchainLLMWrapper(get_llm(temperature=0.0)) # since ragas doesnt understand Langchain LLMs we wrap it in a LangchainLLMWrapper.
    # get_llm() is a function that returns a LLM object with temperature=0.0 (deterministic output)
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    all_dfs = [] # Each call to run_metric() also returns a DataFrame containing per-question scores.
    all_scores = {}

    # ── 1. answer_relevancy ──────────────────────────────────────────────────
    ar_entries = get_answer_relevancy_entries(results, use_all)
    ar_scores = run_metric(
        entries=ar_entries,
        metrics=[answer_relevancy],
        dataset_dict={
            "question": [r["question"] for r in ar_entries],
            "answer":   [r["final_answer"] for r in ar_entries],
            # answer_relevancy uses embeddings on question+answer only.
            # contexts field must still be present in dataset for ragas API
            # but is not used in scoring — pass empty list per entry.
            "contexts": [[] for _ in ar_entries],
        },
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        label="answer_relevancy",
    )
    all_scores.update({k: v for k, v in ar_scores.items() if not k.startswith("_")})
    if "_df" in ar_scores:
        all_dfs.append(ar_scores["_df"])

    # ── 2. faithfulness ──────────────────────────────────────────────────────
    fa_entries = get_faithfulness_entries(results, use_all)
    fa_scores = run_metric(
        entries=fa_entries,
        metrics=[faithfulness],
        dataset_dict={
            "question": [r["question"] for r in fa_entries],
            "answer":   [r["final_answer"] for r in fa_entries],
            "contexts": [r["contexts"] for r in fa_entries],
        },
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        label="faithfulness",
    )
    all_scores.update({k: v for k, v in fa_scores.items() if not k.startswith("_")})
    if "_df" in fa_scores:
        all_dfs.append(fa_scores["_df"])

    # ── 3. context_precision ─────────────────────────────────────────────────
    cp_entries = get_context_precision_entries(results, use_all)
    cp_scores = run_metric(
        entries=cp_entries,
        metrics=[context_precision],
        dataset_dict={
            "question":     [r["question"] for r in cp_entries],
            "answer":       [r["final_answer"] for r in cp_entries],
            "contexts":     [r["contexts"] for r in cp_entries],
            "ground_truth": [r["ground_truth"] for r in cp_entries],
        },
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        label="context_precision",
    )
    all_scores.update({k: v for k, v in cp_scores.items() if not k.startswith("_")})
    if "_df" in cp_scores:
        all_dfs.append(cp_scores["_df"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Final RAGAS Scores ===")
    for metric, score in all_scores.items():
        print(f"  {metric}: {score:.4f}")

    # Save per-question breakdown
    try:
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        combined.to_csv(RAGAS_OUTPUT_PATH, index=False)
        print(f"\nPer-question scores saved to {RAGAS_OUTPUT_PATH}")
    except Exception as e:
        print(f"[WARNING] Could not save CSV ({e}). Scores printed above.")


if __name__ == "__main__":
    run_ragas()