"""
src/ragas_eval.py
RAGAS evaluation for the Agentic Research Assistant.
Evaluates Answer Relevancy, Faithfulness, and Context Precision.
"""

import json
import sys
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from ragas.run_config import RunConfig


load_dotenv()

RESULTS_PATH = "eval/eval_results_with_gt.json"
RAGAS_OUTPUT_PATH = "eval/ragas_results.csv"
PILOT_SIZE = 4  # per metric group for developer testing


# ── RAGAS LLM ADAPTER FIX ─────────────────────────────────────────────────────
# RAGAS 0.2.x inspects .temperature directly on LLM wrappers.
# Wrapping ChatGroq in this custom class guarantees RAGAS always gets a clean
# .temperature attribute without throwing RunnableWithFallbacks errors.

class DirectGroqRagasLLM(BaseRagasLLM):
    def __init__(self, temperature: float = 0.0):
        super().__init__()
        api_key = (
            os.getenv("GROQ_API_KEY1") # the first true value out of all these will be used as api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_API_KEY2")
        )
        if not api_key:
            raise ValueError("No Groq API key found in environment variables.")
            
        self._temperature = temperature
        # Use fast llm openai/gpt-oss-20b to prevent Groq TPM (Tokens Per Minute) 429 errors during parallel judging
        self.langchain_llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=temperature,
            groq_api_key=api_key,
        )
        self.wrapper = LangchainLLMWrapper(self.langchain_llm)

    @property
    def temperature(self) -> float:
        return self._temperature

    # Suppresses the 'is_finished not implemented' logging warnings
    def is_finished(self, response) -> bool:
        return True

    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        return self.wrapper.generate_text(
            prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks
        )

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        return await self.wrapper.agenerate_text(
            prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks
        )


# ── DATA LOADING & FILTERS ────────────────────────────────────────────────────

def load_results() -> list[dict]:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["results"]


def get_answer_relevancy_entries(results: list[dict], use_all: bool) -> list[dict]:
    eligible = [
        r for r in results
        if r["actual_category"].lower() not in ("insufficient_information", "error")
        and r.get("final_answer")
    ]
    return eligible if use_all else eligible[:PILOT_SIZE]


def get_faithfulness_entries(results: list[dict], use_all: bool) -> list[dict]:
    eligible = [
        r for r in results
        if r["actual_category"].lower() not in ("insufficient_information", "error")
        and r.get("contexts")
        and r.get("final_answer")
    ]
    return eligible if use_all else eligible[:PILOT_SIZE]


def get_context_precision_entries(results: list[dict], use_all: bool) -> list[dict]:
    eligible = [
        r for r in results
        if r.get("expected_tool") == "retrieval"
        and r.get("actual_category") != "error"
        and r.get("contexts")
        and len(r.get("contexts", [])) > 0
        and r.get("ground_truth")
        and r.get("final_answer")
    ]
    return eligible if use_all else eligible[:PILOT_SIZE]


def run_metric(entries: list[dict], metrics: list, dataset_dict: dict,
               judge_llm, judge_embeddings, label: str) -> dict:
    if not entries:
        print(f"[SKIP] {label} — no eligible entries.")
        return {}

    ids = [r["id"] for r in entries]
    print(f"\n[{label}] scoring {len(ids)} questions: {ids}")

    dataset = Dataset.from_dict(dataset_dict)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(
            max_workers=1,
            max_retries=10,
            timeout=180
        )
    )

    scores = {}
    try:
        for m in metrics:
            val = result[m.name]
            score = sum(val) / len(val) if isinstance(val, list) else float(val)
            scores[m.name] = score
        for name, score in scores.items():
            print(f"  {name}: {score:.4f}")
    except Exception:
        print(f"  Raw: {result}")

    try:
        df = result.to_pandas()
        df.insert(0, "question_id", ids)
        scores["_df"] = df
    except Exception:
        pass

    return scores


def run_ragas():
    use_all = "--all" in sys.argv
    mode = "all eligible entries" if use_all else f"pilot ({PILOT_SIZE} per metric)"
    print(f"=== RAGAS Evaluation — {mode} ===")

    results = load_results()

    # Instantiate our custom adapter that explicitly exposes .temperature to RAGAS
    judge_llm = DirectGroqRagasLLM(temperature=0.0)

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    all_dfs = []
    all_scores = {}

    # 1. answer_relevancy
    ar_entries = get_answer_relevancy_entries(results, use_all)
    ar_scores = run_metric(
        entries=ar_entries,
        metrics=[answer_relevancy],
        dataset_dict={
            "question": [r["question"] for r in ar_entries],
            "answer":   [r["final_answer"] for r in ar_entries],
            "contexts": [[] for _ in ar_entries],
        },
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        label="answer_relevancy",
    )
    all_scores.update({k: v for k, v in ar_scores.items() if not k.startswith("_")})
    if "_df" in ar_scores:
        all_dfs.append(ar_scores["_df"])

    # 2. faithfulness
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

    # 3. context_precision
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

    print("\n=== Final RAGAS Scores ===")
    for metric, score in all_scores.items():
        print(f"  {metric}: {score:.4f}")

    try:
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        combined.to_csv(RAGAS_OUTPUT_PATH, index=False)
        print(f"\nPer-question scores saved to {RAGAS_OUTPUT_PATH}")
    except Exception as e:
        print(f"[WARNING] Could not save CSV ({e}). Scores printed above.")


if __name__ == "__main__":
    run_ragas()