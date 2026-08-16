"""
eval/ablation_tiering.py
Empirical ablation comparing Homogeneous (120B) vs. Tiered (20B) Routing.
Measures: Component Routing Accuracy (%), Avg Latency (ms), and Unit Economics.
"""

import json
import time
import os
from langchain_groq import ChatGroq
from src.router import RouterDecision, ROUTER_SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()

EVAL_SET_PATH = "eval/ablation_router_set.json"


def evaluate_router_model(model_name: str, cost_per_m_input: float, delay_sec: float = 1.0):
    print(f"\n--- Benchmarking Router: {model_name} ---")
    api_key = os.getenv("GROQ_API_KEY1") or os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=model_name, temperature=0.0, api_key=api_key)
    structured_llm = llm.with_structured_output(RouterDecision)

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    correct = 0
    latencies = []

    for item in eval_set:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]

        t0 = time.perf_counter()
        try:
            decision: RouterDecision = structured_llm.invoke(messages)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            is_correct = decision.tool == item["expected_initial_tool"]
            if is_correct:
                correct += 1
            status = "OK" if is_correct else "FAIL"
            print(
                f"[{status}] {item['id']} ({item['type']}) | "
                f"Expected: {item['expected_initial_tool']} | "
                f"Got: {decision.tool} ({elapsed_ms:.1f}ms)"
            )
        except Exception as e:
            print(f"[ERR] {item['id']} | {e}")

        time.sleep(delay_sec)

    total = len(eval_set)
    acc = (correct / total) * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "model": model_name,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_latency_ms": avg_latency,
        "cost_per_m_input": cost_per_m_input,
    }


if __name__ == "__main__":
    res_20b = evaluate_router_model("openai/gpt-oss-20b", cost_per_m_input=0.075, delay_sec=1.0)

    print("\nCooling down for 10 seconds before baseline benchmark...")
    time.sleep(10)

    res_120b = evaluate_router_model("openai/gpt-oss-120b", cost_per_m_input=0.15, delay_sec=1.2)

    print("\n" + "=" * 70)
    print("FINAL ABLATION RESULTS (Model Tiering Experiment)")
    print("=" * 70)
    print("| Configuration | Model | Accuracy (N=50) | Avg Router Latency | Cost / 1M Input Tokens |")
    print("|---|---|---|---|---|")
    print(f"| Homogeneous Baseline | `{res_120b['model']}` | {res_120b['accuracy']:.1f}% ({res_120b['correct']}/{res_120b['total']}) | {res_120b['avg_latency_ms']:.1f} ms | ${res_120b['cost_per_m_input']} |")
    print(f"| Asymmetric Tiered | `{res_20b['model']}` | {res_20b['accuracy']:.1f}% ({res_20b['correct']}/{res_20b['total']}) | {res_20b['avg_latency_ms']:.1f} ms | ${res_20b['cost_per_m_input']} |")
    print("=" * 70)