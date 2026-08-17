# Engineering Decisions & System Architecture

This document outlines the architectural principles, trade-offs, and technical rationale underlying the design and implementation of the Agentic Research Assistant.

It intends to provide context beyond the source code and explain the engineering thought process.

---

## 1. Summary & Problem Statement

### Motivation

Standard Retrieval-Augmented Generation (RAG) architectures rely primarily on linear, single-pass pipelines (`Retrieve` → `Synthesize`). In technical domains such as machine learning research literature, single-pass architectures encounter three critical failure modes:

* **Parametric Leakage & Hallucination:** When retrieved context lacks required facts, linear systems proceed to synthesis regardless, leaking pre-trained assumptions.
* **Context Contamination:** Dense vector search frequently retrieves semantically adjacent but factually irrelevant text chunks, degrading synthesis accuracy.
* **Inference Inefficiency:** Invoking high-parameter reasoning models for simple deterministic calculations or low-complexity routing adds unnecessary latency and unit cost.

### Core Engineering Philosophy

* **Determinism Over Pure Prompting:** Enforce hard control flow guardrails and deterministic AST parsers rather than relying solely on probabilistic LLM instruction-following.
* **Decoupled Asymmetric Processing:** Segregate task complexity across distinct model tiers to optimize latency and token allocation.
* **System Resilience:** Reduce single-key quota/rate-limit failure modes through automated key rotation and exponential backoff retry strategies.
* **Explainable Evaluation & Metric Transparency:** Measure system performance quantitatively via RAGAS quality benchmarks and custom routing accuracy suites.

---

## 2. Architectural Decisions: Cyclic State Machines vs. Linear Chains

### Decision: LangGraph `StateGraph` Over Linear Execution Chains (LCEL)

Linear chains (e.g., LangChain Expression Language pipelines) operate on an unvalidated happy path. If initial document retrieval returns poor context, a linear pipeline forces immediate synthesis on incomplete data.

**Linear Chain (Fragile):**

```
[ Query ] ──> [ Retrieval ] ──> [ Synthesize ] ──> [ Final Output (Potential Hallucination) ]
```

**Cyclic State Machine (Agentic Research Assistant):**

```
                               ┌─────────────────────────┐
                               ▼                         │ (Insufficient / Missing Tool)
[ Query ] ──> [ Router ] ──> [ Tool Execution ] ──> [ Evaluate ] ──> [ Synthesize ] ──> [ Final Output ]
                                                         │
                                                         └── (Sufficient) ──> [ Synthesize ]
```

* **Dynamic Re-Routing:** An explicit **Evaluate** node checks evidence sufficiency after tool execution.
* **Automated Fallback:** If context is incomplete, the graph routes execution to alternate untried tools (e.g., falling back from local vector search to live web search).

### Deterministic Circuit Breakers

To prevent multi-tool agent loops from entering infinite recursion during unanswerable queries, two guardrails are enforced at the graph level:

* **`MAX_EVIDENCE_STEPS = 4`:** A hard upper bound on state updates. Reaching step 4 forces an immediate transition to `Synthesize` with an explicit missing-information flag appended to system prompt state.
* **`Regex Numerical Short-Circuiting`:** Queries containing direct arithmetic intent or extracted numerical expressions bypass sub-agent escalation, routing straight to local computation.

---

## 3. Asymmetric Model Tiering & Inference Unit Economics

### Decision: GPT OSS 20B (Router) vs. GPT OSS 120B (Evaluator & Synthesizer)

Deploying uniform, high-parameter LLMs across all state nodes introduces unnecessary token expense and processing overhead. Task complexity is segregated across two production tiers:

| Graph Node | Model Assignment | Provider-Reported Throughput | Engineering Rationale |
|---|---|---|---|
| **Router** | GPT OSS 20B (`openai/gpt-oss-20b`) | **1,000 T/s** (250K TPM, $0.075 in / $0.30 out per 1M) | 4-class single-step query classification via Pydantic structured output. Lightweight parameter capacity minimizes cost on pure intent classification. |
| **Evaluate** | GPT OSS 120B (`openai/gpt-oss-120b`) | **500 T/s** (250K TPM, $0.15 in / $0.60 out per 1M) | Multi-step evidence sufficiency evaluation requires high-capacity reasoning across accumulated state context. |
| **Synthesize** | GPT OSS 120B (`openai/gpt-oss-120b`) | **500 T/s** (250K TPM, $0.15 in / $0.60 out per 1M) | Final answer synthesis demands strict grounding adherence to prompt instructions to eliminate parametric knowledge leakage. |

### Empirical Model Tiering Ablation ($N=50$ Adversarial Set)

To empirically validate using `openai/gpt-oss-20b` for intent routing rather than deploying `openai/gpt-oss-120b` uniformly across the graph, an isolated ablation study was conducted over an adversarial boundary dataset (`eval/ablation_router_set.json`). The benchmark specifically tests edge cases: implicit numerical extractions, out-of-index ML models, temporal 2026 search cues, and speculative unanswerable prompts.

| Configuration | Model Tier | Adversarial Router Accuracy ($N=50$) | Avg Router Latency (Benchmark Run) | Cost / 1M Input Tokens |
|:---|:---|:---:|:---:|:---:|
| **Homogeneous Baseline** | `openai/gpt-oss-120b` | **88.0%** (44/50) | 5,088.5 ms | $0.150 |
| **Asymmetric Tiered** | `openai/gpt-oss-20b` | **82.0%** (41/50) | ~5,598 ms (Avg. Latency — Adversarial Ablation Run, N=50) | **$0.075** *(50.0% Cost Reduction)* |

> **Note on latency figures:** The per-query throughput figures in the architecture table above (1,000 T/s / 500 T/s) reflect raw model inference speed as published by the API provider. The ~5,088–5,598 ms figures in this ablation table are *end-to-end benchmark-run averages* across the full adversarial suite — inclusive of network round-trip time, structured-output validation, and retry overhead — and should not be read as a regression in per-token inference speed.

#### Unit Economics & Architecture Insights
* **Tiered Routing Cost Reduction:** `openai/gpt-oss-20b` costs **$0.075** per 1M input tokens vs. **$0.15** per 1M input tokens for `openai/gpt-oss-120b`. Delegating classification to the 20B tier cuts routing token costs by an estimated **50.0%** per routing turn.
* **Cost-Accuracy Trade-off:** Routing accuracy decreased from **88.0% → 82.0%** while classification token expenditure decreased by **50.0%** ($0.075 vs. $0.150 per 1M input tokens); this is a token-cost reduction, not an accuracy or latency improvement.
* **Handling Open-Weight Structured Output Drops (`adv_37`):** During adversarial testing on nuanced parameter queries (`adv_37`), the 20B tier encountered a function-calling schema drop (`HTTP 400 tool_use_failed` with empty payload generation). Fallback/retry handlers in `src/llm.py` intercept dropped tool schemas and fall back gracefully, ensuring pipeline stability without unhandled exceptions.

---

## 4. Hybrid Retrieval & Reranking Architecture

### Ingestion Pipeline: Regex Section Splitting vs. Layout-Aware Parsers (Docling)

The local research corpus comprises 45 ArXiv Machine Learning papers (~5,500+ chunks).

* **Why Layout-Aware Parsers Were Rejected:** Heavy vision/layout parsers (e.g., Docling, Unstructured) introduce massive GPU/OCR binary dependencies and significantly slow down ingestion pipelines.
* **Why Custom Regex Chunking Was Chosen:** ArXiv ML papers feature standardized text section headers. Custom regex parsing (`src/ingest.py`) isolates structural sections (`1. Introduction`, `3. Methodology`) instantly at pure string-parsing speed while avoiding the heavy vision/OCR dependencies required by layout-aware parsers.
* **Execution:** Isolates section blocks prior to applying `RecursiveCharacterTextSplitter` (800 character size / 150 overlap) and annotates metadata (title, section, page, chunk index).

```
Raw PDF Parsing ──> Regex Section Splitting ──> Contextual Chunking ──> Metadata Annotation (Paper, Section, Page)
```

### Retrieval Strategy: Hybrid Search (BM25 + FAISS) vs. Dense Vector Search Alone

* **Why Dense Vector Search Alone Fails:** Dense embeddings (`all-MiniLM-L6-v2`) capture semantic intent well but frequently miss exact alphanumeric ML terminology, hyperparameter strings, or equation tokens (e.g., "ResNet-152", "AdamW", "p_sample_loop").
* **Why Sparse Search Alone Fails:** BM25 guarantees exact keyword hits but completely misses semantic equivalence or paraphrased user queries.
* **Why Hybrid Fusion Was Chosen:** Parallel retrieval via BM25 (sparse keyword) and FAISS (dense semantic) combined 50/50 using Reciprocal Rank Fusion (RRF) delivers both exact term precision and semantic coverage.

```
                    ┌──> BM25 (Sparse) ───┐
[ Incoming Query ] ─┤                     ├─> Reciprocal Rank Fusion ──> Cross-Encoder Rerank ──> Logit Cutoff (-2.5)
                    └──> FAISS (Dense) ───┘
```

### Reranking & Logit Filtering

* **Cross-Encoder Reranking:** Top 15 candidate chunks pass through a `ms-marco-MiniLM-L-6-v2` Cross-Encoder to compute joint semantic relevance.
* **Logit Thresholding (-2.5 Cutoff):** Chunks yielding logit relevance scores below -2.5 are pruned immediately, preventing context window bloat and filtering low-scoring candidate chunks before synthesis.

---

## 5. System Fault Tolerance & Microservice Reliability

### Multi-Token API Key Rotation & Exponential Backoff

* **`RunnableWithFallbacks` Rotation:** Automatically distributes API requests across a pool of API keys (`GROQ_API_KEY1` → `GROQ_API_KEY2`), balancing token distribution across quotas.
* **`tenacity` Backoff Decorators:** Retries failed requests using randomized exponential backoff (2^x delay), absorbing transient rate spikes or network drops.

### AST Math Evaluation via `numexpr`

* **Security:** Replaces Python `eval()` with `numexpr`'s isolated AST-based expression evaluator, avoiding the arbitrary code-execution risk associated with `eval()`.
* **Performance:** Evaluates arithmetic expressions via compiled C-level vector loops with zero system execution privileges.

### Microservice Architecture (FastAPI + Docker)

* **FastAPI Startup Lifespan:** Loads the state graph and FAISS vector index once during application boot via FastAPI's `lifespan` context manager, minimizing endpoint latency.
* **Containerization:** Deployed via a multi-stage `Python 3.12-slim` Docker image for isolated cloud execution.

---

## 6. Quantitative Evaluation Post-Mortem

System evaluation was conducted across two distinct suites:

1. **System Evaluation Benchmark — Task Routing & Execution (`eval/eval_set.json`, $N=50$):** Achieved **94.0% Task Routing / Execution Accuracy** (47/50 queries correctly routed and executed across multi-step retrieval, arithmetic, and temporal web searches). This metric grades tool-routing correctness, not independent final-answer correctness.
2. **RAGAS 0.2.x Quality Metrics:** Evaluated over ground-truth annotated retrieval contexts judged by GPT OSS 20B (`openai/gpt-oss-20b`) to avoid token-per-minute rate limit throttling. Sample size differs per metric because each metric requires different fields: Answer Relevancy needs only the final answer, Faithfulness additionally requires retrieved/web context, and Context Precision further requires a ground-truth reference and applies only to retrieval-routed questions.

### RAGAS Performance Breakdown

| Metric | Sample Size (N) | Mean Score | Median Score | Operational Diagnostic |
|---|---|---|---|---|
| **Context Precision** | 11 | **0.9091** | **1.0000** | Confirms Cross-Encoder reranking and -2.5 logit filtering successfully position true relevant chunks at top ranks. |
| **Faithfulness** | 30 | **0.7633** | **0.8167** | Indicates high adherence to retrieved context during answer generation. |
| **Answer Relevancy** | 47 | **0.6824** | **0.8356** | Median score reflects strong directness. Mean score reflects intentional lower scoring on out-of-corpus queries where the agent correctly issued refusals. |

### Metric Optimization & Remediation

Initial evaluation runs yielded a Faithfulness score of ≈0.65 due to parametric knowledge leakage during answer synthesis when retrieved chunks were thin.

* **Prompt Hardening:** Updated system synthesis prompts to mandate strict grounding: *"Answer ONLY using provided evidence. If evidence is insufficient, explicitly state missing information."*
* **Logit Cutoff Tuning:** Raised Cross-Encoder logit filter to -2.5, stripping low-confidence context before synthesis.
* **Outcome:** The combined prompt-hardening and reranker-threshold intervention lifted mean Faithfulness to **0.7633** and median Faithfulness to **0.8167**; the individual contribution of each change was not isolated.

---

## 7. Future Production Roadmap

1. **Token & Node Streaming via FastAPI (SSE):**
   Transition the FastAPI POST endpoint to Server-Sent Events (SSE) to stream intermediate graph node updates (`router` → `retrieval` → `evaluate`) and real-time answer tokens directly to the client to minimize Time-to-First-Token (TTFT).

2. **Semantic Query Caching:**
   Integrate a Redis-backed semantic caching layer upstream of the query router. Semantically equivalent queries would serve cached final responses immediately, reducing LLM token consumption, with a target of sub-50ms cache-hit latency.

3. **Asynchronous Parallel Tool Execution:**
   Refactor tool execution nodes to leverage Python `asyncio` concurrent gathering (`asyncio.gather`). When the evaluator requests multi-source evidence, execute vector search and web search concurrently rather than blocking sequential graph steps.

4. **Dynamic Multi-Corpus Ingestion API:**
   Extend the ingestion pipeline into an authenticated `/ingest` API endpoint, allowing users to upload custom PDF sets dynamically and update the FAISS index on the fly without restarting the microservice.