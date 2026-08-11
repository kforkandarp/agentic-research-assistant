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
* **System Resilience:** Design for zero single-point-of-failure API dependencies through automated key rotation and exponential backoff retry strategies.
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

### Decision: LLaMA 3.1 8B (Router) vs. LLaMA 3.3 70B (Evaluator & Synthesizer)

Deploying uniform, high-parameter LLMs across all state nodes introduces unnecessary token expense and processing overhead.

| Graph Node | Model Assignment | Latency Profile | Engineering Rationale |
|---|---|---|---|
| **Router** | LLaMA 3.1 8B | Under 200 ms | 4-class single-step query classification via Pydantic structured output requires minimal parameter capacity. 8B yields 100% task fidelity. |
| **Evaluate** | LLaMA 3.3 70B | ~800 ms | Multi-step evidence sufficiency evaluation requires high-capacity reasoning across accumulated state context. |
| **Synthesize** | LLaMA 3.3 70B | ~1,200 ms | Final answer synthesis demands precise adherence to grounded prompt instructions to eliminate parametric knowledge leakage. |

### Unit Economics Calculation

* **Base API Pricing Ratio:** LLaMA 8B costs ≈$0.05–$0.08 per 1M tokens vs. ≈$0.59–$0.70 per 1M tokens for LLaMA 70B.
* **Savings Impact:** Routing queries through the 8B model reduces routing token costs by **≈85%** compared to a homogeneous 70B pipeline, while maintaining a **90.0%** routing accuracy across 50 benchmark test cases.

---

## 4. High-Precision Retrieval Architecture

### Ingestion Pipeline: Regex Section Splitting vs. Layout-Aware Parsers (Docling)

The local research corpus comprises 45 ArXiv Machine Learning papers (~5,500+ chunks).

* **Why Layout-Aware Parsers Were Rejected:** Heavy vision/layout parsers (e.g., Docling, Unstructured) introduce massive GPU/OCR binary dependencies and significantly slow down ingestion pipelines.
* **Why Custom Regex Chunking Was Chosen:** ArXiv ML papers feature standardized text section headers. Custom regex parsing (`src/ingest.py`) isolates structural sections (`1. Introduction`, `3. Methodology`) instantly at pure string-parsing speed with zero heavy ML dependencies.
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
* **Logit Thresholding (-2.5 Cutoff):** Chunks yielding logit relevance scores below -2.5 are pruned immediately, preventing context window bloat and eliminating weak relevance noise.

---

## 5. System Fault Tolerance & Microservice Reliability

### Multi-Token API Key Rotation & Exponential Backoff

* **`RunnableWithFallbacks` Rotation:** Automatically distributes API requests across a pool of API keys (`GROQ_API_KEY1` → `GROQ_API_KEY2`), balancing token distribution across quotas.
* **`tenacity` Backoff Decorators:** Retries failed requests using randomized exponential backoff (2^x delay), absorbing transient rate spikes or network drops.

### AST Math Evaluation via `numexpr`

* **Security:** Replaces dangerous Python `eval()` calls (code injection risk) with isolated AST parsing.
* **Performance:** Evaluates arithmetic expressions via compiled C-level vector loops with zero system execution privileges.

### Microservice Architecture (FastAPI + Docker)

* **FastAPI Startup Lifespan:** Loads the state graph and FAISS vector index once during application boot via FastAPI's `lifespan` context manager, minimizing endpoint latency.
* **Containerization:** Deployed via a multi-stage `Python 3.12-slim` Docker image for isolated cloud execution.

---

## 6. Quantitative Evaluation Post-Mortem

System evaluation was conducted using a **50-Question Benchmark Suite** and **RAGAS 0.2.x Quality Metrics** judged by LLaMA 3.3 70B.

### RAGAS Performance Breakdown

| Metric | Sample Size (N) | Mean Score | Median Score | Operational Diagnostic |
|---|---|---|---|---|
| **Context Precision** | 11 | **0.9091** | **1.0000** | Confirms Cross-Encoder reranking and -2.5 logit filtering successfully position true relevant chunks at top ranks. |
| **Faithfulness** | 30 | **0.7633** | **0.8167** | Indicates high adherence to retrieved context during answer generation. |
| **Answer Relevancy** | 47 | **0.6824** | **0.8356** | Median score reflects strong directness. Mean score reflects intentional lower scoring on out-of-corpus queries where the agent correctly issued refusals. |

### Metric Optimization & Remediation

Initial evaluation runs yielded a Faithfulness score of ≈0.65 due to LLaMA 3.3 70B mixing parametric knowledge into answer synthesis when retrieved chunks were thin.

* **Prompt Hardening:** Updated system synthesis prompts to mandate strict grounding: *"Answer ONLY using provided evidence. If evidence is insufficient, explicitly state missing information."*
* **Logit Cutoff Tuning:** Raised Cross-Encoder logit filter to -2.5, stripping low-confidence context before synthesis.
* **Outcome:** Lifted mean Faithfulness to **0.7633** and median Faithfulness to **0.8167**.

---

## 7. Future Production Roadmap

1. **Token & Node Streaming via FastAPI (SSE):**
   Transition the FastAPI POST endpoint to Server-Sent Events (SSE) to stream intermediate graph node updates (`router` → `retrieval` → `evaluate`) and real-time answer tokens directly to the client to minimize Time-to-First-Token (TTFT).

2. **Semantic Query Caching:**
   Integrate a Redis-backed semantic caching layer upstream of the query router. Semantically equivalent queries will serve cached final responses immediately, reducing LLM token consumption and delivering sub-50ms response latencies.

3. **Asynchronous Parallel Tool Execution:**
   Refactor tool execution nodes to leverage Python `asyncio` concurrent gathering (`asyncio.gather`). When the evaluator requests multi-source evidence, execute vector search and web search concurrently rather than blocking sequential graph steps.

4. **Dynamic Multi-Corpus Ingestion API:**
   Extend the ingestion pipeline into an authenticated `/ingest` API endpoint, allowing users to upload custom PDF sets dynamically and update the FAISS index on the fly without restarting the microservice.