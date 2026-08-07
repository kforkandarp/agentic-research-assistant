# 🔬 Agentic Research Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LangGraph-agentic%20loop-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Retrieval-BM25%20+%20FAISS%20+%20rerank-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Deployed-Streamlit%20Cloud-red?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/Docker-containerized-blue?style=flat-square&logo=docker" />
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi" />
</p>

**Self-correcting single-agent LangGraph system** that routes queries across hybrid retrieval, web search, calculator, and direct answer — over a corpus of 5 foundational ArXiv ML papers. Features a deterministic evaluate loop, streaming Streamlit UI, production FastAPI wrapper, and a Dockerized deployment.

---

## 🚀 Live Demo

| Resource | Link |
|---|---|
| 🎯 Interactive Streamlit UI | [agentic-research-assistant-k.streamlit.app](https://agentic-research-assistant-k.streamlit.app/) |
| 📂 GitHub Repository | [github.com/kforkandarp/agentic-research-assistant](https://github.com/kforkandarp/agentic-research-assistant) |

> App may take 30–60 seconds to wake from sleep on first visit (Streamlit free tier). Subsequent queries respond in ~5–10s depending on tool chain length.

---

## 🏗️ Architecture

![Graph Diagram](graph_diagram.png)

The agent follows a **router → tool → evaluate → (loop or synthesize)** pattern — a practical implementation of the Corrective RAG architecture:

1. **Router** (LLaMA 3.1 8B) — classifies the query into one of four tools: `retrieval`, `web_search`, `calculator`, or `direct_answer`
2. **Tool nodes** — execute the selected tool and append results to shared state via `operator.add` accumulation
3. **Evaluate node** (LLaMA 3.3 70B) — grades evidence sufficiency. If insufficient, routes back to an untried tool. If sufficient, routes to synthesis
4. **Deterministic guardrails** — override the LLM when computation signals are detected (regex), prevent tool repetition, and enforce a `MAX_EVIDENCE_STEPS=4` hard cap
5. **Synthesize node** — produces a grounded answer from accumulated tool outputs, with a caveat injected into the system prompt when evidence was judged insufficient

**Why merge router and evaluator into one node?** Two separate LLM nodes making routing decisions independently caused conflicting outputs — one saying "done", another saying "fetch more". Merging them into a single `evaluate_node` with a clear decision contract eliminated the conflict.

**Why tiered models?** Router is a simple 4-class classification — the 8B model handles it correctly every time and costs fewer tokens. Evaluator requires multi-step reasoning over accumulated evidence — the 8B model failed harder cases, so 70B is used there and for synthesis.

---

## 📊 Evaluation Results

### Routing Accuracy

Evaluated on a **19-question hand-crafted eval set** spanning all tool categories and multi-step chains, plus a **10-question holdout set** run exactly once after all tuning was complete.

| Set | Correct | Total | Accuracy |
|---|---|---|---|
| **Eval set** | 14 | 19 | **73.7%** |
| **Holdout set** | 7 | 10 | **70.0%** |

Misses on both sets are concentrated in two honest failure patterns — not hallucinations:
- **Corpus gap** — ResNet/VGG-19 comparison questions where neither the local corpus nor Tavily returned the specific figures needed; agent correctly returns `insufficient_information` rather than fabricating
- **Multi-step complexity** — questions requiring 3+ tool calls where Groq free-tier rate limits cut the chain short

### RAGAS Quality Metrics

Evaluated using **LLaMA 3.3 70B as judge** via RAGAS on eligible eval set entries.

| Metric | Score | Eligible Questions | What It Measures |
|---|---|---|---|
| **Answer Relevancy** | **0.903** | 12 of 14 | How directly the answer addresses the question |
| **Faithfulness** | **0.651** | 8 | % of answer claims grounded in retrieved evidence |
| **Context Precision** | **0.667** | 3 | % of retrieved chunks that were genuinely useful |

> Answer Relevancy excludes 2 `insufficient_information` entries where the agent correctly refused to answer — these score 0.0 by RAGAS design since there is no substantive answer to evaluate. The 0.903 reflects actual answer quality on questions the agent did answer.

> Faithfulness of 0.651 reflects the multi-tool challenge: when retrieval + web_search + calculator outputs are concatenated, the synthesizer occasionally includes claims from one source that RAGAS judges as unsupported by another. Single-tool retrieval entries score 0.86–1.0.

> Context Precision is computed on only 3 pure-retrieval questions with ground truth — limited sample, treat as directional only.

---

## 🖥️ Deployment Architecture

This project has two deployment surfaces built and verified:

### Streamlit Community Cloud — Public Demo
The Streamlit UI (`app.py`) is deployed publicly on Streamlit Community Cloud. It uses `app.stream()` with node-level updates so each tool firing appears in real time — the router decision, each tool call, the evaluate note between steps, and the final synthesized answer all stream in as they happen rather than appearing all at once.

### FastAPI — Production REST API
A production-grade FastAPI wrapper (`main.py`) wraps the same LangGraph agent and exposes two endpoints:

- `GET /health` — liveness check, returns `graph_loaded: true` once the retriever and agent are initialized
- `POST /query` — accepts a natural language query, runs the full agentic loop via `app.invoke()`, and returns structured JSON

Key design decisions in `main.py`:
- **Graph built once at startup** via FastAPI's `lifespan` context manager — same pattern as `@st.cache_resource` in Streamlit. Avoids rebuilding the FAISS index + BM25 index + embedding model on every request
- **Pydantic request/response models** — `QueryRequest` validates incoming JSON, `QueryResponse` guarantees a consistent schema with `final_answer`, `tools_used`, `routing_reason`, `sufficient`, and `missing_info`
- **Structured error handling** — `503` if the graph failed to build at startup, `500` with a clean message if the graph throws during invocation. Raw tracebacks never reach the caller
- **Full logging** — every request logs the query, every response logs tools used, sufficiency, and answer length. Visible in any cloud provider's log stream

Verified locally: `/health` returns `{"status": "ok", "graph_loaded": true}`, `/query` returns correct structured JSON with the full tool chain, interactive docs available at `/docs`.

### Docker — Self-Hosted / Cloud Deployment
The FastAPI app is fully Dockerized. The image was built and run locally with the agent working end-to-end inside the container — FAISS index loading from a volume mount, Groq API reachable, `/health` and `/query` both verified.

Key Dockerfile decisions:
- **Python 3.12-slim** — matches the development environment exactly; avoids the numpy/torch wheel compatibility issues that arise from mismatched Python versions
- **Layer caching** — `requirements.txt` is copied and installed before application code, so code changes don't trigger a full pip reinstall on rebuild
- **`PYTHONUNBUFFERED=1`** — logs appear immediately in the container output instead of buffering; critical for debugging on cloud platforms
- **`--workers 1`** — Groq free tier has per-minute token limits; multiple workers would amplify 429 errors rather than improve throughput

> HuggingFace Spaces Docker SDK now requires a paid subscription for new Spaces, so the public demo runs on Streamlit Community Cloud instead. The Docker image remains the production artifact for any container-based cloud deployment (Render, Railway, GCP Cloud Run, AWS ECS).

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph (StateGraph, conditional edges, operator.add accumulation) |
| **LLM — Router** | LLaMA 3.1 8B via Groq API |
| **LLM — Eval + Synthesis** | LLaMA 3.3 70B via Groq API |
| **Embeddings** | all-MiniLM-L6-v2 (Sentence Transformers) |
| **Vector Store** | FAISS (CPU) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Retrieval Fusion** | LangChain EnsembleRetriever (equal weights) |
| **Reranking** | ms-marco-MiniLM-L-6-v2 (Cross-Encoder) |
| **Web Search** | Tavily API |
| **Calculator** | numexpr (safe arithmetic — no eval()) |
| **Chunking** | PyPDFLoader + RecursiveCharacterTextSplitter (section-aware) |
| **Evaluation** | RAGAS + custom routing accuracy framework |
| **Observability** | LangSmith |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit (node-level streaming) |
| **Containerization** | Docker (Python 3.12-slim) |
| **Deployment** | Streamlit Community Cloud |

---

## 📄 Local Corpus

725 chunks across 5 foundational ML papers, section-aware chunked (no chunk straddles a section boundary):

| Paper | Sections |
|---|---|
| Attention Is All You Need | 9 |
| ResNet | 7 |
| BERT | 7 |
| DDPM | 8 |
| GPT-3 | 11 |

---

## 📡 API Reference

### `GET /health`

```json
{
  "status": "ok",
  "graph_loaded": true
}
```

### `POST /query`

**Request:**
```json
{
  "query": "What BLEU score did the base Transformer achieve, and how many times larger is GPT-3 than GPT-2?"
}
```

**Response:**
```json
{
  "final_answer": "The base Transformer achieved 27.3 BLEU on WMT 2014 EN-DE...",
  "tools_used": ["retrieval", "web_search", "calculator"],
  "routing_reason": "Question is about specific facts from the local corpus",
  "sufficient": true,
  "missing_info": ""
}
```

**curl example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What attention mechanism does the Transformer paper use?"}'
```

Interactive docs at `http://localhost:8000/docs` when running locally.

---

## ⚙️ Local Setup

```bash
# Clone
git clone https://github.com/kforkandarp/agentic-research-assistant.git
cd agentic-research-assistant

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Environment variables — create .env file
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-research-assistant

# Add your ArXiv PDFs to data/raw_pdfs/ then chunk them
python -m src.chunking

# Run Streamlit UI
streamlit run app.py

# OR run FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker

```bash
# Build
docker build -t agentic-research-assistant .

# Run (PowerShell)
docker run -p 8000:7860 --env-file .env -v "${PWD}/data:/app/data" agentic-research-assistant

# Run (CMD)
docker run -p 8000:7860 --env-file .env -v "%cd%/data:/app/data" agentic-research-assistant
```

Health check at `http://localhost:8000/health` — interactive docs at `http://localhost:8000/docs`.

---

## 🔮 Future Improvements

- **HuggingFace Spaces deployment** — Docker SDK now requires paid tier; deploy there once on a Pro plan for a public API endpoint alongside the Streamlit demo
- **Load testing** — k6 or Locust benchmarks against the FastAPI endpoint reporting p50/p95 latency and requests/sec under concurrent load
- **Streaming FastAPI** — replace `app.invoke()` with `app.astream()` and expose a WebSocket or SSE endpoint for token-level streaming via the REST API
- **FAISS index persistence** — pre-build the index and commit it to the repo (or store in S3/GCS) to eliminate the cold-start rebuild on Streamlit Cloud
- **Larger RAGAS judge model** — re-run RAGAS evaluation with GPT-4o or a larger open model for more reliable faithfulness and context precision scores
- **LangSmith trace integration** — add trace screenshots to this README showing the full tool-call DAG for representative queries
- **Multi-corpus support** — extend beyond the fixed 5-paper corpus to support user-uploaded PDFs with dynamic chunking and index updates
- **Expand eval set** — current 19-question eval set is sufficient for development but small for production confidence; expand to 50+ questions with more edge cases

---

## 🔍 Known Limitations

- **Corpus gap** — ResNet/VGG-19 cross-paper comparisons fail because VGG-19 figures are not in the local corpus and Tavily doesn't reliably surface the exact numbers; agent correctly admits `insufficient_information` rather than fabricating
- **Groq free-tier rate limits** — long multi-step chains occasionally hit the token-per-day limit mid-run; handled gracefully via tenacity retry with exponential backoff
- **FAISS rebuild on cold start** — Streamlit Cloud's ephemeral filesystem means the FAISS index rebuilds from `chunks.json` on each cold start (~20–30 seconds on first query)
- **Context Precision sample** — only 3 questions were eligible for context precision scoring; treat that metric as directional only

---

## 📄 License

MIT