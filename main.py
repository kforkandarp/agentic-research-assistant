"""
FastAPI wrapper for the Agentic Research Assistant.
Endpoints:
  GET  /health  — liveness check (used by HuggingFace Spaces and load balancers)
  POST /query   — runs the full LangGraph agent and returns a structured response

The compiled graph is built ONCE at server startup (inside the lifespan context)
and stored on app.state so every request reuses the same object.
Building it per-request would re-load embeddings + BM25 index on every call — very slow.
"""

import logging
import traceback
from contextlib import asynccontextmanager # it creates a startup/shutdown lifecycle.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import build_graph
from src.state import AgentState

# ── Logging ───────────────────────────────────────────────────────────────────
# basicConfig sets the format for ALL log messages in this process.
# level=INFO means we see INFO, WARNING, ERROR — but not DEBUG spam.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# logger is an object responsible for sending messages to the logging system.

# ── Lifespan — build graph once at startup ────────────────────────────────────
# asynccontextmanager turns this function into a context manager that FastAPI
# calls automatically: everything BEFORE `yield` runs at startup,
# everything AFTER `yield` runs at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building LangGraph agent...")
    try:
        app.state.graph = build_graph()
        logger.info("Graph ready.")
    except Exception as e:
        # If the graph fails to build (e.g. missing API key, corrupt index),
        # log clearly and let the server start anyway — /health will still
        # respond, and /query will return a clean 503 instead of a raw crash.
        logger.error(f"Graph build failed: {e}")
        app.state.graph = None
    yield
    # Shutdown — nothing to clean up for this project (no DB connections etc.)
    logger.info("Server shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Research Assistant",
    description=(
        "Single-agent LangGraph system over 5 ArXiv ML papers. "
        "Hybrid retrieval (BM25 + FAISS + rerank), Tavily web search, "
        "calculator, and direct-answer routing — all on Groq."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows browser-based clients (e.g. a React frontend) to call this API.
# For HuggingFace Spaces, allow_origins=["*"] is fine since it's a public demo.
# In production with sensitive data, replace "*" with your exact frontend URL.


# CORS — not required for the current Streamlit setup (Streamlit is server-side Python,
# not a browser JS client, so the browser's same-origin policy never applies here).
# Included as a forward-looking addition: if a JavaScript/React frontend is added later,
# this middleware is already in place. Safe to remove if keeping Streamlit-only forever.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # since this is a demo, we allow every origin, we make it a list because prod needs multiple origins that can access it
    allow_credentials=True, # credentials are Things proving who you are.
    allow_methods=["*"], # allow HTTP method like get, post, delete, put
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
# Pydantic models do two things for us:
#   1. Validate incoming JSON automatically (wrong type → 422 error with clear message)
#   2. Generate the /docs schema so anyone can see what to send and what they get back

class QueryRequest(BaseModel):
    query: str = Field(
        ...,                        # "..." means this field is REQUIRED
        min_length=3,
        max_length=2000,
        description="The question to send to the agent.",
        examples=["What attention mechanism does the Transformer paper use?"],
    )


class QueryResponse(BaseModel):
    final_answer: str = Field(description="The agent's synthesized answer.")
    tools_used: list[str] = Field(
        description="Ordered list of tools the agent called (e.g. ['retrieval', 'calculator'])."
    )
    routing_reason: str | None = Field(
        description="Why the router picked the first tool. None if routing failed."
    )
    sufficient: bool = Field(
        description="True if the agent judged its evidence sufficient to fully answer the query."
    )
    missing_info: str = Field(
        description="What was missing from the evidence, if sufficient=False. Empty string otherwise."
    )


class HealthResponse(BaseModel):
    status: str
    graph_loaded: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get( # tells fastAPI that function below handles GET requests to the /health endpoint
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["System"],
)
def health():
    """
    Returns 200 as long as the server is running.
    Also tells you whether the graph loaded successfully.
    HuggingFace Spaces pings this to decide whether to mark the Space as healthy.
    """
    return HealthResponse(
        status="ok",
        graph_loaded=app.state.graph is not None,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Run the agentic research assistant",
    tags=["Agent"],
)
def query(request: QueryRequest):
    """
    Accepts a natural-language question, runs the full LangGraph agent
    (router → tool(s) → evaluate → synthesize), and returns the structured result.

    The agent may call multiple tools in sequence before synthesizing — that is
    all handled internally. This endpoint always returns a single JSON response
    when the agent is done (not streaming). Use the Streamlit UI for streaming.
    """
    # Guard: graph failed to build at startup
    if app.state.graph is None:
        raise HTTPException(
            status_code=503, # this means server temporary unavailable, not a client error
            detail=(
                "Agent graph is not available. "
                "Check server logs for the startup error (likely a missing API key or index file)."
            ),
        )

    logger.info(f"Received query: {request.query!r}")

    # Build the initial state — exactly the same structure as before files made: app.py and run_eval.py
    initial_state: AgentState = {
        "query": request.query,
        "next_tool": None,
        "routing_reason": None,
        "tool_outputs": [],
        "final_answer": None,
        "missing_info": "",
        "_grade_sufficient": True,
    }

    try:
        # app.invoke() runs the full graph synchronously and returns the final state.
        # This is the right call for a REST endpoint — we wait for the complete answer
        # before sending the response (unlike app.stream() which Streamlit uses).
        result: AgentState = app.state.graph.invoke(initial_state) # whole ai system executes here

    except Exception as e:
        # Log the full traceback server-side for debugging,
        # but return a clean message to the caller — never expose raw tracebacks.
        logger.error(f"Graph invocation failed:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}",
        )

    # Extract the fields we care about from the final state
    tools_used: list[str] = [record["tool"] for record in result.get("tool_outputs", [])]
    final_answer: str = result.get("final_answer") or "No answer was produced."
    routing_reason: str | None = result.get("routing_reason")
    sufficient: bool = result.get("_grade_sufficient", True)
    missing_info: str = result.get("missing_info") or ""

    logger.info(
        f"Query complete | tools={tools_used} | sufficient={sufficient} | "
        f"answer_len={len(final_answer)}"
    )

    return QueryResponse(
        final_answer=final_answer,
        tools_used=tools_used,
        routing_reason=routing_reason,
        sufficient=sufficient,
        missing_info=missing_info,
    )