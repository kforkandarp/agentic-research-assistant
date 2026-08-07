# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# These are OS-level packages (not Python packages) that some Python libs need
# at install time or runtime:
#   build-essential → gcc, g++, make — needed to compile C extensions (e.g. numexpr, tokenizers)
#   curl            → used below to health-check the server during builds; also useful for debugging
#   git             → some pip installs pull from GitHub directly
# "no-install-recommends" skips optional extras → smaller image.
# We clean apt cache at the end of the same RUN layer to avoid bloating the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container.
# Files we COPY in will land here unless we say otherwise.
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# IMPORTANT: We copy requirements.txt FIRST and install BEFORE copying the rest
# of the code. Why? Docker builds layers. If requirements.txt hasn't changed,
# Docker reuses the cached install layer — even if your code changed.
# This means "docker build" takes 2 minutes the first time but ~10 seconds
# on every subsequent code-only change. If you copy all code first, Docker
# reinstalls everything on every single build.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
# Now copy everything else. This layer re-runs on every code change,
# but that's fast because it's just a file copy, not a pip install.
# The .dockerignore file (you should create one) keeps out __pycache__,
# .env, eval results, and other things the container doesn't need.
COPY . .

# ── HuggingFace Spaces port ───────────────────────────────────────────────────
# HuggingFace Spaces REQUIRES port 7860. This is non-negotiable.
# EXPOSE is documentation — it tells Docker/Spaces which port to route traffic to.
# The actual binding happens in the CMD below.
EXPOSE 7860

# ── Environment defaults ──────────────────────────────────────────────────────
# These are fallback values only. Real secrets (GROQ_API_KEY, TAVILY_API_KEY)
# must be set as HuggingFace Space Secrets — NEVER hardcoded here.
# Setting them here as empty strings means the app starts without crashing
# on import, and fails with a clear error at the first LLM call instead.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GROQ_API_KEY="" \
    TAVILY_API_KEY=""

# PYTHONUNBUFFERED=1    → Python prints logs immediately instead of buffering them.
#                         Without this, you see nothing in HF Spaces logs until the
#                         buffer flushes — makes debugging painful.
# PYTHONDONTWRITEBYTECODE=1 → Don't create .pyc files inside the container.
#                              Keeps the filesystem clean.

# ── Start command ─────────────────────────────────────────────────────────────
# uvicorn is the ASGI server that runs FastAPI.
#   main:app        → file "main.py", variable "app" (the FastAPI() instance)
#   --host 0.0.0.0  → listen on all network interfaces, not just localhost.
#                     Without this, the container is unreachable from outside.
#   --port 7860     → must match EXPOSE above and HF Spaces requirement
#   --workers 1     → one worker process. The Groq free tier has rate limits,
#                     so multiple workers would just cause more 429s.
#                     Scale up only if you move to a paid tier.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]