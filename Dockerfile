# ── Base Image ────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── System Dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code & Data ──────────────────────────────────────────
COPY . .

# ── Expose FastAPI Port ───────────────────────────────────────────────────
EXPOSE 8000

# ── Environment Defaults ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Start Command (FastAPI Backend) ───────────────────────────────────────
# Runs app.py (FastAPI app) on port 8000
# ── Start Command (FastAPI Backend) ───────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]