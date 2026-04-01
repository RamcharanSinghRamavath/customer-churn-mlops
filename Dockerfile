# ── Base ────────────────────────────────────────────────────────────────────
FROM python:3.9-slim

# Metadata
LABEL maintainer="Your Name <your@email.com>"
LABEL project="Customer Churn MLOps"

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project ─────────────────────────────────────────────────────────────
COPY api/     ./api/
COPY models/  ./models/
COPY src/     ./src/

# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start server ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
