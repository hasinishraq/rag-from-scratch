# syntax=docker/dockerfile:1
# UI and API in one image, on port 8000. Build: docker compose build   Run: docker compose up
#
# Size: the lock file resolves torch from PyPI, which on Linux brings the CUDA libraries, so
# the image is several GB although only the CPU is used. Slimming it means locking torch to the
# CPU wheel index, which is a dependency decision (see docs/runbook.md, "Container").

# ---- 1. build the web UI ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY web/ ./
RUN npm run build

# ---- 2. the application ----
FROM python:3.11-slim AS app
# Same uv version as the one that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    RAG_LOG_FORMAT=json \
    # The embedder and reranker are downloaded once into the mounted data folder.
    HF_HOME=/app/data/models
WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY evals ./evals
COPY --from=web /web/dist ./web/dist
RUN uv sync --frozen --no-dev

# Not root: a parsing bug in an uploaded file must not own the container.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]
# The app writes its own access log (with request ids), so uvicorn's is off.
CMD ["uv", "run", "--no-sync", "uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
