# ── Stage 1: Dependencies ──────────────────────────────────────────────────────
FROM python:3.10-slim AS deps

WORKDIR /app

# Install system deps (needed for prophet, statsmodels, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Application ──────────────────────────────────────────────────────
FROM python:3.10-slim AS app

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Install runtime-only system libs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project source (datasets excluded via .dockerignore)
COPY app/ ./app/
COPY ml/ ./ml/
COPY data/ ./data/
COPY pipelines/ ./pipelines/
COPY ["datasets/AQI and Lat Long of Countries.csv", "./datasets/"]
COPY artifacts/ ./artifacts/

# Environment defaults
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV ARTIFACTS_PATH=artifacts
ENV MLFLOW_TRACKING_URI=mlruns

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
