# Multi-stage build for production optimization
FROM python:3.11.9-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Production image ---
FROM python:3.11.9-slim as production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    ENVIRONMENT=production \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

COPY --from=builder /opt/venv /opt/venv

RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

COPY . /app/

RUN mkdir -p /app/templates /app/static /app/reports /app/logs /app/data && \
    chown -R appuser:appuser /app && chmod -R 755 /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--max-requests", "10000", \
     "--max-requests-jitter", "1000", \
     "--preload", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
