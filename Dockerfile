# syntax=docker/dockerfile:1
# Multi-stage build: install into a venv, then copy into a slim runtime image.
FROM python:3.12-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Create an isolated venv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (better layer caching), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

FROM python:3.12-slim AS runtime

# Run as a non-root user.
RUN useradd --create-home --uid 10001 ragforge
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RAGFORGE_HOST=0.0.0.0 \
    RAGFORGE_PORT=8000

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src

USER ragforge
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"RAGFORGE_PORT\"]}/health')" || exit 1

CMD ["uvicorn", "ragforge.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
