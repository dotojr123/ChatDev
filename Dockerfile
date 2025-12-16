# Stage 1: Builder
FROM python:3.12-slim-bookworm as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install runtime dependencies (including tk for legacy GUI support if strictly needed, though headless is preferred)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m chatdev && chown -R chatdev:chatdev /app
USER chatdev

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command (can be overridden)
ENTRYPOINT ["python", "run.py"]
