# Multi-stage build for Cognitive Mirror
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production image
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r cognitive && useradd -r -g cognitive cognitive

# Copy installed packages from builder
COPY --from=builder /root/.local /home/cognitive/.local
ENV PATH=/home/cognitive/.local/bin:$PATH

# Copy application code
COPY --chown=cognitive:cognitive . .

# Switch to non-root user
USER cognitive

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/v1/health')"

EXPOSE 5000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
