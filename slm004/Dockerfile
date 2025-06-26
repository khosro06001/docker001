# Lightweight Multi-stage Dockerfile for Gemma2 Interactive Chatbot
# Connects to host-based Ollama server
# Supports both AMD64 (CPU-only) and ARM64 (GPU-accelerated for Jetson Nano)

# Build stage for ARM64 (Jetson) - using Python base with build tools
FROM python:3.9-slim AS base-arm64-build
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build stage for AMD64 - standard Python slim with build tools
FROM python:3.9-slim AS base-amd64-build
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Select appropriate build stage
FROM base-${TARGETARCH}-build AS builder

# Install Python dependencies in builder stage
COPY requirements_gemma2.txt /tmp/
RUN pip3 install --no-cache-dir --user -r /tmp/requirements_gemma2.txt

# Runtime stage - minimal final image
FROM python:3.9-slim AS runtime-amd64
FROM python:3.9-slim AS runtime-arm64

# Select runtime base
FROM runtime-${TARGETARCH} AS runtime

# Install only essential runtime packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONPATH=/home/chatuser/.local/lib/python3.9/site-packages

# Create non-root user
RUN groupadd -r chatuser && useradd -r -g chatuser -s /bin/bash chatuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/chatuser/.local
RUN chown -R chatuser:chatuser /home/chatuser/.local

# Create app directory
WORKDIR /app
RUN mkdir -p /app/conversations /app/data && \
    chown -R chatuser:chatuser /app

# Copy application files
COPY slm_chatbot_gemma2.py /app/
COPY healthcheck_gemma2.py /app/
RUN chmod +x /app/slm_chatbot_gemma2.py && \
    chown -R chatuser:chatuser /app

# Switch to non-root user
USER chatuser

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD python3 /app/healthcheck_gemma2.py || exit 1

# Expose port for potential future web interface
EXPOSE 8080

# Default command - starts interactive chatbot
CMD ["python3", "/app/slm_chatbot_gemma2.py", "--verbose"]

# Metadata
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Lightweight Gemma2 Interactive Chatbot connecting to host Ollama"
LABEL version="1.0"
LABEL ollama.required="host"
LABEL ollama.model="gemma2:2b-instruct-q4_0"
LABEL architecture.amd64="CPU-only, connects to host Ollama"
LABEL architecture.arm64="GPU-accelerated for Jetson, connects to host Ollama"