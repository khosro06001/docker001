# Multi-stage Dockerfile for Gemma2 Interactive Chatbot via Ollama
# Supports both AMD64 (CPU-only) and ARM64 (GPU-accelerated for Jetson Nano)

FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3 AS base-arm64
FROM python:3.9-slim AS base-amd64

# Select base image based on target architecture
FROM base-${TARGETARCH} AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create app directory and user
WORKDIR /app
RUN groupadd -r chatuser && useradd -r -g chatuser chatuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    ca-certificates \
    readline-common \
    libreadline8 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_gemma2.txt .
RUN pip install --no-cache-dir -r requirements_gemma2.txt

# Copy application files
COPY ollama_gemma2_chatbot.py .
RUN chmod +x ollama_gemma2_chatbot.py

# Create data directories for conversation storage
RUN mkdir -p /app/data /app/conversations && \
    chown -R chatuser:chatuser /app

# Health check
COPY healthcheck_gemma2.py .
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python3 healthcheck_gemma2.py

# Switch to non-root user
USER chatuser

# Expose port (for potential future web interface)
EXPOSE 8080

# Default command - starts interactive chatbot
CMD ["python3", "ollama_gemma2_chatbot.py", "--verbose"]

# Labels
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Gemma2 Interactive Chatbot via Ollama for Multi-Architecture"
LABEL version="1.0"
LABEL ollama.required="true"
LABEL ollama.model="gemma2:2b-instruct-q4_0"
LABEL architecture.amd64="CPU-only"
LABEL architecture.arm64="GPU-accelerated for Jetson"