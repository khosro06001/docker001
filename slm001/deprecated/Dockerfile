# Multi-architecture Dockerfile for slm001 quantized chatbot
# Optimized for Jetson Nano and low-memory devices
# Supports both AMD64 and ARM64 architectures

FROM python:3.9-slim

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Install system dependencies including build tools for bitsandbytes
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create cache directory with proper permissions
RUN mkdir -p /app/cache && chmod 755 /app/cache

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations for ARM64
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY slm001.py .
COPY test_local.py .

# Verify files are copied (for debugging)
RUN ls -la /app/

# Create non-root user for security
RUN useradd -m -u 1000 chatbot && \
    chown -R chatbot:chatbot /app

# Set resource limits for Jetson Nano
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=100000

USER chatbot

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import transformers; print('OK')" || exit 1

# Expose port (if needed for future web interface)
EXPOSE 8080

# Set the default command
CMD ["python", "slm001.py"]