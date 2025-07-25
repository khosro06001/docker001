# Multi-architecture Dockerfile for slm001 PyTorch native quantized chatbot
# Optimized for Jetson Nano and low-memory devices  
# Supports both AMD64 and ARM64 architectures with PyTorch native quantization ONLY
# NO bitsandbytes dependency!

FROM python:3.9-slim

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Install system dependencies
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

# Install Python dependencies - NO bitsandbytes!
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY slm001.py .

# Create a comprehensive test script
RUN echo '#!/usr/bin/env python3\n\
import torch\n\
import transformers\n\
import sys\n\
print("="*60)\n\
print("🚀 Environment Check for PyTorch Native Quantized Chatbot")\n\
print("="*60)\n\
print("PyTorch version:", torch.__version__)\n\
print("Transformers version:", transformers.__version__)\n\
print("CUDA available:", torch.cuda.is_available())\n\
if torch.cuda.is_available():\n\
    print("CUDA version:", torch.version.cuda)\n\
    print("GPU count:", torch.cuda.device_count())\n\
    for i in range(torch.cuda.device_count()):\n\
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")\n\
else:\n\
    print("Running on CPU")\n\
print("PyTorch Quantization available:", hasattr(torch.quantization, "quantize_dynamic"))\n\
try:\n\
    import accelerate\n\
    print("accelerate version:", accelerate.__version__)\n\
except ImportError:\n\
    print("accelerate: Not available")\n\
print("Quantization method: PyTorch Native (NO bitsandbytes)")\n\
print("="*60)\n\
print("✅ Environment check completed successfully!")\n\
print("🤖 Ready to run PyTorch native quantized chatbot")\n\
print("="*60)' > test_env.py && chmod +x test_env.py

# Verify installation
RUN python test_env.py

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