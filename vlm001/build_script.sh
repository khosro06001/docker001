#!/bin/bash

# Docker buildx script to build and deploy multi-architecture image
# Builds for both AMD64 and ARM64 architectures
# Updated for Ollama-based MiniCPM-V image captioning

set -e

# Configuration
# IMAGE_NAME="your-dockerhub-username/vlm-ollama-captioning"
IMAGE_NAME="khosro123/vlm-ollama-captioning"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "=== Docker Multi-Architecture Build and Deploy ==="
echo "Image: $FULL_IMAGE_NAME"
echo "Target: Ollama-based MiniCPM-V Image Captioning"
echo "Date: $(date)"

# Check if buildx is available
if ! docker buildx version >/dev/null 2>&1; then
    echo "Error: Docker buildx is not available"
    echo "Please install Docker buildx or use Docker Desktop"
    exit 1
fi

# Create and use buildx builder if it doesn't exist
BUILDER_NAME="vlm-ollama-builder"
if ! docker buildx inspect $BUILDER_NAME >/dev/null 2>&1; then
    echo "Creating buildx builder: $BUILDER_NAME"
    docker buildx create --name $BUILDER_NAME --driver docker-container --bootstrap
fi

echo "Using buildx builder: $BUILDER_NAME"
docker buildx use $BUILDER_NAME

# Verify required files exist
required_files=("vlm_ollama_minicpm-v.py" "requirements.txt" "Dockerfile" "healthcheck.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file '$file' not found"
        exit 1
    fi
done

# Login to Docker Hub (if not already logged in)
if ! docker info | grep -q "Username:"; then
    echo "Please login to Docker Hub:"
    docker login
fi

echo ""
echo "=== Building multi-architecture image ==="
echo "This will build for AMD64 and ARM64 architectures"
echo "Building may take 10-20 minutes..."
echo ""

# Build and push multi-architecture image
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag $FULL_IMAGE_NAME \
    --push \
    .

echo ""
echo "=== Build completed successfully! ==="
echo "Image pushed to Docker Hub: $FULL_IMAGE_NAME"
echo ""
echo "=== Usage Instructions ==="
echo ""
echo "IMPORTANT: This container requires Ollama to be running with minicpm-v:8b model"
echo ""
echo "Prerequisites:"
echo "1. Install and run Ollama on the host system"
echo "2. Pull the model: ollama pull minicpm-v:8b"
echo "3. Start Ollama server: ollama serve"
echo ""
echo "To run the container:"
echo ""
echo "# Basic usage (connects to host Ollama)"
echo "docker run --rm -it --network host \\"
echo "  -v /path/to/your/images:/app/data \\"
echo "  $FULL_IMAGE_NAME \\"
echo "  python3 vlm_ollama_minicpm-v.py /app/data/your_image.jpg"
echo ""
echo "# With custom question"
echo "docker run --rm -it --network host \\"
echo "  -v /path/to/your/images:/app/data \\"
echo "  $FULL_IMAGE_NAME \\"
echo "  python3 vlm_ollama_minicpm-v.py /app/data/your_image.jpg \\"
echo "  --question \"Describe this image in greatest detail please.\""
echo ""
echo "# For remote Ollama server"
echo "docker run --rm -it \\"
echo "  -v /path/to/your/images:/app/data \\"
echo "  $FULL_IMAGE_NAME \\"
echo "  python3 vlm_ollama_minicpm-v.py /app/data/your_image.jpg \\"
echo "  --url http://your-ollama-server:11434"
echo ""
echo "=== Next steps ==="
echo "1. Set up Ollama on your target system"
echo "2. Pull the minicpm-v:8b model"
echo "3. Test the container with your images"
echo "4. Deploy to Jetson Nano or other ARM64 systems"
