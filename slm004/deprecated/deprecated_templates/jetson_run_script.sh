#!/bin/bash

# Script to pull and run the image captioning container on Jetson Nano
# Uses nvidia-docker with GPU acceleration

set -e

# Configuration
IMAGE_NAME="khosro123/vlm-image-captioning:latest"
CONTAINER_NAME="vlm-captioner"
DATA_DIR="$HOME/vlm_data"
CACHE_DIR="$HOME/vlm_cache"

echo "=== VLM Image Captioning - Jetson Nano Deployment ==="
echo "Image: $IMAGE_NAME"
echo "Date: $(date)"

# Check if running on Jetson
if ! lsmod | grep -q nvgpu; then
    echo "Warning: This doesn't appear to be a Jetson device (nvgpu module not found)"
    echo "Continuing anyway..."
fi

# Check if nvidia-docker is available
if ! which nvidia-docker >/dev/null 2>&1; then
    echo "Error: nvidia-docker not found"
    echo "Please install nvidia-docker runtime for Jetson"
    exit 1
fi

# Create data directories
echo "Creating data directories..."
mkdir -p "$DATA_DIR" "$CACHE_DIR"

# Stop and remove existing container if it exists
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

# Pull latest image
echo "Pulling latest image..."
docker pull $IMAGE_NAME

# Check available disk space
available_space=$(df / | awk 'NR==2 {print $4}')
if [ $available_space -lt 5000000 ]; then
    echo "Warning: Low disk space ($(($available_space/1024))MB available)"
    echo "Consider cleaning up old Docker images: docker system prune"
fi

# Display system info
echo ""
echo "=== System Information ==="
free -h
echo "GPU Status:"
if which tegrastats >/dev/null 2>&1; then
    timeout 3s tegrastats --interval 1000 | head -1 || echo "Unable to get GPU stats"
fi
echo ""

# Function to run image captioning
run_captioning() {
    local image_file="$1"
    local question="${2:-Describe this image in detail.}"
    
    if [ ! -f "$DATA_DIR/$image_file" ]; then
        echo "Error: Image file '$DATA_DIR/$image_file' not found"
        echo "Please copy your images to: $DATA_DIR/"
        return 1
    fi
    
    echo "Processing image: $image_file"
    echo "Question: $question"
    echo ""
    
    nvidia-docker run --rm \
        --gpus all \
        --name $CONTAINER_NAME \
        -v "$DATA_DIR:/app/data:ro" \
        -v "$CACHE_DIR:/app/cache" \
        --memory=3.5g \
        --memory-swap=4g \
        $IMAGE_NAME \
        python3 vlm001.py "/app/data/$image_file" --question "$question" --verbose
}

# Interactive mode or direct execution
if [ $# -eq 0 ]; then
    echo "=== Interactive Mode ==="
    echo "Available images in $DATA_DIR:"
    ls -la "$DATA_DIR"/ 2>/dev/null || echo "No images found"
    echo ""
    echo "Usage examples:"
    echo "  $0 my_image.jpg"
    echo "  $0 my_image.jpg 'What objects are in this image?'"
    echo ""
    echo "To copy images to the data directory:"
    echo "  cp /path/to/your/image.jpg $DATA_DIR/"
    echo ""
    echo "Starting container in interactive mode..."
    
    nvidia-docker run -it --rm \
        --gpus all \
        --name $CONTAINER_NAME \
        -v "$DATA_DIR:/app/data" \
        -v "$CACHE_DIR:/app/cache" \
        --memory=3.5g \
        --memory-swap=4g \
        $IMAGE_NAME bash
        
elif [ $# -eq 1 ]; then
    run_captioning "$1"
elif [ $# -eq 2 ]; then
    run_captioning "$1" "$2"
else
    echo "Usage: $0 [image_filename] [optional_question]"
    echo "  $0                           # Interactive mode"
    echo "  $0 image.jpg                 # Caption with default question"
    echo "  $0 image.jpg 'Custom question?'  # Caption with custom question"
    exit 1
fi

echo ""
echo "=== Completed ==="
