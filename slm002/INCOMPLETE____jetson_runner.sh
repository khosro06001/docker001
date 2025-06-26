#!/bin/bash

# Jetson Nano optimized runner for Gemma2 Chatbot
# Includes GPU acceleration and memory optimization

set -e

# Configuration
IMAGE_NAME="your-dockerhub-username/gemma2-chatbot:latest"
CONTAINER_NAME="gemma2-chatbot"
DATA_DIR="$HOME/gemma2_conversations"
CACHE_DIR="$HOME/gemma2_cache"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Gemma2 Chatbot - Jetson Nano Deployment ===${NC}"
echo "Image: $IMAGE_NAME"
echo "Date: $(date)"

# Verify this is a Jetson device
if ! lsmod | grep -q nvgpu; then
    echo -e "${RED}Error: This doesn't appear to be a Jetson device (nvgpu module not found)${NC}"
    echo "This script is optimized for Jetson Nano. Use the generic runner instead."
    exit 1
fi

echo -e "${GREEN}✓ Jetson device confirmed${NC}"

# Check nvidia-docker
if ! command -v nvidia-docker &> /dev/null; then
    echo -e "${RED}Error: nvidia-docker not found${NC}"
    echo "Please install nvidia-docker runtime for Jetson:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install nvidia-docker2"
    echo "  sudo systemctl restart docker"
    exit 1
fi

echo -e "${GREEN}✓ nvidia-docker available${NC}"

# Create directories
echo "Creating data directories..."
mkdir -p "$DATA_DIR" "$CACHE_DIR"

# System information
echo ""
echo -e "${BLUE}=== Jetson System Information ===${NC}"
echo "Memory:"
free -h
echo ""
echo "GPU Status:"
if command -v tegrastats &> /dev/null; then
    timeout 3s tegrastats --interval 1000 | head -1 2>/dev/null || echo "Unable to get GPU stats"