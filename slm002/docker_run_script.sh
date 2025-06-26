#!/bin/bash

# Script to run Gemma2 chatbot container
# Handles both AMD64 (CPU) and ARM64 (GPU/Jetson) architectures

set -e

# Configuration
IMAGE_NAME="your-dockerhub-username/gemma2-chatbot:latest"
CONTAINER_NAME="gemma2-chatbot"
DATA_DIR="$HOME/gemma2_conversations"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Gemma2 Interactive Chatbot - Docker Runner ===${NC}"
echo "Image: $IMAGE_NAME"
echo "Date: $(date)"
echo ""

# Detect architecture
ARCH=$(uname -m)
echo -e "${BLUE}Architecture detected: ${YELLOW}$ARCH${NC}"

# Check if running on Jetson
IS_JETSON=false
if lsmod 2>/dev/null | grep -q nvgpu; then
    IS_JETSON=true
    echo -e "${GREEN}✓ Jetson device detected (GPU acceleration will be used)${NC}"
elif [ "$ARCH" = "aarch64" ]; then
    echo -e "${YELLOW}⚠ ARM64 detected but no nvgpu module (limited GPU support)${NC}"
else
    echo -e "${YELLOW}⚠ x86_64 detected (CPU-only mode)${NC}"
fi

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found${NC}"
    exit 1
fi

# Check for nvidia-docker on Jetson
if [ "$IS_JETSON" = true ]; then
    if ! command -v nvidia-docker &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-docker not found, using regular docker${NC}"
        echo "For optimal GPU performance, install nvidia-docker runtime"
        DOCKER_CMD="docker"
    else
        DOCKER_CMD="nvidia-docker"
        echo -e "${GREEN}✓ nvidia-docker available${NC}"
    fi
else
    DOCKER_CMD="docker"
fi

# Create data directory for conversation storage
echo "Creating conversation storage directory..."
mkdir -p "$DATA_DIR"

# Stop and remove existing container if it exists
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

# Pull latest image
echo -e "${BLUE}Pulling latest image...${NC}"
docker pull $IMAGE_NAME

# Check system resources
echo ""
echo -e "${BLUE}=== System Information ===${NC}"
echo "Memory:"
free -h
if [ "$IS_JETSON" = true ] && command -v tegrastats &> /dev/null; then
    echo "GPU Status:"
    timeout 3s tegrastats --interval 1000 | head -1 2>/dev/null || echo "Unable to get GPU stats"
fi
echo ""

# Function to run chatbot
run_chatbot() {
    local extra_args="$1"
    
    echo -e "${GREEN}Starting Gemma2 Interactive Chatbot...${NC}"
    echo -e "${YELLOW}Conversation files will be saved to: $DATA_DIR${NC}"
    echo -e "${YELLOW}Commands: /help, /clear, /stats, /save, /quit${NC}"
    echo ""
    
    # Prepare Docker run command
    local docker_args=(
        "--rm"
        "-it"
        "--name" "$CONTAINER_NAME"
        "--network" "host"
        "-v" "$DATA_DIR:/app/conversations"
        "-e" "TERM=xterm-256color"
    )
    
    # Add GPU support for Jetson
    if [ "$IS_JETSON" = true ]; then
        if [ "$DOCKER_CMD" = "nvidia-docker" ]; then
            docker_args+=("--gpus" "all")
        fi
        # Memory limits for Jetson Nano 4GB
        docker_args+=("--memory=3g" "--memory-swap=3.5g")
    else
        # More generous memory for x86_64
        docker_args+=("--memory=2g")
    fi
    
    # Add any extra arguments
    if [ -n "$extra_args" ]; then
        docker_args+=($extra_args)
    fi
    
    # Run the container
    $DOCKER_CMD run "${docker_args[@]}" $IMAGE_NAME python3 ollama_gemma2_chatbot.py --verbose
}

# Parse command line arguments
case "${1:-interactive}" in
    "interactive"|"")
        run_chatbot
        ;;
    "shell"|"bash")
        echo -e "${BLUE}Starting container shell for debugging...${NC}"
        $DOCKER_CMD run --rm -it --network host \
            -v "$DATA_DIR:/app/conversations" \
            --entrypoint /bin/bash \
            $IMAGE_NAME
        ;;
    "test")
        echo -e "${BLUE}Running connection test...${NC}"
        $DOCKER_CMD run --rm --network host \
            $IMAGE_NAME python3 healthcheck_gemma2.py
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  interactive  - Start interactive chatbot (default)"
        echo "  shell        - Open shell in container for debugging"
        echo "  test         - Test Ollama connection and model availability"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Start interactive chat"
        echo "  $0 interactive        # Start interactive chat"
        echo "  $0 shell              # Debug shell"
        echo "  $0 test               # Test connection"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== Session Completed ===${NC}"
echo -e "${YELLOW}Conversations saved in: $DATA_DIR${NC}"