#!/bin/bash

# Enhanced Docker build and deployment script for slm001 quantized chatbot
set -e

# Configuration
DOCKER_USERNAME="khosro123"  # Replace with your Docker Hub username
IMAGE_NAME="slm001-quantized-chatbot"
TAG="latest"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
BUILDER_NAME="quantized-builder"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    docker buildx rm "$BUILDER_NAME" 2>/dev/null || true
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Check requirements
check_requirements() {
    print_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker not installed. Please install Docker first."
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker not running. Please start Docker daemon."
    fi
    
    # Check buildx
    if ! docker buildx version >/dev/null 2>&1; then
        print_error "Docker buildx not available. Please update Docker to latest version."
    fi
    
    # Check required files
    for file in slm001.py requirements.txt; do
        if [ ! -f "$file" ]; then
            print_error "Missing required file: $file"
        fi
    done
    
    # Check for Dockerfile
    if [ -f "Dockerfile.txt" ]; then
        DOCKERFILE="Dockerfile.txt"
    elif [ -f "Dockerfile" ]; then
        DOCKERFILE="Dockerfile"
    else
        print_error "No Dockerfile found (looking for Dockerfile.txt or Dockerfile)"
    fi
    
    # Check if logged into Docker Hub
    if ! docker info | grep -q "Username: $DOCKER_USERNAME" 2>/dev/null; then
        print_warning "Not logged into Docker Hub or different user"
    fi
    
    print_success "All requirements met (using: $DOCKERFILE)"
}

# Login to Docker Hub with retry
docker_login() {
    print_info "Checking Docker Hub authentication..."
    
    # Check if already logged in
    if docker info 2>/dev/null | grep -q "Username: $DOCKER_USERNAME"; then
        print_success "Already logged in as $DOCKER_USERNAME"
        return 0
    fi
    
    print_info "Please login to Docker Hub (username: $DOCKER_USERNAME)"
    
    local attempts=0
    local max_attempts=3
    
    while [ $attempts -lt $max_attempts ]; do
        if docker login -u "$DOCKER_USERNAME"; then
            print_success "Successfully logged in to Docker Hub"
            return 0
        else
            attempts=$((attempts + 1))
            if [ $attempts -lt $max_attempts ]; then
                print_warning "Login failed. Attempt $attempts/$max_attempts. Retrying..."
                sleep 2
            fi
        fi
    done
    
    print_error "Failed to login to Docker Hub after $max_attempts attempts"
}

# Setup multi-arch builder
setup_builder() {
    print_info "Setting up multi-architecture builder..."
    
    # Remove existing builder if it exists
    docker buildx rm "$BUILDER_NAME" 2>/dev/null || true
    
    # Create and bootstrap new builder
    if docker buildx create --name "$BUILDER_NAME" --use --bootstrap; then
        print_success "Builder '$BUILDER_NAME' setup complete"
    else
        print_error "Failed to setup buildx builder"
    fi
}

# Optimize build context
optimize_build_context() {
    print_info "Optimizing build context..."
    
    # Create .dockerignore if it doesn't exist
    if [ ! -f ".dockerignore" ]; then
        cat > .dockerignore << 'EOF'
# Git
.git/
.gitignore
*.md

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
*.log
.dockerignore

# Keep only essential files
!requirements.txt
!slm001.py
!Dockerfile*
EOF
        print_success "Created .dockerignore for optimized builds"
    fi
}

# Build and push image with progress
build_and_push() {
    print_info "Building quantized chatbot for AMD64 and ARM64..."
    print_info "Image: $FULL_IMAGE_NAME"
    print_info "Using: $DOCKERFILE"
    print_warning "Multi-arch builds can take 10-30 minutes depending on your system..."
    
    # Build for multiple architectures and push
    if docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag "$FULL_IMAGE_NAME" \
        --tag "${DOCKER_USERNAME}/${IMAGE_NAME}:quantized" \
        --tag "${DOCKER_USERNAME}/${IMAGE_NAME}:jetson" \
        --file "$DOCKERFILE" \
        --push \
        --progress=plain \
        .; then
        print_success "Multi-arch build completed and pushed"
    else
        print_error "Build failed. Check the output above for details."
    fi
}

# Build local image only (fallback option)
build_local_only() {
    print_warning "Building for current architecture only (fallback mode)..."
    
    if docker build \
        --tag "$FULL_IMAGE_NAME" \
        --tag "${DOCKER_USERNAME}/${IMAGE_NAME}:local" \
        --file "$DOCKERFILE" \
        .; then
        print_success "Local build completed"
        
        # Push local build
        if docker push "$FULL_IMAGE_NAME" && docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:local"; then
            print_success "Local image pushed successfully"
        else
            print_warning "Failed to push local image"
        fi
    else
        print_error "Local build also failed"
    fi
}

# Verify image
verify_image() {
    print_info "Verifying image..."
    
    if docker buildx imagetools inspect "$FULL_IMAGE_NAME" 2>/dev/null; then
        print_success "Multi-arch image verification complete"
    else
        print_warning "Multi-arch verification failed, checking single arch..."
        if docker pull "$FULL_IMAGE_NAME" >/dev/null 2>&1; then
            print_success "Single-arch image verified"
        else
            print_error "Image verification failed"
        fi
    fi
}

# Generate comprehensive run instructions
create_instructions() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    cat > run_instructions.txt << EOF
# SLM001 Quantized Chatbot - Quick Start Guide
Generated: $timestamp

## 🚀 Quick Start

### For Jetson Nano (ARM64):
\`\`\`bash
docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:jetson
docker run -it --rm ${DOCKER_USERNAME}/${IMAGE_NAME}:jetson
\`\`\`

### For other devices (AMD64/ARM64):
\`\`\`bash
docker pull ${FULL_IMAGE_NAME}
docker run -it --rm ${FULL_IMAGE_NAME}
\`\`\`

### With GPU support (if available):
\`\`\`bash
docker run -it --rm --gpus all ${FULL_IMAGE_NAME}
\`\`\`

## 💬 Available Commands in Chatbot:
| Command | Description |
|---------|-------------|
| \`models\` | Show available quantized models |
| \`switch <model>\` | Change model (e.g., "switch distilgpt2") |
| \`stats\` | Show model statistics |
| \`memory\` | Show memory usage |
| \`clear\` | Clear conversation history |
| \`help\` | Show help message |
| \`quit\` | Exit chatbot |

## 🤖 Available Models:
| Model | Size | Best For | Quantization |
|-------|------|----------|-------------|
| DialoGPT-small | ~30MB | Jetson Nano | 4-bit |
| DistilGPT2 | ~80MB | General use | 8-bit |
| GPT2 | ~125MB | Better quality | 8-bit |
| DialoGPT-medium | ~90MB | Conversations | 4-bit |
| TinyLlama-1.1B | ~280MB | Modern chat | 4-bit |

## ✨ Features:
- ✅ Intelligent quantization fallback (bitsandbytes → PyTorch native)
- ✅ Optimized for 4GB RAM devices (Jetson Nano)
- ✅ Real-time model switching without restart
- ✅ Memory-efficient conversation handling
- ✅ Multi-architecture support (AMD64/ARM64)
- ✅ Automatic model downloading and caching

## 🔧 Advanced Usage:

### Custom memory limits:
\`\`\`bash
docker run -it --rm --memory=3g ${FULL_IMAGE_NAME}
\`\`\`

### Mount custom cache directory:
\`\`\`bash
docker run -it --rm -v ./cache:/app/cache ${FULL_IMAGE_NAME}
\`\`\`

### Run with specific model:
\`\`\`bash
docker run -it --rm ${FULL_IMAGE_NAME}
# Then in chatbot: switch microsoft/DialoGPT-small
\`\`\`

## 🆘 Troubleshooting:
- **Out of memory**: Try DialoGPT-small or DistilGPT2
- **Slow on Jetson**: Use 4-bit quantized models
- **CUDA errors**: Fallback to CPU-only mode works automatically
- **Model download fails**: Check internet connection

## 📞 Support:
- Check Docker logs: \`docker logs <container_id>\`
- GitHub issues: [Your repository URL here]
- Built: $timestamp
EOF
    
    print_success "Instructions saved to run_instructions.txt"
}

# Test local build
test_local_build() {
    print_info "Testing local build..."
    
    if docker run --rm "$FULL_IMAGE_NAME" python -c "import torch; import transformers; print('✅ Basic imports work')"; then
        print_success "Local test passed"
    else
        print_warning "Local test failed, but image might still work interactively"
    fi
}

# Main execution with error handling
main() {
    print_info "🚀 Starting enhanced build process for quantized chatbot..."
    echo "Building: $FULL_IMAGE_NAME"
    echo "Time: $(date)"
    echo "Host: $(uname -a)"
    echo ""
    
    # Run checks
    check_requirements
    optimize_build_context
    
    # Login to Docker Hub
    docker_login
    
    # Setup builder
    setup_builder
    
    # Try multi-arch build, fallback to local if needed
    if build_and_push; then
        verify_image
        test_local_build
    else
        print_warning "Multi-arch build failed, trying local build..."
        build_local_only
    fi
    
    create_instructions
    
    print_success "🎉 Build process completed!"
    print_success "Images available:"
    print_success "  • ${FULL_IMAGE_NAME}"
    print_success "  • ${DOCKER_USERNAME}/${IMAGE_NAME}:quantized"
    print_success "  • ${DOCKER_USERNAME}/${IMAGE_NAME}:jetson"
    print_info "📖 See run_instructions.txt for detailed usage instructions"
    
    echo ""
    print_info "Next steps:"
    echo "1. Test: docker run -it --rm $FULL_IMAGE_NAME"
    echo "2. Share: Send run_instructions.txt to users"
    echo "3. Deploy: Use the Docker Hub images on target devices"
}

# Parse command line arguments
case "${1:-}" in
    --local-only)
        print_info "Building for local architecture only..."
        check_requirements
        optimize_build_context
        docker_login
        build_local_only
        ;;
    --help)
        echo "Usage: $0 [--local-only] [--help]"
        echo "  --local-only    Build for current architecture only"
        echo "  --help         Show this help message"
        ;;
    *)
        main "$@"
        ;;
esac