#!/bin/bash

# Multi-architecture Docker build and deployment script for slm001 chatbot
# This script builds for both AMD64 and ARM64 architectures

set -e  # Exit on any error

# Configuration
###DOCKER_USERNAME="your_dockerhub_username"  # Replace with your Docker Hub username
DOCKER_USERNAME="khosro123"  # Replace with your Docker Hub username
IMAGE_NAME="slm001-chatbot"
TAG="latest"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker daemon."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Function to check if buildx is available
check_buildx() {
    if ! docker buildx version &> /dev/null; then
        print_error "Docker buildx is not available. Please update Docker to a newer version."
        exit 1
    fi
    
    print_success "Docker buildx is available"
}

# Function to create and use buildx builder
setup_buildx() {
    print_status "Setting up Docker buildx for multi-architecture builds..."
    
    # Create a new builder instance
    docker buildx create --name multiarch-builder --use --bootstrap 2>/dev/null || true
    
    # Use the builder
    docker buildx use multiarch-builder
    
    # Inspect the builder
    docker buildx inspect --bootstrap
    
    print_success "Multi-architecture builder setup complete"
}

# Function to login to Docker Hub
docker_login() {
    print_status "Logging in to Docker Hub..."
    
    if [ -z "$DOCKER_USERNAME" ] || [ "$DOCKER_USERNAME" = "your_dockerhub_username" ]; then
        print_error "Please set your Docker Hub username in the script"
        exit 1
    fi
    
    echo "Please enter your Docker Hub password:"
    docker login -u "$DOCKER_USERNAME"
    
    print_success "Successfully logged in to Docker Hub"
}

# Function to build multi-architecture image
build_image() {
    print_status "Building multi-architecture Docker image..."
    print_status "Image: $FULL_IMAGE_NAME"
    print_status "Architectures: linux/amd64, linux/arm64"
    
    # Build and push for multiple architectures
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag "$FULL_IMAGE_NAME" \
        --push \
        .
    
    print_success "Multi-architecture image built and pushed successfully"
}

# Function to verify the image
verify_image() {
    print_status "Verifying multi-architecture image..."
    
    # Check if the image exists and show manifest
    docker buildx imagetools inspect "$FULL_IMAGE_NAME"
    
    print_success "Image verification complete"
}

# Function to generate run instructions
generate_instructions() {
    print_status "Generating run instructions..."
    
    cat << EOF > run_instructions.txt
=== SLM001 Chatbot Deployment Instructions ===

1. On your Jetson Nano (or any ARM64 device), run:
   docker pull ${FULL_IMAGE_NAME}

2. Run the chatbot:
   docker run -it --rm ${FULL_IMAGE_NAME}

3. Alternative run with persistent cache:
   docker run -it --rm -v slm001-cache:/app/cache ${FULL_IMAGE_NAME}

4. To run in detached mode with a custom name:
   docker run -d --name slm001-chatbot ${FULL_IMAGE_NAME}

5. To attach to a running container:
   docker exec -it slm001-chatbot /bin/bash

6. To stop the container:
   docker stop slm001-chatbot

=== Troubleshooting ===

If you encounter issues:
- Check Docker version: docker --version
- Check available architectures: docker buildx ls
- View container logs: docker logs slm001-chatbot
- Remove and re-pull: docker rmi ${FULL_IMAGE_NAME} && docker pull ${FULL_IMAGE_NAME}

=== Model Information ===
- Model: microsoft/DialoGPT-small
- Size: ~117MB
- Runs on CPU (optimized for Jetson Nano)
- No internet required after pulling

EOF

    print_success "Instructions saved to run_instructions.txt"
}

# Main execution
main() {
    print_status "Starting multi-architecture build process for slm001 chatbot..."
    
    # Check prerequisites
    check_docker
    check_buildx
    
    # Setup buildx
    setup_buildx
    
    # Login to Docker Hub
    docker_login
    
    # Build the image
    build_image
    
    # Verify the image
    verify_image
    
    # Generate instructions
    generate_instructions
    
    print_success "Build and deployment process completed successfully!"
    print_status "You can now pull and run the image on your Jetson Nano using:"
    print_status "docker pull ${FULL_IMAGE_NAME}"
    print_status "docker run -it --rm ${FULL_IMAGE_NAME}"
}

# Run main function
main
