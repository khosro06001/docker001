#!/bin/bash

# Alternative MiniCPM-V Model Download Script
# Multiple sources when Hugging Face is blocked

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_title() {
    echo -e "${BLUE}[TITLE]${NC} $1"
}

echo "=============================================="
print_title "Alternative MiniCPM-V Model Download"
echo "=============================================="

# Create models directory
MODEL_DIR="./models"
mkdir -p "$MODEL_DIR"

print_status "Available download methods:"
echo "1. Ollama (Recommended - easiest)"
echo "2. ModelScope (China-based, often accessible)"
echo "3. Direct GitHub clone"
echo "4. Manual download links"
echo "5. Docker with pre-built model"

echo ""
read -p "Which method would you like to try? (1-5): " -n 1 -r
echo ""

case $REPLY in
    1)
        print_title "=== METHOD 1: OLLAMA (Recommended) ==="
        
        # Check if Ollama is installed
        if ! command -v ollama &> /dev/null; then
            print_status "Installing Ollama..."
            curl -fsSL https://ollama.com/install.sh | sh || {
                print_error "Failed to install Ollama"
                exit 1
            }
        fi
        
        print_status "Ollama installed successfully"
        print_status "Downloading MiniCPM-V model via Ollama..."
        
        # Pull the model
        ollama pull minicpm-v || {
            print_error "Failed to download model via Ollama"
            exit 1
        }
        
        print_status "Model downloaded successfully!"
        print_status "You can now use: ollama run minicpm-v"
        print_status "Or integrate with your Python script using Ollama API"
        
        # Create Ollama integration script
        cat > run_ollama_vlm.py << 'EOF'
#!/usr/bin/env python3
"""
MiniCPM-V via Ollama API
"""
import requests
import json
import base64
import sys
from pathlib import Path

def image_to_base64(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def caption_image_ollama(image_path, question="Describe this image in detail."):
    """Caption image using Ollama API"""
    try:
        # Convert image to base64
        image_b64 = image_to_base64(image_path)
        
        # Prepare request
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "minicpm-v",
            "prompt": question,
            "images": [image_b64],
            "stream": False
        }
        
        # Make request
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', 'No response received')
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_ollama_vlm.py <image_path> [question]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print("Generating caption...")
    caption = caption_image_ollama(image_path, question)
    
    print("\n" + "="*50)
    print("IMAGE CAPTION:")
    print("="*50)
    print(caption)
    print("="*50)
EOF
        
        chmod +x run_ollama_vlm.py
        print_status "Created run_ollama_vlm.py for easy usage"
        print_status "Usage: python3 run_ollama_vlm.py /path/to/image.jpg"
        ;;
        
    2)
        print_title "=== METHOD 2: MODELSCOPE ==="
        
        print_status "Installing ModelScope..."
        pip3 install modelscope transformers torch pillow || {
            print_error "Failed to install ModelScope dependencies"
            exit 1
        }
        
        # Create ModelScope download script
        cat > download_modelscope.py << 'EOF'
#!/usr/bin/env python3
"""
Download MiniCPM-V from ModelScope
"""
try:
    from modelscope import snapshot_download
    import os
    
    print("Downloading MiniCPM-V from ModelScope...")
    model_dir = snapshot_download('AI-ModelScope/MiniCPM-V-2_6', cache_dir='./models')
    print(f"Model downloaded to: {model_dir}")
    
    # Create symlink for easier access
    if not os.path.exists('./models/minicpm-v'):
        os.symlink(model_dir, './models/minicpm-v')
        print("Created symlink: ./models/minicpm-v")
    
except Exception as e:
    print(f"Error downloading from ModelScope: {e}")
    print("Trying alternative ModelScope model...")
    
    try:
        model_dir = snapshot_download('openbmb/MiniCPM-V-2', cache_dir='./models')
        print(f"Alternative model downloaded to: {model_dir}")
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        exit(1)
EOF
        
        python3 download_modelscope.py
        print_status "ModelScope download completed!"
        ;;
        
    3)
        print_title "=== METHOD 3: DIRECT GITHUB CLONE ==="
        
        print_status "Installing git-lfs..."
        sudo apt update && sudo apt install -y git-lfs || {
            print_warning "Could not install git-lfs via apt, trying manual install..."
            curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
            sudo apt install git-lfs
        }
        
        git lfs install
        
        print_status "Cloning MiniCPM-V repository..."
        cd "$MODEL_DIR"
        
        # Try multiple GitHub sources
        GITHUB_REPOS=(
            "https://github.com/OpenBMB/MiniCPM-V.git"
            "https://github.com/ailabteam/MiniCPM-V.git"
        )
        
        for repo in "${GITHUB_REPOS[@]}"; do
            print_status "Trying repository: $repo"
            if git clone "$repo" minicpm-v-repo; then
                print_status "Successfully cloned from $repo"
                break
            else
                print_warning "Failed to clone from $repo"
            fi
        done
        
        cd ..
        print_status "GitHub clone completed!"
        ;;
        
    4)
        print_title "=== METHOD 4: MANUAL DOWNLOAD LINKS ==="
        
        print_status "Manual download options:"
        echo ""
        echo "Option A - Direct model files:"
        echo "1. Go to: https://github.com/OpenBMB/MiniCPM-V/releases"
        echo "2. Download the latest release"
        echo "3. Extract to ./models/ directory"
        echo ""
        echo "Option B - Alternative mirrors:"
        echo "1. Mirror 1: https://www.modelscope.cn/models/AI-ModelScope/MiniCPM-V-2_6"
        echo "2. Mirror 2: https://gitee.com/OpenBMB/MiniCPM-V"
        echo ""
        echo "Option C - Torrent/P2P (if available):"
        echo "Check academic paper repositories or model sharing communities"
        echo ""
        
        # Try to download specific files
        mkdir -p "$MODEL_DIR/manual"
        cd "$MODEL_DIR/manual"
        
        print_status "Attempting to download key model files..."
        
        # List of potential direct download URLs (these may change)
        DIRECT_URLS=(
            "https://github.com/OpenBMB/MiniCPM-V/raw/main/README.md"
            "https://raw.githubusercontent.com/OpenBMB/MiniCPM-V/main/requirements.txt"
        )
        
        for url in "${DIRECT_URLS[@]}"; do
            filename=$(basename "$url")
            if curl -L -o "$filename" "$url" 2>/dev/null; then
                print_status "Downloaded: $filename"
            else
                print_warning "Could not download: $filename"
            fi
        done
        
        cd ../..
        ;;
        
    5)
        print_title "=== METHOD 5: DOCKER WITH PRE-BUILT MODEL ==="
        
        print_status "Creating Docker setup with model..."
        
        # Create Dockerfile
        cat > Dockerfile.vlm << 'EOF'
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install torch transformers pillow psutil requests modelscope

# Clone model repository
RUN git lfs install && \
    git clone https://github.com/OpenBMB/MiniCPM-V.git || \
    echo "GitHub clone failed, will try ModelScope in runtime"

# Copy application files
COPY . .

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python3", "-c", "print('Docker container ready. Run your VLM script here.')"]
EOF
        
        print_status "Building Docker image..."
        docker build -f Dockerfile.vlm -t minicpm-v-local . || {
            print_error "Docker build failed"
            print_warning "Make sure Docker is installed and running"
            exit 1
        }
        
        print_status "Docker image built successfully!"
        print_status "Run with: docker run -it -v \$(pwd):/app minicpm-v-local"
        ;;
        
    *)
        print_error "Invalid option selected"
        exit 1
        ;;
esac

echo ""
print_title "=== NEXT STEPS ==="

case $REPLY in
    1)
        echo "Ollama method completed. To use:"
        echo "1. Start Ollama: ollama serve"
        echo "2. Test: python3 run_ollama_vlm.py your_image.jpg"
        ;;
    2)
        echo "ModelScope method completed. Update your Python script to use:"
        echo "model_path = './models/minicpm-v'"
        ;;
    3)
        echo "GitHub clone completed. Check ./models/minicpm-v-repo/"
        ;;
    4)
        echo "Manual download - follow the URLs provided above"
        ;;
    5)
        echo "Docker method completed. Run the container and execute your script inside"
        ;;
esac

print_status "Setup completed!"
