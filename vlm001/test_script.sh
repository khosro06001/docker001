#!/bin/bash

# Enhanced test script with better diagnostics and fallback options
set -e

echo "=== Enhanced VLM Image Captioning Test ==="
echo "Date: $(date)"
echo "Platform: $(uname -a)"
echo "Python: $(python3 --version)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Create test directory
TEST_DIR="./test_vlm"
mkdir -p "$TEST_DIR"

# System diagnostics
echo ""
print_status "=== SYSTEM DIAGNOSTICS ==="
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $4 " available"}')"
echo "GPU: $(lspci | grep -i vga || echo 'No VGA device found')"

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
fi

# Check Python environment
echo ""
print_status "=== PYTHON ENVIRONMENT CHECK ==="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment: $VIRTUAL_ENV"
else
    print_warning "Not in a virtual environment"
fi

# Check core dependencies
check_python_package() {
    local package=$1
    local import_name=${2:-$1}
    
    if python3 -c "import $import_name" 2>/dev/null; then
        local version=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null)
        print_status "$package: $version"
        return 0
    else
        print_error "$package: NOT INSTALLED"
        return 1
    fi
}

MISSING_PACKAGES=()

check_python_package "torch" || MISSING_PACKAGES+=("torch")
check_python_package "transformers" || MISSING_PACKAGES+=("transformers")
check_python_package "PIL" "PIL" || MISSING_PACKAGES+=("pillow")
check_python_package "psutil" || MISSING_PACKAGES+=("psutil")
check_python_package "requests" || MISSING_PACKAGES+=("requests")

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_warning "Installing missing packages: ${MISSING_PACKAGES[*]}"
    pip3 install "${MISSING_PACKAGES[@]}" || {
        print_error "Failed to install packages"
        exit 1
    }
fi

# Optional packages
print_status "Optional packages:"
check_python_package "bitsandbytes" || print_warning "bitsandbytes not available (needed for quantization)"
check_python_package "accelerate" || print_warning "accelerate not available (may improve performance)"
check_python_package "GPUtil" || print_warning "GPUtil not available (GPU monitoring disabled)"

# Network connectivity tests
echo ""
print_status "=== NETWORK CONNECTIVITY TESTS ==="

test_connection() {
    local url=$1
    local name=$2
    
    if curl -s --connect-timeout 10 --max-time 30 "$url" > /dev/null; then
        print_status "$name: OK"
        return 0
    else
        print_error "$name: FAILED"
        return 1
    fi
}

CONNECTIVITY_OK=true

test_connection "https://google.com" "Basic internet" || CONNECTIVITY_OK=false
test_connection "https://huggingface.co" "Hugging Face main" || CONNECTIVITY_OK=false
test_connection "https://cdn-lfs.huggingface.co" "Hugging Face CDN" || CONNECTIVITY_OK=false

# DNS test
echo "DNS resolution:"
if nslookup huggingface.co > /dev/null 2>&1; then
    print_status "DNS resolution: OK"
else
    print_error "DNS resolution: FAILED"
    CONNECTIVITY_OK=false
fi

if [ "$CONNECTIVITY_OK" = false ]; then
    print_warning "Network connectivity issues detected!"
    print_warning "Possible solutions:"
    echo "  1. Check firewall settings"
    echo "  2. Try different DNS servers: sudo systemctl restart systemd-resolved"
    echo "  3. Use VPN if behind corporate firewall"
    echo "  4. Set proxy if needed: export https_proxy=http://proxy:port"
    
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Prepare test image
TEST_IMAGE="$TEST_DIR/test_image.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    print_status "Downloading test image..."
    if ! curl -L -o "$TEST_IMAGE" "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png"; then
        print_warning "Failed to download test image from Wikipedia"
        print_status "Trying alternative source..."
        if ! curl -L -o "$TEST_IMAGE" "https://picsum.photos/300/200"; then
            print_error "Could not download any test image"
            print_error "Please place a test image at $TEST_IMAGE and run again"
            exit 1
        fi
    fi
fi

# Verify test image
if [ -f "$TEST_IMAGE" ]; then
    IMAGE_SIZE=$(wc -c < "$TEST_IMAGE")
    if [ "$IMAGE_SIZE" -lt 1000 ]; then
        print_error "Test image seems corrupted (too small: $IMAGE_SIZE bytes)"
        rm -f "$TEST_IMAGE"
        exit 1
    fi
    print_status "Test image ready: $TEST_IMAGE ($IMAGE_SIZE bytes)"
fi

# Check if vlm script exists
SCRIPT_NAME="vlm001.py"
if [ ! -f "$SCRIPT_NAME" ]; then
    print_error "$SCRIPT_NAME not found in current directory"
    print_error "Please ensure the script is in the current directory"
    exit 1
fi

# Make script executable
chmod +x "$SCRIPT_NAME"

# Set environment variables for better performance
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_OFFLINE=0

# Pre-flight check: Test model loading without actually running
echo ""
print_status "=== PRE-FLIGHT MODEL CHECK ==="
python3 -c "
import sys
try:
    from transformers import AutoModel, AutoTokenizer
    print('✓ Transformers import successful')
    
    # Test if we can reach the model repository
    import requests
    response = requests.get('https://huggingface.co/api/models/openbmb/MiniCPM-V-2', timeout=10)
    if response.status_code == 200:
        print('✓ Model repository accessible')
    else:
        print(f'⚠ Model repository returned status: {response.status_code}')
        
except Exception as e:
    print(f'✗ Pre-flight check failed: {e}')
    sys.exit(1)
" || {
    print_error "Pre-flight check failed"
    print_error "This suggests the model loading will likely fail"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# Run the main test
echo ""
print_status "=== RUNNING IMAGE CAPTIONING TEST ==="
print_status "Note: First run may take 5-15 minutes for model download"
print_status "Model size: ~2-4GB depending on variant"
print_status "Press Ctrl+C to cancel"
echo ""

# Create a timeout wrapper to prevent hanging
timeout 1800 python3 "$SCRIPT_NAME" "$TEST_IMAGE" --verbose --debug || {
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        print_error "Test timed out after 30 minutes"
        print_error "This suggests network issues or insufficient resources"
    else
        print_error "Test failed with exit code: $exit_code"
    fi
    
    echo ""
    print_status "TROUBLESHOOTING SUGGESTIONS:"
    echo "1. Check available disk space (need 4GB+ free)"
    echo "2. Try with VPN if behind corporate firewall"
    echo "3. Run with smaller model: export MODEL_NAME=microsoft/DialoGPT-medium"
    echo "4. Try offline mode if model was partially downloaded"
    echo "5. Check system logs: journalctl -f"
    exit $exit_code
}

echo ""
print_status "=== TEST COMPLETED SUCCESSFULLY ==="
print_status "If you see a caption above, the system is working correctly!"
print_status "Performance will be better on GPU-enabled systems."

# Clean up and summary
echo ""
print_status "=== SUMMARY ==="
echo "Script: $SCRIPT_NAME"
echo "Test image: $TEST_IMAGE"
echo "Cache location: ~/.cache/huggingface/"
echo "To run again: python3 $SCRIPT_NAME path/to/image.jpg"
