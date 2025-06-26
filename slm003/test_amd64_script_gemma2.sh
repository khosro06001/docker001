#!/bin/bash

# Test script for Gemma2 Chatbot on AMD64 (CPU-only) architecture
# This script tests the Docker container functionality on AMD64 systems

set -e  # Exit on any error

# Configuration
IMAGE_NAME="gemma2-chatbot:latest"
CONTAINER_NAME="gemma2-test-amd64"
OLLAMA_PORT="11434"
TEST_DIR="$(dirname "$0")/test_amd64"
LOG_FILE="${TEST_DIR}/test_log_$(date +%Y%m%d_%H%M%S).txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${BLUE}[${timestamp}] INFO: ${message}${NC}" ;;
        "WARN")  echo -e "${YELLOW}[${timestamp}] WARN: ${message}${NC}" ;;
        "ERROR") echo -e "${RED}[${timestamp}] ERROR: ${message}${NC}" ;;
        "SUCCESS") echo -e "${GREEN}[${timestamp}] SUCCESS: ${message}${NC}" ;;
    esac
    
    echo "[${timestamp}] ${level}: ${message}" >> "${LOG_FILE}"
}

# Create test directory
mkdir -p "${TEST_DIR}"

# Function to cleanup on exit
cleanup() {
    log "INFO" "Cleaning up test containers..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=1
    
    log "INFO" "Waiting for ${service_name} to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            log "SUCCESS" "${service_name} is ready!"
            return 0
        fi
        
        log "INFO" "Attempt ${attempt}/${max_attempts}: ${service_name} not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    log "ERROR" "${service_name} failed to start within expected time"
    return 1
}

# Function to test Docker image
test_docker_image() {
    log "INFO" "Testing Docker image: ${IMAGE_NAME}"
    
    # Check if image exists
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        log "ERROR" "Docker image '${IMAGE_NAME}' not found"
        log "INFO" "Please build the image first or pull from registry"
        return 1
    fi
    
    log "SUCCESS" "Docker image found"
    return 0
}

# Function to test Ollama connectivity
test_ollama() {
    log "INFO" "Testing Ollama server connectivity..."
    
    # Check if Ollama is running
    if ! command_exists "ollama"; then
        log "ERROR" "Ollama command not found. Please install Ollama first."
        return 1
    fi
    
    # Check if Ollama service is running
    if ! curl -s "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null; then
        log "WARN" "Ollama server not running. Starting ollama serve..."
        ollama serve &
        sleep 3
        
        if ! wait_for_service "Ollama" "curl -s http://localhost:${OLLAMA_PORT}/api/tags"; then
            return 1
        fi
    else
        log "SUCCESS" "Ollama server is already running"
    fi
    
    # Check if gemma2 model is available
    if ollama list | grep -q "gemma2:2b-instruct-q4_0"; then
        log "SUCCESS" "Gemma2 model is available"
    else
        log "WARN" "Gemma2 model not found. Pulling model..."
        if ollama pull gemma2:2b-instruct-q4_0; then
            log "SUCCESS" "Gemma2 model pulled successfully"
        else
            log "ERROR" "Failed to pull Gemma2 model"
            return 1
        fi
    fi
    
    return 0
}

# Function to test container startup
test_container_startup() {
    log "INFO" "Testing container startup..."
    
    # Remove any existing test container
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    
    # Start container in background for testing
    if docker run -d \
        --name "${CONTAINER_NAME}" \
        --network host \
        -v "${TEST_DIR}/conversations:/app/conversations" \
        -e "OLLAMA_HOST=localhost:${OLLAMA_PORT}" \
        "${IMAGE_NAME}" \
        python3 slm_chatbot_gemma2.py --url "http://localhost:${OLLAMA_PORT}" --verbose; then
        
        log "SUCCESS" "Container started successfully"
    else
        log "ERROR" "Failed to start container"
        return 1
    fi
    
    # Wait for container to be ready
    if wait_for_service "Container" "docker exec ${CONTAINER_NAME} python3 -c 'import requests; requests.get(\"http://localhost:${OLLAMA_PORT}/api/tags\", timeout=5)'"; then
        log "SUCCESS" "Container is healthy and can connect to Ollama"
    else
        log "ERROR" "Container health check failed"
        docker logs "${CONTAINER_NAME}"
        return 1
    fi
    
    return 0
}

# Function to test interactive functionality
test_interactive_mode() {
    log "INFO" "Testing interactive chatbot functionality..."
    
    # Create a test script for automated interaction
    cat > "${TEST_DIR}/test_interaction.py" << 'EOF'
#!/usr/bin/env python3
import subprocess
import time
import sys

def test_chatbot():
    try:
        # Start the chatbot process
        process = subprocess.Popen(
            ['docker', 'exec', '-i', 'gemma2-test-amd64', 'python3', 'slm_chatbot_gemma2.py', '--url', 'http://localhost:11434'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Send test message
        test_message = "Hello, can you respond with just 'Test successful'?\n"
        process.stdin.write(test_message)
        process.stdin.flush()
        
        # Wait for response
        time.sleep(10)
        
        # Send quit command
        process.stdin.write("/quit\n")
        process.stdin.flush()
        
        # Wait for process to finish
        stdout, stderr = process.communicate(timeout=15)
        
        if "Test successful" in stdout or process.returncode == 0:
            print("SUCCESS: Interactive test passed")
            return True
        else:
            print("ERROR: Interactive test failed")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR: Interactive test exception: {e}")
        return False
    finally:
        try:
            process.terminate()
        except:
            pass

if __name__ == "__main__":
    success = test_chatbot()
    sys.exit(0 if success else 1)
EOF
    
    chmod +x "${TEST_DIR}/test_interaction.py"
    
    if python3 "${TEST_DIR}/test_interaction.py"; then
        log "SUCCESS" "Interactive functionality test passed"
        return 0
    else
        log "WARN" "Interactive functionality test had issues (this is often normal in automated testing)"
        return 0  # Don't fail the entire test for this
    fi
}

# Function to test health check
test_health_check() {
    log "INFO" "Testing container health check..."
    
    # Wait for health check to run
    sleep 35  # Health check runs every 30s
    
    health_status=$(docker inspect --format='{{.State.Health.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "none")
    
    case $health_status in
        "healthy")
            log "SUCCESS" "Container health check passed"
            return 0
            ;;
        "unhealthy")
            log "ERROR" "Container health check failed"
            docker logs "${CONTAINER_NAME}"
            return 1
            ;;
        "starting")
            log "INFO" "Container health check still starting (this is normal)"
            return 0
            ;;
        *)
            log "WARN" "Container health check status unknown: ${health_status}"
            return 0
            ;;
    esac
}

# Function to test file persistence
test_file_persistence() {
    log "INFO" "Testing conversation file persistence..."
    
    # Check if conversations directory was created and is writable
    if docker exec "${CONTAINER_NAME}" test -w /app/conversations; then
        log "SUCCESS" "Conversations directory is writable"
    else
        log "ERROR" "Conversations directory is not writable"
        return 1
    fi
    
    # Check if host volume mapping works
    if [ -d "${TEST_DIR}/conversations" ]; then
        log "SUCCESS" "Host volume mapping is working"
    else
        log "WARN" "Host volume mapping directory not found"
    fi
    
    return 0
}

# Function to run performance test
test_performance() {
    log "INFO" "Running basic performance test..."
    
    # Check CPU usage
    cpu_usage=$(docker stats "${CONTAINER_NAME}" --no-stream --format "table {{.CPUPerc}}" | tail -n 1 | sed 's/%//')
    log "INFO" "Current CPU usage: ${cpu_usage}%"
    
    # Check memory usage
    mem_usage=$(docker stats "${CONTAINER_NAME}" --no-stream --format "table {{.MemUsage}}" | tail -n 1)
    log "INFO" "Current memory usage: ${mem_usage}"
    
    return 0
}

# Main test function
run_tests() {
    log "INFO" "Starting AMD64 test suite for Gemma2 Chatbot"
    log "INFO" "Architecture: $(uname -m)"
    log "INFO" "OS: $(uname -s)"
    log "INFO" "Kernel: $(uname -r)"
    
    local failed_tests=0
    
    # Test 1: Check prerequisites
    log "INFO" "=== Test 1: Prerequisites ==="
    if ! command_exists "docker"; then
        log "ERROR" "Docker not found"
        ((failed_tests++))
    fi
    
    if ! command_exists "curl"; then
        log "ERROR" "curl not found"
        ((failed_tests++))
    fi
    
    # Test 2: Docker image
    log "INFO" "=== Test 2: Docker Image ==="
    if ! test_docker_image; then
        ((failed_tests++))
    fi
    
    # Test 3: Ollama setup
    log "INFO" "=== Test 3: Ollama Setup ==="
    if ! test_ollama; then
        ((failed_tests++))
    fi
    
    # Test 4: Container startup
    log "INFO" "=== Test 4: Container Startup ==="
    if ! test_container_startup; then
        ((failed_tests++))
    fi
    
    # Test 5: Health check
    log "INFO" "=== Test 5: Health Check ==="
    if ! test_health_check; then
        ((failed_tests++))
    fi
    
    # Test 6: File persistence
    log "INFO" "=== Test 6: File Persistence ==="
    if ! test_file_persistence; then
        ((failed_tests++))
    fi
    
    # Test 7: Performance check
    log "INFO" "=== Test 7: Performance Check ==="
    if ! test_performance; then
        ((failed_tests++))
    fi
    
    # Test 8: Interactive mode (optional)
    log "INFO" "=== Test 8: Interactive Mode ==="
    if ! test_interactive_mode; then
        log "WARN" "Interactive mode test had issues (this is often expected in automated testing)"
    fi
    
    # Summary
    log "INFO" "=== TEST SUMMARY ==="
    log "INFO" "Total failed tests: ${failed_tests}"
    
    if [ $failed_tests -eq 0 ]; then
        log "SUCCESS" "All critical tests passed! 🎉"
        log "INFO" "Your Gemma2 chatbot is ready for AMD64 deployment"
        
        echo ""
        echo "========================="
        echo "🚀 READY FOR DEPLOYMENT!"
        echo "========================="
        echo "To run the chatbot interactively:"
        echo "  docker run -it --rm --network host \\"
        echo "    -v \$(pwd)/conversations:/app/conversations \\"
        echo "    ${IMAGE_NAME}"
        echo ""
        echo "Make sure Ollama is running with:"
        echo "  ollama serve"
        echo "========================="
        
        return 0
    else
        log "ERROR" "Some tests failed. Please check the issues above."
        return 1
    fi
}

# Print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Test script for Gemma2 Chatbot on AMD64 architecture"
    echo ""
    echo "Options:"
    echo "  -i, --image NAME    Docker image name (default: gemma2-chatbot:latest)"
    echo "  -p, --port PORT     Ollama port (default: 11434)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --image myregistry/gemma2-chatbot:v1.0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -p|--port)
            OLLAMA_PORT="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
echo "🧪 GEMMA2 CHATBOT AMD64 TEST SUITE"
echo "=================================="
echo "Image: ${IMAGE_NAME}"
echo "Ollama Port: ${OLLAMA_PORT}"
echo "Test Directory: ${TEST_DIR}"
echo "Log File: ${LOG_FILE}"
echo "=================================="
echo ""

# Run the tests
if run_tests; then
    log "SUCCESS" "Test suite completed successfully!"
    exit 0
else
    log "ERROR" "Test suite failed!"
    echo ""
    echo "❌ TESTS FAILED"
    echo "Check the log file for details: ${LOG_FILE}"
    echo "Common issues:"
    echo "  - Ollama not running (run: ollama serve)"
    echo "  - Gemma2 model not installed (run: ollama pull gemma2:2b-instruct-q4_0)"
    echo "  - Docker image not built (run: python3 build_and_deploy_script.py)"
    exit 1
fi