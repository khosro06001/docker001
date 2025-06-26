#!/bin/bash
# Run this on your Jetson Nano to check nvidia-docker installation

echo "=== Checking NVIDIA Docker Runtime Installation ==="
echo

echo "1. Checking if nvidia-container-runtime is installed:"
if command -v nvidia-container-runtime &> /dev/null; then
    echo "✓ nvidia-container-runtime found"
    nvidia-container-runtime --version
else
    echo "✗ nvidia-container-runtime not found"
fi
echo

echo "2. Checking Docker runtime configuration:"
if [ -f /etc/docker/daemon.json ]; then
    echo "✓ Docker daemon.json exists:"
    cat /etc/docker/daemon.json
else
    echo "✗ No /etc/docker/daemon.json found"
fi
echo

echo "3. Checking available Docker runtimes:"
docker info | grep -A 10 "Runtimes:"
echo

echo "4. Testing NVIDIA GPU access in Docker:"
if docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.4-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU access works in Docker"
else
    echo "✗ NVIDIA GPU access failed - may need to install nvidia-docker2"
    echo "  Install with: sudo apt-get install nvidia-docker2"
fi
echo

echo "5. Checking if Ollama uses GPU:"
echo "Run this command and look for GPU usage:"
echo "nvidia-smi"
echo "Then in another terminal:"
echo "ollama run gemma2:2b-instruct-q4_0 'Hello'"
echo "Watch nvidia-smi output for GPU activity"