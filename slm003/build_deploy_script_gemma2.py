#!/usr/bin/env python3
"""
Multi-architecture Docker Build and Deploy Script for Gemma2 Chatbot
Builds for both AMD64 (CPU) and ARM64 (Jetson Nano GPU) architectures
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

class DockerBuildDeployer:
    def __init__(self, image_name="gemma2-chatbot", registry="docker.io", 
                 dockerfile="Dockerfile", tag="latest"):
        self.image_name = image_name
        self.registry = registry
        self.dockerfile = dockerfile
        self.tag = tag
        self.full_image_name = f"{registry}/{image_name}:{tag}"
        self.build_log = []
        
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.build_log.append(log_entry)
        
    def run_command(self, cmd, check=True, capture_output=False):
        """Run shell command with logging"""
        self.log(f"Running: {cmd}")
        
        try:
            if capture_output:
                result = subprocess.run(cmd, shell=True, check=check, 
                                      capture_output=True, text=True)
                return result.stdout.strip()
            else:
                result = subprocess.run(cmd, shell=True, check=check)
                return result.returncode == 0
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            if check:
                raise
            return False
    
    def check_prerequisites(self):
        """Check if required tools are installed"""
        self.log("Checking prerequisites...")
        
        # Check Docker
        try:
            docker_version = self.run_command("docker --version", capture_output=True)
            self.log(f"Docker found: {docker_version}")
        except:
            self.log("Docker not found! Please install Docker.", "ERROR")
            return False
            
        # Check buildx
        try:
            buildx_version = self.run_command("docker buildx version", capture_output=True)
            self.log(f"Docker buildx found: {buildx_version}")
        except:
            self.log("Docker buildx not found! Please install buildx.", "ERROR")
            return False
            
        # Check if logged into Docker Hub
        try:
            self.run_command("docker info | grep Username", capture_output=True)
            self.log("Docker login verified")
        except:
            self.log("Not logged into Docker Hub. Please run 'docker login'", "WARN")
            
        return True
    
    def create_missing_files(self):
        """Create missing required files"""
        self.log("Creating missing files if needed...")
        
        # Create requirements.txt if it doesn't exist
        requirements_file = "requirements_gemma2.txt"
        if not Path(requirements_file).exists():
            self.log(f"Creating {requirements_file}")
            with open(requirements_file, 'w') as f:
                f.write("""requests>=2.31.0
readline
argparse
pathlib
""")
        
        # Create Dockerfile if it doesn't exist
        if not Path(self.dockerfile).exists():
            self.log(f"Creating {self.dockerfile}")
            dockerfile_content = """# Multi-stage Dockerfile for Gemma2 Interactive Chatbot via Ollama
# Supports both AMD64 (CPU-only) and ARM64 (GPU-accelerated for Jetson Nano)

FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3 AS base-arm64
FROM python:3.9-slim AS base-amd64

# Select base image based on target architecture
FROM base-${TARGETARCH} AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create app directory and user
WORKDIR /app
RUN groupadd -r chatuser && useradd -r -g chatuser chatuser

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    git \\
    ca-certificates \\
    readline-common \\
    libreadline8 \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_gemma2.txt .
RUN pip install --no-cache-dir -r requirements_gemma2.txt

# Copy application files
COPY slm_chatbot_gemma2.py .
RUN chmod +x slm_chatbot_gemma2.py

# Create data directories for conversation storage
RUN mkdir -p /app/data /app/conversations && \\
    chown -R chatuser:chatuser /app

# Health check
COPY healthcheck_gemma2.py .
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \\
    CMD python3 healthcheck_gemma2.py

# Switch to non-root user
USER chatuser

# Expose port (for potential future web interface)
EXPOSE 8080

# Default command - starts interactive chatbot
CMD ["python3", "slm_chatbot_gemma2.py", "--verbose"]

# Labels
LABEL maintainer="AI Developer <ai@example.com>"
LABEL description="Gemma2 Interactive Chatbot via Ollama for Multi-Architecture"
LABEL version="1.0"
LABEL ollama.required="true"
LABEL ollama.model="gemma2:2b-instruct-q4_0"
LABEL architecture.amd64="CPU-only"
LABEL architecture.arm64="GPU-accelerated for Jetson"
"""
            with open(self.dockerfile, 'w') as f:
                f.write(dockerfile_content)
    
    def setup_buildx(self):
        """Setup Docker buildx for multi-architecture builds"""
        self.log("Setting up Docker buildx...")
        
        # Create new builder instance
        builder_name = "gemma2-builder"
        self.run_command(f"docker buildx create --name {builder_name} --use", check=False)
        
        # Bootstrap the builder
        self.run_command("docker buildx inspect --bootstrap")
        
        # List available platforms
        platforms = self.run_command("docker buildx inspect --bootstrap | grep Platforms", 
                                    capture_output=True)
        self.log(f"Available platforms: {platforms}")
        
        return True
    
    def build_multiarch(self, platforms=["linux/amd64", "linux/arm64"], push=True):
        """Build multi-architecture images"""
        self.log(f"Building for platforms: {', '.join(platforms)}")
        
        # Prepare build command
        platform_str = ",".join(platforms)
        cmd = f"docker buildx build --platform {platform_str}"
        
        if push:
            cmd += " --push"
        else:
            cmd += " --load"
            
        cmd += f" -t {self.full_image_name} -f {self.dockerfile} ."
        
        self.log("Starting multi-architecture build...")
        success = self.run_command(cmd)
        
        if success:
            self.log("Multi-architecture build completed successfully!")
        else:
            self.log("Build failed!", "ERROR")
            
        return success
    
    def verify_images(self):
        """Verify that images were built and pushed successfully"""
        self.log("Verifying images...")
        
        # Check if images exist in registry
        try:
            inspect_cmd = f"docker buildx imagetools inspect {self.full_image_name}"
            result = self.run_command(inspect_cmd, capture_output=True)
            self.log("Image verification successful")
            self.log(f"Image details: {result}")
            return True
        except:
            self.log("Image verification failed", "ERROR")
            return False
    
    def generate_deployment_info(self):
        """Generate deployment information"""
        self.log("Generating deployment information...")
        
        deployment_info = {
            "image_name": self.full_image_name,
            "build_time": datetime.now().isoformat(),
            "architectures": ["linux/amd64", "linux/arm64"],
            "registry": self.registry,
            "dockerfile": self.dockerfile,
            "tag": self.tag,
            "deployment_commands": {
                "amd64": f"docker run -it --rm --network host -v $(pwd)/conversations:/app/conversations {self.full_image_name}",
                "arm64_jetson": f"nvidia-docker run -it --rm --network host -v $(pwd)/conversations:/app/conversations {self.full_image_name}"
            },
            "requirements": [
                "Ollama server running on host",
                "gemma2:2b-instruct-q4_0 model pulled in Ollama",
                "For ARM64: nvidia-docker runtime installed"
            ]
        }
        
        with open("deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
            
        self.log("Deployment info saved to deployment_info.json")
        return deployment_info
    
    def save_build_log(self):
        """Save build log to file"""
        log_filename = f"build_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_filename, 'w') as f:
            f.write('\n'.join(self.build_log))
        self.log(f"Build log saved to {log_filename}")

def main():
    parser = argparse.ArgumentParser(description='Build and deploy Gemma2 chatbot')
    parser.add_argument('--image-name', '-n', default='gemma2-chatbot',
                       help='Docker image name (default: gemma2-chatbot)')
    parser.add_argument('--registry', '-r', default='docker.io',
                       help='Docker registry (default: docker.io)')
    parser.add_argument('--tag', '-t', default='latest',
                       help='Image tag (default: latest)')
    parser.add_argument('--dockerfile', '-f', default='Dockerfile',
                       help='Dockerfile path (default: Dockerfile)')
    parser.add_argument('--no-push', action='store_true',
                       help='Build locally without pushing to registry')
    parser.add_argument('--amd64-only', action='store_true',
                       help='Build only for AMD64 architecture')
    parser.add_argument('--arm64-only', action='store_true',
                       help='Build only for ARM64 architecture')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    # Determine platforms
    if args.amd64_only:
        platforms = ["linux/amd64"]
    elif args.arm64_only:
        platforms = ["linux/arm64"]
    else:
        platforms = ["linux/amd64", "linux/arm64"]
    
    # Initialize deployer
    deployer = DockerBuildDeployer(
        image_name=args.image_name,
        registry=args.registry,
        dockerfile=args.dockerfile,
        tag=args.tag
    )
    
    try:
        deployer.log("Starting build and deployment process...")
        
        # Check prerequisites
        if not args.skip_checks:
            if not deployer.check_prerequisites():
                sys.exit(1)
        
        # Create missing files
        deployer.create_missing_files()
        
        # Setup buildx
        deployer.setup_buildx()
        
        # Build images
        success = deployer.build_multiarch(platforms=platforms, push=not args.no_push)
        
        if not success:
            deployer.log("Build failed!", "ERROR")
            sys.exit(1)
        
        # Verify images (only if pushed)
        if not args.no_push:
            deployer.verify_images()
        
        # Generate deployment info
        deployment_info = deployer.generate_deployment_info()
        
        # Save build log
        deployer.save_build_log()
        
        deployer.log("Build and deployment completed successfully!")
        
        # Print usage instructions
        print("\n" + "="*60)
        print("🚀 BUILD COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Image: {deployer.full_image_name}")
        print(f"Platforms: {', '.join(platforms)}")
        print("\n📋 DEPLOYMENT COMMANDS:")
        print("\nAMD64 (CPU-only):")
        print(f"  {deployment_info['deployment_commands']['amd64']}")
        print("\nARM64 Jetson (GPU-accelerated):")
        print(f"  {deployment_info['deployment_commands']['arm64_jetson']}")
        print("\n⚠️  REQUIREMENTS:")
        for req in deployment_info['requirements']:
            print(f"  - {req}")
        print("="*60)
        
    except KeyboardInterrupt:
        deployer.log("Build interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        deployer.log(f"Unexpected error: {e}", "ERROR")
        deployer.save_build_log()
        sys.exit(1)

if __name__ == "__main__":
    main()