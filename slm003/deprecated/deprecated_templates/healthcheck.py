#!/usr/bin/env python3
"""
Health check script for Ollama-based VLM container
Checks if required dependencies are available and Ollama server is reachable
"""



import sys
import requests
import os
from pathlib import Path

def check_python_imports():
    """Check if required Python packages are importable"""
    try:
        import PIL
        from PIL import Image
        import requests
        import base64
        print("✓ Python dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e}")
        return False

def check_ollama_connection():
    """Check if Ollama server is reachable"""
    ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
    if not ollama_host.startswith('http'):
        ollama_host = f"http://{ollama_host}"
    
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama server is reachable")
            return True
        else:
            print(f"✗ Ollama server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Ollama server at {ollama_host}")
        return False
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        return False

def check_script_exists():
    """Check if the main script exists"""
    script_path = Path("/app/vlm_ollama_minicpm-v.py")
    if script_path.exists():
        print("✓ Main script found")
        return True
    else:
        print("✗ Main script not found")
        return False

def check_directories():
    """Check if required directories exist"""
    dirs = ["/app/data", "/app/images"]
    all_exist = True
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory {dir_path} exists")
        else:
            print(f"✗ Directory {dir_path} missing")
            all_exist = False
    return all_exist

def main():
    print("=== Docker Container Health Check ===")
    
    checks = [
        check_python_imports(),
        check_script_exists(),
        check_directories(),
        check_ollama_connection()
    ]
    
    if all(checks):
        print("✓ Container is healthy")
        sys.exit(0)
    else:
        print("✗ Container health check failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
