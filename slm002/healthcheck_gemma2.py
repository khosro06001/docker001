#!/usr/bin/env python3
"""
Health check script for Gemma2 Chatbot Docker container
Verifies Ollama connection and model availability
"""

import requests
import sys
import os

def check_ollama_health():
    """Check if Ollama server is healthy and model is available"""
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        base_url = f"http://{ollama_host}"
        
        # Check server health
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f"Ollama server unhealthy: {response.status_code}")
            return False
        
        # Check if gemma2 model is available
        models = response.json()
        available_models = [model['name'] for model in models.get('models', [])]
        
        target_model = "gemma2:2b-instruct-q4_0"
        if target_model not in available_models:
            print(f"Model {target_model} not available")
            return False
        
        print("Health check passed: Ollama server and Gemma2 model ready")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    if check_ollama_health():
        sys.exit(0)
    else:
        sys.exit(1)