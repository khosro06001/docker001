#!/usr/bin/env python3
"""
Health check script for Gemma2 chatbot container
Verifies that the chatbot can connect to Ollama and the required model is available
"""

import sys
import requests
import os
import json

def main():
    """Health check main function"""
    try:
        # Detect Ollama URL (same logic as in the main chatbot)
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        if not ollama_host.startswith('http'):
            base_url = f"http://{ollama_host}"
        else:
            base_url = ollama_host
        
        # Try multiple URLs if in container
        urls_to_try = [
            base_url,
            "http://host.docker.internal:11434",
            "http://172.17.0.1:11434",
            "http://localhost:11434"
        ]
        
        model_name = "gemma2:2b-instruct-q4_0"
        
        for url in urls_to_try:
            try:
                # Test connection to Ollama
                response = requests.get(f"{url}/api/tags", timeout=10)
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    
                    # Check if required model is available
                    if model_name in available_models:
                        print(f"✓ Health check passed - Connected to {url}, model {model_name} available")
                        sys.exit(0)
                    else:
                        print(f"✗ Model {model_name} not found at {url}")
                        continue
                        
            except requests.exceptions.RequestException:
                continue
        
        # If we get here, no URL worked
        print("✗ Health check failed - Cannot connect to Ollama or model not available")
        sys.exit(1)
        
    except Exception as e:
        print(f"✗ Health check failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()