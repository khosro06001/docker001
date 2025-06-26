#!/usr/bin/env python3
"""
Health check script for Gemma2 chatbot container
Verifies connection to Ollama server and model availability
"""

import sys
import os
import requests
import json
from datetime import datetime

def detect_ollama_url():
    """Detect Ollama URL using same logic as main app"""
    # Check environment variables first
    if os.getenv('OLLAMA_HOST'):
        host = os.getenv('OLLAMA_HOST')
        if not host.startswith('http'):
            host = f"http://{host}"
        return host
    
    # Try common container networking URLs
    possible_urls = [
        "http://host.docker.internal:11434",  # Docker Desktop
        "http://172.17.0.1:11434",           # Default Docker bridge
        "http://localhost:11434"              # Host network mode
    ]
    
    for url in possible_urls:
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                return url
        except:
            continue
    
    return "http://localhost:11434"  # Fallback

def check_ollama_connection(base_url, model_name="gemma2:2b-instruct-q4_0"):
    """Check if Ollama is accessible and model is available"""
    try:
        # Test basic connectivity
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code != 200:
            print(f"ERROR: Ollama server not responding (status: {response.status_code})")
            return False
        
        # Check if required model is available
        models_data = response.json()
        available_models = [model['name'] for model in models_data.get('models', [])]
        
        if model_name not in available_models:
            print(f"ERROR: Required model '{model_name}' not found")
            print(f"Available models: {available_models}")
            return False
        
        # Test model responsiveness with a simple query
        test_payload = {
            "model": model_name,
            "prompt": "System: You are a helpful assistant.\n\nHuman: Hi\nAssistant: ",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 512,
                "max_tokens": 10
            }
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('response'):
                print(f"SUCCESS: Ollama and {model_name} are responding")
                return True
            else:
                print("ERROR: Model returned empty response")
                return False
        else:
            print(f"ERROR: Model generation failed (status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to Ollama at {base_url}")
        return False
    except requests.exceptions.Timeout:
        print("ERROR: Timeout connecting to Ollama")
        return False
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        return False

def main():
    """Main health check function"""
    print(f"Health check started at {datetime.now().isoformat()}")
    
    # Detect Ollama URL
    ollama_url = detect_ollama_url()
    print(f"Testing Ollama at: {ollama_url}")
    
    # Perform health check
    if check_ollama_connection(ollama_url):
        print("HEALTH CHECK PASSED")
        sys.exit(0)
    else:
        print("HEALTH CHECK FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()