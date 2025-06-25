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
