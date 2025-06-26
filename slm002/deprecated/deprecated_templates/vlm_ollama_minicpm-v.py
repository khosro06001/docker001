#!/usr/bin/env python3
"""
MiniCPM-V 8B Image Captioning Application - Ollama Version
Uses Ollama's API to interact with the minicpm-v:8b model only
"""

import argparse
import logging
import sys
import base64
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaImageCaptioner:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model_name = "minicpm-v:8b"
        self.api_url = f"{self.base_url}/api"
        logger.info(f"Using model: {self.model_name}")
        
    def test_ollama_connection(self):
        """Test connection to Ollama server and verify minicpm-v:8b is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"✓ Connected to Ollama server")
                
                # Check if minicpm-v:8b is available
                if self.model_name in available_models:
                    logger.info(f"✓ Model '{self.model_name}' is available")
                    return True
                else:
                    logger.error(f"✗ Model '{self.model_name}' not found")
                    logger.error("Please install it with: ollama pull minicpm-v:8b")
                    return False
            else:
                logger.error(f"Ollama server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("✗ Cannot connect to Ollama server")
            logger.error("Make sure Ollama is running with: ollama serve")
            return False
        except Exception as e:
            logger.error(f"Error testing Ollama connection: {e}")
            return False
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 string with size optimization"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for Ollama compatibility
                max_size = 512
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")
                
                # Convert to base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=70, optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                logger.info(f"Image encoded successfully")
                return img_str
                
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def caption_image(self, image_path: str, question: str = "Describe this image in detail.") -> str:
        """Generate caption for the given image using minicpm-v:8b"""
        try:
            # Validate image path
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": question,
                "images": [image_base64],
                "stream": False
            }
            
            logger.info("Sending request to Ollama...")
            
            # Make request to Ollama
            try:
                response = requests.post(
                    f"{self.api_url}/generate",
                    json=payload,
                    timeout=180,
                    headers={'Content-Type': 'application/json'}
                )
            except requests.exceptions.Timeout:
                logger.error("Request timed out")
                return "Error: Request timed out. Try restarting Ollama server."
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                return "Error: Connection lost. Ollama server may have crashed."
            
            if response.status_code == 200:
                result = response.json()
                caption = result.get('response', '').strip()
                
                if caption:
                    logger.info("Caption generated successfully")
                    return caption
                else:
                    logger.error("Empty response from Ollama")
                    return "Error: Empty response from model"
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return f"Error: Ollama API returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Failed to caption image: {e}")
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using MiniCPM-V 8B via Ollama')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--question', '-q', default='Describe this image in detail.',
                       help='Question/prompt for the model')
    parser.add_argument('--url', '-u', default='http://localhost:11434',
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize captioner
    captioner = OllamaImageCaptioner(base_url=args.url)
    
    # Test connection
    if not captioner.test_ollama_connection():
        sys.exit(1)
    
    # Generate caption
    logger.info("Generating caption...")
    caption = captioner.caption_image(args.image_path, args.question)
    
    # Output result
    print("\n" + "="*50)
    print("IMAGE CAPTION:")
    print("="*50)
    print(caption)
    print("="*50)
    
    # Show model info
    if args.verbose:
        print(f"\nModel used: {captioner.model_name}")
        print(f"Ollama server: {captioner.base_url}")

if __name__ == "__main__":
    main()
