#!/usr/bin/env python3
"""
MiniCPM-V Image Captioning Application - Ollama Version
Uses Ollama's API to interact with locally downloaded MiniCPM-V model
"""

import argparse
import logging
import sys
import base64
import json
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaImageCaptioner:
    def __init__(self, base_url="http://localhost:11434", model_name="minicpm-v:latest"):
        self.base_url = base_url
        self.model_name = model_name
        self.api_url = f"{self.base_url}/api"
        logger.info(f"Initializing Ollama captioner with model: {model_name}")
        
    def test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"✓ Connected to Ollama server")
                logger.info(f"Available models: {available_models}")
                
                # Check if our target model is available
                model_found = any(self.model_name in model for model in available_models)
                if not model_found:
                    # Try to find exact match or suggest alternatives
                    exact_matches = [m for m in available_models if m == self.model_name]
                    partial_matches = [m for m in available_models if self.model_name.split(':')[0] in m]
                    
                    if exact_matches:
                        logger.info(f"✓ Model '{self.model_name}' is available")
                        return True
                    elif partial_matches:
                        logger.warning(f"Exact model '{self.model_name}' not found, but found similar: {partial_matches}")
                        # Auto-select the first partial match
                        self.model_name = partial_matches[0]
                        logger.info(f"✓ Auto-selected model: {self.model_name}")
                        return True
                    else:
                        logger.warning(f"Model '{self.model_name}' not found in available models")
                        logger.info("Available models that might work:")
                        for model in available_models:
                            if any(keyword in model.lower() for keyword in ['minicpm', 'cpm', 'vision', 'v', 'llava']):
                                logger.info(f"  - {model}")
                        return False
                else:
                    logger.info(f"✓ Model '{self.model_name}' is available")
                    return True
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
    
    def list_available_models(self):
        """List all available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def suggest_model_name(self):
        """Suggest the correct model name based on available models"""
        available_models = self.list_available_models()
        
        # Common variations of MiniCPM-V model names in Ollama
        possible_names = [
            'minicpm-v',
            'minicpm-v2',
            'minicpm-v:latest',
            'minicpm-v2:latest',
            'minicpm-v:2b',
            'minicpm-v:8b',
            'openbmb/minicpm-v',
            'openbmb/minicpm-v2'
        ]
        
        for model in available_models:
            if any(name.lower() in model.lower() for name in ['minicpm', 'cpm']):
                logger.info(f"Found potential MiniCPM model: {model}")
                return model
        
        return None
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 string with aggressive size reduction"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # More aggressive resizing for Ollama stability
                max_size = 512  # Reduced from 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size} for Ollama compatibility")
                
                # Convert to base64 with lower quality
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=70, optimize=True)  # Reduced quality
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                logger.info(f"Image encoded successfully (size: {len(img_str)} chars)")
                
                # Warn if still too large
                if len(img_str) > 100000:  # ~100KB base64
                    logger.warning("Image is still quite large, this might cause Ollama to crash")
                    logger.warning("Consider using a smaller image or lower quality")
                
                return img_str
                
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def caption_image(self, image_path: str, question: str = "Describe this image in detail.") -> str:
        """Generate caption for the given image using Ollama"""
        try:
            # Validate image path
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Prepare the request payload - simplified for stability
            payload = {
                "model": self.model_name,
                "prompt": question,
                "images": [image_base64],
                "stream": False
                # Removed options that might cause issues
            }
            
            logger.info("Sending request to Ollama...")
            
            # Make request to Ollama with shorter timeout and better error handling
            try:
                response = requests.post(
                    f"{self.api_url}/generate",
                    json=payload,
                    # timeout=60,  # Shorter timeout
                    timeout=180,  # Shorter timeout
                    headers={'Content-Type': 'application/json'}
                )
            except requests.exceptions.Timeout:
                logger.error("Request timed out - Ollama might be overloaded")
                return "Error: Request timed out. Try restarting Ollama server."
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                return "Error: Connection lost. Ollama server may have crashed. Try restarting with 'ollama serve'"
            
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
    
    def interactive_model_selection(self):
        """Interactively select the correct model if the default doesn't work"""
        available_models = self.list_available_models()
        
        if not available_models:
            logger.error("No models found in Ollama")
            return False
        
        # Filter for vision models
        vision_models = [model for model in available_models 
                        if any(keyword in model.lower() 
                              for keyword in ['minicpm', 'cpm', 'vision', 'llava', 'bakllava'])]
        
        if not vision_models:
            logger.warning("No obvious vision models found. Showing all models:")
            vision_models = available_models[:10]  # Show first 10 models
        
        print("\nAvailable vision models:")
        for i, model in enumerate(vision_models, 1):
            print(f"{i}. {model}")
        
        try:
            choice = input(f"\nSelect model (1-{len(vision_models)}) or press Enter to keep '{self.model_name}': ").strip()
            if choice and choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(vision_models):
                    self.model_name = vision_models[choice_idx]
                    logger.info(f"Selected model: {self.model_name}")
                    return True
        except (ValueError, KeyboardInterrupt):
            pass
        
        return True
    
    def print_troubleshooting_guide(self):
        """Print troubleshooting guide for Ollama"""
        logger.info("\n" + "="*60)
        logger.info("OLLAMA TROUBLESHOOTING GUIDE")
        logger.info("="*60)
        logger.info("1. Make sure Ollama is running:")
        logger.info("   ollama serve")
        logger.info("")
        logger.info("2. Check if MiniCPM-V is installed:")
        logger.info("   ollama list")
        logger.info("")
        logger.info("3. Install MiniCPM-V if not present:")
        logger.info("   ollama pull minicpm-v")
        logger.info("   # or")
        logger.info("   ollama pull minicpm-v:latest")
        logger.info("")
        logger.info("4. Test the model:")
        logger.info("   ollama run minicpm-v")
        logger.info("")
        logger.info("5. Check Ollama status:")
        logger.info("   curl http://localhost:11434/api/tags")
        logger.info("")
        logger.info("6. If using different port, use --port argument")
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using MiniCPM-V via Ollama')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--question', '-q', default='Describe this image in detail.',
                       help='Question/prompt for the model')
    parser.add_argument('--model', '-m', default='minicpm-v:latest',
                       help='Ollama model name (default: minicpm-v:latest)')
    parser.add_argument('--url', '-u', default='http://localhost:11434',
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--port', '-p', type=int,
                       help='Ollama server port (overrides URL port)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Interactive model selection if default fails')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle port override
    if args.port:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(args.url)
        args.url = urlunparse(parsed._replace(netloc=f"{parsed.hostname}:{args.port}"))
    
    # Initialize captioner
    captioner = OllamaImageCaptioner(base_url=args.url, model_name=args.model)
    
    # Test connection
    if not captioner.test_ollama_connection():
        if args.interactive:
            logger.info("Trying interactive model selection...")
            if not captioner.interactive_model_selection():
                captioner.print_troubleshooting_guide()
                sys.exit(1)
        else:
            captioner.print_troubleshooting_guide()
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
