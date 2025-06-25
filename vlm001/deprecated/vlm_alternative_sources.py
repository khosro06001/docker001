#!/usr/bin/env python3
"""
MiniCPM-V Image Captioning with Alternative Model Sources
Supports: Ollama, ModelScope, Local files, and fallback models
"""

import argparse
import logging
import sys
import torch
import requests
import json
import base64
from pathlib import Path
from PIL import Image
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiSourceImageCaptioner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = None
        logger.info(f"Using device: {self.device}")
        
    def image_to_base64(self, image_path):
        """Convert image to base64 for API calls"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def test_ollama_connection(self):
        """Test if Ollama is running and has the model"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                minicpm_models = [m for m in models if 'minicpm' in m.get('name', '').lower()]
                if minicpm_models:
                    logger.info(f"Found Ollama models: {[m['name'] for m in minicpm_models]}")
                    return minicpm_models[0]['name']
                else:
                    logger.warning("Ollama is running but no MiniCPM models found")
                    return None
            return None
        except Exception as e:
            logger.info(f"Ollama not available: {e}")
            return None
    
    def load_model_ollama(self, model_name):
        """Load model via Ollama"""
        try:
            self.model_type = "ollama"
            self.ollama_model = model_name
            logger.info(f"Using Ollama model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup Ollama: {e}")
            return False
    
    def load_model_modelscope(self):
        """Load model from ModelScope"""
        try:
            from modelscope import AutoModel, AutoTokenizer
            
            model_names = [
                "AI-ModelScope/MiniCPM-V-2_6",
                "openbmb/MiniCPM-V-2",
                "AI-ModelScope/MiniCPM-V-2"
            ]
            
            for model_name in model_names:
                try:
                    logger.info(f"Trying ModelScope model: {model_name}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                    
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    
                    self.model_type = "modelscope"
                    logger.info(f"Successfully loaded from ModelScope: {model_name}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"ModelScope model {model_name} failed: {e}")
                    continue
            
            return False
            
        except ImportError:
            logger.error("ModelScope not installed. Install with: pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"ModelScope loading failed: {e}")
            return False
    
    def load_model_local(self):
        """Load model from local directory"""
        local_paths = [
            "./models/minicpm-v",
            "./models/minicpm-v-repo",
            "./models/MiniCPM-V",
            "~/models/minicpm-v"
        ]
        
        for path in local_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                try:
                    logger.info(f"Trying local model: {expanded_path}")
                    
                    from transformers import AutoModel, AutoTokenizer
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(expanded_path),
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    
                    self.model = AutoModel.from_pretrained(
                        str(expanded_path),
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        local_files_only=True
                    )
                    
                    self.model_type = "local"
                    logger.info(f"Successfully loaded local model from: {expanded_path}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Local model {expanded_path} failed: {e}")
                    continue
        
        logger.warning("No valid local models found")
        return False
    
    def load_model_fallback(self):
        """Load fallback models that don't require MiniCPM"""
        fallback_models = [
            ("Salesforce/blip-image-captioning-base", "blip"),
            ("Salesforce/blip-image-captioning-large", "blip"),
            ("nlpconnect/vit-gpt2-image-captioning", "vit-gpt2")
        ]
        
        for model_name, model_type in fallback_models:
            try:
                logger.info(f"Trying fallback model: {model_name}")
                
                if model_type == "blip":
                    from transformers import BlipProcessor, BlipForConditionalGeneration
                    
                    self.processor = BlipProcessor.from_pretrained(model_name)
                    self.model = BlipForConditionalGeneration.from_pretrained(model_name)
                    
                elif model_type == "vit-gpt2":
                    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
                    
                    self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
                    self.processor = ViTImageProcessor.from_pretrained(model_name)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                self.model_type = model_type
                logger.info(f"Successfully loaded fallback model: {model_name}")
                return True
                
            except Exception as e:
                logger.warning(f"Fallback model {model_name} failed: {e}")
                continue
        
        return False
    
    def load_model(self):
        """Load model from any available source"""
        logger.info("Attempting to load model from available sources...")
        
        # Method 1: Try Ollama first (fastest and most reliable)
        ollama_model = self.test_ollama_connection()
        if ollama_model:
            if self.load_model_ollama(ollama_model):
                return
        
        # Method 2: Try ModelScope
        logger.info("Trying ModelScope...")
        if self.load_model_modelscope():
            return
        
        # Method 3: Try local models
        logger.info("Trying local models...")
        if self.load_model_local():
            return
        
        # Method 4: Try fallback models
        logger.info("Trying fallback models...")
        if self.load_model_fallback():
            return
        
        # If all methods fail
        logger.error("All model loading methods failed!")
        self.print_installation_guide()
        sys.exit(1)
    
    def print_installation_guide(self):
        """Print comprehensive installation guide"""
        logger.info("\n" + "="*60)
        logger.info("MODEL INSTALLATION GUIDE")
        logger.info("="*60)
        logger.info("No models could be loaded. Try these options:")
        logger.info("")
        logger.info("OPTION 1 - Ollama (Recommended):")
        logger.info("  curl -fsSL https://ollama.com/install.sh | sh")
        logger.info("  ollama pull minicpm-v")
        logger.info("")
        logger.info("OPTION 2 - ModelScope:")
        logger.info("  pip install modelscope")
        logger.info("  python -c \"from modelscope import snapshot_download; snapshot_download('AI-ModelScope/MiniCPM-V-2_6')\"")
        logger.info("")
        logger.info("OPTION 3 - Manual Download:")
        logger.info("  Run the alternative download script:")
        logger.info("  bash alternative_download.sh")
        logger.info("")
        logger.info("OPTION 4 - Use different model:")
        logger.info("  pip install transformers")
        logger.info("  # Script will auto-download BLIP as fallback")
        logger.info("="*60)
    
    def caption_image(self, image_path: str, question: str = "Describe this image in detail.") -> str:
        """Generate caption for the given image"""
        try:
            # Load and validate image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Processing image: {image_path} ({image.size})")
            
            # Route to appropriate captioning method
            if self.model_type == "ollama":
                return self._caption_with_ollama(image_path, question)
            elif self.model_type == "modelscope":
                return self._caption_with_minicpm(image, question)
            elif self.model_type == "local":
                return self._caption_with_minicpm(image, question)
            elif self.model_type == "blip":
                return self._caption_with_blip(image, question)
            elif self.model_type == "vit-gpt2":
                return self._caption_with_vit_gpt2(image, question)
            else:
                raise Exception(f"Unknown model type: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to caption image: {e}")
            return f"Error: {str(e)}"
    
    def _caption_with_ollama(self, image_path, question):
        """Caption with Ollama API"""
        try:
            image_b64 = self.image_to_base64(image_path)
            
            url = "http://localhost:11434/api/generate"
            data = {
                "model": self.ollama_model,
                "prompt": question,
                "images": [image_b64],
                "stream": False
            }
            
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            caption = result.get('response', 'No response received')
            
            logger.info("Caption generated successfully with Ollama")
            return caption
            
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")
    
    def _caption_with_minicpm(self, image, question):
        """Caption with MiniCPM-V model"""
        msgs = [{'role': 'user', 'content': question}]
        
        with torch.no_grad():
            response = self.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7,
                max_new_tokens=512
            )
        
        logger.info("Caption generated successfully with MiniCPM-V")
        return response
    
    def _caption_with_blip(self, image, question):
        """Caption with BLIP model"""
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=100, num_beams=5)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        logger.info("Caption generated successfully with BLIP")
        return f"[BLIP] {caption}"
    
    def _caption_with_vit_gpt2(self, image, question):
        """Caption with ViT-GPT2 model"""
        with torch.no_grad():
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            output_ids = self.model.generate(pixel_values, max_length=50, num_beams=4)
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        logger.info("Caption generated successfully with ViT-GPT2")
        return f"[ViT-GPT2] {caption}"

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using multiple model sources')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--question', '-q', default='Describe this image in detail.',
                       help='Question/prompt for the model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--force-source', choices=['ollama', 'modelscope', 'local', 'fallback'],
                       help='Force specific model source')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize captioner
    captioner = MultiSourceImageCaptioner()
    
    # Override model loading if force-source is specified
    if args.force_source:
        logger.info(f"Forcing model source: {args.force_source}")
        if args.force_source == "ollama":
            ollama_model = captioner.test_ollama_connection()
            if ollama_model:
                captioner.load_model_ollama(ollama_model)
            else:
                logger.error("Ollama not available")
                sys.exit(1)
        elif args.force_source == "modelscope":
            if not captioner.load_model_modelscope():
                logger.error("ModelScope loading failed")
                sys.exit(1)
        elif args.force_source == "local":
            if not captioner.load_model_local():
                logger.error("Local model loading failed")
                sys.exit(1)
        elif args.force_source == "fallback":
            if not captioner.load_model_fallback():
                logger.error("Fallback model loading failed")
                sys.exit(1)
    else:
        captioner.load_model()
    
    # Generate caption
    caption = captioner.caption_image(args.image_path, args.question)
    
    # Output result
    print("\n" + "="*50)
    print("IMAGE CAPTION:")
    print("="*50)
    print(caption)
    print("="*50)
    print(f"Model type used: {captioner.model_type}")

if __name__ == "__main__":
    main()
