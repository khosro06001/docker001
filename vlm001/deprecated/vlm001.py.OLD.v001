#!/usr/bin/env python3
"""
MiniCPM-V Image Captioning Application
Optimized for NVIDIA Jetson Nano with 4-bit quantization
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
from PIL import Image
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageCaptioner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load MiniCPM-V model with 4-bit quantization"""
        try:
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
            
            # 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model_name = "openbmb/MiniCPM-V-2"
            logger.info(f"Loading model: {model_name}")
            
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            logger.info(f"System RAM: {memory.percent}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
            
            # GPU memory
            if self.device == "cuda" and GPUtil.getGPUs():
                gpu = GPUtil.getGPUs()[0]
                logger.info(f"GPU Memory: {gpu.memoryUtil*100:.1f}% used ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)")
        except Exception as e:
            logger.warning(f"Could not log memory usage: {e}")
    
    def caption_image(self, image_path: str, question: str = "Describe this image in detail.") -> str:
        """Generate caption for the given image"""
        try:
            # Load and validate image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Processing image: {image_path} ({image.size})")
            
            # Generate caption
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
            
            logger.info("Caption generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to caption image: {e}")
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using MiniCPM-V')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--question', '-q', default='Describe this image in detail.',
                       help='Question/prompt for the model (default: "Describe this image in detail.")')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize captioner
    captioner = ImageCaptioner()
    captioner.load_model()
    
    # Generate caption
    caption = captioner.caption_image(args.image_path, args.question)
    
    # Output result
    print("\n" + "="*50)
    print("IMAGE CAPTION:")
    print("="*50)
    print(caption)
    print("="*50)

if __name__ == "__main__":
    main()
