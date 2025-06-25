#!/usr/bin/env python3
"""
Simplified Image Captioning Application for Testing
Uses BLIP-2 which is more reliable for downloads
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
from PIL import Image
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleImageCaptioner:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load BLIP-2 model (more reliable for testing)"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            model_name = "Salesforce/blip2-opt-2.7b"
            logger.info(f"Loading BLIP-2 model: {model_name}")
            
            self.processor = Blip2Processor.from_pretrained(model_name)
            
            # Use CPU-compatible settings
            if self.device == "cpu":
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
            else:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            if self.device == "cuda":
                self.model.to(self.device)
            
            logger.info("Model loaded successfully")
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Trying smaller BLIP model...")
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                model_name = "Salesforce/blip-image-captioning-base"
                logger.info(f"Loading fallback model: {model_name}")
                
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(model_name)
                
                if self.device == "cuda":
                    self.model.to(self.device)
                
                logger.info("Fallback model loaded successfully")
                self._log_memory_usage()
                
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                sys.exit(1)
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        try:
            memory = psutil.virtual_memory()
            logger.info(f"System RAM: {memory.percent}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
            
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"GPU Memory: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB allocated")
        except Exception as e:
            logger.warning(f"Could not log memory usage: {e}")
    
    def caption_image(self, image_path: str, question: str = None) -> str:
        """Generate caption for the given image"""
        try:
            # Load and validate image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Processing image: {image_path} ({image.size})")
            
            # Process image
            if question and question.strip():
                # Question-based captioning
                inputs = self.processor(image, question, return_tensors="pt")
            else:
                # Standard captioning
                inputs = self.processor(image, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True
                )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            logger.info("Caption generated successfully")
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image: {e}")
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using BLIP-2')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--question', '-q', help='Optional question about the image')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize captioner
    captioner = SimpleImageCaptioner()
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
