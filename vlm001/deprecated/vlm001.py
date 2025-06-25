#!/usr/bin/env python3
"""
MiniCPM-V Image Captioning Application
Enhanced with better error handling and fallback options
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
from PIL import Image
import psutil
import time
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageCaptioner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def test_huggingface_connection(self):
        """Test connection to Hugging Face Hub with detailed diagnostics"""
        test_urls = [
            "https://huggingface.co",
            "https://cdn-lfs.huggingface.co",
            "https://huggingface.co/api/models/openbmb/MiniCPM-V-2"
        ]
        
        for url in test_urls:
            try:
                logger.info(f"Testing connection to: {url}")
                response = requests.get(url, timeout=10, stream=True)
                logger.info(f"✓ {url} - Status: {response.status_code}")
            except Exception as e:
                logger.warning(f"✗ {url} - Error: {e}")
        
    def load_model_with_retry(self, model_name, max_retries=3):
        """Load model with retry logic and better error handling"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading attempt {attempt + 1}/{max_retries} for {model_name}")
                
                from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
                import os
                
                # Set environment variables for better stability
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # Increased timeout
                os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Use faster transfer
                os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
                
                # Configure quantization
                if self.device == "cpu":
                    logger.info("CPU device - using float32 without quantization")
                    quantization_config = None
                    torch_dtype = torch.float32
                    device_map = None
                else:
                    logger.info("GPU device - using 4-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    torch_dtype = torch.float16
                    device_map = "auto"
                
                # Load tokenizer first (smaller download)
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True,
                    local_files_only=False,
                    force_download=attempt > 0  # Force download on retry
                )
                
                # Load model
                logger.info("Loading model...")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    resume_download=True,
                    local_files_only=False,
                    force_download=attempt > 0,  # Force download on retry
                    low_cpu_mem_usage=True
                )
                
                logger.info(f"✓ Model {model_name} loaded successfully!")
                self._log_memory_usage()
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Progressive backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {model_name}")
                    return False
        
        return False
        
    def load_model(self):
        """Load MiniCPM-V model with multiple fallback strategies"""
        try:
            # Test connection first
            logger.info("Testing Hugging Face connectivity...")
            self.test_huggingface_connection()
            
            # Try different model variants
            model_variants = [
                "openbmb/MiniCPM-V-2",
                "openbmb/MiniCPM-V-2_6",
                "openbmb/MiniCPM-Llama3-V-2_5",  # Alternative variant
            ]
            
            for model_name in model_variants:
                logger.info(f"Trying model variant: {model_name}")
                if self.load_model_with_retry(model_name):
                    return
            
            # If all models failed, try offline mode
            logger.warning("All online models failed, trying offline mode...")
            self.try_offline_mode()
            
        except Exception as e:
            logger.error(f"Critical error in load_model: {e}")
            self.print_troubleshooting_guide()
            sys.exit(1)
    
    def try_offline_mode(self):
        """Try to load from local cache in offline mode"""
        try:
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            logger.info("Attempting offline mode...")
            from transformers import AutoModel, AutoTokenizer
            
            # Try to load from cache
            cache_dir = os.path.expanduser('~/.cache/huggingface/transformers')
            if os.path.exists(cache_dir):
                logger.info(f"Checking cache directory: {cache_dir}")
                # List cached models
                for item in os.listdir(cache_dir):
                    if 'minicpm' in item.lower():
                        logger.info(f"Found cached model: {item}")
            
            # Try loading cached model
            self.model = AutoModel.from_pretrained(
                "openbmb/MiniCPM-V-2",
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "openbmb/MiniCPM-V-2",
                local_files_only=True,
                trust_remote_code=True
            )
            
            logger.info("✓ Loaded model from cache in offline mode!")
            
        except Exception as e:
            logger.error(f"Offline mode failed: {e}")
            self.suggest_manual_download()
    
    def suggest_manual_download(self):
        """Suggest manual download methods"""
        logger.info("="*60)
        logger.info("MANUAL DOWNLOAD SUGGESTIONS:")
        logger.info("="*60)
        logger.info("Try these methods to download the model manually:")
        logger.info("")
        logger.info("Method 1 - Using git lfs:")
        logger.info("  sudo apt install git-lfs")
        logger.info("  git lfs install")
        logger.info("  git clone https://huggingface.co/openbmb/MiniCPM-V-2")
        logger.info("")
        logger.info("Method 2 - Using huggingface-hub:")
        logger.info("  pip install huggingface-hub")
        logger.info("  python -c \"from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM-V-2')\"")
        logger.info("")
        logger.info("Method 3 - Check proxy/firewall settings:")
        logger.info("  export https_proxy=your_proxy_url")
        logger.info("  export http_proxy=your_proxy_url")
        logger.info("")
        logger.info("Method 4 - Use VPN if behind corporate firewall")
        logger.info("="*60)
        
        # Try a simple fallback model for testing
        self.try_simple_fallback()
    
    def try_simple_fallback(self):
        """Try a simple model for basic testing"""
        try:
            logger.info("Trying simple fallback model for testing...")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model_type = "blip"
            
            logger.info("✓ Fallback BLIP model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Even fallback model failed: {e}")
            logger.error("Please check your internet connection and try again.")
            sys.exit(1)
    
    def print_troubleshooting_guide(self):
        """Print comprehensive troubleshooting guide"""
        logger.info("\n" + "="*60)
        logger.info("TROUBLESHOOTING GUIDE")
        logger.info("="*60)
        logger.info("1. Check internet connection:")
        logger.info("   curl -I https://huggingface.co")
        logger.info("")
        logger.info("2. Update pip and packages:")
        logger.info("   pip install --upgrade pip transformers torch")
        logger.info("")
        logger.info("3. Clear all caches:")
        logger.info("   rm -rf ~/.cache/huggingface/")
        logger.info("   rm -rf ~/.cache/pip/")
        logger.info("")
        logger.info("4. Check disk space:")
        logger.info("   df -h")
        logger.info("   (Model needs ~2-4GB free space)")
        logger.info("")
        logger.info("5. Try with different DNS:")
        logger.info("   sudo systemctl restart systemd-resolved")
        logger.info("")
        logger.info("6. Check for proxy/firewall issues")
        logger.info("="*60)
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            logger.info(f"System RAM: {memory.percent}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
            
            # GPU memory if available
            try:
                import GPUtil
                if self.device == "cuda" and GPUtil.getGPUs():
                    gpu = GPUtil.getGPUs()[0]
                    logger.info(f"GPU Memory: {gpu.memoryUtil*100:.1f}% used ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)")
            except ImportError:
                logger.info("GPUtil not available for GPU memory monitoring")
                
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
            
            # Handle different model types
            if hasattr(self, 'model_type') and self.model_type == "blip":
                return self._caption_with_blip(image, question)
            else:
                return self._caption_with_minicpm(image, question)
            
        except Exception as e:
            logger.error(f"Failed to caption image: {e}")
            return f"Error: {str(e)}"
    
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
        """Caption with BLIP fallback model"""
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=100, num_beams=5)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        logger.info("Caption generated successfully with BLIP (fallback)")
        return f"[Fallback model] {caption}"

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using MiniCPM-V')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--question', '-q', default='Describe this image in detail.',
                       help='Question/prompt for the model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
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
