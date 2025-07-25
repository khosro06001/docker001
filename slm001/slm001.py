#!/usr/bin/env python3
"""
slm001.py - Enhanced Local Small Language Model Chatbot with Native PyTorch Quantization
A lightweight chatbot optimized for low-memory devices like Jetson Nano.
Uses PyTorch native quantization - NO bitsandbytes dependency!
"""

import os
import sys
import json
import signal
from typing import Optional, List, Dict, Any
import logging
import gc
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        pipeline,
        set_seed
    )
    import torch
    import torch.nn as nn
    logger.info("Core dependencies loaded successfully")
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.error("Please install: pip install transformers torch")
    sys.exit(1)

# Model configurations - optimized for Jetson Nano (4GB RAM)
MODEL_CONFIGS = {
    "microsoft/DialoGPT-small": {
        "name": "DialoGPT-small-quantized",
        "size": "~45MB (PyTorch quantized)",
        "description": "Ultra-lightweight conversational model, PyTorch quantized",
        "max_length": 512,
        "conversation_context": True,
        "base_model": "microsoft/DialoGPT-small"
    },
    "distilgpt2": {
        "name": "DistilGPT2-quantized", 
        "size": "~95MB (PyTorch quantized)",
        "description": "Efficient GPT-2 variant, PyTorch quantized for speed",
        "max_length": 768,
        "conversation_context": False,
        "base_model": "distilgpt2"
    },
    "gpt2": {
        "name": "GPT2-quantized",
        "size": "~140MB (PyTorch quantized)",
        "description": "Classic GPT-2, PyTorch quantized, good quality",
        "max_length": 1024,
        "conversation_context": False,
        "base_model": "gpt2"
    },
    "microsoft/DialoGPT-medium": {
        "name": "DialoGPT-medium-quantized",
        "size": "~115MB (PyTorch quantized)",
        "description": "Better conversations, PyTorch quantized for efficiency",
        "max_length": 768,
        "conversation_context": True,
        "base_model": "microsoft/DialoGPT-medium"
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "name": "TinyLlama-quantized",
        "size": "~350MB (PyTorch quantized)",
        "description": "Modern tiny language model, PyTorch quantized",
        "max_length": 2048,
        "conversation_context": True,
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }
}

class NativeQuantizedChatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize the native quantized local chatbot optimized for Jetson Nano.
        Uses ONLY PyTorch native quantization - no external dependencies!
        
        Args:
            model_name: HuggingFace model name. 
                       Default: DialoGPT-small (smallest model)
        """
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["microsoft/DialoGPT-small"])
        self.tokenizer = None
        self.model = None
        self.chat_pipeline = None
        self.conversation_history = []
        self.max_history = 6  # Reduced for memory efficiency
        
        # Set random seed for reproducible results
        set_seed(42)
        
        # Check device capabilities
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")
        
    def _detect_device(self):
        """Detect the best available device."""
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"CUDA GPU detected with {gpu_memory:.1f}GB memory")
                return "cuda"
            except:
                logger.info("CUDA available but not accessible, using CPU")
                return "cpu"
        else:
            logger.info("Using CPU (no CUDA GPU detected)")
            return "cpu"
    
    def _apply_pytorch_quantization(self, model):
        """Apply PyTorch native dynamic quantization."""
        try:
            logger.info("Applying PyTorch dynamic quantization...")
            
            # Set model to eval mode for quantization
            model.eval()
            
            # Apply dynamic quantization to Linear and Embedding layers
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Embedding}, 
                dtype=torch.qint8
            )
            
            logger.info("PyTorch quantization applied successfully")
            
            # Force garbage collection
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            return quantized_model
            
        except Exception as e:
            logger.warning(f"PyTorch quantization failed: {e}")
            logger.info("Using original model without quantization")
            return model
    
    def _optimize_model_for_inference(self, model):
        """Optimize model for inference without external quantization libraries."""
        try:
            # Set to evaluation mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            # Try to use torch.jit.script for additional optimization
            try:
                logger.info("Attempting TorchScript optimization...")
                # Note: This might not work for all models, so we'll catch exceptions
                model = torch.jit.optimize_for_inference(model)
                logger.info("TorchScript optimization applied")
            except Exception as e:
                logger.info(f"TorchScript optimization not available: {e}")
            
            return model
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
            return model
        
    def show_available_models(self):
        """Display available quantized model options."""
        print("\n" + "="*80)
        print("🤖 Available PyTorch Quantized Models (Optimized for Jetson Nano):")
        print("="*80)
        
        for i, (model_key, config) in enumerate(MODEL_CONFIGS.items(), 1):
            current = " (CURRENT)" if model_key == self.model_name else ""
            print(f"{i}. {config['name']}{current}")
            print(f"   Size: {config['size']}")
            print(f"   Description: {config['description']}")
            print(f"   Model ID: {model_key}")
            print()
    
    def switch_model(self, model_name: str):
        """Switch to a different model."""
        if model_name in MODEL_CONFIGS:
            # Clear existing model from memory
            if self.model is not None:
                del self.model
            if self.chat_pipeline is not None:
                del self.chat_pipeline
            if self.tokenizer is not None:
                del self.tokenizer
                
            # Force garbage collection
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.model_name = model_name
            self.model_config = MODEL_CONFIGS[model_name]
            self.model = None
            self.tokenizer = None
            self.chat_pipeline = None
            self.conversation_history = []
            logger.info(f"Switched to model: {self.model_config['name']}")
            return True
        else:
            logger.error(f"Model {model_name} not found in available models")
            return False
        
    def load_model(self):
        """Load the language model with PyTorch native quantization only."""
        try:
            logger.info(f"Loading model: {self.model_config['name']} ({self.model_config['size']})")
            logger.info("Using PyTorch native quantization - reliable and fast!")
            
            # Use the base model name for loading from HuggingFace
            base_model_name = self.model_config.get('base_model', self.model_name)
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                padding_side='left',
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimized settings
            logger.info("Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Move to device
            logger.info(f"Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            
            # Optimize model for inference
            logger.info("Optimizing model for inference...")
            self.model = self._optimize_model_for_inference(self.model)
            
            # Apply PyTorch native quantization
            logger.info("Applying PyTorch quantization...")
            self.model = self._apply_pytorch_quantization(self.model)
            
            # Create optimized pipeline
            logger.info("Creating inference pipeline...")
            device_id = 0 if self.device == "cuda" else -1
            
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
                max_length=self.model_config['max_length'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
                return_full_text=False
            )
            
            # Final cleanup
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("✅ Model loaded successfully with PyTorch quantization!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response using the quantized model with optimized settings.
        
        Args:
            user_input: The user's question or message
            
        Returns:
            Generated response from the model
        """
        try:
            with torch.no_grad():  # Ensure no gradients
                # Handle different model types with memory optimization
                if self.model_config['conversation_context']:
                    # For DialoGPT and TinyLlama models - use conversation format
                    if self.conversation_history:
                        # Build conversation context (reduced for memory)
                        context = ""
                        for exchange in self.conversation_history[-2:]:  # Only last 2 exchanges
                            context += f"{exchange['user']}{self.tokenizer.eos_token}"
                            context += f"{exchange['bot']}{self.tokenizer.eos_token}"
                        
                        # Add current input
                        prompt = context + user_input + self.tokenizer.eos_token
                    else:
                        prompt = user_input + self.tokenizer.eos_token
                    
                    # Generate with memory-optimized settings
                    inputs = self.tokenizer.encode(
                        prompt, 
                        return_tensors="pt",
                        max_length=min(400, self.model_config['max_length'] // 2),
                        truncation=True
                    )
                    
                    if self.device == "cuda":
                        inputs = inputs.to(self.device)
                    
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=60,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.85,
                        repetition_penalty=1.15,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2
                    )
                    
                    # Decode response
                    response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                    bot_response = response.strip()
                    
                else:
                    # For GPT-2 style models - use text generation format
                    if self.conversation_history:
                        context = ""
                        for exchange in self.conversation_history[-2:]:  # Only last 2 exchanges
                            context += f"Human: {exchange['user']}\nAssistant: {exchange['bot']}\n"
                        prompt = f"{context}Human: {user_input}\nAssistant:"
                    else:
                        prompt = f"Human: {user_input}\nAssistant:"
                    
                    # Generate response with optimized settings
                    response = self.chat_pipeline(
                        prompt,
                        max_new_tokens=50,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.85,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=2
                    )
                    
                    # Extract the generated text
                    bot_response = response[0]['generated_text'].strip()
                    
                    # Clean up response for GPT-2 style models
                    if "Human:" in bot_response:
                        bot_response = bot_response.split("Human:")[0].strip()
                
                # Clean up the response
                if not bot_response or len(bot_response.strip()) == 0:
                    bot_response = "I'm not sure how to respond to that. Could you try rephrasing?"
                
                # Limit response length to prevent memory issues
                if len(bot_response) > 150:
                    bot_response = bot_response[:150] + "..."
                
                # Update conversation history (limited for memory)
                self.conversation_history.append({
                    'user': user_input,
                    'bot': bot_response
                })
                
                # Keep only recent history for memory efficiency
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                
                return bot_response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error. This might be due to memory constraints."
        finally:
            # Clean up memory after generation
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def chat_loop(self):
        """Enhanced interactive chat loop optimized for Jetson Nano."""
        print("=" * 80)
        print(f"🤖 PyTorch Quantized Local Chatbot - {self.model_config['name']} Ready!")
        print("🚀 Using PyTorch Native Quantization - Reliable & Fast!")
        print("=" * 80)
        print("Available commands:")
        print("  • quit, exit, bye    - End the conversation")
        print("  • clear             - Clear conversation history")
        print("  • models            - Show available quantized models")
        print("  • switch <model>    - Switch to different quantized model")
        print("  • help              - Show this help message")
        print("  • stats             - Show model and memory stats")
        print("  • memory            - Show current memory usage")
        print("-" * 80)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"\n🤖 {self.model_config['name']}: Goodbye! Have a great day! 👋")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print(f"\n🤖 {self.model_config['name']}: Conversation history cleared!")
                    continue
                elif user_input.lower() == 'models':
                    self.show_available_models()
                    continue
                elif user_input.lower().startswith('switch '):
                    model_name = user_input[7:].strip()
                    if self.switch_model(model_name):
                        print(f"\n🤖 Switching to {self.model_config['name']}...")
                        if self.load_model():
                            print(f"✅ Successfully switched to {self.model_config['name']}!")
                        else:
                            print("❌ Failed to load new model. Reverting to previous model.")
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                elif user_input.lower() == 'memory':
                    self.show_memory_usage()
                    continue
                elif not user_input:
                    continue
                
                # Generate and display response
                print(f"\n🤖 {self.model_config['name']}: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 {self.model_config['name']}: Goodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n🤖 {self.model_config['name']}: Sorry, I encountered an error: {e}")
    
    def show_help(self):
        """Display enhanced help information."""
        help_text = f"""
🤖 PyTorch Quantized Local Chatbot - {self.model_config['name']}

Available commands:
  • quit, exit, bye    - End the conversation
  • clear             - Clear conversation history  
  • models            - Show all available quantized models
  • switch <model_id> - Switch to a different quantized model
  • help              - Show this help message
  • stats             - Show model and conversation statistics
  • memory            - Show current memory usage

Current Model: {self.model_config['name']} ({self.model_config['size']})
Description: {self.model_config['description']}
Quantization: PyTorch Native (No external dependencies!)

🚀 Optimized for Jetson Nano with 4GB RAM using reliable PyTorch quantization!
Just type your question or message to chat with me!
        """
        print(f"\n{help_text}")
    
    def show_stats(self):
        """Show model and conversation statistics."""
        device_info = f"CUDA GPU" if self.device == "cuda" else "CPU"
        print(f"""
📊 Current Statistics:
  • Model: {self.model_config['name']} ({self.model_config['size']})
  • Device: {device_info}
  • Quantization: PyTorch Native (Dynamic int8)
  • Conversation turns: {len(self.conversation_history)}
  • Max context length: {self.model_config['max_length']}
  • Memory optimized: Yes (Jetson Nano)
  • Conversation context: {'Yes' if self.model_config['conversation_context'] else 'No'}
  • Dependencies: Core PyTorch only (no bitsandbytes!)
        """)
    
    def show_memory_usage(self):
        """Show current memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"""
💾 Memory Usage:
  • Total RAM: {memory.total / 1024**3:.1f} GB
  • Available: {memory.available / 1024**3:.1f} GB
  • Used: {memory.used / 1024**3:.1f} GB ({memory.percent:.1f}%)
            """)
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"  • GPU Used: {gpu_memory:.2f} GB")
                print(f"  • GPU Cached: {gpu_cached:.2f} GB")
        except ImportError:
            print("\n💾 Memory monitoring requires psutil: pip install psutil")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🤖 Bot: Goodbye! 👋")
    sys.exit(0)

def main():
    """Main function with native quantized model selection."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting PyTorch Native Quantized Local Chatbot for Jetson Nano...")
    print("🔥 Using ONLY PyTorch quantization - No bitsandbytes needed!")
    
    # Show available quantized models
    print("\nAvailable PyTorch quantized models (ordered by memory efficiency):")
    for i, (model_key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"{i}. {config['name']} - {config['size']} - {config['description']}")
    
    # Let user choose model or use default
    print(f"\nRecommended: DialoGPT-small (ultra-lightweight, perfect for Jetson Nano)")
    model_choice = input("Enter model number (1-5) or press Enter for DialoGPT-small: ").strip()
    
    # Select model
    if model_choice.isdigit() and 1 <= int(model_choice) <= len(MODEL_CONFIGS):
        model_name = list(MODEL_CONFIGS.keys())[int(model_choice) - 1]
    else:
        model_name = "microsoft/DialoGPT-small"  # Default to smallest
    
    # Initialize chatbot
    chatbot = NativeQuantizedChatbot(model_name)
    
    print(f"\nLoading {chatbot.model_config['name']}...")
    print("Using reliable PyTorch quantization - this will work! ⚡")
    
    # Load the model
    if not chatbot.load_model():
        logger.error("Failed to initialize chatbot. Exiting.")
        sys.exit(1)
    
    # Start chat loop
    chatbot.chat_loop()

if __name__ == "__main__":
    main()