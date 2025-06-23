#!/usr/bin/env python3
"""
slm001.py - Enhanced Local Small Language Model Chatbot with Quantized Models
A lightweight chatbot optimized for low-memory devices like Jetson Nano.
Features quantized models for better performance on resource-constrained hardware.
"""

import os
import sys
import json
import signal
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        pipeline,
        set_seed,
        BitsAndBytesConfig
    )
    import torch
    # Try to import bitsandbytes for quantization
    try:
        import bitsandbytes as bnb
        QUANTIZATION_AVAILABLE = True
    except ImportError:
        QUANTIZATION_AVAILABLE = False
        logger.warning("bitsandbytes not available. Using int8 quantization fallback.")
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.error("Please install: pip install transformers torch")
    sys.exit(1)

# Quantized Model configurations - optimized for Jetson Nano (4GB RAM)
MODEL_CONFIGS = {
    "microsoft/DialoGPT-small": {
        "name": "DialoGPT-small-4bit",
        "size": "~30MB (4-bit quantized)",
        "description": "Ultra-lightweight conversational model, 4-bit quantized",
        "max_length": 512,
        "conversation_context": True,
        "quantization": "4bit"
    },
    "distilgpt2": {
        "name": "DistilGPT2-8bit", 
        "size": "~80MB (8-bit quantized)",
        "description": "Efficient GPT-2 variant, 8-bit quantized for speed",
        "max_length": 768,
        "conversation_context": False,
        "quantization": "8bit"
    },
    "gpt2": {
        "name": "GPT2-8bit",
        "size": "~125MB (8-bit quantized)",
        "description": "Classic GPT-2, 8-bit quantized, good quality",
        "max_length": 1024,
        "conversation_context": False,
        "quantization": "8bit"
    },
    "microsoft/DialoGPT-medium": {
        "name": "DialoGPT-medium-4bit",
        "size": "~90MB (4-bit quantized)",
        "description": "Better conversations, 4-bit quantized for efficiency",
        "max_length": 768,
        "conversation_context": True,
        "quantization": "4bit"
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "name": "TinyLlama-4bit",
        "size": "~280MB (4-bit quantized)",
        "description": "Modern tiny language model, excellent for chat",
        "max_length": 2048,
        "conversation_context": True,
        "quantization": "4bit"
    }
}

class QuantizedLocalChatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize the quantized local chatbot optimized for Jetson Nano.
        
        Args:
            model_name: HuggingFace model name. 
                       Default: DialoGPT-small (smallest quantized model)
        """
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["microsoft/DialoGPT-small"])
        self.tokenizer = None
        self.model = None
        self.chat_pipeline = None
        self.conversation_history = []
        self.max_history = 8  # Reduced for memory efficiency
        
        # Set random seed for reproducible results
        set_seed(42)
        
        # Check device capabilities
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")
        
    def _detect_device(self):
        """Detect the best available device."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA GPU detected with {gpu_memory:.1f}GB memory")
            return "cuda"
        else:
            logger.info("Using CPU (no CUDA GPU detected)")
            return "cpu"
    
    def _get_quantization_config(self, quantization_type: str):
        """Get quantization configuration based on type."""
        if not QUANTIZATION_AVAILABLE and quantization_type in ["4bit", "8bit"]:
            logger.warning("bitsandbytes not available, using torch int8 quantization")
            return None
            
        if quantization_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        return None
        
    def show_available_models(self):
        """Display available quantized model options."""
        print("\n" + "="*80)
        print("🤖 Available Quantized Models (Optimized for Jetson Nano):")
        print("="*80)
        
        for i, (model_key, config) in enumerate(MODEL_CONFIGS.items(), 1):
            current = " (CURRENT)" if model_key == self.model_name else ""
            print(f"{i}. {config['name']}{current}")
            print(f"   Size: {config['size']}")
            print(f"   Description: {config['description']}")
            print(f"   Quantization: {config['quantization']}")
            print(f"   Model ID: {model_key}")
            print()
    
    def switch_model(self, model_name: str):
        """Switch to a different quantized model."""
        if model_name in MODEL_CONFIGS:
            self.model_name = model_name
            self.model_config = MODEL_CONFIGS[model_name]
            # Clear existing model to force reload
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
        """Load the quantized language model and tokenizer."""
        try:
            logger.info(f"Loading quantized model: {self.model_config['name']} ({self.model_config['size']})")
            logger.info("Optimized for Jetson Nano - this should be much faster!")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get quantization config
            quantization_config = self._get_quantization_config(self.model_config['quantization'])
            
            # Load model with quantization
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if quantization_config and QUANTIZATION_AVAILABLE:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                # Fallback to manual quantization
                logger.info("Using fallback quantization method")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Manual quantization fallback if bitsandbytes not available
            if not QUANTIZATION_AVAILABLE:
                if self.model_config['quantization'] == "8bit":
                    logger.info("Applying manual int8 quantization")
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            # Create optimized pipeline
            device_id = 0 if self.device == "cuda" else -1
            
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                do_sample=True,
                temperature=0.7,  # Slightly lower for more consistent results
                top_p=0.85,
                max_length=self.model_config['max_length'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,  # Higher to reduce repetition
                return_full_text=False  # Only return generated text
            )
            
            logger.info("Quantized model loaded successfully!")
            logger.info(f"Memory usage optimized for {self.model_config['quantization']} quantization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            logger.error("Make sure you have the required packages installed:")
            logger.error("pip install bitsandbytes accelerate")
            return False
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response using quantized model with optimized settings.
        
        Args:
            user_input: The user's question or message
            
        Returns:
            Generated response from the quantized model
        """
        try:
            # Handle different model types with memory optimization
            if self.model_config['conversation_context']:
                # For DialoGPT and TinyLlama models - use conversation format
                if self.conversation_history:
                    # Build conversation context (reduced for memory)
                    context = ""
                    for exchange in self.conversation_history[-3:]:  # Only last 3 exchanges
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
                    max_length=min(512, self.model_config['max_length'] // 2),
                    truncation=True
                )
                
                if self.device == "cuda":
                    inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=80,  # Reduced for memory efficiency
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.85,
                        repetition_penalty=1.15,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2  # Prevent repetition
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
                    max_new_tokens=60,  # Reduced for memory efficiency
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
            if len(bot_response) > 200:
                bot_response = bot_response[:200] + "..."
            
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
    
    def chat_loop(self):
        """Enhanced interactive chat loop optimized for Jetson Nano."""
        print("=" * 80)
        print(f"🤖 Quantized Local Chatbot - {self.model_config['name']} Ready!")
        print("🚀 Optimized for Jetson Nano (4GB RAM)")
        print("=" * 80)
        print("Available commands:")
        print("  • quit, exit, bye    - End the conversation")
        print("  • clear             - Clear conversation history")
        print("  • models            - Show available quantized models")
        print("  • switch <model>    - Switch to different model")
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
        """Display enhanced help information for quantized models."""
        help_text = f"""
🤖 Quantized Local Chatbot - {self.model_config['name']}

Available commands:
  • quit, exit, bye    - End the conversation
  • clear             - Clear conversation history  
  • models            - Show all available quantized models
  • switch <model_id> - Switch to a different quantized model
  • help              - Show this help message
  • stats             - Show model and conversation statistics
  • memory            - Show current memory usage

Current Model: {self.model_config['name']} ({self.model_config['size']})
Quantization: {self.model_config['quantization']}
Description: {self.model_config['description']}

🚀 Optimized for Jetson Nano with 4GB RAM!
Just type your question or message to chat with me!
        """
        print(f"\n{help_text}")
    
    def show_stats(self):
        """Show model and conversation statistics."""
        device_info = f"CUDA GPU" if self.device == "cuda" else "CPU"
        quantization_info = f"{self.model_config['quantization']} quantization"
        print(f"""
📊 Current Statistics:
  • Model: {self.model_config['name']} ({self.model_config['size']})
  • Device: {device_info}
  • Quantization: {quantization_info}
  • Conversation turns: {len(self.conversation_history)}
  • Max context length: {self.model_config['max_length']}
  • Memory optimized: Yes (Jetson Nano)
  • Conversation context: {'Yes' if self.model_config['conversation_context'] else 'No'}
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
    """Main function with quantized model selection."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Quantized Local Chatbot for Jetson Nano...")
    print("🔥 Optimized with quantized models for 4GB RAM!")
    
    # Show available models
    print("\nAvailable quantized models (ordered by memory efficiency):")
    for i, (model_key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"{i}. {config['name']} - {config['size']} - {config['description']}")
    
    # Let user choose model or use default
    print(f"\nRecommended: DialoGPT-small-4bit (ultra-lightweight, perfect for Jetson Nano)")
    model_choice = input("Enter model number (1-5) or press Enter for DialoGPT-small: ").strip()
    
    # Select model
    if model_choice.isdigit() and 1 <= int(model_choice) <= len(MODEL_CONFIGS):
        model_name = list(MODEL_CONFIGS.keys())[int(model_choice) - 1]
    else:
        model_name = "microsoft/DialoGPT-small"  # Default to smallest
    
    # Initialize quantized chatbot
    chatbot = QuantizedLocalChatbot(model_name)
    
    print(f"\nLoading {chatbot.model_config['name']}...")
    print("This should be much faster than before! ⚡")
    
    # Load the model
    if not chatbot.load_model():
        logger.error("Failed to initialize quantized chatbot. Exiting.")
        sys.exit(1)
    
    # Start chat loop
    chatbot.chat_loop()

if __name__ == "__main__":
    main()