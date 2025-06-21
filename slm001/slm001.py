#!/usr/bin/env python3
"""
slm001.py - Enhanced Local Small Language Model Chatbot
A lightweight chatbot with multiple model options for better conversations.
Designed to work on both AMD64 and ARM64 architectures.
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
        set_seed
    )
    import torch
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.error("Please install: pip install transformers torch")
    sys.exit(1)

# Model configurations - ordered by capability and size
MODEL_CONFIGS = {
    "microsoft/DialoGPT-medium": {
        "name": "DialoGPT-medium",
        "size": "~345MB",
        "description": "Better conversational AI, good balance of size/quality",
        "max_length": 1000,
        "conversation_context": True
    },
    "microsoft/DialoGPT-large": {
        "name": "DialoGPT-large", 
        "size": "~775MB",
        "description": "Best DialoGPT model, high quality conversations",
        "max_length": 1000,
        "conversation_context": True
    },
    "distilgpt2": {
        "name": "DistilGPT2",
        "size": "~320MB", 
        "description": "Lightweight GPT-2, good general purpose model",
        "max_length": 1024,
        "conversation_context": False
    },
    "gpt2": {
        "name": "GPT-2",
        "size": "~500MB",
        "description": "Original GPT-2, very capable for text generation",
        "max_length": 1024,
        "conversation_context": False
    },
    "microsoft/DialoGPT-small": {
        "name": "DialoGPT-small",
        "size": "~117MB",
        "description": "Smallest model, basic conversations only",
        "max_length": 512,
        "conversation_context": True
    }
}

class EnhancedLocalChatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the enhanced local chatbot.
        
        Args:
            model_name: HuggingFace model name. 
                       Default: DialoGPT-medium (good balance of size/quality)
        """
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["microsoft/DialoGPT-medium"])
        self.tokenizer = None
        self.model = None
        self.chat_pipeline = None
        self.conversation_history = []
        self.max_history = 15  # Keep more history for better context
        
        # Set random seed for reproducible results
        set_seed(42)
        
    def show_available_models(self):
        """Display available model options."""
        print("\n" + "="*80)
        print("🤖 Available Models:")
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
        """Load the language model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_config['name']} ({self.model_config['size']})")
            logger.info("This may take a few minutes on first run...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for better compatibility
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Detect device
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if torch.cuda.is_available() else "CPU"
            logger.info(f"Using device: {device_name}")
            
            # Create pipeline with better parameters
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_length=self.model_config['max_length'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input with enhanced conversation handling.
        
        Args:
            user_input: The user's question or message
            
        Returns:
            Generated response from the model
        """
        try:
            # Handle different model types
            if self.model_config['conversation_context']:
                # For DialoGPT models - use conversation format
                if self.conversation_history:
                    # Build conversation context
                    context = ""
                    for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
                        context += f"{exchange['user']}{self.tokenizer.eos_token}"
                        context += f"{exchange['bot']}{self.tokenizer.eos_token}"
                    
                    # Add current input
                    prompt = context + user_input + self.tokenizer.eos_token
                else:
                    prompt = user_input + self.tokenizer.eos_token
                
                # Generate with conversation-specific settings
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                bot_response = response.strip()
                
            else:
                # For GPT-2 style models - use text generation format
                if self.conversation_history:
                    context = ""
                    for exchange in self.conversation_history[-3:]:
                        context += f"Human: {exchange['user']}\nAssistant: {exchange['bot']}\n"
                    prompt = f"{context}Human: {user_input}\nAssistant:"
                else:
                    prompt = f"Human: {user_input}\nAssistant:"
                
                # Generate response
                response = self.chat_pipeline(
                    prompt,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                
                # Extract the generated text
                generated_text = response[0]['generated_text']
                
                # Extract only the new response
                if "Assistant:" in generated_text:
                    bot_response = generated_text.split("Assistant:")[-1].strip()
                    bot_response = bot_response.split("Human:")[0].strip()
                else:
                    bot_response = "I'm not sure how to respond to that."
            
            # Clean up the response
            if not bot_response or len(bot_response.strip()) == 0:
                bot_response = "I'm not sure how to respond to that."
            
            # Update conversation history
            self.conversation_history.append({
                'user': user_input,
                'bot': bot_response
            })
            
            # Keep only recent history
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            return bot_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."
    
    def chat_loop(self):
        """Enhanced interactive chat loop with model management."""
        print("=" * 80)
        print(f"🤖 Enhanced Local Chatbot - {self.model_config['name']} Ready!")
        print("=" * 80)
        print("Available commands:")
        print("  • quit, exit, bye    - End the conversation")
        print("  • clear             - Clear conversation history")
        print("  • models            - Show available models")
        print("  • switch <model>    - Switch to different model")
        print("  • help              - Show this help message")
        print("  • stats             - Show model and conversation stats")
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
🤖 Enhanced Local Chatbot - {self.model_config['name']}

Available commands:
  • quit, exit, bye    - End the conversation
  • clear             - Clear conversation history  
  • models            - Show all available models
  • switch <model_id> - Switch to a different model
  • help              - Show this help message
  • stats             - Show model and conversation statistics

Current Model: {self.model_config['name']} ({self.model_config['size']})
Description: {self.model_config['description']}

Just type your question or message to chat with me!
        """
        print(f"\n{help_text}")
    
    def show_stats(self):
        """Show model and conversation statistics."""
        gpu_info = "GPU Available" if torch.cuda.is_available() else "CPU Only"
        print(f"""
📊 Current Statistics:
  • Model: {self.model_config['name']} ({self.model_config['size']})
  • Device: {gpu_info}
  • Conversation turns: {len(self.conversation_history)}
  • Max context length: {self.model_config['max_length']}
  • Conversation context: {'Yes' if self.model_config['conversation_context'] else 'No'}
        """)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🤖 Bot: Goodbye! 👋")
    sys.exit(0)

def main():
    """Main function with model selection."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Enhanced Local Chatbot...")
    
    # Show available models
    print("\nAvailable models (ordered by capability):")
    for i, (model_key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"{i}. {config['name']} - {config['size']} - {config['description']}")
    
    # Let user choose model or use default
    print(f"\nRecommended: DialoGPT-medium (good balance of quality and size)")
    model_choice = input("Enter model number (1-5) or press Enter for DialoGPT-medium: ").strip()
    
    # Select model
    if model_choice.isdigit() and 1 <= int(model_choice) <= len(MODEL_CONFIGS):
        model_name = list(MODEL_CONFIGS.keys())[int(model_choice) - 1]
    else:
        model_name = "microsoft/DialoGPT-medium"  # Default to medium
    
    # Initialize chatbot
    chatbot = EnhancedLocalChatbot(model_name)
    
    print(f"\nLoading {chatbot.model_config['name']}...")
    
    # Load the model
    if not chatbot.load_model():
        logger.error("Failed to initialize chatbot. Exiting.")
        sys.exit(1)
    
    # Start chat loop
    chatbot.chat_loop()

if __name__ == "__main__":
    main()