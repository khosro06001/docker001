#!/usr/bin/env python3
"""
slm001.py - Local Small Language Model Chatbot
A lightweight chatbot that runs locally using a small language model.
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

class LocalChatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize the local chatbot with a small language model.
        
        Args:
            model_name: HuggingFace model name. Default uses DialoGPT-small
                       which is lightweight and good for conversation.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.chat_pipeline = None
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges
        
        # Set random seed for reproducible results
        set_seed(42)
        
    def load_model(self):
        """Load the language model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info("This may take a few minutes on first run...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for better ARM compatibility
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # Use CPU (better for Jetson Nano)
                do_sample=True,
                temperature=0.7,
                max_length=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's question or message
            
        Returns:
            Generated response from the model
        """
        try:
            # Prepare the prompt with conversation context
            if self.conversation_history:
                # Include recent conversation history for context
                context = ""
                for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                    context += f"Human: {exchange['user']}\nAssistant: {exchange['bot']}\n"
                prompt = f"{context}Human: {user_input}\nAssistant:"
            else:
                prompt = f"Human: {user_input}\nAssistant:"
            
            # Generate response
            response = self.chat_pipeline(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the new response (after "Assistant:")
            if "Assistant:" in generated_text:
                bot_response = generated_text.split("Assistant:")[-1].strip()
                # Clean up the response
                bot_response = bot_response.split("Human:")[0].strip()
            else:
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
        """Main interactive chat loop."""
        print("=" * 60)
        print("🤖 Local Chatbot (slm001) - Ready!")
        print("=" * 60)
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'help' for available commands.")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Bot: Goodbye! Have a great day! 👋")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("\n🤖 Bot: Conversation history cleared!")
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif not user_input:
                    continue
                
                # Generate and display response
                print("\n🤖 Bot: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bot: Goodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n🤖 Bot: Sorry, I encountered an error: {e}")
    
    def show_help(self):
        """Display help information."""
        help_text = """
Available commands:
  • quit, exit, bye  - End the conversation
  • clear           - Clear conversation history
  • help            - Show this help message
  
Just type your question or message to chat with me!
        """
        print(f"\n🤖 Bot: {help_text}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🤖 Bot: Goodbye! 👋")
    sys.exit(0)

def main():
    """Main function to run the chatbot."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting Local Chatbot...")
    
    # Initialize chatbot
    chatbot = LocalChatbot()
    
    # Load the model
    if not chatbot.load_model():
        logger.error("Failed to initialize chatbot. Exiting.")
        sys.exit(1)
    
    # Start chat loop
    chatbot.chat_loop()

if __name__ == "__main__":
    main()