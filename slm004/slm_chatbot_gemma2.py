#!/usr/bin/env python3
"""
Gemma2 2B Interactive Chatbot - Ollama Version (Docker-optimized)
Uses Ollama's API to interact with the gemma2:2b-instruct-q4_0 model
Supports conversation history and both CPU (AMD64) and GPU (ARM64/Jetson) acceleration
Optimized for Docker containers connecting to host-based Ollama
"""

import argparse
import logging
import sys
import json
import signal
import platform
import os
from datetime import datetime
from pathlib import Path
import requests

# Try to import readline for better input handling (optional)
try:
    import readline
except ImportError:
    # readline not available, input will still work but without history/editing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GemmaOllamaChatbot:
    def __init__(self, base_url=None, system_prompt=None):
        # Docker-aware URL detection
        self.base_url = self._detect_ollama_url(base_url)
        self.model_name = "gemma2:2b-instruct-q4_0"
        self.api_url = f"{self.base_url}/api"
        self.conversation_history = []
        self.system_prompt = system_prompt or "You are a helpful, friendly assistant. Provide clear and concise responses."
        self.session_start = datetime.now()
        
        # Detect architecture for GPU optimization hints
        self.arch = platform.machine().lower()
        self.is_jetson = self._detect_jetson()
        self.is_container = self._detect_container()
        
        logger.info(f"Using model: {self.model_name}")
        logger.info(f"Ollama URL: {self.base_url}")
        logger.info(f"Architecture: {self.arch}")
        logger.info(f"Jetson detected: {self.is_jetson}")
        logger.info(f"Container detected: {self.is_container}")
        
    def _detect_ollama_url(self, provided_url):
        """Detect appropriate Ollama URL for container/host environments"""
        if provided_url:
            return provided_url
            
        # Check environment variables first
        if os.getenv('OLLAMA_HOST'):
            host = os.getenv('OLLAMA_HOST')
            if not host.startswith('http'):
                host = f"http://{host}"
            return host
            
        # Container detection and URL selection
        if self._detect_container():
            # In container, try to connect to host
            possible_urls = [
                "http://host.docker.internal:11434",  # Docker Desktop
                "http://172.17.0.1:11434",           # Default Docker bridge
                "http://localhost:11434"              # Host network mode
            ]
            
            for url in possible_urls:
                try:
                    response = requests.get(f"{url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Found Ollama at: {url}")
                        return url
                except:
                    continue
                    
            logger.warning("Could not auto-detect Ollama URL in container")
            return "http://localhost:11434"  # Default fallback
        else:
            # Not in container, use localhost
            return "http://localhost:11434"
    
    def _detect_container(self):
        """Detect if running inside a Docker container"""
        try:
            # Check for Docker-specific files
            if os.path.exists('/.dockerenv'):
                return True
            
            # Check cgroup for docker
            if os.path.exists('/proc/1/cgroup'):
                with open('/proc/1/cgroup', 'r') as f:
                    content = f.read()
                    return 'docker' in content or 'containerd' in content
        except:
            pass
        return False
        
    def _detect_jetson(self):
        """Detect if running on Jetson device"""
        try:
            # Check for tegra in CPU info (Jetson signature)
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    return 'tegra' in cpuinfo or 'jetson' in cpuinfo
        except:
            pass
        return False
    
    def test_ollama_connection(self):
        """Test connection to Ollama server and verify gemma2 model is available"""
        try:
            logger.info(f"Testing connection to: {self.base_url}")
            response = requests.get(f"{self.base_url}/api/tags", timeout=15)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"✓ Connected to Ollama server")
                logger.info(f"Available models: {available_models}")
                
                # Check if gemma2 model is available
                if self.model_name in available_models:
                    logger.info(f"✓ Model '{self.model_name}' is available")
                    return True
                else:
                    logger.error(f"✗ Model '{self.model_name}' not found")
                    logger.error(f"Please install it with: ollama pull {self.model_name}")
                    return False
            else:
                logger.error(f"Ollama server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"✗ Cannot connect to Ollama server at {self.base_url}")
            logger.error("Make sure Ollama is running with: ollama serve")
            if self.is_container:
                logger.error("Container networking tips:")
                logger.error("  - Use --network=host for simplest setup")
                logger.error("  - Or set OLLAMA_HOST environment variable")
                logger.error("  - Or use host.docker.internal (Docker Desktop)")
            return False
        except Exception as e:
            logger.error(f"Error testing Ollama connection: {e}")
            return False
    
    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            response = requests.post(
                f"{self.api_url}/show",
                json={"name": self.model_name},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Could not get model info: {e}")
        return None
    
    def chat(self, user_input: str, stream: bool = False) -> str:
        """Send message to Gemma2 and get response"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user", 
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Prepare conversation context for Ollama
            conversation_context = self._build_conversation_context()
            
            payload = {
                "model": self.model_name,
                "prompt": conversation_context,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096,  # Context window
                }
            }
            
            # Add GPU optimization for Jetson
            if self.is_jetson:
                payload["options"]["num_gpu"] = 1
                payload["options"]["num_thread"] = 4  # Jetson Nano has 4 cores
                logger.debug("Applied Jetson GPU optimizations")
            
            logger.debug("Sending request to Ollama...")
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=120,  # Generous timeout for ARM devices
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get('response', '').strip()
                
                if assistant_response:
                    # Add assistant response to history
                    self.conversation_history.append({
                        "role": "assistant", 
                        "content": assistant_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.debug("Response generated successfully")
                    return assistant_response
                else:
                    logger.error("Empty response from Ollama")
                    return "I'm sorry, I couldn't generate a response. Please try again."
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return f"Error: Unable to get response from the model (status {response.status_code})"
                
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return "I'm taking too long to respond. The model might be overloaded. Please try again."
        except requests.exceptions.ConnectionError:
            logger.error("Connection lost to Ollama server")
            return "Connection lost. Please check if Ollama is still running."
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"An error occurred: {str(e)}"
    
    def _build_conversation_context(self):
        """Build conversation context for Gemma2"""
        context = f"System: {self.system_prompt}\n\n"
        
        # Add recent conversation history (limit to last 10 exchanges to manage context)
        recent_history = self.conversation_history[-20:]  # Last 20 messages (10 exchanges)
        
        for msg in recent_history:
            if msg["role"] == "user":
                context += f"Human: {msg['content']}\n"
            elif msg["role"] == "assistant":
                context += f"Assistant: {msg['content']}\n"
        
        # Add the prompt for the current response
        context += "Assistant: "
        
        return context
    
    def save_conversation(self, filename: str = None):
        """Save conversation history to file"""
        if not filename:
            timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
            filename = f"/app/conversations/gemma2_chat_{timestamp}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            conversation_data = {
                "session_start": self.session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "model": self.model_name,
                "system_prompt": self.system_prompt,
                "architecture": self.arch,
                "is_jetson": self.is_jetson,
                "is_container": self.is_container,
                "ollama_url": self.base_url,
                "conversation": self.conversation_history
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return None
    
    def load_conversation(self, filename: str):
        """Load conversation history from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            self.conversation_history = conversation_data.get('conversation', [])
            self.system_prompt = conversation_data.get('system_prompt', self.system_prompt)
            
            logger.info(f"Loaded conversation with {len(self.conversation_history)} messages")
            return True
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_stats(self):
        """Get conversation statistics"""
        user_messages = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.conversation_history if msg["role"] == "assistant"])
        session_duration = datetime.now() - self.session_start
        
        return {
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_messages": len(self.conversation_history),
            "session_duration": str(session_duration).split('.')[0],  # Remove microseconds
            "model": self.model_name,
            "architecture": self.arch,
            "jetson_optimized": self.is_jetson,
            "container_mode": self.is_container,
            "ollama_url": self.base_url
        }

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n👋 Goodbye! Chat session ended.")
    sys.exit(0)

def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("🤖 GEMMA2 INTERACTIVE CHATBOT (Docker Edition)")
    print("=" * 60)
    print("Commands:")
    print("  /help     - Show this help message")
    print("  /clear    - Clear conversation history")
    print("  /stats    - Show session statistics")
    print("  /save     - Save conversation to file")
    print("  /quit     - Exit the chatbot")
    print("=" * 60)
    print("Start chatting! (Ctrl+C to exit)\n")

def print_help():
    """Print help message"""
    print("\n📋 AVAILABLE COMMANDS:")
    print("  /help     - Show this help message")
    print("  /clear    - Clear conversation history") 
    print("  /stats    - Show session statistics")
    print("  /save     - Save conversation to file")
    print("  /quit     - Exit the chatbot")
    print("\n💡 Tips:")
    print("  - Your conversation history is remembered during this session")
    print("  - Use /clear to start fresh if the context gets too long")
    print("  - Use /save to keep a record of your conversation")
    print("  - Conversations are saved to /app/conversations/ in container")
    print()

def main():
    parser = argparse.ArgumentParser(description='Interactive Gemma2 Chatbot via Ollama (Docker Edition)')
    parser.add_argument('--url', '-u', 
                       help='Ollama server URL (auto-detects if not provided)')
    parser.add_argument('--system-prompt', '-s', 
                       help='Custom system prompt for the assistant')
    parser.add_argument('--load', '-l', 
                       help='Load conversation from JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize chatbot
    chatbot = GemmaOllamaChatbot(base_url=args.url, system_prompt=args.system_prompt)
    
    # Test connection
    if not chatbot.test_ollama_connection():
        sys.exit(1)
    
    # Load conversation if specified
    if args.load:
        if chatbot.load_conversation(args.load):
            print(f"📁 Loaded conversation from: {args.load}")
        else:
            print(f"❌ Failed to load conversation from: {args.load}")
    
    # Show model info if verbose
    if args.verbose:
        model_info = chatbot.get_model_info()
        if model_info:
            print(f"\n📊 Model Info:")
            print(f"   Name: {model_info.get('details', {}).get('family', 'Unknown')}")
            print(f"   Parameters: {model_info.get('details', {}).get('parameter_size', 'Unknown')}")
    
    print_welcome()
    
    # Main chat loop
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command == '/help':
                        print_help()
                    elif command == '/clear':
                        chatbot.clear_history()
                        print("🧹 Conversation history cleared!\n")
                    elif command == '/stats':
                        stats = chatbot.get_stats()
                        print(f"\n📊 SESSION STATISTICS:")
                        print(f"   Messages from you: {stats['user_messages']}")
                        print(f"   Messages from bot: {stats['assistant_messages']}")
                        print(f"   Total messages: {stats['total_messages']}")
                        print(f"   Session duration: {stats['session_duration']}")
                        print(f"   Model: {stats['model']}")
                        print(f"   Architecture: {stats['architecture']}")
                        print(f"   GPU optimized: {stats['jetson_optimized']}")
                        print(f"   Container mode: {stats['container_mode']}")
                        print(f"   Ollama URL: {stats['ollama_url']}")
                        print()
                    elif command == '/save':
                        filename = chatbot.save_conversation()
                        if filename:
                            print(f"💾 Conversation saved to: {filename}\n")
                        else:
                            print("❌ Failed to save conversation\n")
                    elif command in ['/quit', '/exit']:
                        print("\n👋 Goodbye! Chat session ended.")
                        break
                    else:
                        print(f"❓ Unknown command: {user_input}")
                        print("Type /help for available commands\n")
                    
                    continue
                
                # Send message to chatbot
                print("🤖 ", end="", flush=True)
                response = chatbot.chat(user_input)
                print(response)
                print()  # Empty line for readability
                
            except EOFError:
                # Handle Ctrl+D
                print("\n\n👋 Goodbye! Chat session ended.")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\n\n👋 Goodbye! Chat session ended.")
                break
                
        # Auto-save conversation on exit
        if chatbot.conversation_history:
            filename = chatbot.save_conversation()
            if filename:
                print(f"💾 Conversation automatically saved to: {filename}")
                
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()