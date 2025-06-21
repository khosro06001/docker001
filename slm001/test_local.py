#!/usr/bin/env python3
"""
test_local.py - Test script for slm001 chatbot
This script tests the chatbot functionality without Docker
"""

import sys
import os

# Add current directory to path to import slm001
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from slm001 import LocalChatbot
    print("✅ Successfully imported LocalChatbot")
except ImportError as e:
    print(f"❌ Failed to import LocalChatbot: {e}")
    print("Make sure you have installed the requirements:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def test_model_loading():
    """Test if the model can be loaded successfully."""
    print("\n🧪 Testing model loading...")
    
    try:
        chatbot = LocalChatbot()
        success = chatbot.load_model()
        
        if success:
            print("✅ Model loaded successfully!")
            return chatbot
        else:
            print("❌ Failed to load model")
            return None
            
    except Exception as e:
        print(f"❌ Exception during model loading: {e}")
        return None

def test_response_generation(chatbot):
    """Test response generation with sample inputs."""
    print("\n🧪 Testing response generation...")
    
    test_inputs = [
        "Hello, how are you?",
        "What is the weather like?",
        "Tell me a joke",
        "What is Python?",
        "Goodbye"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\nTest {i}/5: '{test_input}'")
        try:
            response = chatbot.generate_response(test_input)
            print(f"Response: {response}")
            print("✅ Response generated successfully!")
        except Exception as e:
            print(f"❌ Failed to generate response: {e}")
            return False
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 SLM001 Chatbot Local Test Suite")
    print("=" * 60)
    
    # Test model loading
    chatbot = test_model_loading()
    if not chatbot:
        print("\n❌ Model loading failed. Cannot proceed with further tests.")
        sys.exit(1)
    
    # Test response generation
    if test_response_generation(chatbot):
        print("\n✅ All tests passed!")
        print("\n🚀 You can now run the full chatbot with: python slm001.py")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()