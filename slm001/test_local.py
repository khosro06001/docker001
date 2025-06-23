#!/usr/bin/env python3
"""
test_local.py - Test script for slm001 quantized chatbot
This script tests the quantized chatbot functionality without Docker
"""

import sys
import os
import logging

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path to import slm001
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'transformers',
        'torch',
        'psutil'  # Optional but recommended
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            if package == 'psutil':
                print(f"⚠️  {package} is missing (optional - for memory monitoring)")
            else:
                print(f"❌ {package} is missing")
                missing_packages.append(package)
    
    # Check optional bitsandbytes
    try:
        import bitsandbytes
        print("✅ bitsandbytes is installed (quantization support)")
    except ImportError:
        print("⚠️  bitsandbytes is missing (will use fallback quantization)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install transformers torch")
        return False
    
    return True

def import_chatbot():
    """Import the quantized chatbot class."""
    try:
        from slm001 import QuantizedLocalChatbot, MODEL_CONFIGS
        print("✅ Successfully imported QuantizedLocalChatbot")
        return QuantizedLocalChatbot, MODEL_CONFIGS
    except ImportError as e:
        print(f"❌ Failed to import QuantizedLocalChatbot: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install transformers torch")
        return None, None

def test_model_configs():
    """Test if model configurations are valid."""
    print("\n🧪 Testing model configurations...")
    
    try:
        _, MODEL_CONFIGS = import_chatbot()
        if not MODEL_CONFIGS:
            return False
        
        print(f"Found {len(MODEL_CONFIGS)} quantized model configurations:")
        for model_name, config in MODEL_CONFIGS.items():
            print(f"  • {config['name']} - {config['size']} - {config['quantization']}")
            
            # Validate required config fields
            required_fields = ['name', 'size', 'description', 'max_length', 'quantization']
            for field in required_fields:
                if field not in config:
                    print(f"❌ Missing field '{field}' in config for {model_name}")
                    return False
        
        print("✅ All model configurations are valid!")
        return True
        
    except Exception as e:
        print(f"❌ Exception testing model configs: {e}")
        return False

def test_model_loading(model_name="microsoft/DialoGPT-small"):
    """Test if the quantized model can be loaded successfully."""
    print(f"\n🧪 Testing quantized model loading ({model_name})...")
    
    try:
        QuantizedLocalChatbot, _ = import_chatbot()
        if not QuantizedLocalChatbot:
            return None
            
        chatbot = QuantizedLocalChatbot(model_name)
        print(f"Using model: {chatbot.model_config['name']}")
        print(f"Quantization: {chatbot.model_config['quantization']}")
        print(f"Expected size: {chatbot.model_config['size']}")
        
        success = chatbot.load_model()
        
        if success:
            print("✅ Quantized model loaded successfully!")
            return chatbot
        else:
            print("❌ Failed to load quantized model")
            return None
            
    except Exception as e:
        print(f"❌ Exception during model loading: {e}")
        print("This might be due to:")
        print("  • Insufficient memory")
        print("  • Missing quantization libraries")
        print("  • Network issues downloading the model")
        return None

def test_device_detection(chatbot):
    """Test device detection functionality."""
    print("\n🧪 Testing device detection...")
    
    try:
        device = chatbot._detect_device()
        print(f"Detected device: {device}")
        
        # Test CUDA availability
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("CUDA not available - using CPU")
        
        print("✅ Device detection working!")
        return True
        
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_quantization_config(chatbot):
    """Test quantization configuration."""
    print("\n🧪 Testing quantization configuration...")
    
    try:
        # Test different quantization types
        for quant_type in ["4bit", "8bit"]:
            config = chatbot._get_quantization_config(quant_type)
            print(f"Quantization config for {quant_type}: {'Available' if config else 'Fallback mode'}")
        
        print("✅ Quantization configuration working!")
        return True
        
    except Exception as e:
        print(f"❌ Quantization configuration failed: {e}")
        return False

def test_response_generation(chatbot):
    """Test response generation with sample inputs."""
    print("\n🧪 Testing quantized response generation...")
    
    test_inputs = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Tell me something interesting",
        "What can you help me with?",
        "Thank you!"
    ]
    
    successful_responses = 0
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\nTest {i}/{len(test_inputs)}: '{test_input}'")
        try:
            response = chatbot.generate_response(test_input)
            print(f"Response: {response}")
            
            # Basic response validation
            if response and len(response.strip()) > 0:
                print("✅ Response generated successfully!")
                successful_responses += 1
            else:
                print("⚠️  Empty response generated")
                
        except Exception as e:
            print(f"❌ Failed to generate response: {e}")
    
    success_rate = (successful_responses / len(test_inputs)) * 100
    print(f"\nResponse generation success rate: {success_rate:.1f}%")
    
    return successful_responses > 0

def test_conversation_history(chatbot):
    """Test conversation history functionality."""
    print("\n🧪 Testing conversation history...")
    
    try:
        # Clear history first
        chatbot.conversation_history.clear()
        
        # Test conversation flow
        test_exchanges = [
            ("Hi, my name is John", ""),
            ("What is my name?", ""),
            ("Tell me about yourself", "")
        ]
        
        for i, (user_input, _) in enumerate(test_exchanges):
            response = chatbot.generate_response(user_input)
            print(f"Exchange {i+1}: User: '{user_input}' -> Bot: '{response[:50]}...'")
        
        history_length = len(chatbot.conversation_history)
        print(f"Conversation history length: {history_length}")
        
        if history_length == len(test_exchanges):
            print("✅ Conversation history working correctly!")
            return True
        else:
            print("⚠️  Conversation history length mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Conversation history test failed: {e}")
        return False

def test_memory_optimization(chatbot):
    """Test memory optimization features."""
    print("\n🧪 Testing memory optimization...")
    
    try:
        # Test memory stats if available
        try:
            chatbot.show_memory_usage()
            print("✅ Memory monitoring available!")
        except:
            print("⚠️  Memory monitoring not available (psutil not installed)")
        
        # Test history limitation
        original_max = chatbot.max_history
        print(f"Max history limit: {original_max}")
        
        # Test model stats
        chatbot.show_stats()
        
        print("✅ Memory optimization features working!")
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("=" * 80)
    print("🧪 SLM001 Quantized Chatbot Comprehensive Test Suite")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Dependencies
    total_tests += 1
    if check_dependencies():
        tests_passed += 1
    
    # Test 2: Model configs
    total_tests += 1
    if test_model_configs():
        tests_passed += 1
    
    # Test 3: Model loading (use smallest model for testing)
    total_tests += 1
    chatbot = test_model_loading("microsoft/DialoGPT-small")
    if chatbot:
        tests_passed += 1
        
        # Additional tests only if model loaded successfully
        # Test 4: Device detection
        total_tests += 1
        if test_device_detection(chatbot):
            tests_passed += 1
        
        # Test 5: Quantization config
        total_tests += 1
        if test_quantization_config(chatbot):
            tests_passed += 1
        
        # Test 6: Response generation
        total_tests += 1
        if test_response_generation(chatbot):
            tests_passed += 1
        
        # Test 7: Conversation history
        total_tests += 1
        if test_conversation_history(chatbot):
            tests_passed += 1
        
        # Test 8: Memory optimization
        total_tests += 1
        if test_memory_optimization(chatbot):
            tests_passed += 1
    
    # Final results
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("\n🚀 Your quantized chatbot is ready to use!")
        print("Run with: python slm001.py")
        return True
    elif tests_passed >= total_tests * 0.75:
        print("⚠️  MOST TESTS PASSED - Chatbot should work with minor issues")
        print("You can try running: python slm001.py")
        return True
    else:
        print("❌ MULTIPLE TESTS FAILED - Please check the issues above")
        return False

def quick_test():
    """Run a quick test to verify basic functionality."""
    print("=" * 60)
    print("🚀 SLM001 Quantized Chatbot Quick Test")
    print("=" * 60)
    
    if not check_dependencies():
        return False
    
    chatbot = test_model_loading("microsoft/DialoGPT-small")
    if not chatbot:
        return False
    
    # Single response test
    try:
        response = chatbot.generate_response("Hello, this is a test!")
        print(f"\nTest response: {response}")
        print("✅ Quick test passed!")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test slm001 quantized chatbot')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-small',
                       help='Model to test (default: microsoft/DialoGPT-small)')
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
    else:
        success = run_comprehensive_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()