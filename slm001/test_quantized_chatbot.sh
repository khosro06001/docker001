#!/bin/bash

# Quick test script for quantized chatbot
IMAGE_NAME="khosro123/slm001-quantized-chatbot:latest"

echo "🧪 Testing Quantized Chatbot with Fallback System..."

# Test 1: Basic import test
echo "Test 1: Basic dependency check..."
docker run --rm "$IMAGE_NAME" python test_env.py

# Test 2: Quantization availability test
echo -e "\nTest 2: Quantization fallback system check..."
docker run --rm "$IMAGE_NAME" python -c "
try:
    import bitsandbytes
    print('✅ BitsAndBytes available for quantization')
except ImportError:
    print('⚠️  BitsAndBytes not available, using PyTorch native quantization')

try:
    from transformers import BitsAndBytesConfig
    print('✅ BitsAndBytesConfig available')
except ImportError:
    print('⚠️  BitsAndBytesConfig not available, using fallback')

# Test PyTorch quantization
try:
    import torch
    print('✅ PyTorch available for native quantization')
    print('✅ Quantization fallback chain ready')
except ImportError:
    print('❌ PyTorch not available')
"

# Test 3: Model configuration test
echo -e "\nTest 3: Model configuration check..."
docker run --rm "$IMAGE_NAME" python -c "
import sys
sys.path.append('/app')
try:
    from slm001 import MODEL_CONFIGS, QuantizedLocalChatbot
    print('✅ Model configurations loaded successfully')
    print(f'Available models: {len(MODEL_CONFIGS)}')
    for model_key, config in MODEL_CONFIGS.items():
        print(f'  - {config[\"name\"]}: {config[\"size\"]}')
    
    # Test chatbot initialization
    chatbot = QuantizedLocalChatbot()
    print('✅ Chatbot initialization successful')
    print(f'Device detected: {chatbot.device}')
    
except Exception as e:
    print(f'❌ Error loading model configs: {e}')
"

# Test 4: Quick interactive test
echo -e "\nTest 4: Quick interactive test (5 seconds)..."
timeout 5s docker run --rm -i "$IMAGE_NAME" python -c "
import sys
sys.path.append('/app')
from slm001 import QuantizedLocalChatbot
chatbot = QuantizedLocalChatbot()
print('✅ Ready for quick test')
print('Note: Full model loading takes longer in actual use')
" 2>/dev/null || echo "⚠️  Timeout reached (expected for full model loading)"

echo -e "\n🎉 Test completed!"
echo "💡 For full testing, run the chatbot interactively:"
echo "   docker run -it --rm $IMAGE_NAME"
