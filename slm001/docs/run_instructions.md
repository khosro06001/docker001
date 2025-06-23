# SLM001 Quantized Chatbot Deployment Instructions

## 🚀 Quick Start

### On Jetson Nano (Recommended):
```bash
# Pull the optimized image
docker pull khosro123/slm001-quantized-chatbot:jetson-nano

# Run the quantized chatbot
docker run -it --rm khosro123/slm001-quantized-chatbot:jetson-nano
```

### On any ARM64/AMD64 device:
```bash
# Pull the latest image
docker pull khosro123/slm001-quantized-chatbot:latest

# Run with interactive mode
docker run -it --rm khosro123/slm001-quantized-chatbot:latest
```

## 🔥 Quantized Model Features

- **Intelligent Quantization Fallback**: Automatically uses best available quantization method
- **Multi-tier Quantization**: bitsandbytes → PyTorch native → no quantization
- **Memory Optimized**: Perfect for 4GB RAM devices like Jetson Nano
- **Real-time Model Switching**: Change models during conversation
- **Multiple Model Options**:
  - DialoGPT-small (~117MB native)
  - DistilGPT2 (~320MB native)
  - GPT2 (~500MB native)
  - DialoGPT-medium (~350MB native)
  - TinyLlama-1.1B (~1.1GB native)

## 🎯 Available Commands

Once the chatbot is running:
- `models` - Show available quantized models
- `switch <model_id>` - Switch to different model
- `stats` - Show model and memory statistics
- `memory` - Show current memory usage
- `clear` - Clear conversation history
- `help` - Show help message
- `quit` - Exit the chatbot

## 💾 Persistent Storage Options

### With model cache persistence:
```bash
docker run -it --rm -v slm001-cache:/app/cache khosro123/slm001-quantized-chatbot:latest
```

### With full data persistence:
```bash
docker run -it --rm -v slm001-data:/app/data -v slm001-cache:/app/cache khosro123/slm001-quantized-chatbot:latest
```

## 🖥️ Advanced Usage

### Run in detached mode:
```bash
docker run -d --name slm001-quantized khosro123/slm001-quantized-chatbot:latest
```

### Attach to running container:
```bash
docker exec -it slm001-quantized /bin/bash
```

### View logs:
```bash
docker logs slm001-quantized
```

### Stop container:
```bash
docker stop slm001-quantized
```

## 🔧 Memory and Performance Tuning

### For Jetson Nano (4GB RAM):
```bash
# Use memory-optimized settings
docker run -it --rm --memory=3g --memory-swap=4g khosro123/slm001-quantized-chatbot:jetson-nano
```

### For devices with more RAM:
```bash
# Allow more memory for better performance
docker run -it --rm --memory=8g khosro123/slm001-quantized-chatbot:latest
```

## 🐛 Troubleshooting

### Check Docker version:
```bash
docker --version
```

### Check available architectures:
```bash
docker buildx ls
```

### Force re-pull image:
```bash
docker rmi khosro123/slm001-quantized-chatbot:latest
docker pull khosro123/slm001-quantized-chatbot:latest
```

### Check container resource usage:
```bash
docker stats slm001-quantized
```

### Debug mode (verbose logging):
```bash
docker run -it --rm -e PYTHONDONTWRITEBYTECODE=0 khosro123/slm001-quantized-chatbot:latest
```

## 📊 Performance Expectations

| Device | Model | Load Time | Memory Usage | Response Time |
|--------|-------|-----------|--------------|---------------|
| Jetson Nano | DialoGPT-small | ~5-10s | ~300-500MB | ~1-3s |
| Jetson Nano | DistilGPT2 | ~10-15s | ~500-800MB | ~2-4s |
| RPi 4 (8GB) | GPT2 | ~15-20s | ~800MB-1.2GB | ~3-5s |
| x86_64 CPU | TinyLlama | ~10-15s | ~1-2GB | ~1-3s |

## 🔄 Model Switching Example

```
💬 You: models
🤖 Bot: [Shows available quantized models]

💬 You: switch distilgpt2
🤖 Bot: Switching to DistilGPT2...
✅ Successfully switched to DistilGPT2!

💬 You: stats
🤖 Bot: [Shows current model statistics]
```

## 🌐 Network Requirements

- **Initial Setup**: Internet required for downloading models
- **Runtime**: No internet required after models are cached
- **Model Cache**: Stored in `/app/cache` (can be persisted)

## ⚠️ Known Limitations

- First model load requires internet connection
- Quantized models may have slightly reduced quality
- Memory usage varies by model and conversation length
- Some models may not work on all architectures

## 📋 Quantization Fallback Chain

1. **bitsandbytes quantization** (if available and supported)
2. **PyTorch native quantization** (fallback for unsupported hardware)
3. **No quantization** (final fallback, uses more memory)

The system automatically detects the best available method.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure your device meets minimum requirements (2GB RAM)
3. Try different models if one doesn't work
4. Check Docker logs for detailed error messages
5. The quantization fallback system should handle most compatibility issues

---

**Image Information:**
- Repository: khosro123/slm001-quantized-chatbot
- Tags: latest, quantized, jetson-nano
- Architectures: linux/amd64, linux/arm64
- Base: python:3.9-slim
- Quantization: Intelligent fallback system (bitsandbytes → PyTorch native)

