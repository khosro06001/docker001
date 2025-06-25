
echo " as root "

# Update Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or if installed via package manager
sudo apt update && sudo apt upgrade ollama
