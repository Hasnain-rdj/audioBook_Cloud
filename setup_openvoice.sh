#!/bin/bash
# OpenVoice V2 Setup for Cloud Deployment

echo "🚀 Setting up OpenVoice V2 for cloud deployment..."

# Create OpenVoice directory structure
mkdir -p OpenVoice/checkpoints/converter
mkdir -p OpenVoice/openvoice

echo "📁 Created directory structure"

# Download OpenVoice V2 source code
echo "📥 Downloading OpenVoice V2 source..."
curl -L -o openvoice.zip https://github.com/myshell-ai/OpenVoice/archive/refs/heads/main.zip
unzip -q openvoice.zip
cp -r OpenVoice-main/* OpenVoice/
rm -rf OpenVoice-main openvoice.zip

echo "✅ OpenVoice V2 setup complete"

# The API will automatically download model checkpoints on first use
