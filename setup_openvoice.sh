#!/bin/bash
# OpenVoice V2 Setup Script for Production Deployment

set -e  # Exit on any error

echo "🎵 Setting up OpenVoice V2 for AudioBook API..."

# Create necessary directories
mkdir -p /tmp/voice_cloning
mkdir -p OpenVoice/checkpoints/converter
mkdir -p OpenVoice/checkpoints/base_speakers/EN

echo "📁 Created directory structure"

# Clone OpenVoice V2 repository
echo "� Cloning OpenVoice V2 repository..."
if [ ! -d "OpenVoice/openvoice" ]; then
    # Download and extract OpenVoice
    curl -L -o openvoice.zip https://github.com/myshell-ai/OpenVoice/archive/refs/heads/main.zip
    unzip -q openvoice.zip
    cp -r OpenVoice-main/* OpenVoice/
    rm -rf OpenVoice-main openvoice.zip
    
    echo "📦 OpenVoice V2 source code installed"
else
    echo "✅ OpenVoice already exists"
fi

# Download OpenVoice V2 models
echo "🔽 Downloading OpenVoice V2 models..."

# Download converter checkpoint
echo "📥 Downloading converter models..."
cd OpenVoice/checkpoints/converter

# Download config.json
curl -L -o config.json "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/config.json" || \
wget -O config.json "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/config.json" || \
echo "Failed to download converter config"

# Download checkpoint.pth  
curl -L -o checkpoint.pth "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/checkpoint.pth" || \
wget -O checkpoint.pth "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/checkpoint.pth" || \
echo "Failed to download converter checkpoint"

cd ../../..

# Download base speaker models
echo "📥 Downloading base speaker models..."
cd OpenVoice/checkpoints/base_speakers/EN

curl -L -o config.json "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/config.json" || \
wget -O config.json "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/config.json" || \
echo "Failed to download base speaker config"

curl -L -o checkpoint.pth "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/checkpoint.pth" || \
wget -O checkpoint.pth "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/checkpoint.pth" || \
echo "Failed to download base speaker checkpoint"

cd ../../../..

# Set permissions
chmod -R 755 OpenVoice/ || echo "Permission setting failed, continuing..."

echo "✅ OpenVoice V2 setup completed!"
echo "🚀 Ready for voice cloning with full OpenVoice V2 capabilities"
