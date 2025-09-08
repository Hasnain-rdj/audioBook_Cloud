#!/bin/bash
# Post-build script for OpenVoice V2 setup

echo "🎵 Starting post-build OpenVoice V2 setup..."

# Make setup script executable
chmod +x setup_openvoice.sh

# Run OpenVoice setup
./setup_openvoice.sh

echo "✅ Post-build setup completed!"
