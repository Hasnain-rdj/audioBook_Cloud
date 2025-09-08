#!/bin/bash
# Post-build script for OpenVoice V2 setup

set -e  # Exit on error

echo "🎵 Starting post-build OpenVoice V2 setup..."

# Make setup script executable
chmod +x setup_openvoice.sh || echo "Failed to make setup script executable"

# Check if we have sufficient space
df -h

# Run OpenVoice setup with error handling
echo "📦 Running OpenVoice V2 setup..."
if ./setup_openvoice.sh; then
    echo "✅ OpenVoice V2 setup completed successfully!"
else
    echo "⚠️ OpenVoice V2 setup failed, continuing with fallback mode"
    echo "The API will use audio effects fallback instead of full voice cloning"
fi

echo "✅ Post-build setup completed!"
