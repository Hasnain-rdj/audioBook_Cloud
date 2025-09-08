#!/bin/bash
# Post-build script for OpenVoice V2 setup (Python 3.9 compatible)

set -e  # Exit on error

echo "🎵 Starting post-build setup for Python 3.9..."

# Check Python version
python --version
echo "Using Python: $(which python)"

# Create temp directory
mkdir -p /tmp/voice_cloning
echo "Created temp directory: /tmp/voice_cloning"

# Test offline TTS availability (don't fail if not available in cloud)
python -c "
try:
    import pyttsx3
    print('✅ pyttsx3 offline TTS available')
except ImportError as e:
    print(f'⚠️ pyttsx3 not available (cloud environment): {e}')
except Exception as e:
    print(f'⚠️ pyttsx3 initialization issue (cloud environment): {e}')
" || echo "TTS test completed with warnings (expected in cloud)"

# Test core dependencies
python -c "
import fastapi
import uvicorn
import PyPDF2
print('✅ Core dependencies loaded successfully')
"

echo "✅ Post-build setup completed successfully"

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
