# Runtime Debugging Guide - AudioBook Cloud API

## Latest Deployment Status (After System Dependencies Fix)

### Changes Applied:
1. **Enhanced Requirements.txt** - Added PyTorch CPU version and comprehensive audio libraries
2. **System Package Installation** - Added apt-get commands in render.yaml for libespeak, festival, ffmpeg
3. **Improved Fallback Logic** - Enhanced TTS and voice effects to handle missing dependencies gracefully

### Expected Results:
- System speech libraries (libespeak, festival) should now be available
- PyTorch CPU version should be installed for OpenVoice functionality
- Fallback mechanisms create text files when audio processing fails

### How to Test:

1. **API Health Check:**
   ```
   curl https://audiobook-cloud.onrender.com/health
   ```

2. **Test Full Pipeline:**
   ```bash
   curl -X POST "https://audiobook-cloud.onrender.com/convert-pdf/" \
   -H "Content-Type: multipart/form-data" \
   -F "file=@test.pdf" \
   -F "voice_style=default" \
   -o result.zip
   ```

3. **Check Logs for:**
   - System dependencies installation during build
   - PyTorch availability messages
   - TTS library status
   - OpenVoice initialization

### Key Debug Points:

#### TTS Issues:
- Look for "libespeak.so.1" availability
- Check "pyttsx3" initialization
- Verify text fallback creation

#### Voice Cloning Issues:
- Confirm PyTorch installation
- Check OpenVoice model loading
- Verify audio file fallback logic

#### Expected Log Messages:
```
INFO: System dependencies installed: libespeak, festival, ffmpeg
INFO: PyTorch CPU version available for OpenVoice
INFO: TTS library initialized successfully
INFO: Voice effects applied using FFmpeg
```

### Fallback Behavior:
- If TTS fails → Creates .txt file with extracted text
- If voice effects fail → Copies original audio or creates text fallback
- If OpenVoice fails → Uses FFmpeg audio processing instead

### Deployment Environment:
- Python: 3.9.18 (enforced)
- Platform: Ubuntu-based Render.com
- System Packages: espeak, festival, ffmpeg
- Memory: Optimized for cloud deployment

### Next Steps if Issues Persist:
1. Check build logs for apt-get installation success
2. Verify PyTorch CPU installation in deployment logs
3. Test individual endpoints for specific failure points
4. Consider alternative lightweight TTS libraries if needed

### Production URL:
https://audiobook-cloud.onrender.com

Status: Deployment in progress with comprehensive system dependencies...
