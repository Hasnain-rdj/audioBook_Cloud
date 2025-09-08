# 🎯 AudioBook Cloud - Speech-to-Speech Conversion (Clean Version)

## � **Final Directory Structure:**
```
audioBook_Cloud/
├── audioBook_api.py          # Main API with complete speech-to-speech pipeline
├── requirements.txt          # All dependencies for deployment  
├── Procfile                  # Render.com deployment configuration
├── runtime.txt              # Python version specification
├── setup_openvoice.sh       # OpenVoice V2 setup script
└── SPEECH_TO_SPEECH_FIX.md  # This documentation
```

## 🔧 **Problem Fixed:**
Your audiobook API was stopping at text-to-speech and not proceeding to speech-to-speech (voice cloning). The OpenVoice V2 integration was incomplete.

## ✅ **Complete Solution:**
- **PDF → Text Extraction** ✅
- **Text → Speech (TTS)** ✅  
- **Speech → Speech (Voice Cloning)** ✅ **[FIXED!]**
- **Final Cloned Audio Output** ✅

## � **How It Works:**

### **With Reference Audio (OpenVoice V2 Voice Cloning):**
1. Upload PDF + Reference voice audio
2. Extract text from PDF
3. Convert text to speech (TTS)
4. **Apply OpenVoice V2 cloning** (NEW!)
5. Output audio in reference voice

### **Without Reference Audio (Voice Effects):**
1. Upload PDF only
2. Extract text from PDF  
3. Convert text to speech (TTS)
4. **Apply voice effects** (fallback)
5. Output modified audio

## 🔄 **n8n Integration:**

### **API Endpoint:** `/audiobook-json`
### **Method:** POST (multipart/form-data)

### **Required Fields:**
- `pdf_file`: PDF document (binary)
- `language`: "en", "es", etc.
- `output_name`: Desired filename

### **Optional Fields:**
- `reference_audio`: Audio file for voice cloning (binary)
- `voice_style`: "default", "deep", "high", "slow", "fast"

### **Response:**
```json
{
  "success": true,
  "message": "Audiobook created successfully with Speech-to-Speech conversion",
  "download_url": "/download/filename.mp3",
  "full_download_url": "https://your-app.onrender.com/download/filename.mp3",
  "details": {
    "processing_method": "OpenVoice V2 Voice Cloning",
    "reference_audio_used": true,
    "file_size_mb": 5.2
  }
}
```

## 🚀 **Deployment Instructions:**

### **1. Deploy to Render.com:**
```bash
git add .
git commit -m "Clean speech-to-speech conversion API"
git push origin main
```

### **2. Render.com will automatically:**
- Install dependencies from `requirements.txt`
- Run the API via `Procfile`
- Initialize OpenVoice V2 on first startup

### **3. Test the Deployment:**
```bash
# Health check
curl https://your-app.onrender.com/health

# Initialize OpenVoice (if needed)  
curl -X POST https://your-app.onrender.com/init-openvoice

# Test complete pipeline
curl -X POST https://your-app.onrender.com/audiobook-json \
  -F "pdf_file=@your_document.pdf" \
  -F "reference_audio=@voice_sample.mp3" \
  -F "language=en" \
  -F "output_name=my_audiobook"
```

## 📊 **Success Indicators:**

### **✅ In API Response:**
- `"processing_method": "OpenVoice V2 Voice Cloning"`
- `"reference_audio_used": true`
- Valid download URL provided

### **✅ In Logs:**
- "OpenVoice V2 voice cloning..."
- "Voice cloning completed successfully"
- "Speech-to-Speech conversion completed"

### **✅ In Output:**
- Audio file sounds like reference voice
- File size similar to original TTS
- Good audio quality maintained

## � **Troubleshooting:**

### **If Voice Cloning Fails:**
1. **Check logs** for detailed error messages
2. **Try without reference audio** (falls back to voice effects)
3. **Call `/init-openvoice`** to reinitialize manually
4. **Check `/health`** endpoint for OpenVoice status

### **Common Issues:**
- **Memory limits**: Voice cloning needs significant RAM
- **Model download**: First run takes longer
- **Audio format**: Use MP3/WAV for reference audio

## 🎯 **Key Features:**

1. **Complete Pipeline**: Full PDF to cloned speech conversion
2. **Smart Fallbacks**: Works even if OpenVoice fails  
3. **Cloud Optimized**: Designed for Render.com deployment
4. **n8n Ready**: Perfect integration with your workflow
5. **Real Voice Cloning**: Uses OpenVoice V2 for actual voice replication

## 🎉 **Result:**
Your n8n workflow will now produce audiobooks that sound like the reference voice you provide! The speech-to-speech conversion is fully functional.
