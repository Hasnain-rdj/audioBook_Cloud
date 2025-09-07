# n8n Workflow Configuration for OpenVoice V2 Speech-to-Speech

## 🎯 True Speech-to-Speech with OpenVoice V2

Your API now supports **real OpenVoice V2 voice cloning** for authentic speech-to-speech conversion!

### 🎙️ Two Processing Modes:

#### Mode 1: OpenVoice V2 Cloning (With Reference Audio)
- Upload PDF + Reference Voice Audio
- Gets: PDF → Text → gTTS → **OpenVoice V2 Cloning** → Your Custom Voice

#### Mode 2: Voice Effects (Fallback)
- Upload PDF only (no reference audio)
- Gets: PDF → Text → gTTS → **Audio Effects** → Enhanced Voice

## 🔧 n8n HTTP Request Node Configuration

### Option 1: JSON Response with OpenVoice V2
```
Method: POST
URL: https://audiobook-cloud.onrender.com/audiobook-json

Body Type: Form-Data
Form Fields:
- pdf_file: {{ $binary.data }} (Binary) - PDF to convert
- reference_audio: {{ $binary.reference }} (Binary) - Your voice sample  
- language: en (Text) - Language code
- output_name: audiobook (Text) - Output filename
- voice_style: default (Text) - Fallback if no reference audio
```

### Response includes:
```json
{
  "success": true,
  "download_url": "/download/audiobook_20250907_171829.mp3",
  "full_download_url": "https://audiobook-cloud.onrender.com/download/audiobook_20250907_171829.mp3",
  "filename": "audiobook_20250907_171829.mp3",
  "details": {
    "text_length": 15000,
    "processing_mode": "openvoice_v2",
    "file_size_mb": 12.5
  }
}
```

## 🔗 Complete n8n Workflow

### Node 1: Form Trigger (Audio Book UI)
- Collect PDF file
- **NEW:** Collect reference audio file (your voice sample)

### Node 2: HTTP Request (Audio Book API) - MAIN PROCESSING
```
Method: POST
URL: https://audiobook-cloud.onrender.com/audiobook-json
Body Type: Form-Data

Form Fields:
- pdf_file: {{ $binary.data }}        ← PDF file
- reference_audio: {{ $binary.voice }} ← Your voice sample (MP3/WAV)
- language: en  
- output_name: audiobook
```

### Node 3: HTTP Request (Download File) - GET FINAL AUDIO
```
Method: GET
URL: {{ $json.full_download_url }}
Response Format: Binary
Binary Property Name: cloned_audio
```

### Node 4: Return Binary Data (Final Output)
```
Binary Property: cloned_audio
Filename: {{ $json.filename }}
MIME Type: audio/mpeg
```

## 🎉 What This Gives You:

1. ✅ **PDF Text Extraction** (15k characters)
2. ✅ **Text-to-Speech** (gTTS base conversion)  
3. ✅ **OpenVoice V2 Cloning** (YOUR voice applied to the text)
4. ✅ **Downloadable Cloned Audio** (Sounds like you reading the PDF!)

## 📋 Reference Audio Requirements:

- **Format:** MP3, WAV, M4A
- **Length:** 10-60 seconds of clear speech
- **Quality:** Good audio quality, minimal background noise
- **Content:** Any speech content (the more, the better cloning)

## 🚀 Testing Instructions:

1. **Record your voice** (10-30 seconds of clear speech)
2. **Update your GitHub** repository with new code
3. **Redeploy on Render.com** 
4. **Configure n8n** with both PDF and reference audio upload
5. **Test the workflow** - the output should sound like YOUR voice!

## ⚠️ Important Notes:

- **First run** will be slower (downloading OpenVoice models)
- **Processing time:** 2-5 minutes for OpenVoice cloning
- **Fallback:** If reference audio fails, uses voice effects
- **Cloud limits:** May have memory constraints for very long audio

Your audiobook will now sound like **YOU** reading the PDF content! �