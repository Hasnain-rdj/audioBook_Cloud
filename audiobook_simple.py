"""
AudioBook API - Minimal Cloud Version
Simple PDF to Audio conversion for Render.com
"""

import os
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from gtts import gTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="AudioBook API",
    description="PDF to Audio Converter",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "AudioBook API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "AudioBook API",
        "version": "1.0.0",
        "endpoints": ["/", "/health", "/process-pdf"]
    }

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Limit text to prevent timeouts (15k characters)
            if len(text) > 15000:
                text = text[:15000] + "..."
                logger.info("Text truncated to 15k characters")
            
            return text.strip()
    
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not extract text from PDF: {str(e)}")

def text_to_speech(text: str, output_path: str, language: str = "en") -> str:
    """Convert text to speech using gTTS"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text to convert")
        
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save audio file
        tts.save(output_path)
        logger.info(f"Audio saved to: {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech conversion failed: {str(e)}")

@app.post("/process-pdf")
async def process_pdf_to_audio(file: UploadFile = File(...)):
    """
    Convert PDF file to audio
    
    Upload a PDF file and get back an MP3 audio file
    """
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = None
    temp_audio_path = None
    
    try:
        # Save uploaded PDF
        temp_pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(temp_pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"PDF saved: {temp_pdf_path}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(temp_pdf_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Convert text to speech
        temp_audio_path = os.path.join(temp_dir, "output.mp3")
        audio_path = text_to_speech(text, temp_audio_path)
        
        # Return audio file
        return FileResponse(
            path=audio_path,
            media_type="audio/mpeg",
            filename=f"audiobook_{file.filename.replace('.pdf', '.mp3')}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")

# Alternative endpoint name for n8n compatibility
@app.post("/process-pdf-to-audio")
async def process_pdf_to_audio_alt(file: UploadFile = File(...)):
    """Alternative endpoint name for n8n workflow compatibility"""
    return await process_pdf_to_audio(file)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
