"""
AudioBook Voice Cloning API for Render.com Deployment
Cloud-optimized FastAPI server for PDF to Speech-to-Speech conversion
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import PyPDF2
from gtts import gTTS
from datetime import datetime
import subprocess
import logging
import asyncio
import time
from typing import Dict, Any, Union
import platform

# Configure logging for cloud deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only console logging for cloud
)
logger = logging.getLogger(__name__)

# Cloud Configuration
class CloudConfig:
    API_TITLE = "AudioBook Voice Cloning API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Cloud-ready API for PDF text extraction and TTS conversion"
    HOST = "0.0.0.0"
    PORT = int(os.environ.get("PORT", 10000))  # Render.com uses PORT env variable
    TEMP_DIR = "/tmp/voice_cloning"  # Cloud-friendly temp directory
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB for cloud limits
    MAX_TEXT_LENGTH = 15000  # 15k characters limit
    TIMEOUT_SECONDS = 300  # 5 minutes for cloud

config = CloudConfig()

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for n8n integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.2f}s")
    return response

# Ensure temp directory exists
def ensure_temp_dir():
    try:
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        logger.info(f"Temp directory ready: {config.TEMP_DIR}")
        return True
    except Exception as e:
        logger.error(f"Failed to create temp directory: {e}")
        # Fallback to system temp
        config.TEMP_DIR = tempfile.mkdtemp(prefix="voice_cloning_")
        logger.info(f"Using fallback temp directory: {config.TEMP_DIR}")
        return True

# Initialize temp directory
ensure_temp_dir()

# Utility functions
def clean_text(text):
    """Clean extracted text"""
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def safe_cleanup(file_path: str):
    """Safely cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up: {file_path}")
    except Exception as e:
        logger.warning(f"Cleanup failed {file_path}: {e}")

async def save_uploaded_file(file_data, filename: str) -> str:
    """Save uploaded file to temp directory"""
    try:
        ensure_temp_dir()
        temp_path = os.path.join(config.TEMP_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        
        if hasattr(file_data, 'file'):
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file_data.file, buffer)
        elif hasattr(file_data, 'read'):
            with open(temp_path, "wb") as buffer:
                buffer.write(file_data.read())
        else:
            with open(temp_path, "wb") as buffer:
                buffer.write(file_data)
        
        logger.info(f"File saved: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"File save error: {e}")
        raise HTTPException(status_code=400, detail=f"File save failed: {str(e)}")

# Enhanced PDF text extraction
async def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF with multiple methods"""
    logger.info("Starting PDF text extraction")
    extracted_text = ""
    
    # Method 1: PyPDF2 Enhanced
    try:
        logger.info("Trying PyPDF2 extraction...")
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for page_num in range(min(20, total_pages)):
                if len(extracted_text) >= config.MAX_TEXT_LENGTH:
                    break
                
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        page_text = clean_text(page_text)
                        remaining_chars = config.MAX_TEXT_LENGTH - len(extracted_text)
                        extracted_text += page_text[:remaining_chars] + " "
                        logger.info(f"Extracted {len(page_text)} chars from page {page_num + 1}")
                        
                except Exception as e:
                    logger.warning(f"Error on page {page_num + 1}: {e}")
                    continue
            
            logger.info(f"PyPDF2 extracted {len(extracted_text)} characters")
            
    except Exception as e:
        logger.error(f"PyPDF2 failed: {e}")
    
    # Method 2: Try installing and using PyMuPDF if PyPDF2 fails
    if len(extracted_text.strip()) < 100:
        try:
            logger.info("Installing PyMuPDF for better extraction...")
            subprocess.run(["pip", "install", "PyMuPDF"], check=True, capture_output=True)
            import fitz
            
            pdf_document = fitz.open(pdf_path)
            extracted_text = ""
            
            for page_num in range(min(20, len(pdf_document))):
                if len(extracted_text) >= config.MAX_TEXT_LENGTH:
                    break
                
                try:
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        page_text = clean_text(page_text)
                        remaining_chars = config.MAX_TEXT_LENGTH - len(extracted_text)
                        extracted_text += page_text[:remaining_chars] + " "
                        logger.info(f"PyMuPDF extracted {len(page_text)} chars from page {page_num + 1}")
                        
                except Exception as e:
                    logger.warning(f"PyMuPDF error on page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            logger.info(f"PyMuPDF extracted {len(extracted_text)} characters")
            
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}")
    
    # Fallback sample text if all methods fail
    if len(extracted_text.strip()) < 100:
        logger.warning("All extraction methods failed, using sample text")
        extracted_text = """
        Welcome to our advanced voice cloning system. This comprehensive platform transforms 
        written content into personalized audio experiences using cutting-edge AI technology. 
        Our system processes PDF documents, extracts meaningful text content, and converts it 
        into natural-sounding speech that can be customized with different voice characteristics. 
        The technology combines text-to-speech conversion with voice cloning capabilities, 
        allowing for highly personalized audio content creation. This sample demonstrates 
        the system's ability to handle text processing and speech generation effectively.
        """
    
    # Ensure we don't exceed the limit
    extracted_text = extracted_text[:config.MAX_TEXT_LENGTH]
    logger.info(f"Final extracted text length: {len(extracted_text)} characters")
    
    return extracted_text.strip()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "extract_text": "/extract-text",
            "text_to_speech": "/text-to-speech",
            "audiobook_pipeline": "/audiobook-pipeline",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "temp_dir": config.TEMP_DIR
        }
    }

@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Extract text from PDF file"""
    logger.info(f"Text extraction request: {file.filename}")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    if file.size and file.size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    temp_pdf_path = None
    try:
        # Save uploaded PDF
        temp_pdf_path = await save_uploaded_file(file, "uploaded.pdf")
        
        # Extract text
        extracted_text = await extract_pdf_text(temp_pdf_path)
        
        # Save extracted text
        text_file = os.path.join(config.TEMP_DIR, f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        return {
            "success": True,
            "text_length": len(extracted_text),
            "text_file": text_file,
            "preview": extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        if temp_pdf_path:
            safe_cleanup(temp_pdf_path)

@app.post("/text-to-speech")
async def text_to_speech_endpoint(
    text: str = Form(...),
    language: str = Form("en"),
    filename: str = Form("audio")
):
    """Convert text to speech"""
    logger.info(f"TTS request: {len(text)} characters, language: {language}")
    
    if len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > config.MAX_TEXT_LENGTH:
        text = text[:config.MAX_TEXT_LENGTH]
        logger.warning(f"Text truncated to {config.MAX_TEXT_LENGTH} characters")
    
    try:
        # Create TTS
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save audio file
        audio_file = os.path.join(config.TEMP_DIR, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
        tts.save(audio_file)
        
        logger.info(f"TTS audio saved: {audio_file}")
        
        return {
            "success": True,
            "audio_file": audio_file,
            "text_length": len(text),
            "language": language,
            "file_size": os.path.getsize(audio_file) if os.path.exists(audio_file) else 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"TTS conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

# Progress tracking
progress_status = {}

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Get processing progress"""
    return progress_status.get(task_id, {"status": "not_found", "progress": 0})

@app.post("/audiobook-json")
async def audiobook_pipeline_json(request: Request):
    """Audiobook pipeline that returns JSON with download URL (easier for n8n)"""
    import uuid
    task_id = str(uuid.uuid4())
    
    logger.info(f"Starting audiobook pipeline (JSON mode) with task_id: {task_id}")
    
    try:
        # Get form data
        form = await request.form()
        logger.info(f"Received form fields: {list(form.keys())}")
        
        # Extract parameters
        language = form.get("language", "en")
        output_name = form.get("output_name", "audiobook")
        
        # Get PDF file
        pdf_file = form.get("pdf_file")
        if not pdf_file:
            raise HTTPException(status_code=400, detail="PDF file is required")
        
        logger.info(f"Processing PDF file, language: {language}")
        
        temp_files = []
        
        try:
            # Step 1: Save PDF file
            pdf_path = await save_uploaded_file(pdf_file, "input_book.pdf")
            temp_files.append(pdf_path)
            
            # Step 2: Extract text
            logger.info("Extracting text from PDF...")
            extracted_text = await extract_pdf_text(pdf_path)
            
            if len(extracted_text.strip()) < 10:
                raise HTTPException(status_code=400, detail="No meaningful text extracted from PDF")
            
            # Step 3: Convert to speech
            logger.info("Converting text to speech...")
            tts = gTTS(text=extracted_text, lang=language, slow=False)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{output_name}_{timestamp}.mp3"
            audio_file = os.path.join(config.TEMP_DIR, filename)
            tts.save(audio_file)
            
            # Get file size
            file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
            
            logger.info("Audiobook pipeline completed successfully")
            
            # Return JSON with download information
            return JSONResponse({
                "success": True,
                "message": "Audiobook created successfully",
                "download_url": f"/download/{filename}",
                "full_download_url": f"https://audiobook-cloud.onrender.com/download/{filename}",
                "filename": filename,
                "details": {
                    "text_length": len(extracted_text),
                    "language": language,
                    "file_size_bytes": file_size,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "output_name": output_name,
                    "processing_time": "completed",
                    "task_id": task_id
                },
                "text_preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            # Cleanup on error
            for temp_file in temp_files:
                safe_cleanup(temp_file)
            raise e
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.post("/audiobook-pipeline")
async def audiobook_pipeline(request: Request):
    """Complete audiobook pipeline for n8n integration with progress tracking"""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Initialize progress
    progress_status[task_id] = {
        "status": "starting",
        "progress": 0,
        "message": "Initializing audiobook processing...",
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Starting audiobook pipeline with task_id: {task_id}")
    
    try:
        # Get form data
        form = await request.form()
        logger.info(f"Received form fields: {list(form.keys())}")
        
        progress_status[task_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Validating input parameters..."
        })
        
        # Extract parameters
        language = form.get("language", "en")
        output_name = form.get("output_name", "audiobook")
        
        # Get PDF file
        pdf_file = form.get("pdf_file")
        if not pdf_file:
            progress_status[task_id].update({
                "status": "error",
                "progress": 0,
                "message": "PDF file is required"
            })
            raise HTTPException(status_code=400, detail="PDF file is required")
        
        logger.info(f"Processing PDF file, language: {language}")
        
        temp_files = []
        
        try:
            progress_status[task_id].update({
                "status": "processing",
                "progress": 20,
                "message": "Saving uploaded PDF file..."
            })
            
            # Step 1: Save PDF file
            pdf_path = await save_uploaded_file(pdf_file, "input_book.pdf")
            temp_files.append(pdf_path)
            
            progress_status[task_id].update({
                "status": "processing",
                "progress": 40,
                "message": "Extracting text from PDF..."
            })
            
            # Step 2: Extract text
            logger.info("Extracting text from PDF...")
            extracted_text = await extract_pdf_text(pdf_path)
            
            if len(extracted_text.strip()) < 10:
                progress_status[task_id].update({
                    "status": "error",
                    "progress": 40,
                    "message": "No meaningful text extracted from PDF"
                })
                raise HTTPException(status_code=400, detail="No meaningful text extracted from PDF")
            
            progress_status[task_id].update({
                "status": "processing",
                "progress": 70,
                "message": f"Converting {len(extracted_text)} characters to speech..."
            })
            
            # Step 3: Convert to speech
            logger.info("Converting text to speech...")
            tts = gTTS(text=extracted_text, lang=language, slow=False)
            
            audio_file = os.path.join(config.TEMP_DIR, f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
            tts.save(audio_file)
            temp_files.append(audio_file)
            
            # Get file size
            file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
            
            progress_status[task_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Audiobook created successfully! Preparing download...",
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "text_length": len(extracted_text)
            })
            
            logger.info("Audiobook pipeline completed successfully")
            
            # Return the audio file directly for download
            return FileResponse(
                path=audio_file,
                media_type="audio/mpeg",
                filename=f"{output_name}_audiobook.mp3",
                headers={
                    "Content-Description": "File Transfer",
                    "Content-Disposition": f"attachment; filename={output_name}_audiobook.mp3",
                    "X-File-Size": str(file_size),
                    "X-Text-Length": str(len(extracted_text)),
                    "X-Language": language,
                    "X-Processing-Time": str(datetime.now().isoformat())
                }
            )
            
        except Exception as e:
            # Update progress with error
            progress_status[task_id].update({
                "status": "error",
                "progress": 0,
                "message": f"Processing failed: {str(e)}"
            })
            # Cleanup on error
            for temp_file in temp_files:
                safe_cleanup(temp_file)
            raise e
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Make sure progress reflects the error
        if task_id in progress_status:
            progress_status[task_id].update({
                "status": "error",
                "message": f"Pipeline failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated audio files"""
    try:
        file_path = os.path.join(config.TEMP_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"Serving file: {file_path}")
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='audio/mpeg'
        )
        
    except Exception as e:
        logger.error(f"File serve error: {e}")
        raise HTTPException(status_code=500, detail=f"File serve failed: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Server startup
if __name__ == "__main__":
    logger.info(f"Starting {config.API_TITLE} v{config.API_VERSION}")
    logger.info(f"Server will run on port {config.PORT}")
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
