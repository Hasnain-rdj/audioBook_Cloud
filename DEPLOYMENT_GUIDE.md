# AudioBook API - Render.com Deployment Guide

## 🎯 Overview
This guide helps you deploy the AudioBook API to Render.com for cloud access in your n8n workflow.

## 📋 Prerequisites
- GitHub account
- Render.com account (free tier available)
- Your project files ready for deployment

## 🚀 Deployment Steps

### 1. Prepare Your Repository
1. Create a new GitHub repository
2. Upload these files to your repository:
   - `audioBook_api.py` (main API file)
   - `requirements_render.txt` (dependencies)
   - `runtime.txt` (Python version)
   - `Procfile` (deployment command)

### 2. Connect to Render.com
1. Go to [render.com](https://render.com)
2. Sign up/Login with your GitHub account
3. Click "New +" → "Web Service"
4. Connect your GitHub repository

### 3. Configure Deployment Settings
```
Name: audiobook-api (or your preferred name)
Environment: Python 3
Region: Oregon (US West) or closest to you
Branch: main
Root Directory: (leave empty)
Build Command: pip install -r requirements_render.txt
Start Command: uvicorn audioBook_api:app --host 0.0.0.0 --port $PORT
```

### 4. Environment Variables (Optional)
Set these if needed:
- `PYTHONPATH`: /opt/render/project/src
- `PORT`: 10000 (Render will set this automatically)

### 5. Deploy
1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Your API will be available at: `https://your-service-name.onrender.com`

## 🔗 API Endpoints
Once deployed, your API will have these endpoints:

### Health Check
- **GET** `/` - Service status
- **GET** `/health` - Detailed health check

### Main Processing
- **POST** `/process-pdf-to-audio` - Convert PDF to audio
  - Upload PDF file
  - Get processed audio file back

## 📝 n8n Integration
Use your deployed URL in n8n HTTP Request node:
```
Method: POST
URL: https://your-service-name.onrender.com/process-pdf-to-audio
Content-Type: multipart/form-data
Body: Binary file (PDF)
```

## ⚠️ Important Notes
1. **Free Tier Limitations:**
   - Service sleeps after 15 minutes of inactivity
   - First request after sleep takes 1-2 minutes
   - 750 hours/month free usage

2. **File Size Limits:**
   - PDF files: Max 10MB
   - Processing time: Max 30 seconds per request

3. **Dependencies:**
   - Only essential packages included for faster deployment
   - Voice cloning features simplified for cloud compatibility

## 🐛 Troubleshooting

### Pydantic Build Error (Rust compilation issues):
**Problem:** `pydantic-core` fails to compile with Rust errors
**Solution:** Use the minimal requirements files provided:

1. **First try:** `requirements_render.txt` (updated with stable versions)
2. **If that fails:** `requirements_minimal.txt` (flexible version ranges)  
3. **Last resort:** `requirements_ultra_minimal.txt` (no version pins)

### Alternative Deployment Options:

#### Option 1: Use Simple API (Recommended for troubleshooting)
```
Main File: audiobook_simple.py
Requirements: requirements_ultra_minimal.txt
Procfile: web: uvicorn audiobook_simple:app --host 0.0.0.0 --port $PORT
```

#### Option 2: Original API with Fixed Dependencies
```
Main File: audioBook_api.py  
Requirements: requirements_render.txt (updated)
Procfile: web: uvicorn audioBook_api:app --host 0.0.0.0 --port $PORT
```

### Common Issues:
1. **Build Fails:** 
   - Try different requirements files in order of complexity
   - Check Python version compatibility (use 3.11.9)
   
2. **Service Won't Start:** 
   - Verify Procfile points to correct Python file
   - Check logs for import errors

3. **Timeout Errors:** 
   - Large PDFs may exceed processing time
   - Text is automatically limited to 30k characters

### Step-by-Step Fix for Current Error:

1. **Update your repository** with these files:
   - Replace `requirements_render.txt` with the updated version
   - Add `audiobook_simple.py` as backup option
   - Update `runtime.txt` to `python-3.11.9`

2. **Try Simple Deployment First:**
   - Change Procfile to: `web: uvicorn audiobook_simple:app --host 0.0.0.0 --port $PORT`
   - Use `requirements_ultra_minimal.txt` as build command

3. **If Simple Works:**
   - Switch back to full API after confirming basic deployment works

## 📞 Support
If you encounter issues:
1. Check Render.com logs
2. Verify all files are in your GitHub repository
3. Test endpoints with Postman first

---
**Ready to deploy!** 🚀
