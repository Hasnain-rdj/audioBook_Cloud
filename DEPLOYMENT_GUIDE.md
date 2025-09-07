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

### Common Issues:
1. **Build Fails:** Check requirements_render.txt syntax
2. **Service Won't Start:** Verify Procfile command
3. **Timeout Errors:** Large PDFs may exceed processing time

### Debug Steps:
1. Check deployment logs in Render dashboard
2. Test health endpoint first: `/health`
3. Start with small PDF files

## 📞 Support
If you encounter issues:
1. Check Render.com logs
2. Verify all files are in your GitHub repository
3. Test endpoints with Postman first

---
**Ready to deploy!** 🚀
