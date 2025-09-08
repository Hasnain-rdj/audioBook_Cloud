#!/usr/bin/env python3
"""
Test script to verify offline AudioBook API functionality
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI not available")
        return False
    
    try:
        import uvicorn
        print(f"✅ Uvicorn available")
    except ImportError:
        print("❌ Uvicorn not available")
        return False
        
    try:
        import PyPDF2
        print(f"✅ PyPDF2 available")
    except ImportError:
        print("❌ PyPDF2 not available")
        return False
    
    try:
        import pyttsx3
        print(f"✅ pyttsx3 available (offline TTS)")
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f"   - Available voices: {len(voices) if voices else 0}")
        engine.stop()
    except ImportError:
        print("⚠️ pyttsx3 not available (cloud fallback mode)")
    except Exception as e:
        print(f"⚠️ pyttsx3 initialization issue: {e}")
    
    return True

def test_api_basic():
    """Test basic API functionality"""
    print("\nTesting API...")
    
    try:
        from audioBook_api import app, config
        print(f"✅ API loaded successfully")
        print(f"   - Title: {config.API_TITLE}")
        print(f"   - Version: {config.API_VERSION}")
        print(f"   - Processing Mode: {config.PROCESSING_MODE}")
        print(f"   - Python Target: {config.PYTHON_VERSION}")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Main test function"""
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_imports()
    api_ok = test_api_basic()
    
    print("=" * 50)
    if imports_ok and api_ok:
        print("✅ All tests passed! API ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
