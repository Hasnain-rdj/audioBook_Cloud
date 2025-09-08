#!/usr/bin/env python3
"""
Simple setup.py for audioBook_Cloud deployment
"""

from setuptools import setup, find_packages

setup(
    name="audiobook-cloud",
    version="2.0.0",
    description="Offline AudioBook Voice Cloning API",
    author="AudioBook Cloud",
    python_requires=">=3.9,<3.10",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.85.0",
        "uvicorn[standard]==0.19.0", 
        "python-multipart==0.0.5",
        "PyPDF2==3.0.1",
        "pyttsx3==2.90",
        "requests==2.28.2",
        "librosa==0.9.2",
        "soundfile==0.11.0",
        "numpy==1.23.5",
        "scipy==1.9.3",
        "Pillow==9.5.0",
        "tqdm==4.64.1"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
    ],
)
