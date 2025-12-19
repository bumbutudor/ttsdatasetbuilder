#!/usr/bin/env python3
"""
Local development server for TTS/STT Dataset Builder.
Run with: python run.py
"""
import uvicorn
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("TTS/STT Dataset Builder - Development Server")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"]
    )

