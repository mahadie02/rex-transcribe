@echo off
REM WhisperX Server - Install dependencies (uses system ffmpeg, excludes torchcodec)
echo Installing dependencies...
pip install -r requirements.txt
echo Removing torchcodec (avoids FFmpeg DLL errors on Windows)...
pip uninstall torchcodec -y 2>nul
echo Done. Run: python server.py
