@echo off
title CineMatch AI - Movie Recommendation System
color 0a

echo ======================================================================
echo    CINEMATCH AI - NEXT-GEN MOVIE RECOMMENDATION SYSTEM
echo    Interactive Cinematic Discovery ^& Machine Learning Engine
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Initializing SQLite Movie Database ^& AI Recommender Index...
python database.py

echo.
echo [2/3] Opening Web Browser at http://127.0.0.1:5001 ...
start http://127.0.0.1:5001

echo.
echo [3/3] Starting Standalone Local Server on Port 5001...
echo Press CTRL+C at any time to stop the server.
echo ======================================================================
echo.

python server.py

pause
