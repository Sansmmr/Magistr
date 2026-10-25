@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0backend"
start "backend" cmd /k "python main.py > ..\web_backend.log 2>&1"
cd /d "%~dp0frontend"
start "frontend" cmd /k "npm run dev > ..\web_frontend.log 2>&1"
