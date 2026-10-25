@echo off
cd /d "%~dp0"
python --version > versions.txt 2>&1
python -m pip freeze >> versions.txt 2>&1
echo Done. Можна закрити.
pause
