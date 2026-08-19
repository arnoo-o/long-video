@echo off
setlocal
cd /d "%~dp0\.."
python scripts\wah_world_inference_gui.py
if errorlevel 1 pause
