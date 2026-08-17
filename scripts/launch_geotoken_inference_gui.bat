@echo off
setlocal
cd /d "%~dp0\.."
python scripts\geotoken_inference_gui.py
if errorlevel 1 pause

