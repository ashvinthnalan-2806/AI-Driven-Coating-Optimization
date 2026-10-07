@echo off
title AI-Driven Coating Formulation & Property Optimization - Web Server
cd /d "%~dp0"
echo ======================================================================
echo Starting AI Coating Prediction Web Server...
echo ======================================================================
if exist "..\AI_Coating_Optimization\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\AI_Coating_Optimization\.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
echo Opening website at http://localhost:5000 in your browser...
start http://localhost:5000
"%PYTHON_EXE%" app.py
pause
