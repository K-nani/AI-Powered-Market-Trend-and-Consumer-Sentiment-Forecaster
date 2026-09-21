@echo off
echo =====================================================================
echo  Starting AI-Powered Market Trend Forecaster & SentiTSMixer Suite
echo  Grounding: Ghosh et al. (IEEE Access 2025)
echo =====================================================================
echo.

set PYTHON_EXEC=%~dp0\.venv\Scripts\python.exe
set STREAMLIT_EXEC=%~dp0\.venv\Scripts\streamlit.exe

if not exist "%STREAMLIT_EXEC%" (
    echo Error: Virtual environment or Streamlit not found at %STREAMLIT_EXEC%
    pause
    exit /b 1
)

echo Launching Streamlit multi-dashboard application...
"%STREAMLIT_EXEC%" run app.py --server.headless=false

pause
