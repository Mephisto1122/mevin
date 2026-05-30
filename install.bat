@echo off
REM ============================================================
REM  Mevin - one-click installer for Windows
REM  Installs Python deps, pulls the AI model, starts the app
REM ============================================================
echo.
echo   Installing Mevin...
echo.

REM --- Check Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo   [!] Python not found. Install it from https://python.org
    echo       Be sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

REM --- Check Ollama ---
where ollama >nul 2>nul
if errorlevel 1 (
    echo   [!] Ollama not found. Download it from https://ollama.com
    echo       Install it, then run this again.
    pause
    exit /b 1
)

echo   Installing Python packages...
python -m pip install --upgrade pip >nul 2>nul
python -m pip install -r requirements.txt

echo.
set /p ONVIF="  Enable auto-discovery for IP cameras/NVRs? (y/n): "
if /i "%ONVIF%"=="y" python -m pip install -r requirements-onvif.txt

echo.
echo   Pulling the AI model (gemma3:4b, ~3.3GB, one time)...
ollama pull gemma3:4b

echo.
echo   Starting Mevin...
echo   Open http://localhost:5555 in your browser.
echo.
python mevin.py
pause
