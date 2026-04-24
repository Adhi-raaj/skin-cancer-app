@echo off
REM ── run.bat — Start the SkinScan AI server (Windows) ─────────────────────
REM Usage: double-click run.bat  OR  run it from cmd/PowerShell

cd /d "%~dp0backend"

REM ── Python venv ──────────────────────────────────────────────────────────
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

REM ── Install deps ─────────────────────────────────────────────────────────
echo Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

REM ── Model check ──────────────────────────────────────────────────────────
if not exist "models\convnext_best.pth" (
    echo.
    echo WARNING: No model checkpoint found!
    echo Place your trained convnext_best.pth in: backend\models\
    echo Train the model using the Colab notebook first.
    echo.
)

if not exist "models" mkdir models

REM ── Start server ─────────────────────────────────────────────────────────
echo.
echo Starting SkinScan AI server...
echo    Frontend:  http://localhost:8000
echo    API docs:  http://localhost:8000/docs
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
