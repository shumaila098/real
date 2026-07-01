@echo off
REM Train (if needed) and launch the scikit-learn prediction service on :8000.
setlocal
cd /d "%~dp0"

if not exist ".venv" (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo Starting prediction service on http://localhost:8000 ...
python -m uvicorn app:app --host 0.0.0.0 --port 8000
