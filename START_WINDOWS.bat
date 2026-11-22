@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto have_venv

set "BASEPY="
for %%V in (3.12 3.11 3.13 3.10 3.14) do (
  if not defined BASEPY (
    py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
  )
)
if not defined BASEPY (
  python -c "import sys" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
  echo No Python found. Install Python 3.11 or newer from python.org and tick "Add to PATH".
  pause
  exit /b 1
)
echo Creating virtual environment with: %BASEPY%
%BASEPY% -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo Could not create the virtual environment.
  pause
  exit /b 1
)

:have_venv
set "PYEXE=%~dp0.venv\Scripts\python.exe"
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed - see messages above.
  pause
  exit /b 1
)
if not exist .env copy .env.example .env
"%PYEXE%" -m app.download_models
echo.
echo Open http://127.0.0.1:8000
"%PYEXE%" -m uvicorn app.main:app --reload
pause
