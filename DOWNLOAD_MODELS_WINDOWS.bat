@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_WINDOWS.bat first to create the environment.
  pause
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
:menu
echo.
echo  Computer-vision model downloader
echo  ---------------------------------
echo  1. List models and status
echo  2. Install recommended models
echo  3. Install ALL models
echo  4. Verify installed files
echo  5. Add a custom model from a URL
echo  6. Quit
set /p C=Choose 1-6: 
if "%C%"=="1" "%PY%" -m app.download_models --list
if "%C%"=="2" "%PY%" -m app.download_models
if "%C%"=="3" "%PY%" -m app.download_models --all
if "%C%"=="4" "%PY%" -m app.download_models --verify
if "%C%"=="5" goto add
if "%C%"=="6" exit /b 0
goto menu
:add
set /p N=Model name (letters, digits, - _): 
set /p U=Direct download URL: 
set /p H=SHA-256 (optional, press Enter to skip): 
if "%H%"=="" ("%PY%" -m app.download_models --add %N% %U%) else ("%PY%" -m app.download_models --add %N% %U% --sha256 %H%)
goto menu
