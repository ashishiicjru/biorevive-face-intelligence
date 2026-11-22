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
echo  Computer-vision programs   (press q or Esc in a camera window to stop)
echo  --------------------------------------------------------------------
echo  1. Expression detector      (happy / sad / angry ... webcam)
echo  2. Phone detector           (webcam, beeps + logs)
echo  3. Study focus monitor      (webcam, session report at the end)
echo  4. Age + texture + skin tone analyser  (a photo)
echo  5. Gender estimator         (optional - error-prone, read the warning)
echo  6. Train a texture model    (needs a labelled dataset such as UTKFace)
echo  7. Quit
set /p C=Choose 1-7: 
if "%C%"=="1" "%PY%" programs\expression_detector.py
if "%C%"=="2" "%PY%" programs\phone_detector.py
if "%C%"=="3" "%PY%" programs\study_focus_monitor.py
if "%C%"=="4" goto age
if "%C%"=="5" goto gender
if "%C%"=="6" goto train
if "%C%"=="7" exit /b 0
goto menu
:age
set /p IMG=Path of the face photo: 
"%PY%" programs\age_texture_analyzer.py --image "%IMG%"
goto menu
:gender
echo.
echo  This is a binary guess about appearance. It can be wrong, it is less reliable for some people,
echo  and it says nothing about anyone's gender identity. Do not use it to make decisions about people.
set /p OK=Type YES to continue: 
if /i not "%OK%"=="YES" goto menu
"%PY%" programs\gender_detector.py --accept-limits
goto menu
:train
set /p TK=Task (age or gender): 
set /p DD=Folder with UTKFace-style images: 
"%PY%" programs\train_texture_model.py --task %TK% --data "%DD%"
goto menu
