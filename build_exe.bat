@echo off
rem Builds a standalone GestureSense.exe (no Python needed on the target PC).
rem Result: dist\GestureSense\GestureSense.exe  -  zip that folder to share it.
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (set PYCMD=py -3) else (set PYCMD=python)
if not exist ".venv\Scripts\python.exe" (
  %PYCMD% -m venv .venv
  if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"
python -m pip install --disable-pip-version-check -q -r requirements.txt pyinstaller
if errorlevel 1 goto :fail
rem Optional: put hand_landmarker.task next to this file first to ship it inside the build.
set MODELARG=
if exist hand_landmarker.task set MODELARG=--add-data "hand_landmarker.task;gesturesense\assets"
pyinstaller --noconfirm --clean --windowed --name GestureSense --icon gesturesense\assets\icon.ico ^
  --add-data "gesturesense\assets;gesturesense\assets" %MODELARG% ^
  --collect-all mediapipe --collect-submodules pycaw --hidden-import comtypes --hidden-import pystray._win32 --hidden-import screen_brightness_control ^
  main.py
if errorlevel 1 goto :fail
echo.
echo Done: dist\GestureSense\GestureSense.exe
goto :end
:fail
echo.
echo Build failed - see the messages above.
:end
pause
