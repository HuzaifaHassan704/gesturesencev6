@echo off
rem Double-click to run GestureSense (the first run installs everything).
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (set PYCMD=py -3) else (set PYCMD=python)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYCMD% -m venv .venv
  if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"
python -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto :fail
python main.py
goto :end
:fail
echo.
echo Setup failed. Make sure Python 3.10-3.12 is installed and you are online.
:end
pause
