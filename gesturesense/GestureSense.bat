@echo off
rem One-click launcher: double-click this (or the desktop shortcut) to start GestureSense.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo First run: setting things up. This takes a few minutes, then GestureSense opens.
  call "%~dp0run.bat"
  exit /b
)
start "" ".venv\Scripts\pythonw.exe" main.py
exit /b
