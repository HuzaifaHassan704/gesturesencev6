@echo off
rem Puts a "GestureSense" icon on your Desktop. Double-click the icon to start the app.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'),'GestureSense.lnk'));" ^
  "$s.TargetPath='%~dp0GestureSense.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0icon.ico'; $s.WindowStyle=7; $s.Description='GestureSense hand-gesture control'; $s.Save()"
if errorlevel 1 (echo Could not create the shortcut.) else (echo Done - look for GestureSense on your Desktop.)
pause
