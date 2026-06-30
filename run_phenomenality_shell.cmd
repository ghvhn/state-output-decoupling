@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\interactive_phenomenality.py"
echo.
echo [Phenomenality shell exited. Press any key to close.]
pause >nul
