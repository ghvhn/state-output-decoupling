@echo off
cd /d "%~dp0"
rem  Optional first argument: a model id/path to run the egg shell on, e.g.
rem      run_phenomenality_shell.cmd Qwen/Qwen2.5-1.5B-Instruct
rem  When launched from the benchmark egg, the model is passed via EGG_MODEL.
".venv\Scripts\python.exe" "scripts\interactive_phenomenality.py" %*
echo.
echo [Phenomenality shell exited. Press any key to close.]
pause >nul
