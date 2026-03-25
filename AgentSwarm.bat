@echo off
setlocal
cd /d "%~dp0"
set "PROMPT=%*"

call :resolve_python3
if errorlevel 1 exit /b 1

%PYTHON_BIN% main.py --prompt "%PROMPT%"
exit /b 0

:resolve_python3
py -3 -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_BIN=py -3"
    exit /b 0
)

python -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_BIN=python"
    exit /b 0
)

echo Python 3 is required to run AgentSwarm. Install Python 3 or ensure `py -3` or `python` resolves to Python 3.
exit /b 1
