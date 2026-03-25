@echo off
setlocal
cd /d "%~dp0"

call :resolve_python3
if errorlevel 1 exit /b 1

%PYTHON_BIN% -m pip install --upgrade pip
if errorlevel 1 exit /b 1

%PYTHON_BIN% -m pip install "langgraph>=1.1,<2" "langchain-core>=1.2,<2"
if errorlevel 1 exit /b 1

%PYTHON_BIN% -m core.host_setup --agent-root "%~dp0"
if errorlevel 1 exit /b 1

echo Installed LangGraph and LangChain Core tool support with %PYTHON_DISPLAY%
exit /b 0

:resolve_python3
py -3 -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_BIN=py -3"
    set "PYTHON_DISPLAY=py -3"
    exit /b 0
)

python -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_BIN=python"
    set "PYTHON_DISPLAY=python"
    exit /b 0
)

echo Python 3 is required to set up AgentSwarm. Install Python 3 or ensure `py -3` or `python` resolves to Python 3.
exit /b 1
