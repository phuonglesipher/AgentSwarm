@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_BIN=python"
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_BIN=py -3"
)

%PYTHON_BIN% -m pip install --upgrade pip
if errorlevel 1 exit /b 1

%PYTHON_BIN% -m pip install "langgraph>=1.1,<2" "langchain-core>=1.2,<2"
if errorlevel 1 exit /b 1

%PYTHON_BIN% -m core.host_setup --agent-root "%~dp0"
if errorlevel 1 exit /b 1

echo Installed LangGraph and LangChain Core tool support with %PYTHON_BIN%
