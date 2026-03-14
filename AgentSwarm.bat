@echo off
setlocal
cd /d "%~dp0"
set "PROMPT=%*"
python main.py --prompt "%PROMPT%"
