@echo off
setlocal
set "PROMPT=%*"
python main.py --prompt "%PROMPT%"
