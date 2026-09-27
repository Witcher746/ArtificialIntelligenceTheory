@echo off
setlocal
cd /d "%~dp0"
set SDL_VIDEODRIVER=windows
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment was not found.
  echo Run: python -m venv .venv
  pause
  exit /b 1
)
start "Virtual Fly Studio" ".venv\Scripts\python.exe" -m fly_ai.studio --gpu rtx
endlocal
