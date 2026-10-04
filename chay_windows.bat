@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Hay chay cai_dat_windows.bat truoc.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
