@echo off
setlocal
cd /d "%~dp0"
py -3 --version
if errorlevel 1 goto no_python
py -3 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-ai.txt
if errorlevel 1 goto failed
echo Cai dat thanh cong. Mo chay_windows.bat de su dung.
pause
exit /b 0
:no_python
echo Hay cai Python 3.11 hoac 3.12, kem Python Launcher, roi chay lai.
pause
exit /b 1
:failed
echo Cai dat chua thanh cong. Kiem tra thong bao loi va ket noi mang.
pause
exit /b 1
