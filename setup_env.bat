@echo off
setlocal

REM Navigate to your project folder
cd /d "D:\Machine Learning\Daisy\Puck\BEP_RFS"

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install miceforest
pip install miceforest==5.4.0

pause