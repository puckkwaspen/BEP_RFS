@echo off
echo === Activating virtual environment ===
call .venv\Scripts\activate.bat

echo === Uninstalling miceforest ===
pip uninstall -y miceforest

echo === Deleting leftover miceforest package folders ===
rmdir /s /q .venv\Lib\site-packages\miceforest
rmdir /s /q .venv\Lib\site-packages\miceforest-*.dist-info

echo === Cleaning up __pycache__ and .pyc files ===
for /r %%i in (*.pyc) do del "%%i"
for /d /r %%i in (__pycache__) do rmdir /s /q "%%i"

echo === Reinstalling miceforest version 5.4.0 ===
pip install miceforest==5.4.0

echo === Verifying installed version ===
pip show miceforest

echo === DONE! ===
pause