@echo off
setlocal

:: Define variables
set VENV_PATH=%APPDATA%\iJewelMatch\venv

:: Check if Python is already installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto :setup_env
)

:: If Python is not installed, download and install Python 3.11.6
echo Python not found. Downloading Python 3.11.6...
powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe -OutFile python_installer.exe"
echo Installing Python 3.11.6...
start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
del python_installer.exe

:: Set Python path explicitly
set PATH=%PATH%;C:\Program Files\Python311;C:\Program Files\Python311\Scripts

:: Verify Python installation
python --version
if %errorlevel% neq 0 (
    echo Python installation failed. Please install Python 3.11.6 manually.
    pause
    exit /b
)

:setup_env
:: Create the directory for the virtual environment
echo Creating directory for virtual environment...
mkdir "%VENV_PATH%"

:: Create a new virtual environment
echo Creating virtual environment...
python -m venv "%VENV_PATH%"

:: Activate the virtual environment
call "%VENV_PATH%\Scripts\activate.bat"

:: Upgrade pip
python -m pip install --upgrade pip
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install required packages
echo Installing required packages...
pip install flask faiss-cpu pillow numpy tqdm werkzeug

:: Run the Flask app
echo Sparkling up your iJewelMatch...
python ijewelmatch.py

endlocal
pause
