@echo off

:: Deactivate and remove the virtual environment if it exists
if exist venv (
    echo Deactivating and removing virtual environment...
    call venv\Scripts\deactivate.bat
    rmdir /s /q venv
) else (
    echo No virtual environment found.
)

:: Optionally delete the Python installer if it was downloaded
if exist python_installer.exe (
    echo Deleting Python installer...
    del python_installer.exe
)

:: Optionally remove Python installation
:: This part is risky and should be used carefully. Removing Python may affect other projects.
:: Uncomment the following lines if you want to remove Python as well.

:: echo Removing Python installation...
:: rmdir /s /q "C:\Program Files\Python312"
:: rmdir /s /q "C:\Program Files\Python312\Scripts"

:: Remove any project files if needed
:: Uncomment and modify the following lines if you want to remove specific project files.
:: echo Deleting project files...
:: del /q ijewelmatch.py

echo Uninstallation complete.
pause
