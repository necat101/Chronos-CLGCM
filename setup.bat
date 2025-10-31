@echo off
echo ===========================================
echo == Setting up Hierarchos Environment...     ==
echo ===========================================

echo.
echo [1/3] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not found in your PATH. Please install Python 3.8+ and try again.
    exit /b 1
)

echo.
echo [2/3] Installing Python dependencies...
pip install -r requirements_kernel.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install Python dependencies.
    exit /b 1
)

echo.
echo [3/3] Compiling and building the Hierarchos C++ kernel...
pip install .
if %errorlevel% neq 0 (
    echo Error: Failed to build the C++ kernel. Make sure you have a C++ compiler (like Visual Studio Build Tools) installed.
    exit /b 1
)

echo.
echo ==============================================================
echo == Setup Complete!                                        ==
echo == The compiled kernel has been placed in the project root. ==
echo ==============================================================
echo.
echo You can now run the program directly, for example:
echo   python Hierarchos.py chat --load-quantized your_model.npz
echo.
pause
