@echo off
setlocal enabledelayedexpansion

:: --- NEW: Argument Parsing ---
set "BUILD_VULKAN=OFF"
:arg_loop
if "%~1"=="" goto :args_done
if /I "%~1"=="--vulkan" (
    echo INFO: --vulkan flag detected. Will attempt to build with Vulkan support.
    set "BUILD_VULKAN=ON"
)
shift
goto :arg_loop
:args_done
:: --- END: Argument Parsing ---

echo ===========================================
echo == Setting up Hierarchos Environment...  ==
echo ===========================================

:: --- NEW: Vulkan Pre-check ---
if "!BUILD_VULKAN!"=="ON" (
    echo.
    echo [INFO] Checking for Vulkan SDK...
    if defined VULKAN_SDK (
        echo   ✅ Found VULKAN_SDK environment variable: !VULKAN_SDK!
        where glslc >nul 2>&1
        if %errorlevel% neq 0 (
            echo   ⚠️  Warning: 'glslc' compiler not found in PATH.
            echo   Please ensure !VULKAN_SDK!\Bin is in your system PATH.
        ) else (
            echo   ✅ Found 'glslc' compiler in PATH.
        )
    ) else (
        echo   ❌ VULKAN_SDK environment variable not set.
        echo   Please install the Vulkan SDK from https://vulkan.lunarg.com/
        echo   and ensure VULKAN_SDK is set, or 'glslc' is in your PATH.
        echo   The build may fail if CMake cannot find Vulkan components.
        pause
    )
)
:: --- END: Vulkan Pre-check ---

:: STEP 1 — CHECK PYTHON
echo.
echo [1/5] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

:: STEP 2 — CHECK/INSTALL BUILD TOOLS
echo.
echo [2/5] Checking for Microsoft C++ Build Tools...
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo Microsoft C++ Build Tools not found. Installing now...
    echo Downloading Visual Studio Build Tools installer...

    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass ^
      " $url='https://aka.ms/vs/17/release/vs_buildtools.exe'; " ^
      " $out=Join-Path $env:TEMP 'vs_buildtools.exe'; " ^
      " Write-Host ('Downloading to: ' + $out); " ^
      " Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing; " ^
      " if (Test-Path $out) {Write-Host ('✅ Saved to: ' + $out)} else {Write-Host '❌ Download failed'; exit 1}"

    if not exist "%TEMP%\vs_buildtools.exe" (
        echo ❌ Download failed. Please download manually from:
        echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
        pause
        exit /b 1
    )

    echo.
    echo Installing required components... (this can take several minutes)
    "%TEMP%\vs_buildtools.exe" --quiet --wait --norestart ^
        --add Microsoft.VisualStudio.Workload.VCTools ^
        --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
        --add Microsoft.VisualStudio.Component.Windows11SDK.22621 ^
        --includeRecommended

    echo.
    echo ✅ Build Tools installation completed.
) else (
    echo ✅ Microsoft C++ Build Tools already installed.
)

:: ==================================================================
:: STEP 3 — FIND AND INITIALIZE COMPILER ENVIRONMENT (REVISED)
:: ==================================================================
echo.
echo [3/5] Initializing compiler environment...

set "VCVARS_PATH="
set "VSWHERE_PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "!VSWHERE_PATH!" (
    echo ❌ Cannot find vswhere.exe at "!VSWHERE_PATH!".
    echo This tool is required to find the build tools.
    echo Your Visual Studio Installer might be corrupted.
    goto :fallback_search
)

echo INFO: Found vswhere.exe. Querying directly for vcvars64.bat...
:: --- FIX: Use a temp file to avoid cmd.exe 'for /f' parsing bugs ---
set "TEMP_VCVARS_PATH_FILE=%TEMP%\vcvars_path_%RANDOM%.txt"
"!VSWHERE_PATH!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" > "!TEMP_VCVARS_PATH_FILE!" 2>nul

if %errorlevel% neq 0 (
    echo INFO: vswhere.exe command failed or returned an error.
    set "VCVARS_PATH="
) else (
    REM Read the first line from the temp file into the variable
    set /p VCVARS_PATH=<"!TEMP_VCVARS_PATH_FILE!"
)

if exist "!TEMP_VCVARS_PATH_FILE!" (
    del "!TEMP_VCVARS_PATH_FILE!"
)
:: --- End of temp file fix ---

if defined VCVARS_PATH (
    if exist "!VCVARS_PATH!" (
        echo ✅ Found vcvars64.bat via vswhere:
        echo !VCVARS_PATH!
        goto :init_env
    )
)

echo INFO: vswhere.exe did not find vcvars64.bat.

:fallback_search
echo INFO: Falling back to broad (slower) search...

echo INFO: Searching for vcvars64.bat in "C:\Program Files\"...
for /f "delims=" %%i in ('dir /b /s "C:\Program Files\Microsoft Visual Studio\*\VC\Auxiliary\Build\vcvars64.bat" 2^>nul') do (
    set "VCVARS_PATH=%%i"
    echo ✅ Found vcvars64.bat via fallback search:
    echo !VCVARS_PATH!
    goto :init_env
)

echo INFO: Searching for vcvars64.bat in "C:\Program Files (x86)\"...
for /f "delims=" %%i in ('dir /b /s "C:\Program Files (x86)\Microsoft Visual Studio\*\VC\Auxiliary\Build\vcvars64.bat" 2^>nul') do (
    set "VCVARS_PATH=%%i"
    echo ✅ Found vcvars64.bat via fallback search:
    echo !VCVARS_PATH!
    goto :init_env
)

if not defined VCVARS_PATH (
    echo ❌ Could not find vcvars64.bat automatically.
    echo Please run setup from an "x64 Native Tools Command Prompt"
    pause
    exit /b 1
)

:init_env
echo Initializing compiler environment...
call "!VCVARS_PATH!" x64 >nul
if %errorlevel% neq 0 (
    echo ⚠️  Warning: call to vcvars64.bat failed. Build may fail.
    goto :check_cl
)

echo ✅ Compiler environment initialized.

:check_cl
echo Checking for cl.exe in PATH...
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: cl.exe not found in PATH even after calling vcvars64.bat.
    echo This indicates a problem with your Build Tools installation.
    echo Please try restarting your PC or running from an
    echo "x64 Native Tools Command Prompt" for Visual Studio.
    pause
    exit /b 1
)

echo ✅ cl.exe is correctly linked in your PATH for this session.

:: ==================================================================
:: END OF REVISED STEP 3
:: ==================================================================


:: STEP 4 — INSTALL PYTHON DEPENDENCIES
echo.
echo [4/5] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements_kernel.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python dependencies.
    pause
    exit /b 1
)

:: STEP 5 — BUILD HIERARCHOS KERNEL
echo.
echo [5/5] Building Hierarchos C++ kernel...

:: --- NEW: Set environment variable for setup.py based on parsed args ---
set "HIERARCHOS_BUILD_VULKAN=!BUILD_VULKAN!"
echo INFO: Setting HIERARCHOS_BUILD_VULKAN=!HIERARCHOS_BUILD_VULKAN!

pip install .
if %errorlevel% neq 0 (
    echo ❌ Build failed. Try restarting your PC and re-running this script as Administrator.
    pause
    exit /b 1
)

echo.
echo ==============================================================
echo == ✅ Setup Complete!                                      ==
echo == The Hierarchos kernel is built and ready to run.       ==
echo ==============================================================
echo.
echo You can now launch Hierarchos like this:
echo   python hierarchos.py chat --model-path ./your_model
echo.
pause
endlocal
