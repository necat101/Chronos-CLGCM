#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Argument Parsing ---
BUILD_VULKAN="OFF"
for arg in "$@"; do
    case $arg in
        --vulkan)
        echo "INFO: --vulkan flag detected. Will attempt to build with Vulkan support."
        BUILD_VULKAN="ON"
        shift # Remove --vulkan from processing
        ;;
    esac
done
# --- END: Argument Parsing ---

echo "============================================"
echo "== Setting up Hierarchos Environment (Linux/macOS) =="
echo "============================================"

# STEP 1: Check Core Dependencies
echo ""
echo "[1/4] Checking for Core Dependencies..."
if ! command -v python3 &> /dev/null || ! command -v pip3 &> /dev/null; then
    echo "❌ Python 3 (python3) or pip3 not found. Please install them."
    echo "   (e.g., 'sudo apt install python3 python3-pip')"
    exit 1
fi
echo "✅ Found python3 and pip3."

# STEP 2: Check Build Tools
echo ""
echo "[2/4] Checking for C++ Build Tools..."
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install it (e.g., 'sudo apt install cmake')."
    exit 1
fi
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "❌ No C++ compiler (g++ or clang++) found. Please install one (e.g., 'sudo apt install build-essential')."
    exit 1
fi
echo "✅ Found CMake and a C++ compiler."

# --- Vulkan Pre-check and Auto-Install ---
if [ "$BUILD_VULKAN" == "ON" ]; then
    echo ""
    echo "[INFO] Checking for Vulkan SDK..."
    
    # --- MODIFIED CHECK: Look for 'glslangValidator' ---
    if command -v glslangValidator &> /dev/null; then
        echo "   ✅ Found 'glslangValidator' compiler in PATH."
    else
        echo "   ❌ 'glslangValidator' compiler not found in PATH."
        echo "   Attempting to automatically install Vulkan build tools (glslang-tools, libvulkan-dev)..."
        echo "   This will require superuser (sudo) permission."
        
        # Set frontend to noninteractive to suppress debconf dialog warnings
        export DEBIAN_FRONTEND=noninteractive
        sudo apt update
        sudo apt install -y glslang-tools libvulkan-dev
        
        # Clear the shell's command cache
        hash -r
        
        # Check again
        if command -v glslangValidator &> /dev/null; then
            echo "   ✅ Successfully installed and found 'glslangValidator' in PATH."
        elif [ -f /usr/bin/glslangValidator ]; then
            echo "   ⚠️  WARNING: 'glslangValidator' was found at /usr/bin/glslangValidator but is NOT in your PATH."
            echo "   This suggests a problem with your shell's environment setup."
            echo "   Continuing anyway, as CMake can often find it directly."
        else
            echo "   ❌ ERROR: Install ran, but 'glslangValidator' could not be found."
            echo "   Please install 'glslang-tools' manually and ensure 'glslangValidator' is in your PATH."
            exit 1
        fi
    fi
    # --- END: Modified Check ---
fi
# --- END: Vulkan Pre-check ---

# STEP 3: Install Python Dependencies (No Venv)
echo ""
echo "[3/4] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements_kernel.txt
if [ $? -ne 0 ]; then echo "❌ Failed to install python dependencies." && exit 1; fi
echo "✅ Python dependencies installed."

# STEP 4: Build Hierarchos Kernel (No Venv)
echo ""
echo "[4/4] Compiling and building the Hierarchos C++ kernel..."

# Set environment variable for setup.py
export HIERARCHOS_BUILD_VULKAN="$BUILD_VULKAN"
echo "INFO: Setting HIERARCHOS_BUILD_VULKAN=$HIERARCHOS_BUILD_VULKAN"

# --- MODIFICATION: Added -v (verbose) flag to show the real error ---
pip3 install -v .
# --- END MODIFICATION ---

if [ $? -ne 0 ]; then echo "❌ Kernel build failed." && exit 1; fi
echo "✅ Build complete."

echo ""
echo "=============================================================="
echo "== ✅ Setup Complete!                                      =="
echo "== The Hierarchos kernel is built and ready to run.         =="
echo "=============================================================="
echo ""
echo "You can now run the program directly, for example:"
echo "  python3 hierarchos.py chat --model-path ./your_model"
echo ""
