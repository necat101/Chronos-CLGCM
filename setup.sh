#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- NEW: Argument Parsing ---
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
echo "[1/5] Checking for Core Dependencies..."
if ! command -v python3 &> /dev/null || ! command -v pip3 &> /dev/null; then
    echo "❌ Python 3 (python3) or pip3 not found. Please install them."
    echo "   (e.g., 'sudo apt install python3 python3-pip python3-venv')"
    exit 1
fi
echo "✅ Found python3 and pip3."

# STEP 2: Check Build Tools
echo ""
echo "[2/5] Checking for C++ Build Tools..."
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install it (e.g., 'sudo apt install cmake')."
    exit 1
fi
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "❌ No C++ compiler (g++ or clang++) found. Please install one (e.g., 'sudo apt install build-essential')."
    exit 1
fi
echo "✅ Found CMake and a C++ compiler."

# --- NEW: Vulkan Pre-check ---
if [ "$BUILD_VULKAN" == "ON" ]; then
    echo ""
    echo "[INFO] Checking for Vulkan SDK..."
    if [ -z "$VULKAN_SDK" ]; then
        echo "   ⚠️  Warning: VULKAN_SDK environment variable not set."
        echo "   Will check for 'glslc' in PATH instead..."
    else
        echo "   ✅ Found VULKAN_SDK environment variable: $VULKAN_SDK"
    fi
    
    if ! command -v glslc &> /dev/null; then
        echo "   ❌ 'glslc' compiler not found in PATH."
        echo "   Please install the Vulkan SDK from https://vulkan.lunarg.com/"
        echo "   (e.g., 'sudo apt install vulkan-sdk' or download from website)"
        echo "   and ensure 'glslc' is in your PATH."
        exit 1
    else
        echo "   ✅ Found 'glslc' compiler in PATH."
    fi
fi
# --- END: Vulkan Pre-check ---

# STEP 3: Setup Virtual Environment
echo ""
echo "[3/5] Creating/Activating Python virtual environment (./venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "✅ Activated virtual environment."

# STEP 4: Install Python Dependencies
echo ""
echo "[4/5] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements_kernel.txt
echo "✅ Python dependencies installed."

# STEP 5: Build Hierarchos Kernel
echo ""
echo "[5/5] Building Hierarchos C++ kernel..."

# --- NEW: Set environment variable for setup.py ---
export HIERARCHOS_BUILD_VULKAN="$BUILD_VULKAN"
echo "INFO: Setting HIERARCHOS_BUILD_VULKAN=$HIERARCHOS_BUILD_VULKAN"

pip3 install .
echo "✅ Build complete."

echo ""
echo "=============================================================="
echo "== ✅ Setup Complete!                                      =="
echo "== The Hierarchos kernel is built and ready to run.         =="
echo "=============================================================="
echo ""
echo "To activate the environment in your shell, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can launch Hierarchos like this:"
echo "  python3 hierarchos.py chat --model-path ./your_model"
echo ""
