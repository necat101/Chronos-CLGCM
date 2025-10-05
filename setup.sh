#!/bin/bash
set -e

echo "==========================================="
echo "== Setting up Chronos Environment...     =="
echo "==========================================="

echo ""
echo "[1/3] Checking for Python..."
python3 --version

echo ""
echo "[2/3] Installing Python dependencies..."
pip install -r requirements_kernel.txt

echo ""
echo "[3/3] Compiling and building the Chronos C++ kernel..."
pip install .

echo ""
echo "=============================================================="
echo "== Setup Complete!                                        =="
echo "== The compiled kernel has been placed in the project root. =="
echo "=============================================================="
echo ""
echo "You can now run the program directly, for example:"
echo "  python3 chronos.py chat --load-quantized your_model.npz"
echo ""
