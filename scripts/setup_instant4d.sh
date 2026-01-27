#!/bin/bash
# Brain Dance - Instant4D CUDA Kernel Setup
# Run on GPU server only (requires CUDA 12.1+)
#
# Usage:
#   chmod +x scripts/setup_instant4d.sh
#   ./scripts/setup_instant4d.sh
#
# Prerequisites:
#   - CUDA toolkit 12.1+ installed (nvcc available)
#   - PyTorch 2.3+ with CUDA support
#   - instant4d submodule initialized

set -e  # Exit on error

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTANT4D="$PROJECT_ROOT/instant4d"

echo "========================================"
echo "Brain Dance - Instant4D Setup"
echo "========================================"
echo ""

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================
echo "[1/6] Checking prerequisites..."

# Check nvcc (CUDA compiler)
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit 12.1+"
    echo "       Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "  CUDA toolkit: $CUDA_VERSION"

# Check Python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

# Check PyTorch with CUDA
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "ERROR: PyTorch with CUDA not available."
    echo "       Install with: pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
echo "  PyTorch: $TORCH_VERSION (CUDA $TORCH_CUDA)"

# Check GPU
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")
echo "  GPU: $GPU_NAME ($GPU_COUNT device(s))"

if [ "$GPU_COUNT" == "0" ]; then
    echo "ERROR: No CUDA GPUs detected"
    exit 1
fi

# =============================================================================
# Step 2: Verify submodule
# =============================================================================
echo ""
echo "[2/6] Verifying Instant4D submodule..."

if [ ! -d "$INSTANT4D" ]; then
    echo "ERROR: instant4d/ directory not found at $INSTANT4D"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "$INSTANT4D/README.md" ]; then
    echo "ERROR: instant4d/ appears to be empty"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

echo "  Found: $INSTANT4D"

# Check for nested submodules
if [ ! -d "$INSTANT4D/SLAM/mega-sam" ] || [ -z "$(ls -A "$INSTANT4D/SLAM/mega-sam" 2>/dev/null)" ]; then
    echo "  Initializing nested submodules..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive
fi

echo "  Nested submodules: OK"

# =============================================================================
# Step 3: Install diff-gaussian-rasterization
# =============================================================================
echo ""
echo "[3/6] Installing diff-gaussian-rasterization..."

DIFF_GAUSS="$INSTANT4D/diff-gaussian-rasterization"
if [ ! -d "$DIFF_GAUSS" ]; then
    echo "ERROR: diff-gaussian-rasterization not found at $DIFF_GAUSS"
    exit 1
fi

cd "$DIFF_GAUSS"
pip install -e . --quiet
echo "  Installed: diff-gaussian-rasterization"

# =============================================================================
# Step 4: Install pointops2
# =============================================================================
echo ""
echo "[4/6] Installing pointops2..."

POINTOPS="$INSTANT4D/submodule/pointops2"
if [ ! -d "$POINTOPS" ]; then
    echo "ERROR: pointops2 not found at $POINTOPS"
    exit 1
fi

cd "$POINTOPS"
pip install -e . --quiet
echo "  Installed: pointops2"

# =============================================================================
# Step 5: Install simple-knn
# =============================================================================
echo ""
echo "[5/6] Installing simple-knn..."

SIMPLE_KNN="$INSTANT4D/submodule/simple-knn"
if [ ! -d "$SIMPLE_KNN" ]; then
    echo "ERROR: simple-knn not found at $SIMPLE_KNN"
    exit 1
fi

cd "$SIMPLE_KNN"
pip install -e . --quiet
echo "  Installed: simple-knn"

# =============================================================================
# Step 6: Install fused-ssim
# =============================================================================
echo ""
echo "[6/6] Installing fused-ssim..."

FUSED_SSIM="$INSTANT4D/submodule/fussed-ssim"
if [ ! -d "$FUSED_SSIM" ]; then
    echo "ERROR: fused-ssim not found at $FUSED_SSIM"
    exit 1
fi

cd "$FUSED_SSIM"
pip install -e . --quiet
echo "  Installed: fused-ssim"

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Verifying installations..."

# Return to project root
cd "$PROJECT_ROOT"

# Verify each package
ERRORS=0

if python -c "from diff_gaussian_rasterization import GaussianRasterizer" 2>/dev/null; then
    echo "  [OK] diff-gaussian-rasterization"
else
    echo "  [FAIL] diff-gaussian-rasterization"
    ERRORS=$((ERRORS + 1))
fi

if python -c "import pointops" 2>/dev/null; then
    echo "  [OK] pointops"
else
    echo "  [FAIL] pointops"
    ERRORS=$((ERRORS + 1))
fi

if python -c "from simple_knn import distCUDA2" 2>/dev/null; then
    echo "  [OK] simple-knn"
else
    echo "  [FAIL] simple-knn"
    ERRORS=$((ERRORS + 1))
fi

if python -c "from fused_ssim import fused_ssim" 2>/dev/null; then
    echo "  [OK] fused-ssim"
else
    echo "  [FAIL] fused-ssim"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "All CUDA kernels installed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: pytest tests/test_instant4d_setup.py -v -m gpu"
    echo "  2. Proceed with Phase 0, Step 2 (Instant4D Adapter)"
else
    echo "WARNING: $ERRORS package(s) failed to install correctly."
    echo "Check the output above for errors."
    exit 1
fi
