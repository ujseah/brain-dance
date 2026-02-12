#!/bin/bash
# Brain Dance - Deformable 3D Gaussians (De3DGS) CUDA Kernel Setup
# Run on GPU server only (requires CUDA 11.6+)
#
# Usage:
#   chmod +x scripts/setup_de3dgs.sh
#   ./scripts/setup_de3dgs.sh
#
# Prerequisites:
#   - CUDA toolkit 11.6+ installed (nvcc available)
#   - PyTorch 1.13.1+ with CUDA support
#   - deformable3dgs submodule initialized
#
# Note: De3DGS uses PyTorch 1.13.1 by default. If using existing brain-dance
# environment with newer PyTorch, the CUDA kernels should still be compatible.

set -e  # Exit on error

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DE3DGS="$PROJECT_ROOT/deformable3dgs"

echo "========================================"
echo "Brain Dance - De3DGS Setup"
echo "========================================"
echo ""

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================
echo "[1/6] Checking prerequisites..."

# Check nvcc (CUDA compiler)
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit 11.6+"
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
    echo "       Install with: pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
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
echo "[2/6] Verifying De3DGS submodule..."

if [ ! -d "$DE3DGS" ]; then
    echo "ERROR: deformable3dgs/ directory not found at $DE3DGS"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "$DE3DGS/train.py" ]; then
    echo "ERROR: deformable3dgs/ appears to be empty"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

echo "  Found: $DE3DGS"

# Check for nested submodules
if [ ! -d "$DE3DGS/submodules/depth-diff-gaussian-rasterization" ] || [ -z "$(ls -A "$DE3DGS/submodules/depth-diff-gaussian-rasterization" 2>/dev/null)" ]; then
    echo "  Initializing nested submodules..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive
fi

echo "  Nested submodules: OK"

# =============================================================================
# Step 2.5: Apply CUDA 12.x compatibility patches (if needed)
# =============================================================================
# Extract major version from CUDA_VERSION (e.g., "12.8" -> "12")
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)

if [ "$CUDA_MAJOR" -ge 12 ] 2>/dev/null; then
    echo ""
    echo "[2.5/6] Applying CUDA 12.x compatibility patches..."

    PATCH_SCRIPT="$PROJECT_ROOT/scripts/patch_cuda12.sh"
    if [ -f "$PATCH_SCRIPT" ]; then
        bash "$PATCH_SCRIPT"
    else
        # Inline patching if script doesn't exist
        echo "  Patch script not found, applying inline patches..."

        SIMPLE_KNN="$DE3DGS/submodules/simple-knn/simple_knn.cu"
        if [ -f "$SIMPLE_KNN" ] && ! grep -q "#include <float.h>" "$SIMPLE_KNN"; then
            sed -i '1i #include <float.h>' "$SIMPLE_KNN"
            echo "  [PATCHED] simple_knn.cu: added float.h"
        fi

        RASTER_IMPL="$DE3DGS/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
        if [ -f "$RASTER_IMPL" ] && ! grep -q "#include <cstdint>" "$RASTER_IMPL"; then
            sed -i '1i #include <cstdint>' "$RASTER_IMPL"
            echo "  [PATCHED] rasterizer_impl.h: added cstdint"
        fi
    fi
fi

# =============================================================================
# Step 3: Install depth-diff-gaussian-rasterization
# =============================================================================
echo ""
echo "[3/6] Installing depth-diff-gaussian-rasterization..."

DIFF_GAUSS="$DE3DGS/submodules/depth-diff-gaussian-rasterization"
if [ ! -d "$DIFF_GAUSS" ]; then
    echo "ERROR: depth-diff-gaussian-rasterization not found at $DIFF_GAUSS"
    exit 1
fi

cd "$DIFF_GAUSS"
if ! pip install -e .; then
    echo "  [FAIL] depth-diff-gaussian-rasterization compilation failed"
    exit 1
fi
echo "  Installed: depth-diff-gaussian-rasterization"

# =============================================================================
# Step 4: Install simple-knn
# =============================================================================
echo ""
echo "[4/6] Installing simple-knn..."

SIMPLE_KNN="$DE3DGS/submodules/simple-knn"
if [ ! -d "$SIMPLE_KNN" ]; then
    echo "ERROR: simple-knn not found at $SIMPLE_KNN"
    exit 1
fi

cd "$SIMPLE_KNN"
if ! pip install -e .; then
    echo "  [FAIL] simple-knn compilation failed"
    exit 1
fi
echo "  Installed: simple-knn"

# =============================================================================
# Step 5: Install De3DGS Python dependencies
# =============================================================================
echo ""
echo "[5/6] Installing De3DGS Python dependencies..."

cd "$DE3DGS"
# Install requirements (excluding the submodule paths which are handled above)
if ! pip install "numpy<2.0" plyfile==0.8.1 tqdm imageio==2.27.0 opencv-python imageio-ffmpeg scipy lpips; then
    echo "  [FAIL] Python dependencies installation failed"
    exit 1
fi
echo "  Installed: Python dependencies"

# dearpygui is optional (for GUI training) - skip if it fails
if pip install --quiet dearpygui 2>/dev/null; then
    echo "  Installed: dearpygui (optional GUI)"
else
    echo "  Skipped: dearpygui (optional GUI, install manually if needed)"
fi

# =============================================================================
# Step 6: Verification
# =============================================================================
echo ""
echo "[6/6] Verifying installation..."
echo ""

# Return to project root
cd "$PROJECT_ROOT"

# Verify each package
ERRORS=0

if python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings"; then
    echo "  [OK] depth-diff-gaussian-rasterization"
else
    echo "  [FAIL] depth-diff-gaussian-rasterization"
    ERRORS=$((ERRORS + 1))
fi

if python -c "from simple_knn._C import distCUDA2"; then
    echo "  [OK] simple-knn"
else
    echo "  [FAIL] simple-knn"
    ERRORS=$((ERRORS + 1))
fi

if python -c "import lpips"; then
    echo "  [OK] lpips"
else
    echo "  [FAIL] lpips"
    ERRORS=$((ERRORS + 1))
fi

if python -c "from plyfile import PlyData"; then
    echo "  [OK] plyfile"
else
    echo "  [FAIL] plyfile"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "All De3DGS dependencies installed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Download test dataset (D-NeRF bouncingballs):"
    echo "     mkdir -p deformable3dgs/data/dnerf"
    echo "     # Download from: https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0"
    echo "     # Extract bouncingballs/ to deformable3dgs/data/dnerf/"
    echo ""
    echo "  2. Test training:"
    echo "     cd deformable3dgs"
    echo "     python train.py -s data/dnerf/bouncingballs --eval --is_blender --iterations 5000"
    echo ""
    echo "  3. Test rendering:"
    echo "     python render.py -m output/bouncingballs --mode render --skip_train"
else
    echo "WARNING: $ERRORS package(s) failed to install correctly."
    echo "Check the output above for errors."
    exit 1
fi
