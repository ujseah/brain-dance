#!/bin/bash
# Brain Dance - CUDA 12.x Compatibility Patches
#
# Patches De3DGS submodules for CUDA 12.x compatibility.
# CUDA 12.x no longer exports FLT_MIN/FLT_MAX from NVIDIA headers.
#
# Reference: https://github.com/graphdeco-inria/gaussian-splatting/issues/1215
#
# Usage:
#   chmod +x scripts/patch_cuda12.sh
#   ./scripts/patch_cuda12.sh
#
# This script is idempotent - safe to run multiple times.

set -e

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DE3DGS="$PROJECT_ROOT/deformable3dgs"

echo "Patching De3DGS submodules for CUDA 12.x compatibility..."

# Check if submodule exists
if [ ! -d "$DE3DGS/submodules" ]; then
    echo "[SKIP] No submodules directory found at $DE3DGS/submodules"
    exit 0
fi

PATCHES_APPLIED=0

# =============================================================================
# Patch 1: simple_knn.cu - add float.h include
# =============================================================================
SIMPLE_KNN="$DE3DGS/submodules/simple-knn/simple_knn.cu"
if [ -f "$SIMPLE_KNN" ]; then
    if ! grep -q "#include <float.h>" "$SIMPLE_KNN"; then
        # Use sed to add include at the beginning, after any existing comments
        # Find the first non-comment, non-empty line and insert before it
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS sed requires empty string for -i
            sed -i '' '1s/^/#include <float.h>\n/' "$SIMPLE_KNN"
        else
            # Linux sed
            sed -i '1s/^/#include <float.h>\n/' "$SIMPLE_KNN"
        fi
        echo "  [PATCHED] simple_knn.cu: added #include <float.h>"
        PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
    else
        echo "  [OK] simple_knn.cu: already has float.h"
    fi
else
    echo "  [SKIP] simple_knn.cu not found"
fi

# =============================================================================
# Patch 2: simple_knn.h - add cfloat include (for FLT_MAX in header)
# =============================================================================
SIMPLE_KNN_H="$DE3DGS/submodules/simple-knn/simple_knn.h"
if [ -f "$SIMPLE_KNN_H" ]; then
    if ! grep -q "#include <cfloat>" "$SIMPLE_KNN_H"; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' '1s/^/#include <cfloat>\n/' "$SIMPLE_KNN_H"
        else
            sed -i '1s/^/#include <cfloat>\n/' "$SIMPLE_KNN_H"
        fi
        echo "  [PATCHED] simple_knn.h: added #include <cfloat>"
        PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
    else
        echo "  [OK] simple_knn.h: already has cfloat"
    fi
fi

# =============================================================================
# Patch 3: rasterizer_impl.h - add cstdint include
# =============================================================================
RASTER_IMPL="$DE3DGS/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
if [ -f "$RASTER_IMPL" ]; then
    if ! grep -q "#include <cstdint>" "$RASTER_IMPL"; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' '1s/^/#include <cstdint>\n/' "$RASTER_IMPL"
        else
            sed -i '1s/^/#include <cstdint>\n/' "$RASTER_IMPL"
        fi
        echo "  [PATCHED] rasterizer_impl.h: added #include <cstdint>"
        PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
    else
        echo "  [OK] rasterizer_impl.h: already has cstdint"
    fi
else
    echo "  [SKIP] rasterizer_impl.h not found"
fi

# =============================================================================
# Patch 4: rasterizer_impl.cu - add cstdint include (if exists)
# =============================================================================
RASTER_IMPL_CU="$DE3DGS/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu"
if [ -f "$RASTER_IMPL_CU" ]; then
    if ! grep -q "#include <cstdint>" "$RASTER_IMPL_CU"; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' '1s/^/#include <cstdint>\n/' "$RASTER_IMPL_CU"
        else
            sed -i '1s/^/#include <cstdint>\n/' "$RASTER_IMPL_CU"
        fi
        echo "  [PATCHED] rasterizer_impl.cu: added #include <cstdint>"
        PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
    else
        echo "  [OK] rasterizer_impl.cu: already has cstdint"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
if [ $PATCHES_APPLIED -gt 0 ]; then
    echo "[OK] Applied $PATCHES_APPLIED patch(es) for CUDA 12.x compatibility"
else
    echo "[OK] All files already patched for CUDA 12.x"
fi
