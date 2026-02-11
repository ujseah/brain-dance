#!/bin/bash
# Brain Dance - CUDA 12.x Compatibility Patches
#
# Patches De3DGS submodules for CUDA 12.x compatibility.
# CUDA 12.x no longer exports FLT_MIN/FLT_MAX/uint32_t from NVIDIA headers.
#
# Evidence:
#   - https://github.com/graphdeco-inria/gaussian-splatting/issues/1215
#   - https://github.com/graphdeco-inria/gaussian-splatting/issues/1296
#   - https://github.com/graphdeco-inria/gaussian-splatting/issues/923
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

PATCHES=0

# Helper function for portable sed insert
# Uses sed 'i' command which is POSIX-compliant and works on both Linux and macOS
patch_file() {
    local file="$1"
    local include="$2"
    local name="$3"

    if [ -f "$file" ]; then
        if ! grep -q "$include" "$file"; then
            # Use sed 'i' command - POSIX compliant, works on Linux and macOS
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS sed requires different syntax for 'i' command
                sed -i '' "1i\\
$include
" "$file"
            else
                # GNU sed (Linux)
                sed -i "1i $include" "$file"
            fi
            echo "  [PATCHED] $name"
            PATCHES=$((PATCHES + 1))
        else
            echo "  [OK] $name: already patched"
        fi
    else
        echo "  [SKIP] $name: file not found"
    fi
}

# =============================================================================
# Patch all required files for CUDA 12.x compatibility
# =============================================================================

# Patch 1: simple_knn.cu - needs float.h for FLT_MAX (used in lines 90-91, 154, 164-166)
patch_file \
    "$DE3DGS/submodules/simple-knn/simple_knn.cu" \
    "#include <float.h>" \
    "simple_knn.cu"

# Patch 2: rasterizer_impl.h - needs cstdint for size_t, uint32_t
patch_file \
    "$DE3DGS/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h" \
    "#include <cstdint>" \
    "rasterizer_impl.h"

# Patch 3: rasterizer_impl.cu - needs cstdint for uint32_t, uint64_t
patch_file \
    "$DE3DGS/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu" \
    "#include <cstdint>" \
    "rasterizer_impl.cu"

# Patch 4: forward.h - needs cstdint for uint32_t in function signatures
patch_file \
    "$DE3DGS/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/forward.h" \
    "#include <cstdint>" \
    "forward.h"

# Patch 5: backward.h - needs cstdint for uint32_t in function signatures
patch_file \
    "$DE3DGS/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/backward.h" \
    "#include <cstdint>" \
    "backward.h"

# =============================================================================
# Summary
# =============================================================================
echo ""
if [ $PATCHES -gt 0 ]; then
    echo "[OK] Applied $PATCHES patch(es) for CUDA 12.x compatibility"
else
    echo "[OK] All files already patched for CUDA 12.x"
fi
