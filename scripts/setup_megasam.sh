#!/bin/bash
# =============================================================================
# Mega-SAM Setup Script
# =============================================================================
#
# This script sets up the Mega-SAM dependencies for Brain Dance's pose estimation.
#
# What it does:
# 1. Verifies prerequisites (CUDA, PyTorch)
# 2. Downloads required checkpoints (Depth-Anything, RAFT)
# 3. Compiles CUDA extensions (lietorch, DROID SLAM)
#
# Requirements:
# - CUDA 11.8+ toolkit installed
# - PyTorch 2.0+ with CUDA support
# - Git LFS for large checkpoint files
#
# Usage:
#   bash scripts/setup_megasam.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MEGASAM_DIR="$PROJECT_ROOT/instant4d/SLAM/mega-sam"
DEPTH_ANYTHING_DIR="$MEGASAM_DIR/Depth-Anything"
CVD_OPT_DIR="$MEGASAM_DIR/cvd_opt"

echo "=========================================="
echo "Mega-SAM Setup for Brain Dance"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: Verify Prerequisites
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 1: Verifying prerequisites...${NC}"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo "  CUDA version: $CUDA_VERSION"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: python not found.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | sed -n 's/Python \([0-9]*\.[0-9]*\).*/\1/p')
echo "  Python version: $PYTHON_VERSION"

# Check PyTorch CUDA
TORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
if [ "$TORCH_CUDA" != "True" ]; then
    echo -e "${RED}ERROR: PyTorch CUDA not available.${NC}"
    echo "Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi
echo "  PyTorch CUDA: Available"

# Check Mega-SAM submodule
if [ ! -d "$MEGASAM_DIR" ]; then
    echo -e "${RED}ERROR: Mega-SAM submodule not found at $MEGASAM_DIR${NC}"
    echo "Initialize with: git submodule update --init --recursive"
    exit 1
fi
echo "  Mega-SAM submodule: Found"

# Check DROID weights (should already exist in submodule)
if [ ! -f "$MEGASAM_DIR/checkpoints/megasam_final.pth" ]; then
    echo -e "${YELLOW}WARNING: megasam_final.pth not found in checkpoints/${NC}"
    echo "This checkpoint should be included in the submodule."
fi

echo -e "${GREEN}Prerequisites verified.${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Download Depth-Anything Checkpoint
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 2: Downloading Depth-Anything checkpoint...${NC}"

DEPTH_ANYTHING_CKPT="$DEPTH_ANYTHING_DIR/checkpoints/depth_anything_vitl14.pth"
mkdir -p "$(dirname "$DEPTH_ANYTHING_CKPT")"

if [ -f "$DEPTH_ANYTHING_CKPT" ]; then
    echo "  Checkpoint already exists: $DEPTH_ANYTHING_CKPT"
else
    echo "  Downloading depth_anything_vitl14.pth (~1.3GB)..."
    wget -q --show-progress -O "$DEPTH_ANYTHING_CKPT" \
        "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"

    if [ -f "$DEPTH_ANYTHING_CKPT" ]; then
        echo -e "${GREEN}  Depth-Anything checkpoint downloaded.${NC}"
    else
        echo -e "${RED}  Failed to download Depth-Anything checkpoint.${NC}"
        echo "  Manual download from: https://huggingface.co/spaces/LiheYoung/Depth-Anything"
    fi
fi
echo ""

# -----------------------------------------------------------------------------
# Step 3: Download RAFT Checkpoint
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 3: Downloading RAFT optical flow checkpoint...${NC}"

RAFT_CKPT="$CVD_OPT_DIR/raft-things.pth"
mkdir -p "$(dirname "$RAFT_CKPT")"

if [ -f "$RAFT_CKPT" ]; then
    echo "  Checkpoint already exists: $RAFT_CKPT"
else
    echo "  Downloading raft-things.pth..."

    # Try gdown first (may fail due to Google Drive rate limits)
    if command -v gdown &> /dev/null; then
        gdown --fuzzy "https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_NDA" -O "$RAFT_CKPT" 2>/dev/null || true
    fi

    # Fallback to Hugging Face mirror if gdown failed
    if [ ! -f "$RAFT_CKPT" ]; then
        echo "  gdown failed, trying Hugging Face mirror..."
        wget -q --show-progress -O "$RAFT_CKPT" \
            "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/flow/raft-things.pth"
    fi

    if [ -f "$RAFT_CKPT" ]; then
        echo -e "${GREEN}  RAFT checkpoint downloaded.${NC}"
    else
        echo -e "${YELLOW}  Could not download RAFT checkpoint automatically.${NC}"
        echo "  Manual download from: https://huggingface.co/DeepBeepMeep/Wan2.1/blob/main/flow/raft-things.pth"
    fi
fi
echo ""

# -----------------------------------------------------------------------------
# Step 4: Compile lietorch CUDA Extensions
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 4: Compiling lietorch CUDA extensions...${NC}"

LIETORCH_DIR="$MEGASAM_DIR/base/thirdparty/lietorch"

if [ -d "$LIETORCH_DIR" ]; then
    cd "$LIETORCH_DIR"

    # Check if already installed
    if python -c "import lietorch" 2>/dev/null; then
        echo "  lietorch already installed"
    else
        echo "  Building lietorch..."
        pip install -e . 2>&1 | tail -5

        # Verify installation
        if python -c "import lietorch; print('lietorch:', lietorch.__file__)" 2>/dev/null; then
            echo -e "${GREEN}  lietorch compiled successfully.${NC}"
        else
            echo -e "${RED}  lietorch compilation failed.${NC}"
            echo "  Check CUDA version compatibility."
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}  lietorch directory not found at $LIETORCH_DIR${NC}"
    echo "  Trying pip install..."
    pip install lietorch || echo "lietorch may need manual installation"
fi

cd "$PROJECT_ROOT"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Compile DROID SLAM Extensions
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 5: Compiling DROID SLAM extensions...${NC}"

DROID_BASE_DIR="$MEGASAM_DIR/base"

if [ -f "$DROID_BASE_DIR/setup.py" ]; then
    cd "$DROID_BASE_DIR"

    # Check if already installed
    if python -c "import droid_backends" 2>/dev/null; then
        echo "  DROID SLAM extensions already installed"
    else
        echo "  Building DROID SLAM extensions..."
        pip install -e . 2>&1 | tail -5

        # Verify installation
        if python -c "import droid_backends; print('droid_backends loaded')" 2>/dev/null; then
            echo -e "${GREEN}  DROID SLAM extensions compiled successfully.${NC}"
        else
            echo -e "${RED}  DROID SLAM compilation failed.${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}  DROID setup.py not found at $DROID_BASE_DIR${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"
echo ""

# -----------------------------------------------------------------------------
# Step 6: Install Python Dependencies
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 6: Installing Python dependencies...${NC}"

# UniDepth
if python -c "import unidepth" 2>/dev/null; then
    echo "  unidepth already installed"
else
    echo "  Installing unidepth..."
    pip install unidepth 2>&1 | tail -3
fi

# xformers (for UniDepth attention optimization)
if python -c "import xformers" 2>/dev/null; then
    echo "  xformers already installed"
else
    echo "  Installing xformers..."
    pip install xformers 2>&1 | tail -3
fi

echo ""

# -----------------------------------------------------------------------------
# Step 7: Verification
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 7: Verifying installation...${NC}"

VERIFICATION_SCRIPT=$(cat << 'EOF'
import sys

checks = []

# Check lietorch
try:
    import lietorch
    checks.append(("lietorch", True, lietorch.__file__))
except ImportError as e:
    checks.append(("lietorch", False, str(e)))

# Check DROID backends
try:
    import droid_backends
    checks.append(("droid_backends", True, "loaded"))
except ImportError as e:
    checks.append(("droid_backends", False, str(e)))

# Check unidepth
try:
    import unidepth
    checks.append(("unidepth", True, "loaded"))
except ImportError as e:
    checks.append(("unidepth", False, str(e)))

# Print results
all_pass = True
for name, success, info in checks:
    status = "OK" if success else "FAIL"
    print(f"  {name}: {status} ({info[:50]}...)" if len(info) > 50 else f"  {name}: {status} ({info})")
    if not success:
        all_pass = False

sys.exit(0 if all_pass else 1)
EOF
)

if python -c "$VERIFICATION_SCRIPT"; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Mega-SAM setup completed successfully!"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: pytest tests/stages/test_video_processing_megasam.py -v"
    echo "  2. Try Mega-SAM: python -c \\"
    echo "       from backend.stages.video_processing import VideoProcessingStage"
    echo "       stage = VideoProcessingStage({'pose_estimator': 'megasam'})"
    echo "       print('Mega-SAM available:', stage._get_megasam_estimator().is_available())\""
else
    echo ""
    echo -e "${RED}=========================================="
    echo "Mega-SAM setup completed with warnings."
    echo "Some components may not be fully functional."
    echo "==========================================${NC}"
    exit 1
fi
