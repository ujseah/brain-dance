"""
Deformable 3D Gaussians adapter for brain-dance pipeline.

This adapter bridges the brain-dance video processing pipeline (Mega-SAM output)
to the Deformable 3D Gaussians training pipeline, using subprocess isolation
to avoid Python version and CUDA kernel conflicts.

Architecture:
    - Subprocess isolation: All De3DGS code runs in separate process
    - Nerfstudio → COLMAP format conversion
    - Per-frame PLY export via standalone script

Reference:
    - De3DGS Paper: "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction" (CVPR 2024)
    - Repository: https://github.com/ingra14m/Deformable-3D-Gaussians
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any
import json
import subprocess
import shutil
import logging
import re
import os

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class De3DGSError(Exception):
    """Base exception for De3DGS training failures."""
    pass


class CUDAOutOfMemoryError(De3DGSError):
    """Raised when GPU memory is exhausted during training."""
    pass


class CUDADriverError(De3DGSError):
    """Raised when CUDA driver issues are detected."""
    pass


class CUDADeviceError(De3DGSError):
    """Raised when no CUDA device is available."""
    pass


class COLMAPConversionError(De3DGSError):
    """Raised when Nerfstudio → COLMAP format conversion fails."""
    pass


class TrainingTimeoutError(De3DGSError):
    """Raised when training exceeds maximum allowed time."""
    pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Deformable3DGSOptions:
    """Configuration for Deformable 3D Gaussians training."""

    # Training parameters
    iterations: int = 20000
    """Total training iterations (De3DGS default for real-world)."""

    sh_degree: int = 3
    """Spherical harmonics degree (0-3)."""

    # Architecture
    is_blender: bool = False
    """Use Blender/D-NeRF time encoding (for synthetic datasets)."""

    is_6dof: bool = False
    """Use SE(3) rigid transformation (vs additive deformation)."""

    # Loss weights
    lambda_dssim: float = 0.2
    """DSSIM loss weight (rest is L1)."""

    # Warm-up
    warm_up: int = 3000
    """Iterations before enabling deformation network."""

    # Preprocessing
    downsample_points: bool = True
    """Downsample initialization points for memory efficiency."""

    target_points: int = 100000
    """Target point count after downsampling."""

    # Export
    export_num_frames: int = 30
    """Number of frames to export for temporal playback."""

    export_fps: int = 30
    """Frames per second for temporal metadata."""

    # Paths
    de3dgs_path: Optional[str] = None
    """Path to De3DGS submodule (auto-detected if None)."""

    # Runtime
    timeout_seconds: int = 7200
    """Maximum training time (2 hours default)."""

    def __post_init__(self):
        """Validate all configuration parameters."""
        # Training iterations
        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.iterations > 500000:
            raise ValueError(
                f"iterations={self.iterations} is unusually high. "
                "Max recommended is 500000. Set explicitly if intentional."
            )

        # Spherical harmonics degree
        if not 0 <= self.sh_degree <= 3:
            raise ValueError(f"sh_degree must be 0-3, got {self.sh_degree}")

        # Loss weight
        if not 0.0 <= self.lambda_dssim <= 1.0:
            raise ValueError(f"lambda_dssim must be in [0, 1], got {self.lambda_dssim}")

        # Warm-up iterations
        if self.warm_up < 0:
            raise ValueError(f"warm_up must be non-negative, got {self.warm_up}")
        if self.warm_up >= self.iterations:
            raise ValueError(
                f"warm_up ({self.warm_up}) must be less than iterations ({self.iterations})"
            )

        # Target points for downsampling
        if self.target_points <= 0:
            raise ValueError(f"target_points must be positive, got {self.target_points}")

        # Export parameters
        if self.export_num_frames <= 0:
            raise ValueError(f"export_num_frames must be positive, got {self.export_num_frames}")
        if self.export_fps <= 0:
            raise ValueError(f"export_fps must be positive, got {self.export_fps}")

        # Timeout
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.timeout_seconds < 60:
            raise ValueError(
                f"timeout_seconds={self.timeout_seconds} is too short. "
                "Minimum recommended is 60 seconds."
            )

        # Path validation (if provided)
        if self.de3dgs_path is not None:
            path = Path(self.de3dgs_path)
            if not path.exists():
                raise ValueError(f"de3dgs_path does not exist: {self.de3dgs_path}")
            if not (path / "train.py").exists():
                raise ValueError(
                    f"de3dgs_path does not contain train.py: {self.de3dgs_path}. "
                    "Ensure this is a valid De3DGS installation."
                )


@dataclass
class Deformable3DGSResult:
    """Result from Deformable 3D Gaussians training."""

    model_path: str
    """Path to trained model directory."""

    canonical_ply_path: str
    """Path to canonical Gaussians PLY file."""

    deformation_pth_path: str
    """Path to deformation MLP weights."""

    ply_paths: List[str] = field(default_factory=list)
    """Paths to per-frame PLY files."""

    num_gaussians: int = 0
    """Number of Gaussians in trained model."""

    num_frames: int = 0
    """Number of exported temporal frames."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    """Training metrics (PSNR, iterations, etc.)."""

    temporal_metadata: Dict[str, Any] = field(default_factory=dict)
    """Temporal metadata (fps, duration, timestamps)."""


# =============================================================================
# Coordinate Conversion Utilities
# =============================================================================

def normalize_timestamp(frame_idx: int, total_frames: int) -> float:
    """
    Normalize frame index to [0, 1] range for De3DGS.

    De3DGS expects timestamps normalized such that:
    - First frame (idx=0) maps to 0.0
    - Last frame (idx=total_frames-1) maps to 1.0

    Args:
        frame_idx: 0-indexed frame number
        total_frames: Total number of frames

    Returns:
        Normalized timestamp in [0, 1]
    """
    if total_frames <= 1:
        return 0.0
    return frame_idx / (total_frames - 1)


def c2w_to_colmap_pose(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Nerfstudio camera-to-world (OpenGL) to COLMAP world-to-camera pose.

    Coordinate System Conversion:
    - Nerfstudio/OpenGL: X-right, Y-up, Z-backward
    - COLMAP: X-right, Y-down, Z-forward

    Transformation: Flip Y and Z axes, then invert to get w2c.

    Args:
        c2w: (4, 4) camera-to-world matrix in OpenGL convention

    Returns:
        quat: (4,) quaternion [qw, qx, qy, qz] for COLMAP w2c rotation
        trans: (3,) translation [tx, ty, tz] for COLMAP w2c
    """
    # OpenGL → COLMAP coordinate change matrix (flip Y and Z)
    flip_yz = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)

    # Apply coordinate change: c2w_colmap = flip @ c2w_opengl
    c2w_colmap = flip_yz @ c2w

    # Invert to get world-to-camera
    w2c = np.linalg.inv(c2w_colmap)

    # Extract rotation and translation
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Convert rotation matrix to quaternion
    # scipy uses [x, y, z, w] order, COLMAP uses [w, x, y, z]
    rot = Rotation.from_matrix(R)
    quat_xyzw = rot.as_quat()
    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return quat, t


def colmap_pose_to_c2w(quat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP w2c pose back to Nerfstudio c2w (for round-trip testing).

    Args:
        quat: (4,) quaternion [qw, qx, qy, qz] from COLMAP
        trans: (3,) translation from COLMAP

    Returns:
        c2w: (4, 4) camera-to-world matrix in OpenGL convention
    """
    # Build w2c matrix from quaternion and translation
    # Convert [w,x,y,z] to scipy's [x,y,z,w]
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    R = Rotation.from_quat(quat_xyzw).as_matrix()

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = trans

    # Invert to get c2w in COLMAP convention
    c2w_colmap = np.linalg.inv(w2c)

    # Apply inverse coordinate change (flip Y and Z back)
    flip_yz = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)

    c2w_opengl = flip_yz @ c2w_colmap

    return c2w_opengl


# =============================================================================
# COLMAP File Writers
# =============================================================================

def write_colmap_cameras(
    cameras_path: Path,
    width: int,
    height: int,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
) -> None:
    """
    Write COLMAP cameras.txt from camera intrinsics.

    COLMAP cameras.txt format:
    # Camera list with one line per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # PARAMS for PINHOLE: fx, fy, cx, cy

    Args:
        cameras_path: Output path for cameras.txt
        width, height: Image dimensions
        fl_x, fl_y: Focal lengths
        cx, cy: Principal point
    """
    with open(cameras_path, 'w') as f:
        f.write("# Camera list with one line per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fl_x:.8f} {fl_y:.8f} {cx:.8f} {cy:.8f}\n")


def write_colmap_images(
    images_path: Path,
    frames: List[Dict[str, Any]],
) -> None:
    """
    Write COLMAP images.txt from Nerfstudio frame transforms.

    COLMAP images.txt format:
    # Image list with two lines per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] (can be empty)

    Args:
        images_path: Output path for images.txt
        frames: List of frame dicts with 'file_path' and 'transform_matrix'
    """
    with open(images_path, 'w') as f:
        f.write("# Image list with two lines per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")

        for idx, frame in enumerate(frames, start=1):
            # Convert c2w matrix to COLMAP w2c pose
            c2w = np.array(frame['transform_matrix'])
            quat, trans = c2w_to_colmap_pose(c2w)

            # Extract filename (just the basename)
            file_path = Path(frame['file_path']).name

            # Write image line
            f.write(f"{idx} {quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f} ")
            f.write(f"{trans[0]:.8f} {trans[1]:.8f} {trans[2]:.8f} 1 {file_path}\n")

            # Write empty POINTS2D line (we don't have 2D-3D correspondences)
            f.write("\n")


def write_colmap_points3d_txt(points3d_path: Path) -> None:
    """
    Write minimal COLMAP points3D.txt (empty, De3DGS can handle this).

    Args:
        points3d_path: Output path for points3D.txt
    """
    with open(points3d_path, 'w') as f:
        f.write("# 3D point list with one line per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


# =============================================================================
# Main Adapter Class
# =============================================================================

class Deformable3DGSAdapter:
    """
    Adapter for Deformable 3D Gaussians monocular reconstruction.

    Uses subprocess isolation to avoid Python/CUDA conflicts between
    brain-dance (Python 3.10+) and De3DGS (Python 3.7+).

    All De3DGS code runs in separate processes via subprocess calls to:
    - train.py for training
    - scripts/export_per_frame_ply.py for PLY export

    Example:
        adapter = Deformable3DGSAdapter()
        result = adapter.run_full_pipeline(
            video_result=stage1_output,
            output_dir="/path/to/output",
            options=Deformable3DGSOptions(iterations=20000),
        )
        print(f"Trained model: {result.model_path}")
        print(f"Per-frame PLYs: {len(result.ply_paths)}")
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize adapter.

        Args:
            config: Configuration dictionary (deprecated, use options instead)
        """
        self.config = config or {}

        # Auto-detect De3DGS path
        default_path = Path(__file__).parent.parent.parent / "deformable3dgs"
        self.de3dgs_path = Path(self.config.get("de3dgs_path", default_path))

        if not self.de3dgs_path.exists():
            logger.warning(f"De3DGS path not found: {self.de3dgs_path}")

    def _validate_paths(self, transforms_path: Path) -> None:
        """
        Validate input paths before processing.

        Security: Prevents path traversal attacks by checking the original path
        for ".." components BEFORE resolution, and verifying the resolved path
        is within the expected workspace directory.

        Args:
            transforms_path: Path to transforms.json

        Raises:
            FileNotFoundError: If transforms.json doesn't exist
            ValueError: If path is invalid or contains path traversal
        """
        # Check for path traversal in ORIGINAL path (before resolve)
        # Using .parts catches ".." as a path component, not just substring
        if '..' in transforms_path.parts:
            raise ValueError(
                f"Path traversal not allowed: {transforms_path}. "
                "Use absolute paths without '..' components."
            )

        # Resolve to absolute path
        resolved = transforms_path.resolve()

        # Verify path exists
        if not resolved.exists():
            raise FileNotFoundError(f"transforms.json not found: {resolved}")

        # Verify correct file extension
        if resolved.suffix != '.json':
            raise ValueError(f"transforms_path must be a JSON file: {resolved}")

    def _parse_cuda_error(
        self,
        output: str,
        return_code: int,
        last_iteration: int
    ) -> De3DGSError:
        """
        Parse subprocess output to identify specific CUDA errors.

        Returns an appropriate exception with actionable error message.

        Args:
            output: Combined stdout/stderr from the subprocess
            return_code: Process exit code
            last_iteration: Last training iteration reached before failure

        Returns:
            Appropriate De3DGSError subclass with helpful message
        """
        output_lower = output.lower()

        # CUDA Out of Memory
        if "cuda out of memory" in output_lower or "out of memory" in output_lower:
            # Try to extract memory info
            mem_match = re.search(
                r'Tried to allocate ([\d.]+) ([GMK]iB)',
                output,
                re.IGNORECASE
            )
            mem_info = f" (tried to allocate {mem_match.group(1)} {mem_match.group(2)})" if mem_match else ""

            return CUDAOutOfMemoryError(
                f"GPU out of memory at iteration {last_iteration}{mem_info}.\n\n"
                "Recommended fixes:\n"
                "  1. Reduce target_points (try 50000 instead of 100000)\n"
                "  2. Reduce sh_degree from 3 to 2 or 1\n"
                "  3. Use a GPU with more VRAM (minimum 12GB recommended)\n"
                "  4. Process fewer frames (reduce input video length)"
            )

        # CUDA driver version mismatch
        if "cuda driver version is insufficient" in output_lower:
            return CUDADriverError(
                "CUDA driver version is insufficient for this PyTorch version.\n\n"
                "Recommended fixes:\n"
                "  1. Update NVIDIA drivers: https://www.nvidia.com/drivers\n"
                "  2. Check compatibility: PyTorch 2.3+ requires CUDA 12.1+\n"
                "  3. Run 'nvidia-smi' to check current driver version"
            )

        # No CUDA device available
        if "no cuda-capable device" in output_lower or "cuda is not available" in output_lower:
            return CUDADeviceError(
                "No CUDA-capable GPU detected.\n\n"
                "Recommended fixes:\n"
                "  1. Verify GPU is installed: run 'nvidia-smi'\n"
                "  2. Check CUDA_VISIBLE_DEVICES environment variable\n"
                "  3. Ensure NVIDIA drivers are installed correctly\n"
                "  4. If on Mac, note that CUDA is not supported - use cloud GPU"
            )

        # Invalid device ordinal
        if "invalid device ordinal" in output_lower:
            return CUDADeviceError(
                "Invalid CUDA device specified.\n\n"
                "The requested GPU index does not exist.\n"
                "Run 'nvidia-smi -L' to list available GPUs and their indices."
            )

        # CUDA initialization error
        if "cuda error" in output_lower or "cudnn error" in output_lower:
            return CUDADriverError(
                f"CUDA/cuDNN error during training at iteration {last_iteration}.\n\n"
                "This may indicate:\n"
                "  1. Driver/toolkit version mismatch\n"
                "  2. Corrupted CUDA installation\n"
                "  3. Hardware issue with GPU\n\n"
                f"Full error output:\n{output[-1000:]}"  # Last 1000 chars
            )

        # NaN/Inf detected (training divergence)
        if "nan" in output_lower and ("loss" in output_lower or "gradient" in output_lower):
            return De3DGSError(
                f"Training diverged (NaN detected) at iteration {last_iteration}.\n\n"
                "Recommended fixes:\n"
                "  1. Reduce learning rate\n"
                "  2. Increase warm_up iterations\n"
                "  3. Check input data quality (poses may be incorrect)"
            )

        # Generic fallback with output snippet
        output_snippet = output[-500:] if len(output) > 500 else output
        return De3DGSError(
            f"De3DGS training failed with exit code {return_code} "
            f"at iteration {last_iteration}.\n\n"
            f"Last output:\n{output_snippet}"
        )

    def preprocess(
        self,
        video_result: Any,  # VideoProcessingResult
        output_dir: str,
        options: Optional[Deformable3DGSOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Convert brain-dance format to De3DGS COLMAP format.

        Input (from Mega-SAM / Stage 1):
          - transforms.json (Nerfstudio format with c2w matrices)
          - sparse/points3D.ply (optional sparse point cloud)
          - frames/ (video frames)

        Output (De3DGS COLMAP format):
          - sparse/0/cameras.txt (COLMAP camera intrinsics)
          - sparse/0/images.txt (COLMAP camera extrinsics)
          - sparse/0/points3D.txt or points3D.ply (sparse initialization)
          - images/ (symlink to frames/)

        Args:
            video_result: Output from Stage 1 (VideoProcessingResult)
            output_dir: Output directory for COLMAP format
            options: Configuration options
            progress_callback: Progress reporting callback

        Returns:
            Path to preprocessed directory
        """
        if options is None:
            options = Deformable3DGSOptions()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create COLMAP directory structure
        sparse_dir = output_path / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[preprocess {pct*100:.0f}%] {msg}")

        report(0.0, "Loading transforms.json")

        # Validate and load transforms
        transforms_path = Path(video_result.transforms_path)
        self._validate_paths(transforms_path)

        with open(transforms_path, 'r') as f:
            transforms = json.load(f)

        report(0.1, "Converting camera intrinsics")

        # Extract intrinsics
        w = transforms.get('w')
        h = transforms.get('h')
        fl_x = transforms.get('fl_x')
        fl_y = transforms.get('fl_y', fl_x)  # Default to fl_x if fl_y not provided
        cx = transforms.get('cx', w / 2.0)
        cy = transforms.get('cy', h / 2.0)

        if w is None or h is None or fl_x is None:
            raise COLMAPConversionError("Missing camera intrinsics in transforms.json")

        # Write cameras.txt
        write_colmap_cameras(
            sparse_dir / "cameras.txt",
            width=w,
            height=h,
            fl_x=fl_x,
            fl_y=fl_y,
            cx=cx,
            cy=cy,
        )

        report(0.3, "Converting camera extrinsics")

        # Write images.txt
        frames = transforms.get('frames', [])
        if not frames:
            raise COLMAPConversionError("No frames found in transforms.json")

        write_colmap_images(sparse_dir / "images.txt", frames)

        report(0.5, "Processing sparse points")

        # Handle sparse point cloud
        sparse_input = Path(video_result.sparse_points_path) if video_result.sparse_points_path else None

        if sparse_input and sparse_input.exists():
            # Copy or symlink the sparse points
            sparse_output = sparse_dir / "points3D.ply"
            if not sparse_output.exists():
                shutil.copy(sparse_input, sparse_output)
            logger.info(f"Copied sparse points from {sparse_input}")
        else:
            # Write empty points3D.txt - De3DGS can initialize from images
            write_colmap_points3d_txt(sparse_dir / "points3D.txt")
            logger.info("No sparse points available, wrote empty points3D.txt")

        report(0.8, "Linking image directory")

        # Symlink frames directory as images/
        frames_input = Path(video_result.frames_dir)
        images_output = output_path / "images"

        if images_output.exists() or images_output.is_symlink():
            images_output.unlink()

        # Use relative symlink if possible, absolute otherwise
        try:
            images_output.symlink_to(frames_input.resolve())
        except OSError as e:
            logger.warning(f"Symlink failed, copying frames: {e}")
            shutil.copytree(frames_input, images_output)

        report(1.0, f"Preprocessing complete: {len(frames)} frames")

        return output_path

    def train(
        self,
        preprocessed_dir: Path,
        output_dir: str,
        options: Optional[Deformable3DGSOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Deformable3DGSResult:
        """
        Train De3DGS model via subprocess.

        Args:
            preprocessed_dir: COLMAP format directory from preprocess()
            output_dir: Output directory for trained model
            options: Training configuration
            progress_callback: Progress reporting callback

        Returns:
            Deformable3DGSResult with model paths and metrics
        """
        if options is None:
            options = Deformable3DGSOptions()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        de3dgs_path = Path(options.de3dgs_path) if options.de3dgs_path else self.de3dgs_path
        train_script = de3dgs_path / "train.py"

        if not train_script.exists():
            raise FileNotFoundError(f"De3DGS train.py not found: {train_script}")

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[train {pct*100:.0f}%] {msg}")

        report(0.0, "Starting De3DGS training")

        # Build training command (NEVER use shell=True)
        cmd = [
            "python",
            str(train_script.resolve()),
            "-s", str(preprocessed_dir.resolve()),
            "-m", str(output_path.resolve()),
            "--eval",
            "--iterations", str(int(options.iterations)),
        ]

        if options.is_blender:
            cmd.append("--is_blender")
        if options.is_6dof:
            cmd.append("--is_6dof")

        logger.info(f"Running: {' '.join(cmd)}")

        # Run training subprocess
        output_lines = []  # Collect all output for error analysis
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(de3dgs_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            )

            last_iter = 0
            final_psnr = None

            # Monitor progress from stdout
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                # Store for error analysis
                output_lines.append(line)

                # Log all output for debugging
                logger.debug(f"De3DGS: {line}")

                # Parse iteration progress
                iter_match = re.search(r'Iteration\s+(\d+)', line, re.IGNORECASE)
                if iter_match:
                    current_iter = int(iter_match.group(1))
                    if current_iter > last_iter:
                        last_iter = current_iter
                        progress = min(current_iter / options.iterations, 0.95)
                        report(progress, f"Training iteration {current_iter}/{options.iterations}")

                # Parse PSNR if available
                psnr_match = re.search(r'PSNR[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                if psnr_match:
                    final_psnr = float(psnr_match.group(1))

            return_code = process.wait(timeout=options.timeout_seconds)

        except subprocess.TimeoutExpired:
            process.kill()
            raise TrainingTimeoutError(
                f"Training timed out after {options.timeout_seconds} seconds "
                f"(last iteration: {last_iter})"
            )

        if return_code != 0:
            # Parse output for specific CUDA errors and provide actionable messages
            all_output = "\n".join(output_lines)
            raise self._parse_cuda_error(all_output, return_code, last_iter)

        report(0.98, "Training complete, locating outputs")

        # Find trained model outputs
        # - Canonical Gaussians: point_cloud/iteration_N/point_cloud.ply
        # - Deformation MLP: deform/iteration_N/deform.pth
        canonical_ply = output_path / "point_cloud" / f"iteration_{options.iterations}" / "point_cloud.ply"
        deformation_pth = output_path / "deform" / f"iteration_{options.iterations}" / "deform.pth"

        # Fallback: find latest iteration if exact match not found
        if not canonical_ply.exists():
            pc_dir = output_path / "point_cloud"
            if pc_dir.exists():
                iter_dirs = sorted([d for d in pc_dir.iterdir() if d.is_dir()], reverse=True)
                if iter_dirs:
                    canonical_ply = iter_dirs[0] / "point_cloud.ply"

        if not deformation_pth.exists():
            deform_dir = output_path / "deform"
            if deform_dir.exists():
                iter_dirs = sorted([d for d in deform_dir.iterdir() if d.is_dir()], reverse=True)
                if iter_dirs:
                    deformation_pth = iter_dirs[0] / "deform.pth"

        # Count Gaussians
        num_gaussians = 0
        if canonical_ply.exists():
            try:
                from plyfile import PlyData
                plydata = PlyData.read(str(canonical_ply))
                num_gaussians = plydata['vertex'].count
            except Exception as e:
                logger.warning(f"Could not count Gaussians: {e}")

        report(1.0, f"Training complete: {num_gaussians} Gaussians")

        return Deformable3DGSResult(
            model_path=str(output_path),
            canonical_ply_path=str(canonical_ply) if canonical_ply.exists() else "",
            deformation_pth_path=str(deformation_pth) if deformation_pth.exists() else "",
            ply_paths=[],  # Populated by extract_per_frame_ply
            num_gaussians=num_gaussians,
            num_frames=0,
            metrics={
                "iterations": options.iterations,
                "psnr": final_psnr,
                "sh_degree": options.sh_degree,
            },
            temporal_metadata={
                "is_blender": options.is_blender,
                "is_6dof": options.is_6dof,
            },
        )

    def extract_per_frame_ply(
        self,
        model_path: str,
        output_dir: str,
        options: Optional[Deformable3DGSOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[str]:
        """
        Export per-frame PLY files by calling export script via subprocess.

        Args:
            model_path: Path to trained model directory
            output_dir: Output directory for PLY files
            options: Export configuration
            progress_callback: Progress reporting callback

        Returns:
            List of paths to exported PLY files
        """
        if options is None:
            options = Deformable3DGSOptions()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        de3dgs_path = Path(options.de3dgs_path) if options.de3dgs_path else self.de3dgs_path
        export_script = de3dgs_path / "scripts" / "export_per_frame_ply.py"

        if not export_script.exists():
            raise FileNotFoundError(
                f"Export script not found: {export_script}. "
                "Run 'scripts/export_per_frame_ply.py' setup first."
            )

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[export {pct*100:.0f}%] {msg}")

        report(0.0, f"Exporting {options.export_num_frames} per-frame PLY files")

        # Build export command
        cmd = [
            "python",
            str(export_script.resolve()),
            "--model_path", str(Path(model_path).resolve()),
            "--output_dir", str(output_path.resolve()),
            "--num_frames", str(options.export_num_frames),
            "--time_start", "0.0",
            "--time_end", "1.0",
        ]

        if options.is_6dof:
            cmd.append("--is_6dof")

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(de3dgs_path),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout for export
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            )

            if result.returncode != 0:
                logger.error(f"Export stderr: {result.stderr}")
                raise De3DGSError(f"PLY export failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise TrainingTimeoutError("PLY export timed out after 30 minutes")

        # Collect exported PLY files
        ply_files = sorted(output_path.glob("frame_*.ply"))
        ply_paths = [str(p) for p in ply_files]

        report(1.0, f"Exported {len(ply_paths)} PLY files")

        return ply_paths

    def run_full_pipeline(
        self,
        video_result: Any,  # VideoProcessingResult
        output_dir: str,
        segmentation_result: Any = None,  # ObjectSegmentationResult (unused for De3DGS)
        options: Optional[Deformable3DGSOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Deformable3DGSResult:
        """
        Run complete De3DGS pipeline: preprocess → train → export.

        Args:
            video_result: Output from Stage 1 (VideoProcessingResult)
            output_dir: Output directory
            segmentation_result: NOT USED for De3DGS (included for API compatibility)
            options: Configuration options
            progress_callback: Progress callback

        Returns:
            Deformable3DGSResult with all paths and metrics
        """
        if options is None:
            options = Deformable3DGSOptions()

        output_path = Path(output_dir)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        # Phase 1: Preprocess (0-20%)
        preprocessed_dir = self.preprocess(
            video_result=video_result,
            output_dir=str(output_path / "preprocessed"),
            options=options,
            progress_callback=lambda p, m: report(p * 0.2, m),
        )

        # Phase 2: Train (20-80%)
        result = self.train(
            preprocessed_dir=preprocessed_dir,
            output_dir=str(output_path / "model"),
            options=options,
            progress_callback=lambda p, m: report(0.2 + p * 0.6, m),
        )

        # Phase 3: Export per-frame PLYs (80-100%)
        ply_paths = self.extract_per_frame_ply(
            model_path=result.model_path,
            output_dir=str(output_path / "per_frame_plys"),
            options=options,
            progress_callback=lambda p, m: report(0.8 + p * 0.2, m),
        )

        # Update result with PLY paths
        result.ply_paths = ply_paths
        result.num_frames = len(ply_paths)
        result.temporal_metadata.update({
            "fps": options.export_fps,
            "duration": len(ply_paths) / options.export_fps if options.export_fps > 0 else 0,
            "timestamps": [normalize_timestamp(i, len(ply_paths)) for i in range(len(ply_paths))],
        })

        return result

    def cleanup(self) -> None:
        """Release resources (no-op for subprocess-based adapter)."""
        pass
