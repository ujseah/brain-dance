"""Stage 3: 4D Gaussian Splatting Training with Deformable 3D Gaussians.

This stage trains a Deformable 3D Gaussian Splatting model from processed video data,
enabling temporal reconstruction where users can navigate in 3D while time progresses.

Architecture Notes:
    - Uses Deformable3DGSAdapter internally via subprocess isolation
    - No segmentation required (De3DGS handles dynamics implicitly via deformation MLP)
    - Outputs per-frame PLY files for Stage 5 (Web Export)
    - Legacy Instant4D adapter available via GAUSSIAN_ADAPTER=instant4d env var

Reference:
    - De3DGS Paper: "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction" (CVPR 2024)
    - Repository: https://github.com/ingra14m/Deformable-3D-Gaussians
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Any
import logging
import os

from .video_processing import VideoProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class GaussianTrainingResult:
    """Result of 4D Gaussian training stage."""

    ply_path: str
    """Path to first PLY file (for backward compatibility)."""

    num_gaussians: int
    """Number of Gaussians in the trained model."""

    metrics: dict = field(default_factory=dict)
    """Quality metrics (PSNR, SSIM, etc.)."""

    config_path: Optional[str] = None
    """Path to training configuration file."""

    # New fields for 4D support
    ply_paths: List[str] = field(default_factory=list)
    """Paths to all per-frame PLY files (for 4D reconstruction)."""

    model_path: Optional[str] = None
    """Path to trained 4D model checkpoint (.pth)."""

    num_frames: int = 0
    """Number of temporal frames extracted."""

    temporal_metadata: dict = field(default_factory=dict)
    """Temporal metadata (fps, duration, timestamps)."""


class GaussianTrainingStage:
    """
    Stage 3: Train Deformable 3D Gaussian Splatting model from processed video data.

    This stage uses Deformable 3D Gaussians (De3DGS) to create a true 4D reconstruction
    that supports temporal playback - users can navigate in 3D space while time progresses.

    De3DGS Architecture:
    - Canonical 3D Gaussians in a reference frame
    - Per-Gaussian deformation MLP: f(position, time) -> (delta_pos, delta_rot, delta_scale)
    - Implicitly learns motion without explicit segmentation

    Pipeline:
    1. Preprocess Stage 1 outputs to COLMAP format
    2. Train canonical Gaussians + deformation MLP (~10-30 minutes)
    3. Export per-frame PLY files for web viewing

    Input:
        - VideoProcessingResult from Stage 1 (frames + poses)
        - No segmentation required (unlike legacy Instant4D)

    Output:
        - Per-frame PLY files in output_dir/per_frame_plys/
        - Trained canonical Gaussians + deformation MLP
        - Quality metrics (PSNR)

    Requirements:
        - CUDA 11.6+ with compiled De3DGS kernels
        - ~12-16GB VRAM for training

    Rollback:
        Set GAUSSIAN_ADAPTER=instant4d environment variable to use legacy adapter.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Gaussian training stage.

        Args:
            config: Configuration dictionary with keys:
                - adapter: Adapter type - "deformable3dgs" (default) or "instant4d" (legacy)
                - iterations: Training iterations (default: 20000 for De3DGS, 5000 for Instant4D)
                - device: CUDA device (default: "cuda:0")
                - export_num_frames: Frames to export (default: 30)
                - is_blender: Use D-NeRF time encoding (default: False)
                - is_6dof: Use SE(3) rigid transformation (default: False)
        """
        self.config = config or {}
        self._adapter = None
        self._adapter_type = self.config.get(
            "adapter",
            os.environ.get("GAUSSIAN_ADAPTER", "deformable3dgs")
        )

    def _get_adapter(self):
        """Lazy-load adapter based on configuration."""
        if self._adapter is None:
            if self._adapter_type == "deformable3dgs":
                from ..adapters.deformable3dgs import Deformable3DGSAdapter
                self._adapter = Deformable3DGSAdapter(self.config)
            elif self._adapter_type == "instant4d":
                # Legacy fallback - keep for rollback capability
                from ..adapters.instant4d import Instant4DAdapter
                self._adapter = Instant4DAdapter(self.config)
            else:
                raise ValueError(
                    f"Unknown adapter type: {self._adapter_type}. "
                    f"Available: 'deformable3dgs', 'instant4d'"
                )
        return self._adapter

    def train(
        self,
        processed: VideoProcessingResult,
        output_dir: str,
        segmentation: Any = None,  # Kept for API compatibility, ignored by De3DGS
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GaussianTrainingResult:
        """
        Train 4D Gaussian model from processed video data.

        Args:
            processed: Result from video processing stage (Stage 1).
            output_dir: Directory to store outputs.
            segmentation: DEPRECATED - Not used by De3DGS.
                         Kept for API compatibility with legacy Instant4D.
            progress_callback: Optional callback(progress, message).

        Returns:
            GaussianTrainingResult with paths to trained model and PLY files.

        Raises:
            ImportError: If adapter is not properly installed.
            RuntimeError: If training fails.
        """
        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct*100:.0f}%] {msg}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report(0.0, f"Initializing 4D Gaussian training with {self._adapter_type}")

        adapter = self._get_adapter()

        try:
            if self._adapter_type == "deformable3dgs":
                # Use De3DGS adapter
                from ..adapters.deformable3dgs import Deformable3DGSOptions

                options = Deformable3DGSOptions(
                    iterations=self.config.get("iterations", 20000),
                    export_num_frames=self.config.get("export_num_frames", 30),
                    is_blender=self.config.get("is_blender", False),
                    is_6dof=self.config.get("is_6dof", False),
                )

                report(0.02, "Running De3DGS pipeline")
                result = adapter.run_full_pipeline(
                    video_result=processed,
                    output_dir=str(output_path),
                    options=options,
                    progress_callback=lambda p, m: report(0.02 + p * 0.96, m),
                )

                report(0.98, "Finalizing results")

                training_result = GaussianTrainingResult(
                    ply_path=result.ply_paths[0] if result.ply_paths else "",
                    num_gaussians=result.num_gaussians,
                    metrics=result.metrics,
                    config_path=None,
                    ply_paths=result.ply_paths,
                    model_path=result.model_path,
                    num_frames=result.num_frames,
                    temporal_metadata=result.temporal_metadata,
                )

            else:
                # Legacy Instant4D path
                from ..adapters.instant4d import Instant4DOptions

                options = Instant4DOptions(
                    iterations=self.config.get("iterations", 5000),
                    enable_pruning=self.config.get("enable_pruning", True),
                    motion_threshold=self.config.get("motion_threshold", 0.5),
                    gaussian_dim=self.config.get("gaussian_dim", 4),
                    rot_4d=self.config.get("rot_4d", True),
                )

                report(0.02, "Running Instant4D pipeline (legacy)")
                result = adapter.run_full_pipeline(
                    video_result=processed,
                    segmentation_result=segmentation,
                    output_dir=str(output_path),
                    options=options,
                    progress_callback=lambda p, m: report(0.02 + p * 0.96, m),
                )

                report(0.98, "Finalizing results")

                training_result = GaussianTrainingResult(
                    ply_path=result.ply_paths[0] if result.ply_paths else "",
                    num_gaussians=result.num_gaussians,
                    metrics=result.metrics,
                    config_path=result.config_path,
                    ply_paths=result.ply_paths,
                    model_path=result.model_path,
                    num_frames=result.num_frames,
                    temporal_metadata=result.temporal_metadata,
                )

            report(1.0, f"4D training complete: {training_result.num_frames} frames, "
                       f"{training_result.num_gaussians} Gaussians")

            return training_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"4D Gaussian training failed: {e}") from e

        finally:
            adapter.cleanup()

    def cleanup(self) -> None:
        """Release resources."""
        if self._adapter is not None:
            self._adapter.cleanup()
            self._adapter = None
