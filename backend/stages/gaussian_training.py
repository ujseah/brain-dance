"""Stage 3: 4D Gaussian Splatting Training with Instant4D.

This stage trains a 4D Gaussian Splatting model from processed video data,
enabling temporal reconstruction where users can navigate in 3D while time
progresses.

Architecture Notes:
    - Uses Instant4DAdapter internally (not exposed in ADAPTERS registry)
    - Accepts optional ObjectSegmentationResult from Stage 2 for dynamic/static separation
    - Outputs per-frame PLY files for Stage 5 (Web Export)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List
import logging

from .video_processing import VideoProcessingResult
from .object_segmentation import ObjectSegmentationResult

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
    Stage 3: Train 4D Gaussian Splatting model from processed video data.

    This stage uses Instant4D to create a true 4D reconstruction that supports
    temporal playback - users can navigate in 3D space while time progresses.

    Pipeline:
    1. Preprocess Stage 1/2 outputs to Instant4D format
    2. Apply voxel-based grid pruning (92% Gaussian reduction)
    3. Train 4D Gaussians (~2-5 minutes)
    4. Export per-frame PLY files for web viewing

    Input:
        - VideoProcessingResult from Stage 1 (frames + poses)
        - ObjectSegmentationResult from Stage 2 (optional, for dynamic/static separation)

    Output:
        - Per-frame PLY files in output_dir/plys/
        - Trained 4D model checkpoint
        - Quality metrics (PSNR, SSIM)

    Requirements:
        - CUDA 12.1+ with compiled Instant4D kernels
        - ~24GB VRAM for training
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Gaussian training stage.

        Args:
            config: Configuration dictionary with keys:
                - iterations: Training iterations (default: 5000)
                - device: CUDA device (default: "cuda:0")
                - enable_pruning: Enable voxel pruning (default: True)
                - motion_threshold: Static/dynamic threshold (default: 0.5)
        """
        self.config = config or {}
        self._adapter = None

    def _get_adapter(self):
        """Lazy-load Instant4DAdapter."""
        if self._adapter is None:
            from ..adapters.instant4d import Instant4DAdapter

            self._adapter = Instant4DAdapter(self.config)
        return self._adapter

    def train(
        self,
        processed: VideoProcessingResult,
        output_dir: str,
        segmentation: Optional[ObjectSegmentationResult] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GaussianTrainingResult:
        """
        Train 4D Gaussian model from processed video data.

        Args:
            processed: Result from video processing stage (Stage 1).
            output_dir: Directory to store outputs.
            segmentation: Result from object segmentation stage (Stage 2), optional.
                         If provided, enables dynamic/static region separation.
            progress_callback: Optional callback(progress, message).

        Returns:
            GaussianTrainingResult with paths to trained model and PLY files.

        Raises:
            ImportError: If Instant4D is not properly installed.
            RuntimeError: If training fails.
        """
        from ..adapters.instant4d import Instant4DOptions

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct*100:.0f}%] {msg}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report(0.0, "Initializing 4D Gaussian training")

        # Build options from config
        options = Instant4DOptions(
            iterations=self.config.get("iterations", 5000),
            enable_pruning=self.config.get("enable_pruning", True),
            motion_threshold=self.config.get("motion_threshold", 0.5),
            gaussian_dim=self.config.get("gaussian_dim", 4),
            rot_4d=self.config.get("rot_4d", True),
        )

        # Get adapter and run pipeline
        adapter = self._get_adapter()

        try:
            report(0.02, "Running Instant4D pipeline")
            result = adapter.run_full_pipeline(
                video_result=processed,
                segmentation_result=segmentation,
                output_dir=str(output_path),
                options=options,
                progress_callback=lambda p, m: report(0.02 + p * 0.96, m),
            )

            report(0.98, "Finalizing results")

            # Build result
            training_result = GaussianTrainingResult(
                # Backward compatibility: ply_path is first PLY
                ply_path=result.ply_paths[0] if result.ply_paths else "",
                num_gaussians=result.num_gaussians,
                metrics=result.metrics,
                config_path=result.config_path,
                # New 4D fields
                ply_paths=result.ply_paths,
                model_path=result.model_path,
                num_frames=result.num_frames,
                temporal_metadata=result.temporal_metadata,
            )

            report(1.0, f"4D training complete: {result.num_frames} frames, "
                       f"{result.num_gaussians} Gaussians")

            return training_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"4D Gaussian training failed: {e}") from e

        finally:
            # Cleanup to release GPU memory
            adapter.cleanup()

    def cleanup(self) -> None:
        """Release resources."""
        if self._adapter is not None:
            self._adapter.cleanup()
            self._adapter = None
