"""Stage 3: 3D Gaussian Splatting Training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from .video_processing import VideoProcessingResult


@dataclass
class GaussianTrainingResult:
    """Result of 3DGS training stage."""

    ply_path: str
    """Path to trained Gaussian splat PLY file."""

    num_gaussians: int
    """Number of Gaussians in the trained model."""

    metrics: dict = field(default_factory=dict)
    """Quality metrics (PSNR, SSIM, etc.)."""

    config_path: Optional[str] = None
    """Path to Nerfstudio config file."""


class GaussianTrainingStage:
    """
    Stage 3: Train 3D Gaussian Splatting model from processed video data.

    This stage:
    1. Initializes Gaussians from sparse point cloud
    2. Trains using Nerfstudio's Splatfacto
    3. Exports trained model to PLY format

    Output: scene.ply, quality_metrics.json
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.num_iterations = self.config.get("num_iterations", 30000)
        self.method = self.config.get("method", "splatfacto")

    def train(
        self,
        processed: VideoProcessingResult,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GaussianTrainingResult:
        """
        Train 3DGS model from processed video data.

        Args:
            processed: Result from video processing stage.
            output_dir: Directory to store outputs.
            progress_callback: Optional callback(progress, message).

        Returns:
            GaussianTrainingResult with path to trained PLY.
        """

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report(0.0, "Initializing 3DGS training")

        # Step 1: Prepare data in Nerfstudio format
        report(0.05, "Preparing training data")
        data_path = self._prepare_nerfstudio_data(processed, output_path)

        # Step 2: Train Splatfacto
        report(0.1, f"Training {self.method} ({self.num_iterations} iterations)")
        config_path = self._train_splatfacto(data_path, output_path, progress_callback)

        # Step 3: Export to PLY
        report(0.95, "Exporting to PLY format")
        ply_path = output_path / "scene.ply"
        num_gaussians = self._export_ply(config_path, ply_path)

        # Step 4: Compute quality metrics
        metrics = self._compute_metrics(config_path)

        report(1.0, "3DGS training complete")

        return GaussianTrainingResult(
            ply_path=str(ply_path),
            num_gaussians=num_gaussians,
            metrics=metrics,
            config_path=str(config_path) if config_path else None,
        )

    def _prepare_nerfstudio_data(self, processed: VideoProcessingResult, output_path: Path) -> Path:
        """Prepare data in Nerfstudio format."""
        # TODO: Convert transforms.json to Nerfstudio format if needed
        # The transforms.json from hloc should already be compatible
        raise NotImplementedError("Nerfstudio data preparation not yet implemented")

    def _train_splatfacto(
        self,
        data_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """Train Splatfacto model."""
        # TODO: Implement Nerfstudio training
        # ns-train splatfacto --data {data_path} --output-dir {output_path}
        # with progress monitoring
        raise NotImplementedError("Splatfacto training not yet implemented")

    def _export_ply(self, config_path: Path, ply_path: Path) -> int:
        """Export trained model to PLY format."""
        # TODO: Implement PLY export
        # ns-export gaussian-splat --load-config {config_path} --output-dir {ply_path.parent}
        raise NotImplementedError("PLY export not yet implemented")

    def _compute_metrics(self, config_path: Path) -> dict:
        """Compute quality metrics on held-out views."""
        # TODO: Load model and compute PSNR, SSIM, LPIPS
        return {}
