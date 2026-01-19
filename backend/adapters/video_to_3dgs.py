"""Video to 3DGS adapter - Transform video into explorable 3D Gaussian Splat."""

from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

from .base import WorldModelAdapter, ModelCapabilities


@dataclass
class VideoProcessingOptions:
    """Options for video processing."""

    pose_estimator: str = "hloc"
    """Pose estimation method: 'hloc' (recommended) or 'dust3r' (fallback)."""

    frame_interval: int = 1
    """Extract every Nth frame from video."""

    max_frames: int = 300
    """Maximum number of frames to extract."""


@dataclass
class TrainingOptions:
    """Options for 3DGS training."""

    num_iterations: int = 30000
    """Number of training iterations."""

    quality: str = "balanced"
    """Quality preset: 'fast', 'balanced', 'high'."""


@dataclass
class HoleFillingOptions:
    """Options for AI hole-filling."""

    enabled: bool = True
    """Whether to run hole-filling."""

    quality: str = "balanced"
    """Quality preset: 'fast', 'balanced', 'high'."""

    num_views: int = 8
    """Number of views to render around each hole."""


@dataclass
class ExportOptions:
    """Options for web export."""

    format: str = "spz"
    """Export format: 'spz' (recommended), 'ksplat', 'ply'."""

    compression: str = "balanced"
    """Compression level: 'fast', 'balanced', 'high'."""


@dataclass
class VideoTo3DGSResult:
    """Result of the full video-to-3DGS pipeline."""

    splat_path: str
    """Path to final 3DGS file (PLY or compressed)."""

    viewer_path: Optional[str] = None
    """Path to web viewer bundle."""

    num_frames: int = 0
    """Number of frames processed."""

    num_gaussians: int = 0
    """Number of Gaussians in final model."""

    holes_filled: int = 0
    """Number of holes filled by AI."""

    metrics: dict = field(default_factory=dict)
    """Quality metrics and timing info."""


class VideoTo3DGSAdapter(WorldModelAdapter):
    """
    Adapter for video-to-3DGS reconstruction pipeline.

    This adapter transforms video footage into an explorable 3D Gaussian Splat
    viewable in web browsers, with optional AI-powered hole filling.

    Pipeline stages:
    1. Video Processing - Frame extraction + camera pose estimation
    2. 3DGS Training - Train Splatfacto model from frames
    3. Hole Filling (optional) - AI fills unseen regions
    4. Web Export - Convert to compressed web format

    Requirements:
        - CUDA 12.1+ (no Mac support for GPU stages)
        - ~24GB VRAM for training
        - Nerfstudio, hloc, pycolmap dependencies
    """

    def __init__(self, config: dict = None):
        """
        Initialize VideoTo3DGS adapter.

        Args:
            config: Configuration options:
                - device: CUDA device (default: "cuda:0")
                - mock: If True, skip model loading (for testing)
                - versecrafter_path: Path to VerseCrafter submodule (for MoGe-V2)
        """
        self.config = config or {}
        self.device = self.config.get("device", "cuda:0")
        self.mock = self.config.get("mock", False)
        self.versecrafter_path = Path(
            self.config.get("versecrafter_path", "./versecrafter")
        )

        # Lazy-loaded stages
        self._video_stage = None
        self._training_stage = None
        self._hole_filling_stage = None
        self._export_stage = None

    @property
    def name(self) -> str:
        return "Video to 3DGS"

    @property
    def description(self) -> str:
        return "Transform video into explorable 3D Gaussian Splat with AI hole-filling"

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_video_input=True,
            supports_3dgs_output=True,
            supports_hole_filling=True,
            supports_web_export=True,
            max_video_duration=300.0,  # 5 minutes
            max_resolution=(1920, 1080),
            supported_formats=["spz", "ksplat", "ply"],
        )

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        options: Optional[VideoProcessingOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 1: Process video to extract frames and estimate poses.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to store outputs.
            options: Video processing options.
            progress_callback: Optional callback(progress, message).

        Returns:
            VideoProcessingResult with paths to frames and poses.
        """
        from ..stages.video_processing import VideoProcessingStage

        options = options or VideoProcessingOptions()

        if self._video_stage is None:
            self._video_stage = VideoProcessingStage({
                "pose_estimator": options.pose_estimator,
                "frame_interval": options.frame_interval,
                "max_frames": options.max_frames,
            })

        return self._video_stage.process(video_path, output_dir, progress_callback)

    def train_3dgs(
        self,
        processed_dir: str,
        output_dir: str,
        options: Optional[TrainingOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 2: Train 3DGS model from processed video data.

        Args:
            processed_dir: Directory containing processed video data.
            output_dir: Directory to store training outputs.
            options: Training options.
            progress_callback: Optional callback(progress, message).

        Returns:
            GaussianTrainingResult with path to trained PLY.
        """
        from ..stages.gaussian_training import GaussianTrainingStage
        from ..stages.video_processing import VideoProcessingResult

        options = options or TrainingOptions()

        if self._training_stage is None:
            self._training_stage = GaussianTrainingStage({
                "num_iterations": options.num_iterations,
            })

        # Load processed result
        processed = VideoProcessingResult(
            frames_dir=str(Path(processed_dir) / "frames"),
            num_frames=0,  # Will be determined during training
            transforms_path=str(Path(processed_dir) / "transforms.json"),
        )

        return self._training_stage.train(processed, output_dir, progress_callback)

    def fill_holes(
        self,
        ply_path: str,
        output_dir: str,
        options: Optional[HoleFillingOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 3: Fill holes in 3DGS using AI inpainting.

        Args:
            ply_path: Path to trained PLY file.
            output_dir: Directory to store outputs.
            options: Hole-filling options.
            progress_callback: Optional callback(progress, message).

        Returns:
            HoleFillingResult with path to refined PLY.
        """
        from ..stages.hole_filling import HoleFillingStage
        from ..stages.gaussian_training import GaussianTrainingResult

        options = options or HoleFillingOptions()

        if not options.enabled:
            return None

        if self._hole_filling_stage is None:
            self._hole_filling_stage = HoleFillingStage({
                "quality": options.quality,
                "num_inpaint_views": options.num_views,
            })

        trained = GaussianTrainingResult(
            ply_path=ply_path,
            num_gaussians=0,  # Will be determined during filling
        )

        return self._hole_filling_stage.fill(trained, output_dir, progress_callback)

    def export_for_web(
        self,
        ply_path: str,
        output_dir: str,
        scene_name: str = "scene",
        options: Optional[ExportOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 4: Export 3DGS for web viewing.

        Args:
            ply_path: Path to PLY file to export.
            output_dir: Directory to store outputs.
            scene_name: Name for the scene.
            options: Export options.
            progress_callback: Optional callback(progress, message).

        Returns:
            WebExportResult with paths to viewer bundle.
        """
        from ..stages.web_export import WebExportStage

        options = options or ExportOptions()

        if self._export_stage is None:
            self._export_stage = WebExportStage({
                "format": options.format,
                "compression_level": options.compression,
            })

        return self._export_stage.export(ply_path, output_dir, scene_name, progress_callback)

    def run_full_pipeline(
        self,
        video_path: str,
        output_dir: str,
        video_options: Optional[VideoProcessingOptions] = None,
        training_options: Optional[TrainingOptions] = None,
        hole_filling_options: Optional[HoleFillingOptions] = None,
        export_options: Optional[ExportOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VideoTo3DGSResult:
        """
        Run the complete video-to-3DGS pipeline.

        Args:
            video_path: Path to input video file.
            output_dir: Base directory for all outputs.
            video_options: Options for video processing stage.
            training_options: Options for training stage.
            hole_filling_options: Options for hole-filling stage.
            export_options: Options for export stage.
            progress_callback: Optional callback(progress, message).

        Returns:
            VideoTo3DGSResult with paths to all outputs.
        """

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        hole_filling_options = hole_filling_options or HoleFillingOptions()

        # Stage 1: Video Processing (0-25%)
        report(0.0, "Stage 1: Processing video")
        processed = self.process_video(
            video_path,
            str(output_path / "processed"),
            video_options,
            lambda p, m: report(p * 0.25, f"[Video] {m}"),
        )

        # Stage 2: 3DGS Training (25-70%)
        report(0.25, "Stage 2: Training 3DGS")
        trained = self.train_3dgs(
            str(output_path / "processed"),
            str(output_path / "trained"),
            training_options,
            lambda p, m: report(0.25 + p * 0.45, f"[Training] {m}"),
        )

        # Stage 3: Hole Filling (70-90%)
        final_ply = trained.ply_path
        holes_filled = 0

        if hole_filling_options.enabled:
            report(0.70, "Stage 3: Filling holes")
            filled = self.fill_holes(
                trained.ply_path,
                str(output_path / "filled"),
                hole_filling_options,
                lambda p, m: report(0.70 + p * 0.20, f"[Hole-filling] {m}"),
            )
            if filled:
                final_ply = filled.ply_path
                holes_filled = filled.num_holes_filled

        # Stage 4: Web Export (90-100%)
        report(0.90, "Stage 4: Exporting for web")
        exported = self.export_for_web(
            final_ply,
            str(output_path / "export"),
            "scene",
            export_options,
            lambda p, m: report(0.90 + p * 0.10, f"[Export] {m}"),
        )

        report(1.0, "Pipeline complete")

        return VideoTo3DGSResult(
            splat_path=exported.splat_path,
            viewer_path=exported.viewer_path,
            num_frames=processed.num_frames,
            num_gaussians=trained.num_gaussians,
            holes_filled=holes_filled,
            metrics={
                "training_metrics": trained.metrics,
                "export_format": exported.format,
                "file_size_mb": exported.file_size_mb,
            },
        )

    # =========================================================================
    # Legacy interface compatibility (from base WorldModelAdapter)
    # =========================================================================

    def preprocess(self, image_path: str, options=None, progress_callback=None):
        """Legacy interface - not used for video-to-3DGS pipeline."""
        raise NotImplementedError(
            "VideoTo3DGSAdapter uses process_video() instead of preprocess(). "
            "Use run_full_pipeline() for the complete video-to-3DGS workflow."
        )

    def generate(self, preprocessed, trajectory, options=None, progress_callback=None):
        """Legacy interface - not used for video-to-3DGS pipeline."""
        raise NotImplementedError(
            "VideoTo3DGSAdapter doesn't use trajectories. "
            "Use run_full_pipeline() for the complete video-to-3DGS workflow."
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._hole_filling_stage:
            self._hole_filling_stage.cleanup()

        self._video_stage = None
        self._training_stage = None
        self._hole_filling_stage = None
        self._export_stage = None
