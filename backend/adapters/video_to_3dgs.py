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
class SegmentationOptions:
    """Options for object segmentation (Stage 2)."""

    enabled: bool = True
    """Whether to run object segmentation. Critical for accurate scene completion."""

    keyframe_interval: int = 10
    """Run SAM-2 automatic mask generation every Nth frame."""

    min_object_size: int = 100
    """Minimum object size in pixels to track."""

    model_size: str = "large"
    """SAM-2 model size: 'tiny', 'small', 'base', 'large'."""


@dataclass
class TrainingOptions:
    """Options for 3DGS training."""

    num_iterations: int = 30000
    """Number of training iterations."""

    quality: str = "balanced"
    """Quality preset: 'fast', 'balanced', 'high'."""


@dataclass
class SceneCompletionOptions:
    """Options for AI scene completion (Stage 4)."""

    enabled: bool = True
    """Whether to run scene completion. This is the core feature of Brain Dance."""

    quality: str = "balanced"
    """Quality preset: 'fast', 'balanced', 'high'."""

    num_views: int = 8
    """Number of views to render around each detected gap."""


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

    regions_completed: int = 0
    """Number of regions completed by AI scene completion."""

    metrics: dict = field(default_factory=dict)
    """Quality metrics and timing info."""


class VideoTo3DGSAdapter(WorldModelAdapter):
    """
    Adapter for video-to-3DGS reconstruction pipeline.

    This adapter transforms video footage into an explorable 3D Gaussian Splat
    viewable in web browsers, with AI-powered scene completion.

    Pipeline stages:
    1. Video Processing - Frame extraction + camera pose estimation
    2. Object Segmentation - SAM-2 segmentation & tracking (critical for scene completion)
    3. 3DGS Training - Train Splatfacto model from frames
    4. AI Scene Completion - Generate unseen regions using object-aware inpainting
    5. Web Export - Convert to compressed web format

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
        self._segmentation_stage = None
        self._training_stage = None
        self._scene_completion_stage = None
        self._export_stage = None

    @property
    def name(self) -> str:
        return "Video to 3DGS"

    @property
    def description(self) -> str:
        return "Transform video into explorable 3D Gaussian Splat with AI scene completion"

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

    def segment_objects(
        self,
        frames_dir: str,
        output_dir: str,
        options: Optional[SegmentationOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 2: Segment and track objects across frames.

        This stage is CRITICAL for accurate scene completion. Without object-aware
        segmentation, gaps get filled with "texture soup" - averaged nearby
        colors that don't respect object boundaries.

        Args:
            frames_dir: Directory containing extracted frames.
            output_dir: Directory to store segmentation outputs.
            options: Segmentation options.
            progress_callback: Optional callback(progress, message).

        Returns:
            ObjectSegmentationResult with paths to masks and metadata.
        """
        from ..stages.object_segmentation import ObjectSegmentationStage

        options = options or SegmentationOptions()

        if not options.enabled:
            return None

        if self._segmentation_stage is None:
            self._segmentation_stage = ObjectSegmentationStage({
                "keyframe_interval": options.keyframe_interval,
                "min_object_size": options.min_object_size,
                "model_size": options.model_size,
            })

        return self._segmentation_stage.segment(frames_dir, output_dir, progress_callback)

    def train_3dgs(
        self,
        processed_dir: str,
        output_dir: str,
        options: Optional[TrainingOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 3: Train 3DGS model from processed video data.

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

    def complete_scene(
        self,
        ply_path: str,
        output_dir: str,
        masks_dir: Optional[str] = None,
        options: Optional[SceneCompletionOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 4: AI Scene Completion - Generate unseen regions.

        This is the core differentiator of Brain Dance. Uses object masks from
        Stage 2 for object-aware completion, ensuring AI-generated regions
        respect object boundaries and scene context.

        Args:
            ply_path: Path to trained PLY file.
            output_dir: Directory to store outputs.
            masks_dir: Path to masks from object segmentation (from Stage 2).
            options: Scene completion options.
            progress_callback: Optional callback(progress, message).

        Returns:
            SceneCompletionResult with path to completed PLY.
        """
        from ..stages.scene_completion import SceneCompletionStage
        from ..stages.gaussian_training import GaussianTrainingResult

        options = options or SceneCompletionOptions()

        if not options.enabled:
            return None

        if self._scene_completion_stage is None:
            self._scene_completion_stage = SceneCompletionStage({
                "quality": options.quality,
                "num_inpaint_views": options.num_views,
                "masks_dir": masks_dir,  # Pass object masks for object-aware completion
            })

        trained = GaussianTrainingResult(
            ply_path=ply_path,
            num_gaussians=0,  # Will be determined during completion
        )

        return self._scene_completion_stage.complete(trained, output_dir, progress_callback)

    def export_for_web(
        self,
        ply_path: str,
        output_dir: str,
        scene_name: str = "scene",
        options: Optional[ExportOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Stage 5: Export 3DGS for web viewing.

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
        segmentation_options: Optional[SegmentationOptions] = None,
        training_options: Optional[TrainingOptions] = None,
        scene_completion_options: Optional[SceneCompletionOptions] = None,
        export_options: Optional[ExportOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VideoTo3DGSResult:
        """
        Run the complete video-to-3DGS pipeline.

        Args:
            video_path: Path to input video file.
            output_dir: Base directory for all outputs.
            video_options: Options for video processing stage.
            segmentation_options: Options for object segmentation stage (1.5).
            training_options: Options for training stage.
            scene_completion_options: Options for AI scene completion stage.
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

        segmentation_options = segmentation_options or SegmentationOptions()
        scene_completion_options = scene_completion_options or SceneCompletionOptions()

        # Stage 1: Video Processing (0-20%)
        report(0.0, "Stage 1: Processing video")
        processed = self.process_video(
            video_path,
            str(output_path / "processed"),
            video_options,
            lambda p, m: report(p * 0.20, f"[Video] {m}"),
        )

        # Stage 2: Object Segmentation (20-35%)
        # Critical for accurate hole-filling - identifies object boundaries
        masks_dir = None
        if segmentation_options.enabled:
            report(0.20, "Stage 2: Segmenting objects")
            segmented = self.segment_objects(
                processed.frames_dir,
                str(output_path / "segmented"),
                segmentation_options,
                lambda p, m: report(0.20 + p * 0.15, f"[Segmentation] {m}"),
            )
            if segmented:
                masks_dir = segmented.masks_dir

        # Stage 3: 3DGS Training (35-70%)
        report(0.35, "Stage 3: Training 3DGS")
        trained = self.train_3dgs(
            str(output_path / "processed"),
            str(output_path / "trained"),
            training_options,
            lambda p, m: report(0.35 + p * 0.35, f"[Training] {m}"),
        )

        # Stage 4: AI Scene Completion (70-90%)
        final_ply = trained.ply_path
        regions_completed = 0

        if scene_completion_options.enabled:
            report(0.70, "Stage 4: Completing scene")
            completed = self.complete_scene(
                trained.ply_path,
                str(output_path / "completed"),
                masks_dir,  # Pass object masks for object-aware completion
                scene_completion_options,
                lambda p, m: report(0.70 + p * 0.20, f"[Scene Completion] {m}"),
            )
            if completed:
                final_ply = completed.ply_path
                regions_completed = completed.num_regions_completed

        # Stage 5: Web Export (90-100%)
        report(0.90, "Stage 5: Exporting for web")
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
            regions_completed=regions_completed,
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
        if self._scene_completion_stage:
            self._scene_completion_stage.cleanup()

        self._video_stage = None
        self._segmentation_stage = None
        self._training_stage = None
        self._scene_completion_stage = None
        self._export_stage = None
