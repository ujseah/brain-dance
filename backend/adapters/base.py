"""Abstract base class for world model adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class ModelCapabilities:
    """Describes what a world model adapter can do."""

    # Video generation capabilities (legacy VerseCrafter-style)
    camera_control: bool = False
    """Can control camera trajectory (position, rotation, FOV)."""

    object_control: bool = False
    """Can control individual object motion."""

    depth_estimation: bool = False
    """Provides depth estimation from input image."""

    segmentation: bool = False
    """Provides object segmentation."""

    # 3DGS reconstruction capabilities (new video-to-3DGS pipeline)
    supports_video_input: bool = False
    """Can process video input for reconstruction."""

    supports_3dgs_output: bool = False
    """Outputs explorable 3D Gaussian Splat."""

    supports_hole_filling: bool = False
    """Supports AI-powered hole filling for unseen regions."""

    supports_web_export: bool = False
    """Can export to web-viewable formats (SPZ, KSPLAT)."""

    # Limits
    max_duration: float = 10.0
    """Maximum video duration in seconds (for generation)."""

    max_video_duration: float = 300.0
    """Maximum input video duration for reconstruction (5 minutes)."""

    max_resolution: tuple = (1920, 1080)
    """Maximum output resolution (width, height)."""

    supported_formats: list = field(default_factory=lambda: ["mp4"])
    """Supported output formats."""


@dataclass
class DetectedObject:
    """A detected object in the scene."""

    id: str
    """Unique identifier for this object."""

    label: str
    """Human-readable label (e.g., "car", "person")."""

    mask_path: Optional[str] = None
    """Path to segmentation mask image."""

    center: Optional[tuple] = None
    """3D center position (x, y, z)."""

    bounds: Optional[dict] = None
    """Bounding box or 3D bounds."""

    gaussian: Optional[dict] = None
    """3D Gaussian representation (for VerseCrafter-style models)."""


@dataclass
class CameraInfo:
    """Camera intrinsics and initial pose."""

    intrinsics: Optional[list] = None
    """3x3 intrinsic matrix as nested list."""

    initial_pose: Optional[list] = None
    """4x4 pose matrix as nested list."""

    fov: Optional[float] = None
    """Field of view in degrees."""


@dataclass
class PreprocessResult:
    """Result of preprocessing an input image."""

    depth_map: Optional[str] = None
    """Base64-encoded depth map or path to file."""

    depth_map_path: Optional[str] = None
    """Path to depth map file."""

    objects: list = field(default_factory=list)
    """List of DetectedObject instances."""

    camera: Optional[CameraInfo] = None
    """Estimated camera information."""

    scene_bounds: Optional[dict] = None
    """Scene bounding box: {"min": [x,y,z], "max": [x,y,z]}."""

    scene_data: dict = field(default_factory=dict)
    """Model-specific scene data (for passing to generate)."""


@dataclass
class GenerateResult:
    """Result of video generation."""

    video_path: str
    """Path to generated video file."""

    control_map_path: Optional[str] = None
    """Path to control map visualization (optional)."""

    duration: Optional[float] = None
    """Actual duration of generated video."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata about the generation."""


class WorldModelAdapter(ABC):
    """
    Abstract base class for world model adapters.

    Each adapter wraps a specific world model (VerseCrafter, Runway, etc.)
    and provides a consistent interface for the Brain Dance editor.

    Example:
        adapter = VerseCrafterAdapter(config={"model_path": "./models"})
        preprocess = adapter.preprocess("input.jpg")
        result = adapter.generate(preprocess, trajectory)
        print(result.video_path)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this adapter."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this adapter/model does."""
        pass

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """
        Get the capabilities of this model.

        Returns:
            ModelCapabilities describing what this model can do.
        """
        pass

    @abstractmethod
    def preprocess(
        self,
        image_path: str,
        options: Optional[dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> PreprocessResult:
        """
        Run preprocessing pipeline on input image.

        This typically includes:
        - Depth estimation
        - Object detection/segmentation
        - 3D scene reconstruction
        - Camera estimation

        Args:
            image_path: Path to input image file.
            options: Model-specific preprocessing options.
            progress_callback: Optional callback(progress, message) for progress updates.

        Returns:
            PreprocessResult with depth, objects, camera info, etc.
        """
        pass

    @abstractmethod
    def generate(
        self,
        preprocessed: PreprocessResult,
        trajectory: dict,
        options: Optional[dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GenerateResult:
        """
        Generate video from preprocessed scene and trajectory.

        Args:
            preprocessed: Result from preprocess().
            trajectory: Camera and object trajectories in standard format:
                {
                    "duration": float,  # seconds
                    "fps": int,
                    "camera": {
                        "keyframes": [
                            {"time": float, "position": [x,y,z], "rotation": [qx,qy,qz,qw]},
                            ...
                        ],
                        "interpolation": "linear" | "cubic" | "bezier"
                    },
                    "objects": [
                        {
                            "id": str,
                            "keyframes": [{"time": float, "position": [...], "rotation": [...]}]
                        },
                        ...
                    ]
                }
            options: Model-specific generation options (resolution, seed, etc.).
            progress_callback: Optional callback(progress, message) for progress updates.

        Returns:
            GenerateResult with video path and metadata.
        """
        pass

    def validate_trajectory(self, trajectory: dict) -> list:
        """
        Validate a trajectory against this model's capabilities.

        Args:
            trajectory: Trajectory to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        caps = self.get_capabilities()

        # Check duration
        duration = trajectory.get("duration", 0)
        if duration > caps.max_duration:
            errors.append(
                f"Duration {duration}s exceeds max {caps.max_duration}s"
            )

        # Check object control
        if trajectory.get("objects") and not caps.object_control:
            errors.append("This model does not support object control")

        # Check camera control
        if trajectory.get("camera") and not caps.camera_control:
            errors.append("This model does not support camera control")

        return errors

    def cleanup(self) -> None:
        """
        Clean up resources (unload models, close connections, etc.).

        Called when the adapter is no longer needed.
        """
        pass
