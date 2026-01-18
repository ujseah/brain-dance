"""VerseCrafter adapter - 4D geometric control over camera and object motion."""

import os
import sys
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from .base import (
    WorldModelAdapter,
    PreprocessResult,
    GenerateResult,
    ModelCapabilities,
    DetectedObject,
    CameraInfo,
)


class VerseCrafterAdapter(WorldModelAdapter):
    """
    Adapter for Tencent's VerseCrafter model.

    VerseCrafter provides explicit 4D geometric control over camera and
    multi-object motion in video generation. It uses a GeoAdapter attached
    to a frozen Wan2.1 diffusion backbone.

    Requirements:
        - CUDA 12.1+ (no Mac support)
        - ~24GB VRAM for full pipeline
        - VerseCrafter submodule at ./versecrafter

    See: https://github.com/TencentARC/VerseCrafter
    """

    def __init__(self, config: dict = None):
        """
        Initialize VerseCrafter adapter.

        Args:
            config: Configuration options:
                - versecrafter_path: Path to VerseCrafter submodule
                - device: CUDA device (default: "cuda:0")
                - mock: If True, skip model loading (for testing)
        """
        self.config = config or {}
        self.versecrafter_path = Path(
            self.config.get("versecrafter_path", "./versecrafter")
        )
        self.device = self.config.get("device", "cuda:0")
        self.mock = self.config.get("mock", False)

        # Lazy-loaded models
        self._depth_model = None
        self._segmentation_model = None
        self._generation_model = None

    @property
    def name(self) -> str:
        return "VerseCrafter"

    @property
    def description(self) -> str:
        return "4D geometric control over camera and multi-object motion"

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            camera_control=True,
            object_control=True,
            depth_estimation=True,
            segmentation=True,
            max_duration=10.0,
            max_resolution=(1280, 720),  # VerseCrafter default
            supported_formats=["mp4"],
        )

    def _ensure_versecrafter_path(self):
        """Add VerseCrafter to Python path if needed."""
        vc_path = str(self.versecrafter_path.absolute())
        if vc_path not in sys.path:
            sys.path.insert(0, vc_path)

    def _load_depth_model(self):
        """Lazy-load MoGe depth estimation model."""
        if self._depth_model is None and not self.mock:
            self._ensure_versecrafter_path()
            # TODO: Import and initialize MoGe model
            # from versecrafter.inference import depth_estimation
            # self._depth_model = depth_estimation.load_model(self.device)
            pass
        return self._depth_model

    def _load_segmentation_model(self):
        """Lazy-load Grounded-SAM-2 segmentation model."""
        if self._segmentation_model is None and not self.mock:
            self._ensure_versecrafter_path()
            # TODO: Import and initialize SAM-2 model
            # from versecrafter.inference import segmentation
            # self._segmentation_model = segmentation.load_model(self.device)
            pass
        return self._segmentation_model

    def _load_generation_model(self):
        """Lazy-load Wan2.1 + GeoAdapter generation model."""
        if self._generation_model is None and not self.mock:
            self._ensure_versecrafter_path()
            # TODO: Import and initialize generation model
            # from versecrafter.inference import video_generation
            # self._generation_model = video_generation.load_model(self.device)
            pass
        return self._generation_model

    def preprocess(
        self,
        image_path: str,
        options: Optional[dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> PreprocessResult:
        """
        Run VerseCrafter preprocessing pipeline.

        Steps:
        1. Depth estimation (MoGe-V2)
        2. Object segmentation (Grounded-SAM-2)
        3. 3D Gaussian fitting

        Args:
            image_path: Path to input image.
            options: Preprocessing options:
                - detect_objects: bool (default True)
                - object_prompts: list of text prompts for object detection
            progress_callback: Optional progress callback.

        Returns:
            PreprocessResult with depth, objects, and scene data.
        """
        options = options or {}

        def report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        report(0.0, "Starting preprocessing")

        # Mock mode for testing without GPU
        if self.mock:
            report(1.0, "Mock preprocessing complete")
            return PreprocessResult(
                depth_map_path=None,
                objects=[
                    DetectedObject(id="mock_obj_1", label="object", center=(0, 0, 1))
                ],
                camera=CameraInfo(fov=60.0),
                scene_data={"mock": True, "image_path": image_path},
            )

        # Step 1: Depth estimation
        report(0.1, "Running depth estimation (MoGe-V2)")
        depth_model = self._load_depth_model()
        # TODO: depth_map = depth_model.predict(image_path)
        depth_map_path = None

        # Step 2: Object segmentation
        report(0.4, "Running object segmentation (Grounded-SAM-2)")
        if options.get("detect_objects", True):
            seg_model = self._load_segmentation_model()
            prompts = options.get("object_prompts", [])
            # TODO: masks = seg_model.segment(image_path, prompts)
            objects = []
        else:
            objects = []

        # Step 3: 3D Gaussian fitting
        report(0.7, "Fitting 3D Gaussians")
        # TODO: Fit 3D Gaussians to segmented objects

        # Step 4: Estimate camera
        report(0.9, "Estimating camera parameters")
        camera = CameraInfo(fov=60.0)  # TODO: Estimate from depth

        report(1.0, "Preprocessing complete")

        return PreprocessResult(
            depth_map_path=depth_map_path,
            objects=objects,
            camera=camera,
            scene_data={
                "image_path": image_path,
                "depth_path": depth_map_path,
                # Additional VerseCrafter-specific data
            },
        )

    def generate(
        self,
        preprocessed: PreprocessResult,
        trajectory: dict,
        options: Optional[dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GenerateResult:
        """
        Generate video using VerseCrafter.

        Steps:
        1. Render 4D control maps from trajectory
        2. Run diffusion with GeoAdapter conditioning

        Args:
            preprocessed: Result from preprocess().
            trajectory: Camera and object trajectories.
            options: Generation options:
                - resolution: (width, height)
                - seed: Random seed
                - guidance_scale: Classifier-free guidance scale
                - num_inference_steps: Diffusion steps
            progress_callback: Optional progress callback.

        Returns:
            GenerateResult with video path.
        """
        options = options or {}

        def report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        # Validate trajectory
        errors = self.validate_trajectory(trajectory)
        if errors:
            raise ValueError(f"Invalid trajectory: {errors}")

        report(0.0, "Starting generation")

        # Mock mode
        if self.mock:
            report(1.0, "Mock generation complete")
            return GenerateResult(
                video_path="/tmp/mock_output.mp4",
                duration=trajectory.get("duration", 4.0),
                metadata={"mock": True, "adapter": self.name},
            )

        # Step 1: Render control maps
        report(0.1, "Rendering 4D control maps")
        control_maps = self._render_control_maps(preprocessed, trajectory)

        # Step 2: Run diffusion
        report(0.3, "Running video diffusion")
        gen_model = self._load_generation_model()

        resolution = options.get("resolution", (1280, 720))
        seed = options.get("seed", None)
        guidance = options.get("guidance_scale", 7.5)
        steps = options.get("num_inference_steps", 50)

        # TODO: Actually run generation
        # output = gen_model.generate(
        #     image=preprocessed.scene_data["image_path"],
        #     control_maps=control_maps,
        #     resolution=resolution,
        #     seed=seed,
        #     guidance_scale=guidance,
        #     num_inference_steps=steps,
        # )

        # For now, placeholder
        output_path = "/tmp/versecrafter_output.mp4"

        report(1.0, "Generation complete")

        return GenerateResult(
            video_path=output_path,
            control_map_path=None,  # TODO: Save control maps
            duration=trajectory.get("duration", 4.0),
            metadata={
                "adapter": self.name,
                "resolution": resolution,
                "seed": seed,
                "guidance_scale": guidance,
            },
        )

    def _render_control_maps(self, preprocessed: PreprocessResult, trajectory: dict):
        """
        Render 4D control maps from trajectory.

        This converts camera and object trajectories into the visual
        control signal format expected by VerseCrafter's GeoAdapter.
        """
        # TODO: Implement control map rendering
        # This involves:
        # 1. Interpolating keyframes to full frame rate
        # 2. Rendering camera motion as optical flow / depth changes
        # 3. Rendering object motion as per-object displacement maps
        return None

    def cleanup(self) -> None:
        """Unload models to free GPU memory."""
        self._depth_model = None
        self._segmentation_model = None
        self._generation_model = None
        # Force garbage collection
        import gc
        gc.collect()
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# =============================================================================
# Coordinate System Utilities
# =============================================================================
#
# VerseCrafter uses Blender's coordinate system internally (Z-up, Y-forward)
# but depth maps and images use OpenCV convention (Y-down, Z-forward).
#
# OpenCV:  X = right, Y = down,  Z = forward (into screen)
# Blender: X = right, Y = forward, Z = up
#

# Transformation matrix: OpenCV → Blender
# Multiplying a point in OpenCV coords by this matrix gives Blender coords
COORD_TRANSFORM_CV2BLENDER = np.array([
    [1,  0,  0],   # Blender X = OpenCV X (right)
    [0,  0,  1],   # Blender Y = OpenCV Z (forward)
    [0, -1,  0],   # Blender Z = -OpenCV Y (up, since OpenCV Y points down)
], dtype=np.float32)

# Inverse transformation: Blender → OpenCV
COORD_TRANSFORM_BLENDER2CV = np.array([
    [1,  0,  0],   # OpenCV X = Blender X (right)
    [0,  0, -1],   # OpenCV Y = -Blender Z (down)
    [0,  1,  0],   # OpenCV Z = Blender Y (forward)
], dtype=np.float32)


def opencv_to_blender(points: np.ndarray) -> np.ndarray:
    """
    Convert 3D points from OpenCV to Blender coordinate system.

    Args:
        points: Array of shape (N, 3) or (3,) in OpenCV coordinates.

    Returns:
        Array of same shape in Blender coordinates.
    """
    return points @ COORD_TRANSFORM_CV2BLENDER.T


def blender_to_opencv(points: np.ndarray) -> np.ndarray:
    """
    Convert 3D points from Blender to OpenCV coordinate system.

    Args:
        points: Array of shape (N, 3) or (3,) in Blender coordinates.

    Returns:
        Array of same shape in OpenCV coordinates.
    """
    return points @ COORD_TRANSFORM_BLENDER2CV.T


def transform_camera_matrix_cv2blender(extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 camera extrinsic matrix from OpenCV to Blender convention.

    Args:
        extrinsic: 4x4 world-to-camera matrix in OpenCV convention.

    Returns:
        4x4 camera-to-world matrix in Blender convention.
    """
    # Extract rotation and translation
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # Convert rotation
    R_blender = COORD_TRANSFORM_CV2BLENDER @ R @ COORD_TRANSFORM_BLENDER2CV

    # Convert translation
    t_blender = COORD_TRANSFORM_CV2BLENDER @ t

    # Build camera-to-world (Blender convention)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_blender.T  # Transpose for c2w
    c2w[:3, 3] = -R_blender.T @ t_blender

    return c2w
