"""Stage 2: Object Segmentation - SAM-2 segmentation and tracking."""

import hashlib
import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List

import torch
import numpy as np
from PIL import Image

# Configure structured logging
logger = logging.getLogger(__name__)

# SAM-2 imports with graceful handling
try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning(
        "SAM-2 not installed. Install with: "
        "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
    )

# SAM-2 model configuration
# Maps model size names to their config files, checkpoint names, URLs, and VRAM requirements
#
# NOTE: SHA256 checksums are intentionally None for prototype phase.
# Downloads are protected by HTTPS (Meta's certificate). For production deployment,
# compute and add checksums by downloading each model and running:
#   sha256sum sam2_hiera_*.pt
# This provides defense-in-depth against supply chain attacks.
SAM2_MODELS = {
    "tiny": {
        "config": "sam2_hiera_t.yaml",
        "checkpoint": "sam2_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "sha256": None,
        "vram_gb": 4.0,
    },
    "small": {
        "config": "sam2_hiera_s.yaml",
        "checkpoint": "sam2_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sha256": None,
        "vram_gb": 8.0,
    },
    "base_plus": {
        "config": "sam2_hiera_b+.yaml",
        "checkpoint": "sam2_hiera_base_plus.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sha256": None,
        "vram_gb": 12.0,
    },
    "large": {
        "config": "sam2_hiera_l.yaml",
        "checkpoint": "sam2_hiera_large.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "sha256": None,
        "vram_gb": 16.0,
    },
}

# Quality presets for mask generator
# Controls trade-off between speed and segmentation quality
SEGMENTATION_PRESETS = {
    "fast": {
        "points_per_side": 48,
        "pred_iou_thresh": 0.6,
        "stability_score_thresh": 0.75,
    },
    "balanced": {
        "points_per_side": 32,
        "pred_iou_thresh": 0.7,
        "stability_score_thresh": 0.85,
    },
    "thorough": {
        "points_per_side": 64,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.9,
    },
}


@dataclass
class SegmentedObject:
    """Represents a single segmented object tracked across frames."""

    object_id: int
    """Unique identifier for this object."""

    label: Optional[str] = None
    """Optional semantic label (e.g., 'person', 'car', 'building')."""

    mask_paths: List[str] = field(default_factory=list)
    """List of mask file paths for each frame where object appears."""

    frame_indices: List[int] = field(default_factory=list)
    """Frame indices where this object is visible."""

    confidence: float = 1.0
    """Average segmentation confidence for this object."""


@dataclass
class ObjectSegmentationResult:
    """Result of object segmentation stage."""

    masks_dir: str
    """Directory containing per-object mask directories."""

    num_objects: int
    """Number of unique objects detected and tracked."""

    objects: List[SegmentedObject] = field(default_factory=list)
    """List of segmented objects with their masks."""

    metadata_path: str = ""
    """Path to object_metadata.json with tracking info."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata."""


class ObjectSegmentationStage:
    """
    Stage 2: Segment objects in frames and track across video.

    This stage is CRITICAL for accurate hole-filling. Without object-aware
    segmentation, holes get filled with "texture soup" - averaged nearby colors
    that don't respect object boundaries.

    Process:
    1. Run SAM-2 on keyframes to detect all objects
    2. Track object identities across all frames using SAM-2's video tracking
    3. Generate per-object binary masks with consistent IDs

    Output: masks/{object_id}/*.png, object_metadata.json
    """

    def __init__(self, config: dict = None):
        """
        Initialize ObjectSegmentationStage.

        Args:
            config: Configuration options:
                - keyframe_interval: Sample every Nth frame for detection (default: 10)
                - min_object_size: Minimum pixels for valid object (default: 100)
                - model_size: SAM-2 model variant (default: "large")
                - allow_cpu: Allow CPU execution for testing (default: False)
        """
        self.config = config or {}
        self.keyframe_interval = self.config.get("keyframe_interval", 10)
        self.min_object_size = self.config.get("min_object_size", 100)
        self.quality_preset = self.config.get("quality_preset", "balanced")

        # Dual-model architecture: image model for Phase 2, video predictor for Phase 3
        self.image_model = None
        self.mask_generator = None
        self.video_predictor = None

        # Device selection with CPU fallback for testing
        self.allow_cpu = self.config.get("allow_cpu", False)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif self.allow_cpu:
            logger.warning(
                "[SEG-WARN-001] Running on CPU. This will be very slow. "
                "Use allow_cpu=True only for testing."
            )
            self.device = "cpu"
        else:
            raise RuntimeError(
                "[SEG-ERR-001] SAM-2 requires CUDA. No GPU detected.\n"
                "For testing without GPU, set allow_cpu=True in config."
            )

        # Cache directory for checkpoints
        self.cache_dir = Path(
            os.environ.get(
                "SAM2_CHECKPOINT_DIR",
                os.path.expanduser("~/.cache/brain-dance/sam2"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metrics for observability
        self.metrics = {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def _select_model_for_vram(self, requested_size: str) -> str:
        """
        Select the best model size based on available VRAM.

        If requested model fits, use it. Otherwise, select largest that fits.

        Args:
            requested_size: User-requested model size

        Returns:
            Selected model size (may differ from requested if VRAM insufficient)
        """
        if self.device == "cpu":
            logger.info("CPU mode: using requested model size without VRAM check")
            return requested_size

        # Get available VRAM
        available_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Reserve 10% or minimum 2GB for overhead (other operations, fragmentation)
        # This scales better across GPU sizes (8GB vs 48GB)
        overhead_gb = max(2.0, available_vram_gb * 0.1)
        usable_vram_gb = available_vram_gb - overhead_gb

        requested_vram = SAM2_MODELS[requested_size]["vram_gb"]

        if requested_vram <= usable_vram_gb:
            logger.info(
                f"Using requested model '{requested_size}' "
                f"({requested_vram}GB / {usable_vram_gb:.1f}GB available)"
            )
            return requested_size

        # Find largest model that fits
        sizes_by_vram = sorted(
            SAM2_MODELS.items(),
            key=lambda x: x[1]["vram_gb"],
            reverse=True,
        )

        for size, info in sizes_by_vram:
            if info["vram_gb"] <= usable_vram_gb:
                logger.warning(
                    f"[SEG-WARN-002] Requested '{requested_size}' needs {requested_vram}GB, "
                    f"but only {usable_vram_gb:.1f}GB available. "
                    f"Auto-selecting '{size}' ({info['vram_gb']}GB)."
                )
                return size

        raise RuntimeError(
            f"[SEG-ERR-002] Insufficient VRAM. Need at least 4GB for 'tiny' model, "
            f"but only {usable_vram_gb:.1f}GB available."
        )

    def _get_checkpoint_path(self, model_size: str) -> Path:
        """
        Get path to model checkpoint, downloading if necessary.

        Args:
            model_size: Model size name (tiny, small, base_plus, large)

        Returns:
            Path to checkpoint file
        """
        model_info = SAM2_MODELS[model_size]
        checkpoint_filename = model_info["checkpoint"]
        checkpoint_path = self.cache_dir / checkpoint_filename

        if checkpoint_path.exists():
            logger.info(f"Using cached checkpoint: {checkpoint_path}")
            return checkpoint_path

        # Need to download
        logger.info(f"Checkpoint not found at {checkpoint_path}")
        return self._download_checkpoint(model_size, checkpoint_path)

    def _download_checkpoint(self, model_size: str, target_path: Path) -> Path:
        """
        Download SAM-2 checkpoint with retry logic and optional checksum verification.

        Args:
            model_size: Model size name
            target_path: Path to save checkpoint

        Returns:
            Path to downloaded checkpoint
        """
        import urllib.request

        model_info = SAM2_MODELS[model_size]
        url = model_info["url"]
        expected_sha256 = model_info.get("sha256")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Downloading SAM-2 {model_size} (attempt {attempt + 1}/{max_retries})"
                )
                logger.info(f"URL: {url}")

                # Download with progress
                temp_path = target_path.with_suffix(".tmp")

                def report_progress(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(100, (block_num * block_size / total_size) * 100)
                        if block_num % 500 == 0:
                            logger.info(f"Download progress: {percent:.1f}%")

                # Set socket timeout to prevent indefinite hangs on slow/stalled connections
                import socket
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(300)  # 5 minutes for large model files
                try:
                    urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)
                finally:
                    socket.setdefaulttimeout(old_timeout)

                # Verify checksum if available
                if expected_sha256:
                    logger.info("Verifying checkpoint integrity...")
                    actual_sha256 = self._compute_sha256(temp_path)
                    if actual_sha256 != expected_sha256:
                        temp_path.unlink()
                        raise ValueError(
                            f"[SEG-ERR-003] Checksum mismatch. "
                            f"Expected {expected_sha256[:16]}..., "
                            f"got {actual_sha256[:16]}..."
                        )
                    logger.info("Checksum verified successfully")

                # Move to final location
                temp_path.rename(target_path)
                logger.info(f"Checkpoint saved to {target_path}")
                return target_path

            except Exception as e:
                logger.error(f"Download failed: {e}")

                # Clean up temp file if exists
                temp_path = target_path.with_suffix(".tmp")
                if temp_path.exists():
                    temp_path.unlink()

                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"[SEG-ERR-004] Failed to download checkpoint after {max_retries} attempts.\n"
                        f"Manual download: {url}\n"
                        f"Save to: {target_path}"
                    ) from e

    def _compute_sha256(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            Hex string of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_quality_preset(self) -> dict:
        """
        Get mask generator parameters based on quality preset.

        Returns:
            Dictionary with points_per_side, pred_iou_thresh, stability_score_thresh
        """
        preset_name = self.quality_preset
        if preset_name not in SEGMENTATION_PRESETS:
            logger.warning(
                f"[SEG-WARN-003] Invalid quality preset '{preset_name}', using 'balanced'"
            )
            preset_name = "balanced"
        return SEGMENTATION_PRESETS[preset_name]

    def _validate_path(self, path: str, param_name: str) -> Path:
        """
        Validate and resolve a path, preventing path traversal attacks.

        Args:
            path: Path string to validate
            param_name: Parameter name for error messages

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or attempts traversal
        """
        resolved = Path(path).resolve()

        # Check for path traversal attempts
        if ".." in Path(path).parts:
            raise ValueError(
                f"[SEG-ERR-009] Invalid {param_name}: path traversal not allowed"
            )

        return resolved

    def _load_image_model(self) -> None:
        """
        Load SAM-2 image model and automatic mask generator for Phase 2.

        This model is used for keyframe segmentation to discover objects.
        Must be unloaded before loading video predictor to manage VRAM.

        Raises:
            RuntimeError: If SAM-2 not installed or model loading fails
        """
        if not SAM2_AVAILABLE:
            raise RuntimeError(
                "[SEG-ERR-005] SAM-2 not installed.\n"
                "Install: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        start_time = time.perf_counter()

        # Auto-select model based on VRAM
        requested_size = self.config.get("model_size", "large")
        if requested_size not in SAM2_MODELS:
            raise ValueError(
                f"[SEG-ERR-006] Invalid model size: {requested_size}. "
                f"Valid options: {list(SAM2_MODELS.keys())}"
            )

        selected_size = self._select_model_for_vram(requested_size)
        model_info = SAM2_MODELS[selected_size]

        # Get or download checkpoint
        checkpoint_path = self._get_checkpoint_path(selected_size)

        try:
            logger.info(f"Loading SAM-2 image model ({selected_size})...")

            # Build the base SAM-2 model
            self.image_model = build_sam2(
                config_file=model_info["config"],
                ckpt_path=str(checkpoint_path),
            )
            self.image_model.to(self.device)

            # Create mask generator with quality preset
            preset = self._get_quality_preset()
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.image_model,
                points_per_side=preset["points_per_side"],
                pred_iou_thresh=preset["pred_iou_thresh"],
                stability_score_thresh=preset["stability_score_thresh"],
                min_mask_region_area=self.min_object_size,
            )

            load_time = time.perf_counter() - start_time

            # Record metrics
            self.metrics["image_model_load_time_seconds"] = load_time
            self.metrics["model_size"] = selected_size
            self.metrics["requested_model_size"] = requested_size
            self.metrics["quality_preset"] = self.quality_preset
            self.metrics["device"] = self.device

            if torch.cuda.is_available():
                self.metrics["image_model_vram_allocated_gb"] = (
                    torch.cuda.memory_allocated() / 1e9
                )
                self.metrics["image_model_vram_reserved_gb"] = (
                    torch.cuda.memory_reserved() / 1e9
                )

            logger.info(
                f"SAM-2 image model loaded in {load_time:.2f}s on {self.device}"
            )
            logger.info(f"Quality preset: {self.quality_preset} ({preset})")

        except Exception as e:
            raise RuntimeError(
                f"[SEG-ERR-007] Failed to load SAM-2 image model.\n"
                f"Model: {selected_size}\n"
                f"Config: {model_info['config']}\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Error: {e}"
            ) from e

    def _unload_image_model(self) -> None:
        """
        Unload image model and mask generator to free VRAM.

        CRITICAL: Must be called after Phase 2 (keyframe segmentation)
        and before Phase 3 (video tracking) to avoid OOM on 24GB GPUs.
        """
        if self.mask_generator is not None:
            del self.mask_generator
            self.mask_generator = None
            logger.info("Mask generator unloaded")

        if self.image_model is not None:
            del self.image_model
            self.image_model = None
            logger.info("SAM-2 image model unloaded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleared after image model unload")

    def _load_video_predictor(self) -> None:
        """
        Load SAM-2 video predictor model for Phase 3 tracking.

        This model is used for object tracking across frames.
        Should only be loaded after image model is unloaded to manage VRAM.

        Raises:
            RuntimeError: If SAM-2 not installed or model loading fails
        """
        if not SAM2_AVAILABLE:
            raise RuntimeError(
                "[SEG-ERR-005] SAM-2 not installed.\n"
                "Install: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        start_time = time.perf_counter()

        # Auto-select model based on VRAM
        requested_size = self.config.get("model_size", "large")
        if requested_size not in SAM2_MODELS:
            raise ValueError(
                f"[SEG-ERR-006] Invalid model size: {requested_size}. "
                f"Valid options: {list(SAM2_MODELS.keys())}"
            )

        selected_size = self._select_model_for_vram(requested_size)
        model_info = SAM2_MODELS[selected_size]

        # Get or download checkpoint
        checkpoint_path = self._get_checkpoint_path(selected_size)

        try:
            logger.info(f"Loading SAM-2 video predictor ({selected_size})...")

            self.video_predictor = build_sam2_video_predictor(
                config_file=model_info["config"],
                ckpt_path=str(checkpoint_path),
                device=self.device,
            )

            load_time = time.perf_counter() - start_time

            # Record metrics for observability
            self.metrics["video_predictor_load_time_seconds"] = load_time
            self.metrics["model_size"] = selected_size
            self.metrics["requested_model_size"] = requested_size
            self.metrics["device"] = self.device

            if torch.cuda.is_available():
                self.metrics["video_predictor_vram_allocated_gb"] = (
                    torch.cuda.memory_allocated() / 1e9
                )
                self.metrics["video_predictor_vram_reserved_gb"] = (
                    torch.cuda.memory_reserved() / 1e9
                )

            logger.info(
                f"SAM-2 video predictor loaded in {load_time:.2f}s on {self.device}"
            )

        except Exception as e:
            raise RuntimeError(
                f"[SEG-ERR-007] Failed to load SAM-2 video predictor.\n"
                f"Model: {selected_size}\n"
                f"Config: {model_info['config']}\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Error: {e}"
            ) from e

    def _unload_video_predictor(self) -> None:
        """Unload video predictor to free VRAM."""
        if self.video_predictor is not None:
            del self.video_predictor
            self.video_predictor = None
            logger.info("SAM-2 video predictor unloaded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleared after video predictor unload")

    def cleanup(self) -> None:
        """Clean up all models and free GPU memory."""
        # Unload image model if loaded
        if self.mask_generator is not None:
            del self.mask_generator
            self.mask_generator = None

        if self.image_model is not None:
            del self.image_model
            self.image_model = None
            logger.info("SAM-2 image model unloaded")

        # Unload video predictor if loaded
        if self.video_predictor is not None:
            del self.video_predictor
            self.video_predictor = None
            logger.info("SAM-2 video predictor unloaded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleared")

    def segment(
        self,
        frames_dir: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ObjectSegmentationResult:
        """
        Segment and track objects across video frames.

        Uses dual-model architecture:
        - Phase 2: Load image model → segment keyframes → unload
        - Phase 3: Load video predictor → track objects → unload

        Args:
            frames_dir: Directory containing extracted frames.
            output_dir: Directory to store segmentation outputs.
            progress_callback: Optional callback(progress, message).

        Returns:
            ObjectSegmentationResult with paths to masks and metadata.

        Note:
            For videos >300 frames on 24GB GPUs, consider reducing frame count
            or using a smaller model size to avoid OOM errors.
        """

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct*100:.0f}%] {msg}")

        # Validate and resolve paths (security: prevent path traversal)
        frames_path = self._validate_path(frames_dir, "frames_dir")
        output_path = self._validate_path(output_dir, "output_dir")
        output_path.mkdir(parents=True, exist_ok=True)

        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Get sorted frame files
        frame_files = sorted(frames_path.glob("*.jpg")) + sorted(
            frames_path.glob("*.png")
        )
        if not frame_files:
            raise ValueError(f"[SEG-ERR-012] No frames found in {frames_dir}")

        report(0.0, f"Found {len(frame_files)} frames for segmentation")

        # ===== PHASE 2: Keyframe Segmentation =====
        # Load image model with mask generator
        report(0.05, "Loading SAM-2 image model for keyframe segmentation")
        self._load_image_model()

        try:
            # Sample keyframes (always includes first and last)
            keyframe_indices = self._sample_keyframes(frame_files)
            report(0.1, f"Selected {len(keyframe_indices)} keyframes for detection")

            # Segment keyframes to discover objects
            def keyframe_progress(pct: float, msg: str):
                # Map keyframe progress (0-1) to overall progress (0.1-0.3)
                overall_pct = 0.1 + pct * 0.2
                report(overall_pct, msg)

            initial_detections = self._segment_keyframes(
                frame_files, keyframe_indices, keyframe_progress
            )
            report(0.3, f"Discovered {len(initial_detections)} unique objects in keyframes")

        finally:
            # CRITICAL: Unload image model to free VRAM before loading video predictor
            self._unload_image_model()
            report(0.32, "Image model unloaded, VRAM freed")

        # Handle case where no objects detected
        if not initial_detections:
            logger.warning(
                "[SEG-WARN-006] No objects detected. Returning empty result."
            )
            metadata_path = output_path / "object_metadata.json"
            self._save_metadata([], metadata_path)

            return ObjectSegmentationResult(
                masks_dir=str(masks_dir),
                num_objects=0,
                objects=[],
                metadata_path=str(metadata_path),
                metadata={
                    "num_frames": len(frame_files),
                    "keyframe_interval": self.keyframe_interval,
                    "model_size": self.metrics.get("model_size", "unknown"),
                    "quality_preset": self.quality_preset,
                    "device": self.device,
                    "warning": "No objects detected in video",
                },
            )

        # ===== PHASE 3: Object Tracking =====
        report(0.35, "Loading SAM-2 video predictor for object tracking")
        self._load_video_predictor()

        try:
            report(0.4, "Tracking objects across all frames")
            tracked_objects = self._track_objects(
                frame_files,
                initial_detections,
                output_path,  # Pass output_path, not masks_dir
                progress_callback=lambda p, m: report(0.4 + p * 0.55, m),
            )
            report(0.95, f"Tracked {len(tracked_objects)} objects across {len(frame_files)} frames")

        finally:
            # Cleanup video predictor
            self._unload_video_predictor()

        # Metadata is now saved by _track_objects with enhanced format
        metadata_path = output_path / "object_metadata.json"

        report(1.0, "Object segmentation complete")

        return ObjectSegmentationResult(
            masks_dir=str(masks_dir),
            num_objects=len(tracked_objects),
            objects=tracked_objects,
            metadata_path=str(metadata_path),
            metadata={
                "num_frames": len(frame_files),
                "keyframe_interval": self.keyframe_interval,
                "num_keyframes": len(keyframe_indices),
                "model_size": self.metrics.get("model_size", "unknown"),
                "quality_preset": self.quality_preset,
                "device": self.device,
            },
        )

    def _sample_keyframes(self, frame_files: List[Path]) -> List[int]:
        """
        Select keyframe indices for object detection.

        Always includes first and last frames for boundary coverage.
        If keyframe_interval is larger than num_frames, returns all frames.

        Args:
            frame_files: List of frame file paths

        Returns:
            Sorted list of keyframe indices
        """
        num_frames = len(frame_files)

        if num_frames == 0:
            return []

        if num_frames <= 2:
            # Very short video: use all frames
            return list(range(num_frames))

        # Always include first and last frame
        keyframes = {0, num_frames - 1}

        # Add intermediate keyframes at regular intervals
        for idx in range(self.keyframe_interval, num_frames - 1, self.keyframe_interval):
            keyframes.add(idx)

        return sorted(keyframes)

    def _segment_single_keyframe(
        self, image_path: Path, keyframe_idx: int
    ) -> List[dict]:
        """
        Run automatic mask generation on a single keyframe.

        Args:
            image_path: Path to the frame image
            keyframe_idx: Index of this keyframe in the video

        Returns:
            List of detection dictionaries with mask, bbox, area, confidence, frame_idx
        """
        if self.mask_generator is None:
            raise RuntimeError(
                "[SEG-ERR-008] Mask generator not initialized. "
                "Call _load_image_model() first."
            )

        # Load image as RGB numpy array
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Validate image format
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise RuntimeError(
                f"[SEG-ERR-011] Expected RGB image, got shape {image_np.shape}"
            )

        # Run mask generation with mixed precision for efficiency
        with torch.cuda.amp.autocast():
            masks = self.mask_generator.generate(image_np)

        # Convert to detection dictionaries
        detections = []
        for mask_info in masks:
            # Filter by minimum object size (already done by mask_generator,
            # but double-check in case of API changes)
            if mask_info["area"] < self.min_object_size:
                continue

            detections.append({
                "mask": mask_info["segmentation"],  # Binary mask (H, W)
                "bbox": mask_info["bbox"],  # [x, y, width, height]
                "area": mask_info["area"],  # Pixel count
                "confidence": mask_info["stability_score"],  # 0-1
                "frame_idx": keyframe_idx,
            })

        return detections

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two binary masks.

        Args:
            mask1: First binary mask (H, W)
            mask2: Second binary mask (H, W)

        Returns:
            IoU value between 0 and 1
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)

    def _merge_keyframe_detections(
        self, detections: List[dict], iou_threshold: float = 0.5
    ) -> List[dict]:
        """
        Merge overlapping detections across keyframes.

        Uses IoU-based clustering to identify the same object detected
        in multiple keyframes. Keeps the detection with highest confidence.

        Args:
            detections: List of all detections from all keyframes
            iou_threshold: Minimum IoU to consider detections as same object

        Returns:
            Deduplicated list of detections
        """
        if not detections:
            return []

        # Sort by confidence descending (keep best detection per cluster)
        sorted_detections = sorted(
            detections, key=lambda d: d["confidence"], reverse=True
        )

        merged = []
        used = [False] * len(sorted_detections)

        for i, det_i in enumerate(sorted_detections):
            if used[i]:
                continue

            # This detection is a new cluster representative
            merged.append(det_i)
            used[i] = True

            # Find all overlapping detections and mark as used
            for j, det_j in enumerate(sorted_detections):
                if used[j] or i == j:
                    continue

                # Only compare detections from different keyframes
                if det_i["frame_idx"] == det_j["frame_idx"]:
                    continue

                iou = self._compute_mask_iou(det_i["mask"], det_j["mask"])
                if iou >= iou_threshold:
                    used[j] = True

        logger.info(
            f"Merged {len(detections)} detections into {len(merged)} unique objects"
        )
        return merged

    def _segment_keyframes(
        self,
        frame_files: List[Path],
        keyframe_indices: List[int],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[dict]:
        """
        Run automatic mask generation on keyframes to discover objects.

        Uses SAM-2's automatic mask generator to find all objects in keyframes,
        then merges overlapping detections across keyframes.

        Args:
            frame_files: List of all frame file paths
            keyframe_indices: Indices of keyframes to process
            progress_callback: Optional callback for progress reporting

        Returns:
            List of unique object detections (dict with mask, bbox, area, confidence, frame_idx)
        """
        if self.mask_generator is None:
            raise RuntimeError(
                "[SEG-ERR-008] Mask generator not initialized. "
                "Call _load_image_model() first."
            )

        all_detections = []
        num_keyframes = len(keyframe_indices)

        for i, keyframe_idx in enumerate(keyframe_indices):
            if keyframe_idx >= len(frame_files):
                logger.warning(
                    f"[SEG-WARN-004] Keyframe index {keyframe_idx} out of range, skipping"
                )
                continue

            frame_path = frame_files[keyframe_idx]

            # Report progress per keyframe
            if progress_callback:
                progress = i / num_keyframes
                progress_callback(progress, f"Segmenting keyframe {i+1}/{num_keyframes}")

            logger.info(f"Segmenting keyframe {i+1}/{num_keyframes}: {frame_path.name}")

            try:
                detections = self._segment_single_keyframe(frame_path, keyframe_idx)
                all_detections.extend(detections)
                logger.info(f"  Found {len(detections)} objects in keyframe {keyframe_idx}")

            except Exception as e:
                # Fail-fast for prototype: any keyframe failure raises error
                raise RuntimeError(
                    f"[SEG-ERR-010] Failed to segment keyframe {keyframe_idx} ({frame_path}).\n"
                    f"Error: {e}"
                ) from e

        # Handle case where no objects were detected
        if not all_detections:
            logger.warning(
                "[SEG-WARN-005] No objects detected in any keyframe. "
                "Video may be empty or min_object_size threshold too high."
            )
            return []

        # Merge overlapping detections across keyframes
        merged_detections = self._merge_keyframe_detections(all_detections)

        if progress_callback:
            progress_callback(1.0, f"Discovered {len(merged_detections)} unique objects")

        return merged_detections

    # =========================================================================
    # Phase 3: Video Object Tracking Helper Methods
    # =========================================================================

    def _mask_to_bbox(self, mask: np.ndarray) -> List[float]:
        """
        Extract bounding box from binary mask in XYXY format.

        Args:
            mask: Binary mask array (H, W)

        Returns:
            Bounding box as [x1, y1, x2, y2] in pixel coordinates,
            or [0, 0, 0, 0] if mask is empty
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return [0.0, 0.0, 0.0, 0.0]

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return [float(x1), float(y1), float(x2), float(y2)]

    def _mask_to_centroid(self, mask: np.ndarray) -> List[float]:
        """
        Compute centroid (center of mass) of binary mask.

        Args:
            mask: Binary mask array (H, W)

        Returns:
            Centroid as [cx, cy] in pixel coordinates,
            or center of image if mask is empty
        """
        if not mask.any():
            h, w = mask.shape
            return [float(w / 2), float(h / 2)]

        # Compute center of mass
        rows, cols = np.where(mask)
        cy = float(np.mean(rows))
        cx = float(np.mean(cols))

        return [cx, cy]

    def _normalize_bbox(
        self, bbox: List[float], img_height: int, img_width: int
    ) -> List[float]:
        """
        Normalize bounding box coordinates to 0-1 range.

        Args:
            bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates
            img_height: Image height in pixels
            img_width: Image width in pixels

        Returns:
            Normalized bounding box as [x1, y1, x2, y2] in 0-1 range
        """
        x1, y1, x2, y2 = bbox
        return [
            x1 / img_width,
            y1 / img_height,
            x2 / img_width,
            y2 / img_height,
        ]

    def _init_video_state(self, frames_dir: Path) -> dict:
        """
        Initialize SAM-2 video predictor state with frame directory.

        Uses directory path for memory efficiency (SAM-2 loads frames on-demand).

        Args:
            frames_dir: Directory containing video frames (*.jpg or *.png)

        Returns:
            SAM-2 inference state dictionary

        Raises:
            RuntimeError: If video predictor not loaded or initialization fails
        """
        if self.video_predictor is None:
            raise RuntimeError(
                "[SEG-ERR-013] Video predictor not initialized. "
                "Call _load_video_predictor() first."
            )

        logger.info(f"Initializing video state from {frames_dir}")
        start_time = time.perf_counter()

        try:
            # Use autocast for BFloat16 compatibility with SAM-2 video predictor
            # See: https://github.com/facebookresearch/sam2/issues/577
            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device == "cuda"
                else nullcontext()
            )

            with autocast_ctx:
                # Use directory path - SAM-2 loads frames on demand
                inference_state = self.video_predictor.init_state(
                    video_path=str(frames_dir),
                    offload_video_to_cpu=self.config.get("offload_video_to_cpu", False),
                    offload_state_to_cpu=self.config.get("offload_state_to_cpu", False),
                )

            init_time = time.perf_counter() - start_time
            self.metrics["video_state_init_time_seconds"] = init_time
            logger.info(f"Video state initialized in {init_time:.2f}s")

            return inference_state

        except Exception as e:
            raise RuntimeError(
                f"[SEG-ERR-014] Failed to initialize video state.\n"
                f"Frames directory: {frames_dir}\n"
                f"Error: {e}"
            ) from e

    def _add_object_prompts(
        self, inference_state: dict, detections: List[dict]
    ) -> dict:
        """
        Add bounding box prompts from keyframe detections to video state.

        Converts Phase 2 mask detections to bounding box prompts for SAM-2
        video tracking. Each detection becomes a tracked object.

        Args:
            inference_state: SAM-2 video inference state from _init_video_state()
            detections: List of detection dicts from Phase 2 keyframe segmentation
                Each dict has: mask, bbox (XYWH), area, confidence, frame_idx

        Returns:
            Dictionary mapping object_id to detection info for reference
        """
        logger.info(f"Adding {len(detections)} object prompts from keyframes")

        object_info = {}

        # Use autocast for BFloat16 compatibility with SAM-2 video predictor
        # See: https://github.com/facebookresearch/sam2/issues/577
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )

        with autocast_ctx:
            for obj_id, detection in enumerate(detections):
                frame_idx = detection["frame_idx"]
                mask = detection["mask"]
                img_height, img_width = mask.shape

                # Convert bbox from XYWH to XYXY format
                x, y, w, h = detection["bbox"]
                bbox_xyxy = [x, y, x + w, y + h]

                # Normalize to 0-1 range for SAM-2
                bbox_normalized = self._normalize_bbox(bbox_xyxy, img_height, img_width)

                try:
                    # Add bounding box prompt to video predictor
                    _, out_obj_ids, out_masks = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        box=np.array(bbox_normalized),
                        normalize_coords=True,
                    )

                    object_info[obj_id] = {
                        "source_frame": frame_idx,
                        "source_confidence": detection["confidence"],
                        "source_bbox": detection["bbox"],
                    }

                    logger.debug(
                        f"Added prompt for object {obj_id} at frame {frame_idx} "
                        f"with bbox {bbox_normalized}"
                    )

                except Exception as e:
                    logger.error(
                        f"[SEG-ERR-015] Failed to add prompt for object {obj_id} "
                        f"at frame {frame_idx}: {e}"
                    )
                    raise

        logger.info(f"Successfully added {len(object_info)} object prompts")
        return object_info

    def _propagate_masks(
        self,
        inference_state: dict,
        num_frames: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> dict:
        """
        Propagate masks through all video frames using SAM-2 video tracking.

        Iterates through the SAM-2 propagation generator, collecting masks
        and computing per-frame metadata (bbox, centroid, area) for Stage 4.

        Args:
            inference_state: SAM-2 video inference state with prompts added
            num_frames: Total number of frames in video
            progress_callback: Optional callback(progress, message)

        Returns:
            Dictionary mapping frame_idx -> list of object data dicts:
                [{object_id, mask, bbox, centroid, area}, ...]
        """
        logger.info(f"Propagating masks through {num_frames} frames")
        start_time = time.perf_counter()

        all_masks = {}
        frames_processed = 0

        # Use autocast for BFloat16 compatibility with SAM-2 video predictor
        # See: https://github.com/facebookresearch/sam2/issues/577
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )

        try:
            with autocast_ctx:
                for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(
                    inference_state=inference_state,
                ):
                    frames_processed += 1

                    if progress_callback:
                        progress = frames_processed / num_frames
                        progress_callback(
                            progress, f"Tracking frame {frames_processed}/{num_frames}"
                        )

                    frame_data = []

                    # Process each object's mask for this frame
                    for i, obj_id in enumerate(obj_ids):
                        # Convert logits to binary mask
                        # mask_logits shape: (num_objects, 1, H, W); squeeze to (H, W)
                        mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()

                        # Skip empty masks (object not visible in this frame)
                        if not mask.any():
                            continue

                        # Compute per-frame metadata for Stage 4
                        bbox = self._mask_to_bbox(mask)
                        centroid = self._mask_to_centroid(mask)
                        area = int(mask.sum())

                        frame_data.append({
                            "object_id": int(obj_id),
                            "mask": mask,
                            "bbox": bbox,
                            "centroid": centroid,
                            "area": area,
                        })

                    if frame_data:
                        all_masks[frame_idx] = frame_data

            propagate_time = time.perf_counter() - start_time
            self.metrics["propagation_time_seconds"] = propagate_time
            self.metrics["frames_tracked"] = frames_processed

            fps = frames_processed / propagate_time if propagate_time > 0 else 0
            logger.info(
                f"Propagation complete: {frames_processed} frames in {propagate_time:.2f}s "
                f"({fps:.1f} fps)"
            )

            return all_masks

        except Exception as e:
            raise RuntimeError(
                f"[SEG-ERR-016] Mask propagation failed at frame {frames_processed}.\n"
                f"Error: {e}"
            ) from e

    def _validate_tracking_quality(self, all_masks: dict) -> dict:
        """
        Validate tracking quality by computing temporal IoU between adjacent frames.

        Emits warnings (not errors) for quality issues to allow pipeline to continue.

        Args:
            all_masks: Dictionary from _propagate_masks() mapping frame_idx -> object data

        Returns:
            Quality metrics dictionary with mean_iou, min_iou, warnings
        """
        logger.info("Validating tracking quality")

        quality_metrics = {
            "mean_iou": 0.0,
            "min_iou": 1.0,
            "per_object_iou": {},
            "warnings": [],
        }

        if len(all_masks) < 2:
            logger.warning("[SEG-WARN-010] Not enough frames for quality validation")
            return quality_metrics

        # Group masks by object ID across frames
        object_tracks = {}  # obj_id -> [(frame_idx, mask), ...]
        for frame_idx, frame_data in all_masks.items():
            for obj_info in frame_data:
                obj_id = obj_info["object_id"]
                if obj_id not in object_tracks:
                    object_tracks[obj_id] = []
                object_tracks[obj_id].append((frame_idx, obj_info["mask"]))

        all_ious = []

        for obj_id, track in object_tracks.items():
            if len(track) < 2:
                continue

            # Sort by frame index
            track.sort(key=lambda x: x[0])
            object_ious = []

            # Compute IoU between adjacent frames
            for i in range(len(track) - 1):
                frame1, mask1 = track[i]
                frame2, mask2 = track[i + 1]

                # Only compute for truly adjacent frames
                if frame2 - frame1 == 1:
                    iou = self._compute_mask_iou(mask1, mask2)
                    object_ious.append(iou)
                    all_ious.append(iou)

                    # Emit warning for low IoU
                    if iou < 0.7:
                        warning = (
                            f"Object {obj_id}: low IoU ({iou:.2f}) "
                            f"between frames {frame1}-{frame2}"
                        )
                        quality_metrics["warnings"].append(warning)
                        logger.warning(f"[SEG-WARN-011] {warning}")

            if object_ious:
                quality_metrics["per_object_iou"][obj_id] = {
                    "mean": float(np.mean(object_ious)),
                    "min": float(np.min(object_ious)),
                    "max": float(np.max(object_ious)),
                }

        if all_ious:
            quality_metrics["mean_iou"] = float(np.mean(all_ious))
            quality_metrics["min_iou"] = float(np.min(all_ious))

        # Log summary
        if quality_metrics["warnings"]:
            logger.warning(
                f"[SEG-WARN-012] Quality validation found {len(quality_metrics['warnings'])} "
                f"warnings. Pipeline will continue."
            )
        else:
            logger.info("Quality validation passed - no warnings")

        logger.info(
            f"Quality metrics: mean_iou={quality_metrics['mean_iou']:.3f}, "
            f"min_iou={quality_metrics['min_iou']:.3f}"
        )

        return quality_metrics

    def _export_tracks(
        self,
        all_masks: dict,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> tuple:
        """
        Export tracked masks to disk and generate enhanced metadata.

        Creates directory structure: masks/{object_id}/{frame_idx:04d}.png
        Generates enhanced metadata with per-frame bbox, centroid, area for Stage 4.

        Args:
            all_masks: Dictionary from _propagate_masks()
            output_dir: Base output directory
            progress_callback: Optional callback(progress, message)

        Returns:
            Tuple of (List[SegmentedObject], enhanced_metadata_dict)
        """
        logger.info(f"Exporting tracks to {output_dir}")
        start_time = time.perf_counter()

        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Reorganize by object ID
        object_data = {}  # obj_id -> {frames: [], per_frame_data: {}}
        frame_to_objects = {}  # frame_idx -> [obj_ids]

        total_masks = sum(len(frame_data) for frame_data in all_masks.values())
        masks_saved = 0

        for frame_idx, frame_data in all_masks.items():
            frame_to_objects[frame_idx] = []

            for obj_info in frame_data:
                obj_id = obj_info["object_id"]
                frame_to_objects[frame_idx].append(obj_id)

                if obj_id not in object_data:
                    object_data[obj_id] = {
                        "frames": [],
                        "per_frame_data": {},
                        "mask_paths": [],
                    }

                # Track frame data
                object_data[obj_id]["frames"].append(frame_idx)
                object_data[obj_id]["per_frame_data"][str(frame_idx)] = {
                    "bbox": obj_info["bbox"],
                    "centroid": obj_info["centroid"],
                    "area": obj_info["area"],
                }

                # Save mask PNG
                obj_dir = masks_dir / str(obj_id)
                obj_dir.mkdir(exist_ok=True)
                mask_filename = f"{frame_idx:04d}.png"
                mask_path = obj_dir / mask_filename

                self._save_mask(obj_info["mask"], mask_path)
                object_data[obj_id]["mask_paths"].append(str(mask_path))

                masks_saved += 1
                if progress_callback and masks_saved % 50 == 0:
                    progress = masks_saved / total_masks
                    progress_callback(progress, f"Saving masks ({masks_saved}/{total_masks})")

        # Build SegmentedObject list
        segmented_objects = []
        for obj_id in sorted(object_data.keys()):
            data = object_data[obj_id]
            # Sort frames for consistency
            sorted_frames = sorted(data["frames"])

            obj = SegmentedObject(
                object_id=obj_id,
                label=None,  # No semantic labels in MVP
                mask_paths=data["mask_paths"],
                frame_indices=sorted_frames,
                confidence=1.0,  # Could compute from tracking quality
            )
            segmented_objects.append(obj)

        # Build enhanced metadata for Stage 4
        enhanced_metadata = {
            "num_frames": len(all_masks),
            "frame_to_objects": {
                str(k): v for k, v in frame_to_objects.items()
            },
            "per_object_frame_data": {
                str(obj_id): data["per_frame_data"]
                for obj_id, data in object_data.items()
            },
        }

        export_time = time.perf_counter() - start_time
        self.metrics["export_time_seconds"] = export_time
        self.metrics["masks_saved"] = masks_saved

        logger.info(
            f"Exported {len(segmented_objects)} objects, {masks_saved} masks "
            f"in {export_time:.2f}s"
        )

        return segmented_objects, enhanced_metadata

    def _save_mask(self, mask: np.ndarray, path: Path) -> None:
        """
        Save binary mask as 8-bit grayscale PNG.

        Args:
            mask: Binary mask array (H, W) with dtype bool or 0/1 values
            path: Output path for PNG file
        """
        # Convert to 8-bit (0 = background, 255 = object)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        img = Image.fromarray(mask_uint8, mode="L")
        img.save(path, compress_level=6)

    def _save_enhanced_metadata(
        self,
        objects: List[SegmentedObject],
        metadata_path: Path,
        quality_metrics: dict,
        enhanced_metadata: dict,
    ) -> dict:
        """
        Save enhanced metadata JSON for Stage 4 compatibility.

        Includes per-frame spatial data (bbox, centroid, area) and
        frame-to-objects mapping for efficient spatial queries.

        Args:
            objects: List of SegmentedObject
            metadata_path: Path to save JSON file
            quality_metrics: Quality metrics from _validate_tracking_quality()
            enhanced_metadata: Enhanced metadata from _export_tracks()

        Returns:
            Complete metadata dictionary
        """
        metadata = {
            "num_objects": len(objects),
            "num_frames": enhanced_metadata.get("num_frames", 0),
            "objects": [
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "num_frames": len(obj.frame_indices),
                    "frame_indices": obj.frame_indices,
                    "confidence": obj.confidence,
                    "per_frame_data": enhanced_metadata.get("per_object_frame_data", {}).get(
                        str(obj.object_id), {}
                    ),
                }
                for obj in objects
            ],
            "frame_to_objects": enhanced_metadata.get("frame_to_objects", {}),
            "quality_metrics": quality_metrics,
            "metrics": self.metrics,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved enhanced metadata to {metadata_path}")
        return metadata

    def _track_objects(
        self,
        frame_files: List[Path],
        initial_detections: List[dict],
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[SegmentedObject]:
        """
        Track objects across all frames using SAM-2 video predictor.

        SAM-2's video tracking propagates masks forward from keyframe prompts,
        maintaining consistent object identities throughout the video.

        Process:
        1. Initialize video state with frame directory
        2. Add bounding box prompts from keyframe detections
        3. Propagate masks through all frames
        4. Validate tracking quality (emit warnings, not errors)
        5. Export masks to disk with enhanced metadata

        Args:
            frame_files: List of all frame file paths
            initial_detections: List of detection dicts from Phase 2 keyframe segmentation
                Each dict has: mask, bbox, area, confidence, frame_idx
            output_dir: Directory to save mask PNGs
            progress_callback: Optional callback(progress, message)

        Returns:
            List of SegmentedObject with mask_paths and frame_indices populated

        Raises:
            RuntimeError: If tracking fails critically
        """
        logger.info(
            f"Starting object tracking: {len(initial_detections)} objects, "
            f"{len(frame_files)} frames"
        )
        start_time = time.perf_counter()

        # Get frames directory from first frame file
        frames_dir = frame_files[0].parent
        num_frames = len(frame_files)

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.debug(f"[{pct*100:.0f}%] {msg}")

        try:
            # Step 1: Initialize video state
            report(0.0, "Initializing video state")
            inference_state = self._init_video_state(frames_dir)

            # Step 2: Add object prompts from keyframe detections
            report(0.1, "Adding object prompts from keyframes")
            object_info = self._add_object_prompts(inference_state, initial_detections)

            # Warn about memory usage for long videos
            if num_frames > 500:
                # Get frame dimensions from first frame
                first_frame = Image.open(frame_files[0])
                img_width, img_height = first_frame.size
                first_frame.close()

                # Estimate: each mask is H*W bytes (bool array), one per object per frame
                num_objects = len(initial_detections)
                estimated_gb = (num_frames * num_objects * img_height * img_width) / (1024 ** 3)
                if estimated_gb > 4.0:
                    logger.warning(
                        f"[SEG-WARN-014] Long video detected: {num_frames} frames with "
                        f"{num_objects} objects may require ~{estimated_gb:.1f}GB RAM "
                        f"for mask storage. Consider reducing frame count or video resolution."
                    )

            # Step 3: Propagate masks through video
            report(0.15, "Propagating masks through video")
            all_masks = self._propagate_masks(
                inference_state,
                num_frames,
                lambda p, m: report(0.15 + p * 0.55, m),
            )

            # Handle case where no masks were propagated
            if not all_masks:
                logger.warning("[SEG-WARN-013] No masks propagated - objects may not be visible")
                return []

            # Step 4: Validate tracking quality
            report(0.70, "Validating tracking quality")
            quality_metrics = self._validate_tracking_quality(all_masks)

            # Step 5: Export masks and generate enhanced metadata
            report(0.75, "Exporting masks to disk")
            segmented_objects, enhanced_metadata = self._export_tracks(
                all_masks,
                output_dir,
                lambda p, m: report(0.75 + p * 0.20, m),
            )

            # Step 6: Save enhanced metadata
            report(0.95, "Saving metadata")
            metadata_path = output_dir / "object_metadata.json"
            self._save_enhanced_metadata(
                segmented_objects,
                metadata_path,
                quality_metrics,
                enhanced_metadata,
            )

            # Record total tracking time
            total_time = time.perf_counter() - start_time
            self.metrics["total_tracking_time_seconds"] = total_time

            report(1.0, f"Tracking complete: {len(segmented_objects)} objects")
            logger.info(
                f"Object tracking complete: {len(segmented_objects)} objects "
                f"tracked in {total_time:.2f}s"
            )

            return segmented_objects

        except Exception as e:
            logger.error(f"[SEG-ERR-017] Object tracking failed: {e}")
            raise RuntimeError(
                f"[SEG-ERR-017] Object tracking failed.\n"
                f"Frames directory: {frames_dir}\n"
                f"Initial detections: {len(initial_detections)}\n"
                f"Error: {e}"
            ) from e

    def _save_metadata(
        self, objects: List[SegmentedObject], metadata_path: Path
    ) -> None:
        """Save object metadata to JSON file."""
        metadata = {
            "num_objects": len(objects),
            "objects": [
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "num_frames": len(obj.frame_indices),
                    "frame_indices": obj.frame_indices,
                    "confidence": obj.confidence,
                }
                for obj in objects
            ],
            "metrics": self.metrics,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
