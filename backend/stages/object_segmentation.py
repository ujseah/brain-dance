"""Stage 2: Object Segmentation - SAM-2 segmentation and tracking."""

import hashlib
import json
import logging
import os
import time
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
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

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
        self.model = None

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

                urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)

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

    def _load_sam2_model(self) -> None:
        """
        Load SAM-2 video predictor model.

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
            logger.info(f"Loading SAM-2 {selected_size} model...")

            self.model = build_sam2_video_predictor(
                config_file=model_info["config"],
                ckpt_path=str(checkpoint_path),
                device=self.device,
            )

            load_time = time.perf_counter() - start_time

            # Record metrics for observability
            self.metrics["model_load_time_seconds"] = load_time
            self.metrics["model_size"] = selected_size
            self.metrics["requested_model_size"] = requested_size
            self.metrics["device"] = self.device

            if torch.cuda.is_available():
                self.metrics["vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                self.metrics["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

            logger.info(
                f"SAM-2 loaded successfully in {load_time:.2f}s on {self.device}"
            )
            logger.info(f"Metrics: {self.metrics}")

        except Exception as e:
            raise RuntimeError(
                f"[SEG-ERR-007] Failed to load SAM-2 model.\n"
                f"Model: {selected_size}\n"
                f"Config: {model_info['config']}\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Error: {e}"
            ) from e

    def cleanup(self) -> None:
        """Clean up model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("SAM-2 model unloaded")

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

        Args:
            frames_dir: Directory containing extracted frames.
            output_dir: Directory to store segmentation outputs.
            progress_callback: Optional callback(progress, message).

        Returns:
            ObjectSegmentationResult with paths to masks and metadata.
        """

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct*100:.0f}%] {msg}")

        frames_path = Path(frames_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Get sorted frame files
        frame_files = sorted(frames_path.glob("*.jpg")) + sorted(
            frames_path.glob("*.png")
        )
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")

        report(0.0, f"Found {len(frame_files)} frames for segmentation")

        # Step 1: Load SAM-2 model
        report(0.05, "Loading SAM-2 model")
        self._load_sam2_model()

        # Step 2: Segment keyframes to discover objects
        report(0.1, "Segmenting keyframes to discover objects")
        keyframe_indices = list(range(0, len(frame_files), self.keyframe_interval))
        initial_objects = self._segment_keyframes(frame_files, keyframe_indices)
        report(0.3, f"Discovered {len(initial_objects)} objects in keyframes")

        # Step 3: Track objects across all frames
        report(0.3, "Tracking objects across all frames")
        tracked_objects = self._track_objects(frame_files, initial_objects, masks_dir)
        report(0.9, f"Tracked {len(tracked_objects)} objects across {len(frame_files)} frames")

        # Step 4: Save metadata
        report(0.95, "Saving segmentation metadata")
        metadata_path = output_path / "object_metadata.json"
        self._save_metadata(tracked_objects, metadata_path)

        report(1.0, "Object segmentation complete")

        return ObjectSegmentationResult(
            masks_dir=str(masks_dir),
            num_objects=len(tracked_objects),
            objects=tracked_objects,
            metadata_path=str(metadata_path),
            metadata={
                "num_frames": len(frame_files),
                "keyframe_interval": self.keyframe_interval,
                "model_size": self.metrics.get("model_size", "unknown"),
                "device": self.device,
            },
        )

    def _segment_keyframes(
        self, frame_files: List[Path], keyframe_indices: List[int]
    ) -> List[SegmentedObject]:
        """
        Run automatic mask generation on keyframes to discover objects.

        Uses SAM-2's automatic mask generator to find all objects in keyframes,
        then initializes tracking for each unique object.

        Note: This is a Phase 2 implementation - currently raises NotImplementedError.
        """
        # TODO: Implement in Phase 2
        # 1. For each keyframe:
        #    - Run SAM-2 automatic mask generation
        #    - Filter by min_object_size
        #    - Store initial masks and prompts for tracking
        # 2. Merge overlapping detections across keyframes
        # 3. Return list of unique objects to track
        raise NotImplementedError(
            "Keyframe segmentation not yet implemented (Phase 2)"
        )

    def _track_objects(
        self,
        frame_files: List[Path],
        initial_objects: List[SegmentedObject],
        output_dir: Path,
    ) -> List[SegmentedObject]:
        """
        Track objects across all frames using SAM-2 video predictor.

        SAM-2's video tracking propagates masks forward and backward in time,
        maintaining consistent object identities.

        Note: This is a Phase 3 implementation - currently raises NotImplementedError.
        """
        # TODO: Implement in Phase 3
        # 1. Initialize SAM-2 video predictor with video frames
        # 2. Add initial prompts (masks/points) for each object from keyframes
        # 3. Propagate masks through video
        # 4. Save masks to output_dir/{object_id}/{frame_idx:04d}.png
        # 5. Update SegmentedObject with mask_paths and frame_indices
        raise NotImplementedError(
            "Object tracking not yet implemented (Phase 3)"
        )

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

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
