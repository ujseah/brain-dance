"""Stage 1.5: Object Segmentation - SAM-2 segmentation and tracking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List


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
    Stage 1.5: Segment objects in frames and track across video.

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
        self.config = config or {}
        self.keyframe_interval = self.config.get("keyframe_interval", 10)
        self.min_object_size = self.config.get("min_object_size", 100)  # pixels
        self.model_size = self.config.get("model_size", "large")  # tiny, small, base, large
        self._sam2_model = None
        self._sam2_predictor = None

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

        frames_path = Path(frames_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Get sorted frame files
        frame_files = sorted(frames_path.glob("*.jpg")) + sorted(frames_path.glob("*.png"))
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")

        report(0.0, f"Found {len(frame_files)} frames for segmentation")

        # Step 1: Load SAM-2 model
        report(0.05, "Loading SAM-2 model")
        self._load_sam2_model()

        # Step 2: Segment keyframes to discover objects
        report(0.1, "Segmenting keyframes to discover objects")
        keyframe_indices = range(0, len(frame_files), self.keyframe_interval)
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
                "model_size": self.model_size,
            },
        )

    def _load_sam2_model(self):
        """Load SAM-2 model for segmentation and tracking."""
        # TODO: Implement SAM-2 model loading
        # from sam2.build_sam import build_sam2_video_predictor
        # from sam2.sam2_image_predictor import SAM2ImagePredictor
        #
        # Model checkpoints:
        # - sam2_hiera_tiny.pt
        # - sam2_hiera_small.pt
        # - sam2_hiera_base_plus.pt
        # - sam2_hiera_large.pt
        #
        # self._sam2_predictor = build_sam2_video_predictor(
        #     config_file=f"sam2_hiera_{self.model_size}.yaml",
        #     ckpt_path=f"checkpoints/sam2_hiera_{self.model_size}.pt",
        # )
        raise NotImplementedError("SAM-2 model loading not yet implemented")

    def _segment_keyframes(
        self, frame_files: List[Path], keyframe_indices: range
    ) -> List[SegmentedObject]:
        """
        Run automatic mask generation on keyframes to discover objects.

        Uses SAM-2's automatic mask generator to find all objects in keyframes,
        then initializes tracking for each unique object.
        """
        # TODO: Implement keyframe segmentation
        # 1. For each keyframe:
        #    - Run SAM-2 automatic mask generation
        #    - Filter by min_object_size
        #    - Store initial masks and prompts for tracking
        # 2. Merge overlapping detections across keyframes
        # 3. Return list of unique objects to track
        raise NotImplementedError("Keyframe segmentation not yet implemented")

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
        """
        # TODO: Implement object tracking
        # 1. Initialize SAM-2 video predictor with video frames
        # 2. Add initial prompts (masks/points) for each object from keyframes
        # 3. Propagate masks through video
        # 4. Save masks to output_dir/{object_id}/{frame_idx:04d}.png
        # 5. Update SegmentedObject with mask_paths and frame_indices
        raise NotImplementedError("Object tracking not yet implemented")

    def _save_metadata(self, objects: List[SegmentedObject], metadata_path: Path):
        """Save object metadata to JSON file."""
        import json

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
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
