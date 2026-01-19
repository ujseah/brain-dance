"""Stage 1: Video Processing - Frame extraction and camera pose estimation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable


@dataclass
class VideoProcessingResult:
    """Result of video processing stage."""

    frames_dir: str
    """Directory containing extracted frames."""

    num_frames: int
    """Number of frames extracted."""

    transforms_path: str
    """Path to transforms.json with camera poses."""

    sparse_points_path: Optional[str] = None
    """Path to sparse point cloud (if available)."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata (fps, resolution, etc.)."""


class VideoProcessingStage:
    """
    Stage 1: Extract frames from video and estimate camera poses.

    This stage:
    1. Extracts frames from input video using ffmpeg
    2. Runs camera pose estimation (hloc + GLOMAP or DUSt3R fallback)
    3. Generates sparse point cloud via COLMAP SfM

    Output: frames/, transforms.json, sparse/points3D.ply
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.pose_estimator = self.config.get("pose_estimator", "hloc")
        self.frame_interval = self.config.get("frame_interval", 1)  # Extract every Nth frame
        self.max_frames = self.config.get("max_frames", 300)

    def process(
        self,
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VideoProcessingResult:
        """
        Process video to extract frames and estimate poses.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to store outputs.
            progress_callback: Optional callback(progress, message).

        Returns:
            VideoProcessingResult with paths to outputs.
        """

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frames_dir = output_path / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Step 1: Extract frames
        report(0.0, "Extracting frames from video")
        num_frames = self._extract_frames(video_path, frames_dir)
        report(0.3, f"Extracted {num_frames} frames")

        # Step 2: Estimate camera poses
        report(0.3, "Estimating camera poses")
        transforms_path = output_path / "transforms.json"
        sparse_path = output_path / "sparse"

        if self.pose_estimator == "hloc":
            self._run_hloc_pipeline(frames_dir, transforms_path, sparse_path)
        elif self.pose_estimator == "dust3r":
            self._run_dust3r_pipeline(frames_dir, transforms_path)
        else:
            raise ValueError(f"Unknown pose estimator: {self.pose_estimator}")

        report(1.0, "Video processing complete")

        return VideoProcessingResult(
            frames_dir=str(frames_dir),
            num_frames=num_frames,
            transforms_path=str(transforms_path),
            sparse_points_path=str(sparse_path / "points3D.ply") if sparse_path.exists() else None,
            metadata={
                "video_path": video_path,
                "pose_estimator": self.pose_estimator,
            },
        )

    def _extract_frames(self, video_path: str, output_dir: Path) -> int:
        """Extract frames from video using ffmpeg."""
        # TODO: Implement ffmpeg frame extraction
        # ffmpeg -i video.mp4 -vf "select=not(mod(n\,{interval}))" -vsync vfr frames/%04d.jpg
        raise NotImplementedError("Frame extraction not yet implemented")

    def _run_hloc_pipeline(self, frames_dir: Path, transforms_path: Path, sparse_path: Path):
        """Run hloc + GLOMAP for pose estimation."""
        # TODO: Implement hloc pipeline
        # 1. Extract features with SuperPoint
        # 2. Match features with LightGlue
        # 3. Run GLOMAP for global SfM
        # 4. Export to transforms.json (Nerfstudio format)
        raise NotImplementedError("hloc pipeline not yet implemented")

    def _run_dust3r_pipeline(self, frames_dir: Path, transforms_path: Path):
        """Run DUSt3R for pose estimation (fallback)."""
        # TODO: Implement DUSt3R pipeline
        # 1. Load DUSt3R model
        # 2. Process frame pairs
        # 3. Global alignment
        # 4. Export to transforms.json
        raise NotImplementedError("DUSt3R pipeline not yet implemented")
