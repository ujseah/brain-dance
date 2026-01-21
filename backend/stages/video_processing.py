"""Stage 1: Video Processing - Frame extraction and camera pose estimation."""

import subprocess
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


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

        # Step 2: Estimate camera poses with automatic fallback
        report(0.3, "Estimating camera poses")
        transforms_path = output_path / "transforms.json"
        sparse_path = output_path / "sparse"

        if self.pose_estimator == "hloc":
            try:
                self._run_hloc_pipeline(frames_dir, transforms_path, sparse_path)
                report(0.9, "hloc pipeline completed")
            except RuntimeError as e:
                # Automatic fallback to DUSt3R
                logger.warning(f"hloc pipeline failed: {e}")
                report(0.5, "Falling back to DUSt3R...")
                self._run_dust3r_pipeline(frames_dir, transforms_path)
                report(0.9, "DUSt3R fallback completed")
        elif self.pose_estimator == "dust3r":
            self._run_dust3r_pipeline(frames_dir, transforms_path)
            report(0.9, "DUSt3R pipeline completed")
        else:
            raise ValueError(f"Unknown pose estimator: {self.pose_estimator}")

        report(1.0, "Video processing complete")

        # Merge video metadata into result
        metadata = {
            "video_path": video_path,
            "pose_estimator": self.pose_estimator,
        }
        if hasattr(self, "_video_metadata"):
            metadata.update(self._video_metadata)

        # Check if sparse point cloud file exists (not just the directory)
        sparse_ply_path = sparse_path / "points3D.ply"

        return VideoProcessingResult(
            frames_dir=str(frames_dir),
            num_frames=num_frames,
            transforms_path=str(transforms_path),
            sparse_points_path=str(sparse_ply_path) if sparse_ply_path.exists() else None,
            metadata=metadata,
        )

    def _extract_frames(self, video_path: str, output_dir: Path) -> int:
        """
        Extract frames from video using ffmpeg.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to save extracted frames.

        Returns:
            Number of frames extracted.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Probe video to get metadata
        probe_cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            str(video_path),
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")

        probe_data = json.loads(probe_result.stdout)
        if not probe_data.get("streams"):
            raise ValueError(f"No video stream found in: {video_path}")

        video_stream = probe_data["streams"][0]
        self._video_metadata = {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
        }

        # Parse frame rate (can be "30/1" or "29.97")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        self._video_metadata["fps"] = fps

        # Get total frame count if available
        nb_frames = video_stream.get("nb_frames")
        if nb_frames:
            total_frames = int(nb_frames)
        else:
            # Estimate from duration
            duration = float(video_stream.get("duration", 0))
            total_frames = int(duration * fps)
        self._video_metadata["total_frames"] = total_frames

        # Calculate how many frames we'll actually extract
        frames_after_interval = (total_frames + self.frame_interval - 1) // self.frame_interval
        frames_to_extract = min(frames_after_interval, self.max_frames)

        # Build ffmpeg command
        # Use select filter to pick every Nth frame, limit total output
        output_pattern = str(output_dir / "%04d.jpg")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files
            "-i", str(video_path),
            "-vf", f"select=not(mod(n\\,{self.frame_interval}))",
            "-vsync", "vfr",  # Variable frame rate (only output selected frames)
            "-frames:v", str(frames_to_extract),  # Limit number of output frames
            "-q:v", "2",  # High quality JPEG (1-31, lower is better)
            output_pattern,
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        # Count actual extracted frames
        extracted_frames = list(output_dir.glob("*.jpg"))
        num_frames = len(extracted_frames)

        if num_frames == 0:
            raise RuntimeError("No frames were extracted from video")

        self._video_metadata["extracted_frames"] = num_frames
        return num_frames

    def _run_hloc_pipeline(self, frames_dir: Path, transforms_path: Path, sparse_path: Path):
        """
        Run hloc + GLOMAP pipeline for pose estimation.

        Pipeline:
        1. Extract SuperPoint features
        2. Generate frame pairs (sequential)
        3. Match features with LightGlue
        4. Run GLOMAP global SfM
        5. Validate quality metrics
        6. Export to transforms.json (Nerfstudio format)
        7. Export sparse point cloud
        """
        try:
            from hloc import extract_features, match_features, pairs_from_sequential, reconstruction
            import pycolmap
        except ImportError as e:
            raise ImportError(
                f"hloc dependencies not installed: {e}\n"
                "Install with: pip install hloc pycolmap"
            )

        sparse_path.mkdir(parents=True, exist_ok=True)
        sfm_dir = sparse_path / "0"

        # Step 1: Extract SuperPoint features
        logger.info("Extracting SuperPoint features...")
        feature_path = sparse_path / "features.h5"
        feature_conf = {
            'model': {
                'name': 'superpoint',
                'max_keypoints': 2048,
                'nms_radius': 3,
            }
        }

        extract_features.main(
            conf=feature_conf,
            image_dir=frames_dir,
            feature_path=feature_path
        )

        # Step 2: Generate sequential frame pairs
        logger.info("Generating frame pairs...")
        pairs_path = sparse_path / "pairs.txt"
        image_list = sorted(frames_dir.glob("*.jpg"))

        pairs_from_sequential.main(
            output=pairs_path,
            image_list=image_list,
            num_matched=10  # Match each frame with next 10 frames
        )

        # Step 3: Match features with LightGlue
        logger.info("Matching features with LightGlue...")
        match_path = sparse_path / "matches.h5"
        match_conf = {
            'model': {
                'name': 'lightglue',
                'features': 'superpoint',
                'depth_confidence': -1,
                'width_confidence': -1,
            }
        }

        match_features.main(
            conf=match_conf,
            pairs=pairs_path,
            features=feature_path,
            matches=match_path
        )

        # Step 4: Run GLOMAP for global SfM
        logger.info("Running GLOMAP reconstruction...")
        try:
            reconstruction.main(
                sfm_dir=sfm_dir,
                image_dir=frames_dir,
                pairs=pairs_path,
                features=feature_path,
                matches=match_path,
                mapper='glomap'
            )
        except Exception as e:
            logger.warning(f"GLOMAP failed: {e}, trying COLMAP...")
            # Fallback to COLMAP if GLOMAP fails
            reconstruction.main(
                sfm_dir=sfm_dir,
                image_dir=frames_dir,
                pairs=pairs_path,
                features=feature_path,
                matches=match_path,
                mapper='colmap'
            )

        # Step 5: Validate quality metrics
        logger.info("Validating reconstruction quality...")
        total_frames = len(image_list)
        quality_metrics = self._compute_quality_metrics(sfm_dir, total_frames)

        should_fallback, reason = self._should_use_dust3r_fallback(quality_metrics)
        if should_fallback:
            logger.warning(f"Quality check failed: {reason}")
            logger.warning("Triggering DUSt3R fallback...")
            raise RuntimeError(f"hloc quality insufficient: {reason}")

        logger.info(f"Reconstruction quality: {quality_metrics}")

        # Step 6: Export to transforms.json (Nerfstudio format)
        logger.info("Exporting to transforms.json...")
        self._export_transforms_json(sfm_dir, transforms_path)

        # Step 7: Export sparse point cloud
        logger.info("Exporting sparse point cloud...")
        sparse_ply = sparse_path / "points3D.ply"
        self._export_sparse_points(sfm_dir, sparse_ply)

        logger.info("hloc pipeline completed successfully")

    def _compute_quality_metrics(self, sparse_dir: Path, total_frames: int) -> dict:
        """
        Compute reconstruction quality metrics.

        Returns dict with:
        - reprojection_error_mean: Mean reprojection error in pixels
        - track_length_mean: Mean number of views per 3D point
        - track_length_max: Maximum track length
        - coverage_ratio: Percentage of frames with poses
        - num_registered: Number of registered frames
        - num_points: Number of 3D points
        """
        try:
            import pycolmap
        except ImportError:
            raise ImportError("pycolmap not installed. Install with: pip install pycolmap")

        reconstruction = pycolmap.Reconstruction(sparse_dir)

        # Validate reconstruction is not empty
        if len(reconstruction.cameras) == 0:
            raise RuntimeError("Reconstruction contains no cameras")
        if len(reconstruction.images) == 0:
            raise RuntimeError("Reconstruction contains no registered images")

        # 1. Reprojection errors
        reproj_errors = []
        for image in reconstruction.images.values():
            for point2D in image.points2D:
                if point2D.point3D_id != -1:
                    reproj_errors.append(point2D.error)

        # 2. Track lengths
        track_lengths = []
        for point3D in reconstruction.points3D.values():
            track_lengths.append(len(point3D.track.elements))

        # 3. Coverage
        num_registered = len(reconstruction.images)
        coverage = num_registered / total_frames if total_frames > 0 else 0

        return {
            'reprojection_error_mean': np.mean(reproj_errors) if reproj_errors else float('inf'),
            'track_length_mean': np.mean(track_lengths) if track_lengths else 0,
            'track_length_max': np.max(track_lengths) if track_lengths else 0,
            'coverage_ratio': coverage,
            'num_registered': num_registered,
            'num_points': len(reconstruction.points3D)
        }

    def _should_use_dust3r_fallback(self, quality_metrics: dict) -> Tuple[bool, str]:
        """
        Determine if DUSt3R fallback should be triggered.

        Quality thresholds from ROADMAP.md:
        - Coverage > 80% for success, < 60% triggers fallback
        - Reprojection error < 2px for success, > 3px triggers fallback
        - Track length should be sufficient

        Returns:
            (should_fallback, reason)
        """
        # Criterion 1: Insufficient pose coverage
        if quality_metrics['coverage_ratio'] < 0.6:
            return True, f"Low coverage: {quality_metrics['coverage_ratio']:.1%} < 60%"

        # Criterion 2: High reprojection error
        if quality_metrics['reprojection_error_mean'] > 3.0:
            return True, f"High error: {quality_metrics['reprojection_error_mean']:.2f}px > 3.0px"

        # Criterion 3: Short track lengths
        if quality_metrics['track_length_mean'] < 3.0:
            return True, f"Short tracks: {quality_metrics['track_length_mean']:.1f} < 3.0"

        # Criterion 4: Very few 3D points
        if quality_metrics['num_points'] < 100:
            return True, f"Few points: {quality_metrics['num_points']} < 100"

        return False, "Quality thresholds met"

    def _export_transforms_json(self, sparse_dir: Path, output_path: Path):
        """
        Export COLMAP reconstruction to Nerfstudio transforms.json format.

        Uses nerfstudio's built-in converter if available, otherwise
        falls back to manual conversion.
        """
        try:
            from nerfstudio.process_data.colmap_utils import colmap_to_json

            colmap_to_json(
                recon_dir=sparse_dir,
                output_dir=output_path.parent,
                camera_model="OPENCV"
            )
        except ImportError:
            logger.warning("nerfstudio not installed, using manual conversion")
            self._manual_export_transforms_json(sparse_dir, output_path)

    def _manual_export_transforms_json(self, sparse_dir: Path, output_path: Path):
        """Manual conversion from COLMAP to Nerfstudio format."""
        import pycolmap

        reconstruction = pycolmap.Reconstruction(sparse_dir)

        # Validate reconstruction
        if len(reconstruction.cameras) == 0:
            raise RuntimeError("Cannot export: No cameras in reconstruction")
        if len(reconstruction.images) == 0:
            raise RuntimeError("Cannot export: No images registered in reconstruction")

        # Get camera (assume single camera)
        camera = list(reconstruction.cameras.values())[0]

        transforms = {
            "camera_model": "OPENCV",
            "fl_x": float(camera.focal_length_x),
            "fl_y": float(camera.focal_length_y),
            "cx": float(camera.principal_point_x),
            "cy": float(camera.principal_point_y),
            "w": int(camera.width),
            "h": int(camera.height),
            "k1": float(camera.params[4]) if len(camera.params) > 4 else 0.0,
            "k2": float(camera.params[5]) if len(camera.params) > 5 else 0.0,
            "p1": float(camera.params[6]) if len(camera.params) > 6 else 0.0,
            "p2": float(camera.params[7]) if len(camera.params) > 7 else 0.0,
            "frames": []
        }

        # Convert each image pose
        for image_id, image in reconstruction.images.items():
            # COLMAP: world-to-camera
            qvec = image.qvec
            tvec = image.tvec

            # Convert to rotation matrix
            R_w2c = pycolmap.qvec_to_rotmat(qvec)

            # Create 4x4 w2c matrix
            w2c = np.eye(4)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = tvec

            # Invert to get camera-to-world
            c2w = np.linalg.inv(w2c)

            # Convert OpenCV to OpenGL convention (flip Y and Z axes)
            c2w[:3, 1:3] *= -1

            frame = {
                "file_path": f"./frames/{image.name}",
                "transform_matrix": c2w.tolist()
            }
            transforms["frames"].append(frame)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

    def _export_sparse_points(self, sparse_dir: Path, output_ply: Path):
        """Export sparse point cloud to PLY format."""
        import pycolmap

        reconstruction = pycolmap.Reconstruction(sparse_dir)

        points = []
        colors = []
        for point3D in reconstruction.points3D.values():
            points.append(point3D.xyz)
            colors.append(point3D.color)

        if not points:
            logger.warning("No 3D points to export")
            return

        points = np.array(points)
        colors = np.array(colors)

        # Write PLY file (ASCII format)
        with open(output_ply, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

        logger.info(f"Exported {len(points)} points to {output_ply}")

    def _run_dust3r_pipeline(self, frames_dir: Path, transforms_path: Path):
        """
        Run DUSt3R pipeline for pose estimation (fallback).

        Pipeline:
        1. Load DUSt3R model
        2. Load and prepare frames
        3. Create image pairs
        4. Run inference
        5. Global alignment
        6. Export to transforms.json
        """
        try:
            import torch
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        except ImportError as e:
            raise ImportError(
                f"DUSt3R dependencies not installed: {e}\n"
                "Install with: pip install git+https://github.com/naver/dust3r.git roma trimesh"
            )

        logger.info("Starting DUSt3R fallback pipeline...")

        model = None  # Initialize outside try block for cleanup
        try:
            # Step 1: Load DUSt3R model
            logger.info("Loading DUSt3R model...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

            try:
                model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
                model.eval()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DUSt3R model: {e}\n"
                    "Possible causes:\n"
                    "- Network connectivity issues\n"
                    "- Insufficient disk space (~3GB required in ~/.cache/huggingface/)\n"
                    "- Corrupted cache (try: rm -rf ~/.cache/huggingface/hub/models--naver--*)\n"
                ) from e

            # Step 2: Load frames
            logger.info("Loading frames...")
            frame_paths = sorted(frames_dir.glob("*.jpg"))
            images = load_images([str(p) for p in frame_paths], size=512, verbose=False)

            # Step 3: Create pairs (adaptive strategy based on frame count)
            num_frames = len(images)
            if num_frames <= 30:
                scene_graph = 'complete'
                logger.info(f"Using complete pairing for {num_frames} frames")
            elif num_frames <= 100:
                scene_graph = 'swin'
                logger.info(f"Using sliding window pairing for {num_frames} frames")
            else:
                scene_graph = 'one-ref'
                logger.info(f"Using one-reference pairing for {num_frames} frames")

            pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)

            # Step 4: Run inference
            logger.info("Running DUSt3R inference...")
            output = inference(pairs, model, device, batch_size=1, verbose=False)

            # Step 5: Global alignment
            logger.info("Running global alignment...")
            scene = global_aligner(
                output,
                device=device,
                mode=GlobalAlignerMode.PointCloudOptimizer
            )
            scene.compute_global_alignment(
                init='mst',
                niter=300,
                schedule='cosine',
                lr=0.01
            )

            # Step 6: Export to transforms.json
            logger.info("Exporting to transforms.json...")
            self._dust3r_to_transforms_json(scene, frame_paths, transforms_path)

            logger.info("DUSt3R pipeline completed successfully")

        finally:
            # Cleanup GPU memory
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _dust3r_to_transforms_json(self, scene, frame_paths: list, output_path: Path):
        """Convert DUSt3R scene to Nerfstudio transforms.json format."""
        from PIL import Image

        # Get camera parameters
        focals = scene.get_focals()
        principal_points = scene.get_principal_points()
        poses = scene.get_im_poses()

        # Get image dimensions
        first_frame = Image.open(frame_paths[0])
        width, height = first_frame.size

        transforms = {
            "camera_model": "OPENCV",
            "fl_x": float(focals[0][0]),
            "fl_y": float(focals[0][1]),
            "cx": float(principal_points[0][0]),
            "cy": float(principal_points[0][1]),
            "w": width,
            "h": height,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "frames": []
        }

        # Convert each camera pose
        for i, (frame_path, w2c) in enumerate(zip(frame_paths, poses)):
            # DUSt3R outputs world-to-camera, convert to camera-to-world
            c2w = np.linalg.inv(w2c)

            # Convert from OpenCV to OpenGL convention (flip Y and Z)
            c2w[:3, 1:3] *= -1

            frame = {
                "file_path": f"./frames/{frame_path.name}",
                "transform_matrix": c2w.tolist()
            }
            transforms["frames"].append(frame)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Exported {len(transforms['frames'])} camera poses to {output_path}")
