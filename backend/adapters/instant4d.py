"""Instant4D adapter for 4D Gaussian Splatting reconstruction.

This adapter wraps the Instant4D pipeline to enable true 4D reconstruction
from video input, replacing the static 3DGS approach with temporal Gaussians.

Key capabilities:
- Preprocessing: Convert Stage 1/2 outputs to Instant4D format
- Grid Pruning: Voxel-based 92% Gaussian reduction
- 4D Training: Optimize 4D Gaussians (~2-5 min)
- Per-frame Export: Extract PLYs at arbitrary timestamps

Requirements:
    - CUDA 12.1+
    - PyTorch 2.3+
    - Instant4D CUDA kernels compiled (run scripts/setup_instant4d.sh)
    - ~24GB VRAM for training
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any
import json
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Instant4DOptions:
    """Options for Instant4D 4D Gaussian training."""

    # Training parameters
    iterations: int = 5_000
    """Number of optimization iterations."""

    batch_size: int = 1
    """Training batch size."""

    gaussian_dim: int = 4
    """Gaussian dimension: 3 for static, 4 for temporal."""

    time_duration: Tuple[float, float] = (0.0, 3.0)
    """Temporal range for 4D Gaussians."""

    rot_4d: bool = True
    """Enable 4D rotation representation."""

    num_pts: int = 100_000
    """Initial number of points."""

    # Preprocessing
    use_megasam: bool = False
    """Use Mega-SAM preprocessing (requires additional deps)."""

    motion_threshold: float = 0.5
    """Threshold for static/dynamic separation."""

    # Grid pruning
    enable_pruning: bool = True
    """Enable voxel-based grid pruning."""

    static_voxel_scale: float = 2.0
    """Voxel size scale for static regions."""

    dynamic_voxel_scale: float = 0.5
    """Voxel size scale for dynamic regions."""

    # Export
    export_fps: float = 30.0
    """Frames per second for PLY export."""

    opacity_threshold: float = 0.01
    """Minimum opacity for including Gaussians in export."""

    # Video rendering
    replay_num_frames: int = 120
    """Number of frames in temporal replay video (120 = 4 seconds at 30fps)."""

    replay_fps: int = 30
    """Frame rate for temporal replay video."""

    replay_wobble_factor: float = 0.05
    """Camera wobble amplitude for temporal replay (0 = fixed viewpoint)."""


@dataclass
class Instant4DResult:
    """Result of Instant4D 4D Gaussian training."""

    model_path: str
    """Path to trained 4D Gaussian model checkpoint (.pth)."""

    ply_paths: List[str] = field(default_factory=list)
    """Paths to per-frame PLY files extracted from 4D model."""

    num_gaussians: int = 0
    """Total number of 4D Gaussians in trained model."""

    num_frames: int = 0
    """Number of frames/timestamps extracted."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    """Training metrics (PSNR, SSIM, timing, etc.)."""

    config_path: Optional[str] = None
    """Path to training configuration used."""

    preview_video_path: Optional[str] = None
    """Path to rendered preview video (frozen-moment wobble diagnostic)."""

    temporal_replay_path: Optional[str] = None
    """Path to temporal replay video (4D scene evolution over time)."""

    temporal_metadata: Dict[str, Any] = field(default_factory=dict)
    """Temporal metadata (fps, duration, timestamps)."""


# =============================================================================
# Instant4D Adapter
# =============================================================================


class Instant4DAdapter:
    """
    Adapter for Instant4D 4D Gaussian Splatting pipeline.

    This adapter enables:
    1. Preprocessing - Convert Stage 1 output to Instant4D format
    2. Grid Pruning - Voxel-based 92% Gaussian reduction
    3. 4D Training - Optimize 4D Gaussians (~2-5 min)
    4. Per-frame Export - Extract PLYs at arbitrary timestamps

    Integration with Stage 2:
    - Uses SAM-2 masks to populate prob_motion field
    - Enables dynamic/static region separation

    Example:
        >>> adapter = Instant4DAdapter()
        >>> result = adapter.run_full_pipeline(
        ...     video_result=stage1_output,
        ...     segmentation_result=stage2_output,  # Optional
        ...     output_dir="/path/to/output",
        ...     options=Instant4DOptions(iterations=5000),
        ... )
        >>> print(f"Exported {result.num_frames} PLY files")
    """

    # Path to Instant4D submodule (relative to this file)
    INSTANT4D_PATH = Path(__file__).parent.parent.parent / "instant4d"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Instant4D adapter.

        Args:
            config: Optional configuration dictionary with keys:
                - device: CUDA device (default: "cuda:0")
                - instant4d_path: Override Instant4D submodule path
        """
        self.config = config or {}
        self.device = self.config.get("device", "cuda:0")

        # Allow overriding Instant4D path
        if "instant4d_path" in self.config:
            self.INSTANT4D_PATH = Path(self.config["instant4d_path"])

        # Lazy-loaded models
        self._gaussian_model = None
        self._scene = None
        self._path_added = False

        # Validate installation
        self._validate_installation()

    def _validate_installation(self) -> None:
        """
        Validate Instant4D submodule is properly installed.

        Raises:
            ImportError: If Instant4D is not found or incomplete
        """
        if not self.INSTANT4D_PATH.exists():
            raise ImportError(
                f"Instant4D submodule not found at {self.INSTANT4D_PATH}. "
                "Run: git submodule update --init --recursive"
            )

        # Check for essential files
        required_files = [
            "scene/gaussian_model.py",
            "gaussian_renderer/__init__.py",
            "scene/__init__.py",
        ]

        missing = []
        for f in required_files:
            if not (self.INSTANT4D_PATH / f).exists():
                missing.append(f)

        if missing:
            raise ImportError(
                f"Instant4D installation incomplete. Missing: {missing}. "
                "Run: git submodule update --init --recursive"
            )

        logger.info(f"Instant4D found at {self.INSTANT4D_PATH}")

    def _add_instant4d_to_path(self) -> None:
        """Add Instant4D and submodule paths for imports."""
        if self._path_added:
            return

        paths = [
            self.INSTANT4D_PATH,
            self.INSTANT4D_PATH / "submodule" / "fussed-ssim",
            self.INSTANT4D_PATH / "submodule",
            self.INSTANT4D_PATH / "submodule" / "pointops2",
            self.INSTANT4D_PATH / "submodule" / "simple-knn",
        ]
        for p in paths:
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
        self._path_added = True
        logger.debug(f"Added Instant4D paths to Python path")

    def _validate_cuda_kernels(self) -> None:
        """
        Validate CUDA kernels are compiled and importable.

        Raises:
            ImportError: If CUDA kernels are not available
        """
        self._add_instant4d_to_path()

        try:
            from gaussian_renderer import GaussianRasterizer

            assert GaussianRasterizer is not None
        except ImportError as e:
            raise ImportError(
                f"CUDA kernel 'diff-gaussian-rasterization' not available: {e}. "
                "Run: bash scripts/setup_instant4d.sh"
            ) from e

        try:
            from simple_knn._C import distCUDA2

            assert distCUDA2 is not None
        except ImportError as e:
            raise ImportError(
                f"CUDA kernel 'simple-knn' not available: {e}. "
                "Run: bash scripts/setup_instant4d.sh"
            ) from e

        logger.info("CUDA kernels validated successfully")

    # =========================================================================
    # Main Pipeline Methods
    # =========================================================================

    def preprocess(
        self,
        video_result: "VideoProcessingResult",
        segmentation_result: Optional["ObjectSegmentationResult"],
        output_dir: str,
        options: Optional[Instant4DOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Preprocess video data for Instant4D training.

        Converts Stage 1/2 outputs to Instant4D's filtered_cvd.npz format:
        - xyz: 3D point positions
        - rgb: Point colors
        - prob_motion: Motion probability (from Stage 2 masks or Mega-SAM)
        - time_stamp: Temporal coordinate
        - scale_time: Temporal scale
        - intrinsic: Camera intrinsics
        - cam_c2w: Camera-to-world transforms

        When Mega-SAM was used for pose estimation, the video_result metadata
        may contain:
        - depth_maps_dir: Path to dense depth maps (better initialization)
        - motion_prob_path: Path to Mega-SAM's motion probability

        Args:
            video_result: Output from Stage 1 (frames + poses)
            segmentation_result: Output from Stage 2 (masks for motion), optional
            output_dir: Directory to store preprocessed data
            options: Preprocessing options
            progress_callback: Progress updates (pct: float, msg: str)

        Returns:
            Path to preprocessed data directory (contains filtered_cvd.npz)
        """
        options = options or Instant4DOptions()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct*100:.0f}%] {msg}")

        report(0.0, "Starting preprocessing")

        # Step 1: Load camera transforms from Stage 1
        report(0.05, "Loading camera transforms")
        transforms = self._load_transforms(video_result.transforms_path)
        frames = transforms["frames"]
        num_frames = len(frames)

        # Step 2: Extract intrinsics
        report(0.1, "Extracting camera intrinsics")
        intrinsic = self._extract_intrinsics(transforms)

        # Step 3: Convert poses from Nerfstudio to Instant4D format
        report(0.15, "Converting camera poses")
        cam_c2w = self._convert_poses_nerfstudio_to_instant4d(frames)

        # Step 4: Load or generate point cloud
        report(0.2, "Loading point cloud")

        # Check if Mega-SAM depth maps are available (preferred)
        depth_maps_dir = video_result.metadata.get("depth_maps_dir")
        frame_timestamps = None
        if depth_maps_dir and Path(depth_maps_dir).exists():
            try:
                xyz, rgb, frame_timestamps = self._load_points_from_megasam_depth(
                    depth_maps_dir, video_result.frames_dir, cam_c2w, intrinsic,
                    time_duration=options.time_duration,
                )
                report(0.3, f"Loaded {xyz.shape[0]} points from Mega-SAM depth")
            except Exception as e:
                logger.warning(f"Failed to load Mega-SAM depth: {e}, falling back to sparse")
                xyz, rgb, frame_timestamps = self._load_or_generate_points(
                    video_result, cam_c2w, intrinsic, num_frames, options
                )
                report(0.3, f"Loaded/generated {xyz.shape[0]} points (fallback)")
        else:
            xyz, rgb, frame_timestamps = self._load_or_generate_points(
                video_result, cam_c2w, intrinsic, num_frames, options
            )
            report(0.3, f"Loaded/generated {xyz.shape[0]} points")

        # Step 5: Compute motion probabilities
        report(0.4, "Computing motion probabilities")

        # Check if Mega-SAM motion probability is available (preferred)
        motion_prob_path = video_result.metadata.get("motion_prob_path")
        if motion_prob_path and Path(motion_prob_path).exists():
            try:
                prob_motion = self._load_megasam_motion_prob(
                    motion_prob_path, xyz, cam_c2w, intrinsic
                )
                num_dynamic = np.sum(prob_motion > options.motion_threshold)
                report(0.5, f"Loaded Mega-SAM motion prob, {num_dynamic} dynamic points")
            except Exception as e:
                logger.warning(f"Failed to load Mega-SAM motion prob: {e}")
                prob_motion = self._compute_motion_fallback(
                    segmentation_result, xyz, frames, cam_c2w, intrinsic, options
                )
                report(0.5, "Motion probability computed (fallback)")
        else:
            prob_motion = self._compute_motion_fallback(
                segmentation_result, xyz, frames, cam_c2w, intrinsic, options
            )
            num_dynamic = np.sum(prob_motion > options.motion_threshold)
            report(0.5, f"Motion probability computed, {num_dynamic} dynamic points")

        # Step 6: Assign timestamps to points
        report(0.55, "Assigning temporal coordinates")
        time_stamp, scale_time = self._assign_timestamps(
            xyz, prob_motion, num_frames, options,
            frame_timestamps=frame_timestamps,
            total_depth_frames=num_frames,
        )

        # Step 7: Apply voxel grid pruning
        if options.enable_pruning:
            report(0.6, "Applying voxel grid pruning")
            original_count = xyz.shape[0]
            xyz, rgb, prob_motion, time_stamp, scale_time = self._voxel_filter(
                xyz, rgb, prob_motion, time_stamp, scale_time, intrinsic, cam_c2w, options
            )
            reduction = (1 - xyz.shape[0] / original_count) * 100
            report(0.75, f"Pruned to {xyz.shape[0]} points ({reduction:.0f}% reduction)")
        else:
            report(0.75, "Skipping voxel pruning")

        # Step 8: Save filtered_cvd.npz
        report(0.8, "Saving preprocessed data")
        np.savez(
            output_path / "filtered_cvd.npz",
            xyz=xyz.astype(np.float32),
            rgb=rgb.astype(np.float32),
            prob_motion=prob_motion.astype(np.float32),
            time_stamp=time_stamp.astype(np.float32),
            scale_time=scale_time.astype(np.float32),
            intrinsic=intrinsic.astype(np.float32),
            cam_c2w=cam_c2w.astype(np.float32),
        )

        # Step 9: Create transforms_train.json and transforms_test.json
        report(0.9, "Creating Instant4D transforms")
        self._create_instant4d_transforms(
            transforms, video_result.frames_dir, output_path, num_frames, cam_c2w
        )

        report(1.0, "Preprocessing complete")
        return output_path

    def train(
        self,
        preprocessed_dir: str,
        output_dir: str,
        options: Optional[Instant4DOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Instant4DResult:
        """
        Train 4D Gaussian model.

        Runs Instant4D's optimization pipeline:
        1. Load preprocessed data (filtered_cvd.npz + transforms)
        2. Initialize 4D Gaussians from point cloud
        3. Run optimization loop with densification
        4. Save checkpoint and metrics

        Training is ~2-5 minutes on typical video.

        Args:
            preprocessed_dir: Path to preprocessed data
            output_dir: Directory for training outputs
            options: Training hyperparameters
            progress_callback: Progress updates (pct: float, msg: str)

        Returns:
            Instant4DResult with model path and metrics
        """
        options = options or Instant4DOptions()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct*100:.0f}%] {msg}")

        report(0.0, "Initializing training")

        # Validate CUDA kernels before training
        self._validate_cuda_kernels()

        # Add Instant4D to path
        self._add_instant4d_to_path()

        # Import Instant4D modules
        report(0.02, "Loading Instant4D modules")
        from scene.gaussian_model import GaussianModel
        from scene import Scene

        # Build configuration
        report(0.05, "Building training configuration")
        model_params, opt_params, pipe_params = self._build_training_config(
            preprocessed_dir, output_path, options
        )

        # Initialize model
        report(0.08, "Initializing 4D Gaussian model")
        gaussians = GaussianModel(
            sh_degree=0,
            gaussian_dim=options.gaussian_dim,
            time_duration=list(options.time_duration),
            rot_4d=options.rot_4d,
            force_sh_3d=False,
            sh_degree_t=2,  # Temporal SH (matches upstream eval_shfs_4d=True)
        )

        # Load scene
        report(0.1, "Loading scene data")
        scene = Scene(model_params, gaussians, time_duration=list(options.time_duration))

        # Override cameras_extent using point cloud extent
        # For monocular video, getNerfppNorm() returns a tiny baseline which
        # weakens densification thresholds (percent_dense * extent). Use the
        # actual point cloud spatial extent instead.
        npz_path = Path(preprocessed_dir) / "filtered_cvd.npz"
        if npz_path.exists():
            pc_data = np.load(npz_path)
            pc_xyz = pc_data["xyz"]
            pc_extent = np.linalg.norm(
                pc_xyz.max(axis=0) - pc_xyz.min(axis=0)
            )
            original_extent = scene.cameras_extent
            new_extent = max(pc_extent * 0.5, original_extent)
            if new_extent > original_extent:
                scene.cameras_extent = new_extent
                logger.info(
                    f"cameras_extent override: {original_extent:.4f} -> "
                    f"{new_extent:.4f} (pc_extent={pc_extent:.4f})"
                )

        # Setup optimizer
        report(0.12, "Setting up optimizer")
        gaussians.training_setup(opt_params)

        # Training loop
        import torch
        from gaussian_renderer import render

        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        training_cameras = scene.getTrainCameras()

        if len(training_cameras) == 0:
            raise RuntimeError("No training cameras found. Check preprocessed data.")

        total_iterations = options.iterations
        metrics_history = {"psnr": [], "loss": []}

        logger.info(
            f"Scene cameras_extent={scene.cameras_extent:.4f}, "
            f"initial gaussians={gaussians.get_xyz.shape[0]}"
        )
        report(0.15, f"Starting training ({total_iterations} iterations)")

        for iteration in range(1, total_iterations + 1):
            # Update learning rate (exponential decay for position params)
            gaussians.update_learning_rate(iteration)

            # Sample training view — CameraDataset.__getitem__ returns (image, camera)
            camera_idx = iteration % len(training_cameras)
            gt_image, viewpoint = training_cameras[camera_idx]

            # Move to GPU
            gt_image = gt_image.cuda()
            viewpoint = viewpoint.cuda()

            # Render
            render_pkg = render(viewpoint, gaussians, pipe_params, bg_color)
            image = render_pkg["render"]

            # Compute loss
            loss = self._compute_loss(image, gt_image, alpha=render_pkg.get("alpha"))
            loss.backward()

            # Track metrics
            with torch.no_grad():
                psnr = self._compute_psnr(image, gt_image)
                metrics_history["psnr"].append(psnr)
                metrics_history["loss"].append(loss.item())

            # Progress update every 100 iterations
            if iteration % 100 == 0 or iteration == 1:
                pct = 0.15 + 0.75 * (iteration / total_iterations)
                report(pct, f"Iteration {iteration}/{total_iterations}, PSNR: {psnr:.2f}")

            # Densification and pruning (must read gradients before zero_grad)
            if iteration < opt_params.densify_until_iter:
                gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]],
                )

                # Compute temporal gradient for 4D Gaussians
                t_grad = None
                if gaussians.gaussian_dim == 4:
                    t_grad = gaussians._t.grad.clone().detach()

                gaussians.add_densification_stats(
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    t_grad,
                )

                if (
                    iteration > opt_params.densify_from_iter
                    and iteration % opt_params.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt_params.opacity_reset_interval else None
                    )
                    before_count = gaussians.get_xyz.shape[0]
                    gaussians.densify_and_prune(
                        opt_params.densify_grad_threshold,
                        opt_params.thresh_opa_prune,
                        scene.cameras_extent,
                        size_threshold,
                        opt_params.densify_grad_t_threshold,
                    )
                    after_count = gaussians.get_xyz.shape[0]
                    if iteration % 500 == 0 or abs(after_count - before_count) > 1000:
                        logger.info(
                            f"Iter {iteration}: gaussians {before_count} -> "
                            f"{after_count}, extent={scene.cameras_extent:.4f}, "
                            f"size_thresh={size_threshold}"
                        )

            # Optimizer step (after densification reads gradients)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # Save checkpoint
        report(0.92, "Saving model checkpoint")
        checkpoint_path = output_path / "model.pth"
        torch.save(gaussians.capture(), checkpoint_path)

        # Save config
        config_path = output_path / "config.yaml"
        self._save_config(config_path, options)

        # Compute final metrics
        report(0.95, "Computing final metrics")
        final_metrics = self._compute_final_metrics(
            scene, gaussians, pipe_params, bg_color
        )
        final_metrics["training_history"] = metrics_history

        # Store for later use
        self._gaussian_model = gaussians
        self._scene = scene
        self._pipe_params = pipe_params

        report(1.0, "Training complete")

        return Instant4DResult(
            model_path=str(checkpoint_path),
            num_gaussians=gaussians.get_xyz.shape[0],
            metrics=final_metrics,
            config_path=str(config_path),
        )

    def extract_per_frame_ply(
        self,
        model_path: str,
        output_dir: str,
        timestamps: Optional[List[float]] = None,
        export_fps: float = 30.0,
        opacity_threshold: float = 0.01,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[str]:
        """
        Extract per-frame PLY files from trained 4D model.

        Queries the 4D Gaussian model at each timestamp to get:
        - Current 3D position (mean + temporal offset)
        - Current covariance (marginal at timestamp)
        - Current opacity (weighted by temporal Gaussian)
        - Color (from spherical harmonics)

        Args:
            model_path: Path to trained model checkpoint
            output_dir: Directory for PLY outputs
            timestamps: List of timestamps to export (default: derived from training)
            export_fps: Frames per second for fallback frame count when timestamps not provided
            opacity_threshold: Minimum effective opacity to include a Gaussian
            progress_callback: Progress updates (pct: float, msg: str)

        Returns:
            List of paths to exported PLY files
        """
        import torch

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct*100:.0f}%] {msg}")

        report(0.0, "Starting per-frame PLY export")

        # Load model if not already in memory
        if self._gaussian_model is None:
            report(0.05, "Loading model checkpoint")
            self._gaussian_model = self._load_model(model_path)

        gaussians = self._gaussian_model

        # Determine timestamps to export
        if timestamps is None:
            t_min, t_max = gaussians.time_duration
            duration = t_max - t_min
            num_frames = max(1, int(duration * export_fps))
            timestamps = np.linspace(t_min, t_max, num_frames).tolist()

        ply_paths = []
        num_timestamps = len(timestamps)

        report(0.1, f"Exporting {num_timestamps} frames")

        for i, timestamp in enumerate(timestamps):
            with torch.no_grad():
                # Get temporal marginal weight
                marginal_t = gaussians.get_marginal_t(timestamp)

                # Get current position at this timestamp
                if hasattr(gaussians, "rot_4d") and gaussians.rot_4d:
                    try:
                        _, delta_mean = gaussians.get_current_covariance_and_mean_offset(
                            scaling_modifier=1.0, timestamp=timestamp
                        )
                        xyz = (gaussians.get_xyz + delta_mean).cpu().numpy()
                    except Exception:
                        # Fallback if method not available
                        xyz = gaussians.get_xyz.cpu().numpy()
                else:
                    xyz = gaussians.get_xyz.cpu().numpy()

                # Filter by temporal weight
                opacity = gaussians.get_opacity.cpu().numpy().squeeze()
                marginal_np = marginal_t.cpu().numpy().squeeze()
                effective_opacity = opacity * marginal_np
                active_mask = effective_opacity > opacity_threshold

                # Extract all Gaussian properties for active points
                xyz_active = xyz[active_mask]
                features_dc = gaussians._features_dc.cpu().numpy()[active_mask, 0, :]  # [N, 3]
                opacity_raw = gaussians._opacity.cpu().numpy()[active_mask]  # [N, 1] logit-space
                scaling_raw = gaussians._scaling.cpu().numpy()[active_mask, :3]  # [N, 3] log-space (skip t)
                rotation_raw = gaussians._rotation.cpu().numpy()[active_mask]  # [N, 4] quaternions

            # Write standard 3DGS PLY format
            ply_path = output_path / f"frame_{i:04d}.ply"
            self._write_splat_ply(
                ply_path, xyz_active, features_dc, opacity_raw, scaling_raw, rotation_raw
            )
            ply_paths.append(str(ply_path))

            # Progress update
            pct = 0.1 + 0.85 * ((i + 1) / num_timestamps)
            if (i + 1) % 5 == 0 or i == 0 or i == num_timestamps - 1:
                report(pct, f"Exported frame {i+1}/{num_timestamps} ({xyz_active.shape[0]} points)")

        report(1.0, f"Export complete: {len(ply_paths)} PLY files")

        return ply_paths

    def run_full_pipeline(
        self,
        video_result: "VideoProcessingResult",
        output_dir: str,
        segmentation_result: Optional["ObjectSegmentationResult"] = None,
        options: Optional[Instant4DOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Instant4DResult:
        """
        Run complete Instant4D pipeline: preprocess -> train -> export.

        This is the main entry point for Stage 3 integration.

        Args:
            video_result: Output from Stage 1 (frames + poses)
            segmentation_result: Output from Stage 2 (masks), optional
            output_dir: Directory for all outputs
            options: Pipeline options
            progress_callback: Progress updates (pct: float, msg: str)

        Returns:
            Instant4DResult with model path, PLY paths, and metrics
        """
        options = options or Instant4DOptions()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        # Phase 1: Preprocessing (0-20%)
        report(0.0, "Phase 1: Preprocessing")
        preprocessed_dir = self.preprocess(
            video_result,
            segmentation_result,
            str(output_path / "preprocessed"),
            options,
            lambda p, m: report(p * 0.2, f"[Preprocess] {m}"),
        )

        # Phase 2: Training (20-80%)
        report(0.2, "Phase 2: Training 4D Gaussians")
        result = self.train(
            str(preprocessed_dir),
            str(output_path / "trained"),
            options,
            lambda p, m: report(0.2 + p * 0.6, f"[Train] {m}"),
        )

        # Phase 3: Export (80-95%)
        report(0.8, "Phase 3: Exporting per-frame PLYs")

        # Derive frame count from transforms_train.json written during preprocessing
        transforms_train_path = output_path / "preprocessed" / "transforms_train.json"
        if transforms_train_path.exists():
            with open(transforms_train_path) as f:
                train_transforms = json.load(f)
            num_train_frames = len(train_transforms.get("frames", []))
        else:
            num_train_frames = 0

        # Build explicit timestamps matching training frame count
        t_min, t_max = options.time_duration
        if num_train_frames > 0:
            export_timestamps = np.linspace(t_min, t_max, num_train_frames).tolist()
        else:
            export_timestamps = None  # fallback to fps-based default

        ply_paths = self.extract_per_frame_ply(
            result.model_path,
            str(output_path / "plys"),
            timestamps=export_timestamps,
            export_fps=options.export_fps,
            opacity_threshold=options.opacity_threshold,
            progress_callback=lambda p, m: report(0.8 + p * 0.15, f"[Export] {m}"),
        )

        # Update result
        result.ply_paths = ply_paths
        result.num_frames = len(ply_paths)
        result.temporal_metadata = {
            "fps": options.export_fps,
            "duration_seconds": len(ply_paths) / options.export_fps,
            "num_frames": len(ply_paths),
            "timestamps": list(
                np.linspace(
                    options.time_duration[0], options.time_duration[1], len(ply_paths)
                )
            ),
        }

        # Phase 4: Preview videos (95-100%)
        preview_dir = str(output_path / "preview")

        report(0.95, "Phase 4a: Rendering diagnostic wobble video")
        preview_path = self.render_preview_video(preview_dir)
        if preview_path:
            result.preview_video_path = preview_path
            report(0.96, f"Wobble preview: {preview_path}")
        else:
            report(0.96, "Wobble preview skipped (no test cameras or rendering failed)")

        report(0.97, "Phase 4b: Rendering temporal replay video")
        replay_path = self.render_temporal_replay_video(
            preview_dir,
            num_frames=options.replay_num_frames,
            fps=options.replay_fps,
            wobble_factor=options.replay_wobble_factor,
        )
        if replay_path:
            result.temporal_replay_path = replay_path
            report(0.99, f"Temporal replay: {replay_path}")
        else:
            report(0.99, "Temporal replay skipped")

        report(1.0, "Instant4D pipeline complete")
        return result

    def render_preview_video(self, output_dir: str) -> Optional[str]:
        """Render a preview video from trained model as a quality check.

        Uses Instant4D's render_evaluate_sora() to render test camera views
        with wobble animation. Requires eval=True so test cameras are available.

        Args:
            output_dir: Directory for video output

        Returns:
            Path to output MP4, or None if rendering failed
        """
        if self._scene is None or self._gaussian_model is None or self._pipe_params is None:
            logger.warning("Cannot render preview: model/scene not loaded")
            return None

        import torch

        scene = self._scene
        gaussians = self._gaussian_model
        pipe_params = self._pipe_params

        # Check for test cameras (requires eval=True during training)
        if not hasattr(scene, "test_cameras") or not scene.test_cameras:
            logger.warning("No test cameras available (was eval=True set?)")
            return None

        # Check test_cameras dict has entries
        test_cam_keys = list(scene.test_cameras.keys())
        if not test_cam_keys or len(scene.test_cameras[test_cam_keys[0]]) == 0:
            logger.warning("Test camera list is empty")
            return None

        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            import os as _os

            # Log OpenCV codec availability for diagnostics
            try:
                logger.info(
                    f"OpenCV {cv2.__version__}, "
                    f"FFMPEG: {'yes' if 'FFMPEG' in cv2.getBuildInformation() else 'no'}"
                )
            except Exception:
                pass

            # Monkey-patch cv2.VideoWriter with codec fallback.
            # render_evaluate_sora (in the instant4d submodule) hard-codes
            # avc1 (H.264), which silently fails on Colab. Since the
            # submodule isn't updated by git pull, we patch cv2.VideoWriter
            # here so the submodule code picks up the fallback automatically.
            _OriginalVideoWriter = cv2.VideoWriter
            _CODECS = [("avc1", ".mp4"), ("mp4v", ".mp4"), ("XVID", ".avi")]

            class _FallbackVideoWriter:
                """cv2.VideoWriter wrapper that tries multiple codecs."""

                def __init__(self, filename, fourcc, fps, frameSize, isColor=True):
                    self._writer = None
                    self._path = filename
                    base, _ = _os.path.splitext(filename)
                    for codec, ext in _CODECS:
                        try_path = base + ext
                        fc = cv2.VideoWriter_fourcc(*codec)
                        w = _OriginalVideoWriter(
                            try_path, fc, fps, frameSize, isColor
                        )
                        if w.isOpened():
                            self._writer = w
                            self._path = try_path
                            logger.info(f"VideoWriter using codec: {codec}")
                            break
                        w.release()
                    if self._writer is None:
                        logger.warning("No working video codec found")
                        self._writer = _OriginalVideoWriter()

                def isOpened(self):
                    return self._writer.isOpened()

                def write(self, frame):
                    return self._writer.write(frame)

                def release(self):
                    return self._writer.release()

            cv2.VideoWriter = _FallbackVideoWriter

            try:
                logger.info("Rendering preview video (quality check)")
                with torch.no_grad():
                    scene.render_evaluate_sora(
                        str(output_path), gaussians, pipe_params, bg_color
                    )
            finally:
                cv2.VideoWriter = _OriginalVideoWriter

            # Find the generated video (render_evaluate_sora writes to test/ subdir)
            # Filter to nonzero-size files — cv2.VideoWriter silently creates empty
            # files when the codec (avc1/H.264) isn't available on the platform.
            test_dir = output_path / "test"
            if test_dir.exists():
                video_files = sorted(
                    f
                    for f in test_dir.iterdir()
                    if f.name.startswith("novel_view_")
                    and f.suffix in (".mp4", ".avi")
                    and f.stat().st_size > 0
                )
            else:
                video_files = []

            if video_files:
                video_path = str(video_files[-1])  # last variant (largest wobble)
                logger.info(
                    f"Preview video saved: {video_path} "
                    f"({len(video_files)} variant(s))"
                )
                return video_path
            else:
                logger.warning(
                    "render_evaluate_sora completed but no valid video file found "
                    "(H.264 codec may not be available — "
                    "try: pip install opencv-python-headless)"
                )
                return None
        except Exception as e:
            logger.warning(f"Preview video rendering failed: {e}")
            return None

    def render_temporal_replay_video(
        self,
        output_dir: str,
        num_frames: int = 120,
        fps: int = 30,
        wobble_factor: float = 0.05,
    ) -> Optional[str]:
        """Render temporal replay video showing the 4D scene evolving over time.

        Unlike the diagnostic wobble video (which freezes time and moves the camera),
        this advances the timestamp each frame while applying gentle camera motion,
        producing a video that shows the full temporal evolution of the 4D model.

        Args:
            output_dir: Directory for video output.
            num_frames: Total frames in output video.
            fps: Video frame rate.
            wobble_factor: Amplitude of camera wobble (0 = fixed viewpoint).

        Returns:
            Path to output MP4, or None if rendering failed.
        """
        if self._scene is None or self._gaussian_model is None or self._pipe_params is None:
            logger.warning("Cannot render temporal replay: model/scene not loaded")
            return None

        import torch

        scene = self._scene
        gaussians = self._gaussian_model
        pipe_params = self._pipe_params

        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(
                f"Rendering temporal replay: {num_frames} frames, "
                f"wobble_factor={wobble_factor}"
            )
            video_path = scene.render_temporal_replay(
                str(output_path), gaussians, pipe_params, bg_color,
                num_frames=num_frames, fps=fps, wobble_factor=wobble_factor,
            )

            if video_path and os.path.exists(video_path):
                logger.info(f"Temporal replay video saved: {video_path}")
                return video_path
            else:
                logger.warning("render_temporal_replay completed but no video produced")
                return None
        except Exception as e:
            logger.warning(f"Temporal replay rendering failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release GPU memory and resources."""
        if self._gaussian_model is not None:
            del self._gaussian_model
            self._gaussian_model = None

        if self._scene is not None:
            del self._scene
            self._scene = None

        if hasattr(self, "_pipe_params"):
            del self._pipe_params

        # Clear CUDA cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Instant4D adapter resources cleaned up")

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _load_transforms(self, transforms_path: str) -> Dict[str, Any]:
        """Load and validate transforms.json from Stage 1."""
        with open(transforms_path) as f:
            transforms = json.load(f)

        # Validate required fields
        if "frames" not in transforms:
            raise ValueError(f"transforms.json missing 'frames' key: {transforms_path}")

        return transforms

    def _extract_intrinsics(self, transforms: Dict[str, Any]) -> np.ndarray:
        """Extract camera intrinsics matrix from transforms."""
        intrinsic = np.eye(3, dtype=np.float32)

        # Handle different intrinsic formats
        if "fl_x" in transforms:
            intrinsic[0, 0] = transforms["fl_x"]
            intrinsic[1, 1] = transforms.get("fl_y", transforms["fl_x"])
            intrinsic[0, 2] = transforms.get("cx", transforms.get("w", 640) / 2)
            intrinsic[1, 2] = transforms.get("cy", transforms.get("h", 480) / 2)
        elif "camera_angle_x" in transforms:
            # Nerfstudio format with angle
            w = transforms.get("w", 640)
            h = transforms.get("h", 480)
            fov_x = transforms["camera_angle_x"]
            fl_x = w / (2 * np.tan(fov_x / 2))
            intrinsic[0, 0] = fl_x
            intrinsic[1, 1] = fl_x
            intrinsic[0, 2] = w / 2
            intrinsic[1, 2] = h / 2
        else:
            # Default fallback
            logger.warning("No intrinsics found in transforms, using defaults")
            intrinsic[0, 0] = 500
            intrinsic[1, 1] = 500
            intrinsic[0, 2] = 320
            intrinsic[1, 2] = 240

        return intrinsic

    def _convert_poses_nerfstudio_to_instant4d(
        self, frames: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Convert camera poses from Nerfstudio (OpenGL) to Instant4D (COLMAP-like) convention.

        OpenGL: Y-up, Z-back (looking at -Z)
        COLMAP/Instant4D: Y-down, Z-forward

        Args:
            frames: List of frame dicts with "transform_matrix"

        Returns:
            Camera-to-world matrices (N, 4, 4)
        """
        cam_c2w = []

        for frame in frames:
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)

            # Ensure 4x4
            if c2w.shape == (3, 4):
                c2w = np.vstack([c2w, [0, 0, 0, 1]])

            # Convert from OpenGL to COLMAP convention
            # Flip Y and Z axes
            c2w[:3, 1:3] *= -1

            cam_c2w.append(c2w)

        return np.stack(cam_c2w)

    def _load_sparse_points(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load sparse point cloud from PLY file."""
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError("plyfile required. Install with: pip install plyfile")

        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

        # Try to load colors
        if "red" in vertex:
            rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.5

        return xyz.astype(np.float32), rgb.astype(np.float32)

    def _generate_initial_points(
        self,
        cam_c2w: np.ndarray,
        intrinsic: np.ndarray,
        num_frames: int,
        num_pts: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial point cloud from camera frustums."""
        # Simple approach: sample points in a bounding box around cameras
        camera_centers = cam_c2w[:, :3, 3]
        center = camera_centers.mean(axis=0)
        extent = (camera_centers.max(axis=0) - camera_centers.min(axis=0)).max() * 2

        # Random points in bounding box
        xyz = (np.random.rand(num_pts, 3) - 0.5) * extent + center
        rgb = np.random.rand(num_pts, 3).astype(np.float32)

        return xyz.astype(np.float32), rgb

    def _load_or_generate_points(
        self,
        video_result: "VideoProcessingResult",
        cam_c2w: np.ndarray,
        intrinsic: np.ndarray,
        num_frames: int,
        options: Instant4DOptions,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load sparse points or generate initial points.

        Returns:
            xyz, rgb, timestamps (timestamps is None when frame correlation
            is not available, e.g. sparse points or random generation).
        """
        if video_result.sparse_points_path and Path(video_result.sparse_points_path).exists():
            xyz, rgb = self._load_sparse_points(video_result.sparse_points_path)
            return xyz, rgb, None
        else:
            xyz, rgb = self._generate_initial_points(
                cam_c2w, intrinsic, num_frames, options.num_pts
            )
            return xyz, rgb, None

    def _load_points_from_megasam_depth(
        self,
        depth_maps_dir: str,
        frames_dir: str,
        cam_c2w: np.ndarray,
        intrinsic: np.ndarray,
        subsample_rate: int = 3,
        max_points: int = 200_000,
        time_duration: Tuple[float, float] = (0.0, 3.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load point cloud from Mega-SAM depth maps via back-projection.

        Each point receives a timestamp derived from its source frame index,
        matching the upstream Instant4D formula: ``frame_idx / total * t_range``.

        Args:
            depth_maps_dir: Directory containing depth NPZ files
            frames_dir: Directory containing frame images
            cam_c2w: (N, 4, 4) camera-to-world matrices
            intrinsic: (3, 3) camera intrinsic matrix
            subsample_rate: Subsample every N-th frame to reduce points
            max_points: Maximum points to return
            time_duration: (t_min, t_max) temporal range for timestamp mapping

        Returns:
            xyz: (M, 3) point coordinates
            rgb: (M, 3) point colors [0-1]
            timestamps: (M,) per-point timestamp derived from source frame
        """
        import cv2
        from glob import glob

        # Find depth file
        depth_files = sorted(glob(f"{depth_maps_dir}/*_droid.npz"))
        if not depth_files:
            raise FileNotFoundError(f"No Mega-SAM depth files in {depth_maps_dir}")

        # Load the NPZ file
        data = np.load(depth_files[0])
        depths = data["depths"]  # (N, H, W)
        images = data.get("images")  # (N, H, W, 3) or None
        total_depth_frames = depths.shape[0]

        all_xyz = []
        all_rgb = []
        all_timestamps = []

        t_min, t_max = time_duration

        # Subsample frames (matching upstream prune.py ::3 pattern)
        frame_indices = list(range(0, total_depth_frames, subsample_rate))
        num_subsampled = len(frame_indices)

        for sub_idx, idx in enumerate(frame_indices):
            depth = depths[idx]
            H, W = depth.shape

            # Create pixel grid
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            u = u.flatten() + 0.5
            v = v.flatten() + 0.5
            z = depth.flatten()

            # Filter valid depths
            valid = (z > 0.01) & (z < 100.0) & np.isfinite(z)

            # Subsample pixels
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > max_points // len(frame_indices):
                valid_indices = np.random.choice(
                    valid_indices,
                    max_points // len(frame_indices),
                    replace=False
                )

            u, v, z = u[valid_indices], v[valid_indices], z[valid_indices]

            # Back-project to camera space
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            xyz_cam = np.stack([x, y, z], axis=1)

            # Transform to world coordinates
            if idx < cam_c2w.shape[0]:
                c2w = cam_c2w[idx]
                xyz_world = (c2w[:3, :3] @ xyz_cam.T + c2w[:3, 3:4]).T
                all_xyz.append(xyz_world)

                # Frame-correlated timestamp (upstream prune.py:110)
                # Uses subsampled index / subsampled count to map [0, t_range]
                if num_subsampled > 1:
                    t = sub_idx / num_subsampled * (t_max - t_min) + t_min
                else:
                    t = (t_min + t_max) / 2
                all_timestamps.append(
                    np.full(xyz_world.shape[0], t, dtype=np.float32)
                )

                # Get colors if available
                if images is not None:
                    img = images[idx]
                    pixel_u = np.clip(u.astype(int), 0, W - 1)
                    pixel_v = np.clip(v.astype(int), 0, H - 1)
                    colors = img[pixel_v, pixel_u] / 255.0
                    all_rgb.append(colors)
                else:
                    all_rgb.append(np.ones((xyz_world.shape[0], 3)) * 0.5)

        xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
        rgb = np.concatenate(all_rgb, axis=0).astype(np.float32)
        timestamps = np.concatenate(all_timestamps, axis=0).astype(np.float32)

        # Final subsampling if too many points
        if xyz.shape[0] > max_points:
            indices = np.random.choice(xyz.shape[0], max_points, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]
            timestamps = timestamps[indices]

        logger.info(f"Loaded {xyz.shape[0]} points from Mega-SAM depth")
        return xyz, rgb, timestamps

    def _load_megasam_motion_prob(
        self,
        motion_prob_path: str,
        xyz: np.ndarray,
        cam_c2w: np.ndarray,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        """
        Load motion probability from Mega-SAM and interpolate to 3D points.

        Mega-SAM provides per-pixel motion probability maps. We project 3D points
        to the image and sample the motion probability.

        Args:
            motion_prob_path: Path to motion_prob.npy (N, H/8, W/8)
            xyz: (M, 3) 3D point coordinates
            cam_c2w: (N, 4, 4) camera-to-world matrices
            intrinsic: (3, 3) camera intrinsics

        Returns:
            prob_motion: (M,) motion probability for each point
        """
        import cv2

        motion_maps = np.load(motion_prob_path)  # (N, H/8, W/8)
        num_points = xyz.shape[0]
        motion_sum = np.zeros(num_points, dtype=np.float32)
        visible_count = np.zeros(num_points, dtype=np.float32)

        for frame_idx in range(min(motion_maps.shape[0], cam_c2w.shape[0])):
            motion_map = motion_maps[frame_idx]  # (H/8, W/8)
            h, w = motion_map.shape

            # Get camera matrices
            c2w = cam_c2w[frame_idx]
            w2c = np.linalg.inv(c2w)

            # Project 3D points to camera space
            xyz_cam = (w2c[:3, :3] @ xyz.T + w2c[:3, 3:4]).T
            z = xyz_cam[:, 2]
            valid = z > 0.1

            # Project to image (at 1/8 resolution)
            uv = (intrinsic @ xyz_cam.T).T
            uv = uv[:, :2] / (uv[:, 2:3] + 1e-8)
            uv = uv / 8.0  # Scale to motion map resolution

            # Sample motion probability
            u = np.clip(uv[:, 0].astype(int), 0, w - 1)
            v = np.clip(uv[:, 1].astype(int), 0, h - 1)

            motion_values = motion_map[v, u]

            # Accumulate for visible points
            motion_sum[valid] += motion_values[valid]
            visible_count[valid] += 1

        # Average motion probability
        prob_motion = np.zeros(num_points, dtype=np.float32)
        has_observations = visible_count > 0
        prob_motion[has_observations] = motion_sum[has_observations] / visible_count[has_observations]

        logger.info(
            f"Loaded Mega-SAM motion prob: "
            f"{np.sum(prob_motion > 0.5)} dynamic, {np.sum(prob_motion <= 0.5)} static points"
        )
        return prob_motion

    def _compute_motion_fallback(
        self,
        segmentation_result: Optional["ObjectSegmentationResult"],
        xyz: np.ndarray,
        frames: List[Dict[str, Any]],
        cam_c2w: np.ndarray,
        intrinsic: np.ndarray,
        options: Instant4DOptions,
    ) -> np.ndarray:
        """Compute motion probability using SAM-2 masks or assume static."""
        if segmentation_result and segmentation_result.objects:
            return self._compute_motion_from_masks(
                segmentation_result, xyz, frames, cam_c2w, intrinsic
            )
        else:
            logger.warning(
                "No segmentation or Mega-SAM motion data. Assuming all points static."
            )
            return np.zeros(xyz.shape[0], dtype=np.float32)

    def _compute_motion_from_masks(
        self,
        segmentation_result: "ObjectSegmentationResult",
        xyz: np.ndarray,
        frames: List[Dict[str, Any]],
        cam_c2w: np.ndarray,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-point motion probability from Stage 2 masks.

        Strategy:
        1. For each 3D point, project to each frame
        2. Check if point falls inside any object mask
        3. Aggregate across frames to get motion probability
        """
        import cv2

        num_points = xyz.shape[0]
        motion_counts = np.zeros(num_points, dtype=np.float32)
        visible_counts = np.zeros(num_points, dtype=np.float32)

        # Build mask lookup by frame index
        frame_masks: Dict[int, List[np.ndarray]] = {}
        for obj in segmentation_result.objects:
            for frame_idx, mask_path in zip(obj.frame_indices, obj.mask_paths):
                if frame_idx not in frame_masks:
                    frame_masks[frame_idx] = []
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        frame_masks[frame_idx].append(mask)
                except Exception as e:
                    logger.warning(f"Failed to load mask {mask_path}: {e}")

        # Project points to each frame
        for frame_idx in range(len(frames)):
            if frame_idx not in frame_masks or not frame_masks[frame_idx]:
                continue

            # Get camera matrices
            c2w = cam_c2w[frame_idx]
            w2c = np.linalg.inv(c2w)

            # Project 3D points to camera space
            xyz_cam = (w2c[:3, :3] @ xyz.T + w2c[:3, 3:4]).T
            z = xyz_cam[:, 2]
            valid = z > 0.1  # Points in front of camera

            # Project to image
            uv = (intrinsic @ xyz_cam.T).T
            uv = uv[:, :2] / (uv[:, 2:3] + 1e-8)

            # Check mask coverage
            for mask in frame_masks[frame_idx]:
                h, w = mask.shape
                in_frame = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < w)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < h)
                    & valid
                )

                in_frame_indices = np.where(in_frame)[0]
                if len(in_frame_indices) == 0:
                    continue

                u_int = uv[in_frame, 0].astype(int)
                v_int = uv[in_frame, 1].astype(int)

                # Check if in mask
                in_mask = mask[v_int, u_int] > 127
                motion_counts[in_frame_indices[in_mask]] += 1

            visible_counts[valid] += 1

        # Compute probability
        valid_visible = visible_counts > 0
        prob_motion = np.zeros_like(motion_counts)
        prob_motion[valid_visible] = (
            motion_counts[valid_visible] / visible_counts[valid_visible]
        )

        return prob_motion

    def _assign_timestamps(
        self,
        xyz: np.ndarray,
        prob_motion: np.ndarray,
        num_frames: int,
        options: Instant4DOptions,
        frame_timestamps: Optional[np.ndarray] = None,
        total_depth_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign temporal coordinates and scales to points.

        When frame_timestamps are provided (from depth loading), dynamic points
        keep their frame-correlated timestamps and get tight temporal scales
        matching upstream prune.py. Static points get midpoint timestamp with
        full-range scale.

        When frame_timestamps is None, falls back to random assignment for
        backward compatibility.

        Upstream reference (instant4d/script/prune.py):
          - Static timestamp: midpoint (line 250)
          - Static scale_t: full t_range (line 251)
          - Dynamic scale_t: t_range / ((B-1) * 10) where B = original
            frame count before subsampling (line 252)
        """
        t_min, t_max = options.time_duration
        t_mid = (t_min + t_max) / 2
        t_range = t_max - t_min

        num_points = xyz.shape[0]
        # Static defaults: midpoint timestamp, full temporal range
        time_stamp = np.full(num_points, t_mid, dtype=np.float32)
        scale_time = np.full(num_points, t_range, dtype=np.float32)

        dynamic_mask = prob_motion > options.motion_threshold
        num_dynamic = np.sum(dynamic_mask)

        if num_dynamic > 0:
            if frame_timestamps is not None:
                # Use frame-correlated timestamps from depth loading
                time_stamp[dynamic_mask] = frame_timestamps[dynamic_mask]
            else:
                # Fallback: random timestamps (legacy behavior)
                time_stamp[dynamic_mask] = np.random.uniform(
                    t_min, t_max, num_dynamic
                ).astype(np.float32)

            # Tight temporal scale for dynamic points
            # Use original depth frame count (before subsampling) for scale
            B = total_depth_frames if total_depth_frames is not None else num_frames
            if B > 1:
                scale_time[dynamic_mask] = t_range / ((B - 1) * 10)
            else:
                scale_time[dynamic_mask] = t_range

        logger.info(
            f"Timestamps assigned: {num_dynamic} dynamic "
            f"(scale_t={scale_time[dynamic_mask].mean():.6f} if any), "
            f"{num_points - num_dynamic} static "
            f"(scale_t={t_range}, t_mid={t_mid})"
        )

        return time_stamp, scale_time

    def _voxel_filter(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        prob_motion: np.ndarray,
        time_stamp: np.ndarray,
        scale_time: np.ndarray,
        intrinsic: np.ndarray,
        cam_c2w: np.ndarray,
        options: Instant4DOptions,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply voxel-based grid pruning to reduce point count.

        Uses different voxel sizes for static vs dynamic regions.
        """
        try:
            import point_cloud_utils as pcu
        except ImportError:
            logger.warning(
                "point_cloud_utils not available. Skipping voxel filtering. "
                "Install with: pip install point-cloud-utils"
            )
            return xyz, rgb, prob_motion, time_stamp, scale_time

        # Compute mean depth for voxel size
        focal = intrinsic[0, 0]
        camera_centers = cam_c2w[:, :3, 3]
        mean_depth = np.linalg.norm(xyz.mean(axis=0) - camera_centers.mean(axis=0))

        # Split static/dynamic
        dynamic_mask = prob_motion > options.motion_threshold
        static_mask = ~dynamic_mask

        results = []  # list of (xyz, rgb, prob, time, scale) tuples

        # Filter static points
        if np.any(static_mask):
            voxel_size = mean_depth / focal * options.static_voxel_scale
            s_xyz, s_rgb, s_prob, s_time, s_scale = pcu.downsample_point_cloud_on_voxel_grid(
                voxel_size,
                xyz[static_mask],
                rgb[static_mask],
                prob_motion[static_mask],
                time_stamp[static_mask],
                scale_time[static_mask],
            )
            results.append((s_xyz, s_rgb, s_prob, s_time, s_scale))

        # Filter dynamic points
        if np.any(dynamic_mask):
            voxel_size = mean_depth / focal * options.dynamic_voxel_scale
            d_xyz, d_rgb, d_prob, d_time, d_scale = pcu.downsample_point_cloud_on_voxel_grid(
                voxel_size,
                xyz[dynamic_mask],
                rgb[dynamic_mask],
                prob_motion[dynamic_mask],
                time_stamp[dynamic_mask],
                scale_time[dynamic_mask],
            )
            results.append((d_xyz, d_rgb, d_prob, d_time, d_scale))

        if not results:
            return xyz, rgb, prob_motion, time_stamp, scale_time

        # Concatenate static + dynamic results
        return tuple(np.concatenate([r[i] for r in results], axis=0) for i in range(5))

    def _create_instant4d_transforms(
        self,
        transforms: Dict[str, Any],
        frames_dir: str,
        output_path: Path,
        num_frames: int,
        cam_c2w: "np.ndarray",
    ) -> None:
        """Create transforms_train.json and transforms_test.json for Instant4D.

        Args:
            transforms: Original Stage 1 transforms (used for intrinsics and file_path)
            frames_dir: Path to frame images
            output_path: Output directory
            num_frames: Number of frames
            cam_c2w: (N, 4, 4) camera-to-world matrices in COLMAP convention.
                Instant4D's readCamerasFromCutter expects COLMAP convention
                (the OpenGL→COLMAP flip at line 377 is commented out).
                The point cloud in filtered_cvd.npz is also in COLMAP world
                space, so cameras and points must use the same convention.
        """
        # Symlink Stage 1 frames into preprocessed directory so Instant4D can find them
        frames_link = output_path / "frames"
        if not frames_link.exists():
            frames_link.symlink_to(Path(frames_dir).resolve())

        frames = transforms["frames"]

        # Split 90/10 train/test
        train_count = int(num_frames * 0.9)
        train_frames = frames[:train_count]
        test_frames = frames[train_count:]

        # Add timestamp and overwrite transform_matrix with COLMAP-convention poses.
        # The original transforms from Stage 1 are in OpenGL convention, but
        # Instant4D's reader expects COLMAP convention (Y-down, Z-forward).
        for i, frame in enumerate(frames):
            frame["time"] = i / max(1, num_frames - 1) * 3.0  # Normalize to [0, 3]
            if i < cam_c2w.shape[0]:
                frame["transform_matrix"] = cam_c2w[i].tolist()

        # Create base structure
        base = {
            "fl_x": transforms.get("fl_x", 500),
            "fl_y": transforms.get("fl_y", transforms.get("fl_x", 500)),
            "cx": transforms.get("cx", transforms.get("w", 640) / 2),
            "cy": transforms.get("cy", transforms.get("h", 480) / 2),
            "w": transforms.get("w", 640),
            "h": transforms.get("h", 480),
        }

        # Write train transforms
        train_data = {**base, "frames": train_frames}
        with open(output_path / "transforms_train.json", "w") as f:
            json.dump(train_data, f, indent=2)

        # Write test transforms
        test_data = {**base, "frames": test_frames if test_frames else train_frames[:1]}
        with open(output_path / "transforms_test.json", "w") as f:
            json.dump(test_data, f, indent=2)

    def _build_training_config(
        self, preprocessed_dir: str, output_path: Path, options: Instant4DOptions
    ) -> Tuple[Any, Any, Any]:
        """Build Instant4D training configuration objects."""
        from argparse import Namespace

        # Model parameters (must match Instant4D's ModelParams defaults)
        model_params = Namespace(
            source_path=preprocessed_dir,
            model_path=str(output_path),
            sh_degree=0,
            images="images",
            resolution=1,
            white_background=False,
            data_device="cuda",
            eval=True,
            extension="",  # file_path in transforms JSON already includes extension
            num_extra_pts=0,
            loaded_pth="",
            frame_ratio=1,
            dataloader=False,
        )

        # Optimization parameters (must match Instant4D's OptimizationParams defaults)
        opt_params = Namespace(
            iterations=options.iterations,
            position_lr_init=0.00016,
            position_t_lr_init=-1.0,  # <0 means fallback to position_lr_init
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=options.iterations,
            feature_lr=0.0025,
            opacity_lr=0.05,
            scaling_lr=0.005,
            rotation_lr=0.001,
            densify_from_iter=500,
            densify_until_iter=min(15000, options.iterations),
            densify_grad_threshold=0.0002,
            densify_grad_t_threshold=0.000005,
            densification_interval=100,
            opacity_reset_interval=options.iterations + 1,  # disable size pruning; opacity sparsity handles floaters
            densify_until_num_points=-1,
            final_prune_from_iter=-1,
            sh_increase_interval=1000,
            thresh_opa_prune=0.005,
            percent_dense=0.01,
            lambda_dssim=0.2,
            lambda_opa_mask=0.0,
            lambda_rigid=0.0,
            lambda_motion=0.0,
        )

        # Pipeline parameters
        pipe_params = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            env_map_res=0,
            env_optimize_until=1000000000,
            env_optimize_from=0,
            eval_shfs_4d=True,
        )

        return model_params, opt_params, pipe_params

    def _compute_loss(
        self,
        image: "torch.Tensor",
        gt_image: "torch.Tensor",
        alpha: "Optional[torch.Tensor]" = None,
    ) -> "torch.Tensor":
        """Compute training loss (L1 + SSIM + opacity sparsity)."""
        import torch
        import torch.nn.functional as F

        # L1 loss
        l1_loss = F.l1_loss(image, gt_image)

        # Simple SSIM approximation (use fused_ssim if available)
        try:
            from fused_ssim import fused_ssim

            ssim_loss = 1 - fused_ssim(
                image.unsqueeze(0), gt_image.unsqueeze(0), padding="same"
            )
        except ImportError:
            # Fallback: just use L1
            ssim_loss = torch.tensor(0.0, device=image.device)

        loss = 0.8 * l1_loss + 0.2 * ssim_loss

        # Opacity sparsity: penalize unnecessary rendered opacity to suppress floaters.
        # Floaters are screen-space Gaussians the optimizer places at the camera plane;
        # this gently discourages high total alpha in the rendered image.
        if alpha is not None:
            loss = loss + 0.001 * alpha.mean()

        return loss

    def _compute_psnr(self, image: "torch.Tensor", gt_image: "torch.Tensor") -> float:
        """Compute PSNR between rendered and ground truth images."""
        import torch

        mse = torch.mean((image - gt_image) ** 2)
        if mse == 0:
            return float("inf")
        return (10 * torch.log10(1.0 / mse)).item()

    def _compute_final_metrics(
        self, scene: Any, gaussians: Any, pipe_params: Any, bg_color: "torch.Tensor"
    ) -> Dict[str, Any]:
        """Compute final evaluation metrics on test views."""
        import torch
        from gaussian_renderer import render

        test_cameras = scene.getTestCameras()
        if len(test_cameras) == 0:
            return {"psnr": 0.0, "ssim": 0.0, "num_test_views": 0}

        psnr_values = []
        num_eval = min(5, len(test_cameras))

        with torch.no_grad():
            for i in range(num_eval):
                gt_image, viewpoint = test_cameras[i]
                viewpoint = viewpoint.cuda()
                render_pkg = render(viewpoint, gaussians, pipe_params, bg_color)
                image = render_pkg["render"]
                gt_image = gt_image.cuda()
                psnr = self._compute_psnr(image, gt_image)
                psnr_values.append(psnr)

        return {
            "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
            "num_test_views": len(psnr_values),
            "num_gaussians": gaussians.get_xyz.shape[0],
        }

    def _load_model(self, model_path: str) -> Any:
        """Load trained Gaussian model from checkpoint."""
        import torch

        self._add_instant4d_to_path()
        from scene.gaussian_model import GaussianModel

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cuda")

        # Infer model parameters from checkpoint
        sh_degree = checkpoint.get("active_sh_degree", 0)

        # Create model
        gaussians = GaussianModel(
            sh_degree=sh_degree,
            gaussian_dim=4,
            time_duration=[0.0, 3.0],
            rot_4d=True,
            force_sh_3d=False,
        )

        # Restore state
        gaussians.restore(checkpoint, None)

        return gaussians

    def _save_config(self, config_path: Path, options: Instant4DOptions) -> None:
        """Save training configuration to YAML."""
        import yaml

        config = {
            "iterations": options.iterations,
            "batch_size": options.batch_size,
            "gaussian_dim": options.gaussian_dim,
            "time_duration": list(options.time_duration),
            "rot_4d": options.rot_4d,
            "num_pts": options.num_pts,
            "enable_pruning": options.enable_pruning,
            "motion_threshold": options.motion_threshold,
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def _write_splat_ply(
        self,
        path: Path,
        xyz: np.ndarray,
        features_dc: np.ndarray,
        opacity: np.ndarray,
        scaling: np.ndarray,
        rotation: np.ndarray,
    ) -> None:
        """Write standard 3DGS PLY file compatible with splat viewers.

        All values are pre-activation (raw parameters):
        - opacity: logit-space (not sigmoid-activated)
        - scaling: log-space (not exp-activated)
        - rotation: unnormalized quaternions

        With sh_degree=0, sh_degree_t=0: 17 properties per vertex
        (3 pos + 3 normals + 3 f_dc + 1 opacity + 3 scale + 4 rotation).
        """
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise ImportError("plyfile required. Install with: pip install plyfile")

        num_points = xyz.shape[0]

        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]

        elements = np.empty(num_points, dtype=dtype)
        elements["x"] = xyz[:, 0]
        elements["y"] = xyz[:, 1]
        elements["z"] = xyz[:, 2]
        elements["nx"] = 0.0
        elements["ny"] = 0.0
        elements["nz"] = 0.0
        elements["f_dc_0"] = features_dc[:, 0]
        elements["f_dc_1"] = features_dc[:, 1]
        elements["f_dc_2"] = features_dc[:, 2]
        elements["opacity"] = opacity.squeeze()
        elements["scale_0"] = scaling[:, 0]
        elements["scale_1"] = scaling[:, 1]
        elements["scale_2"] = scaling[:, 2]
        elements["rot_0"] = rotation[:, 0]
        elements["rot_1"] = rotation[:, 1]
        elements["rot_2"] = rotation[:, 2]
        elements["rot_3"] = rotation[:, 3]

        vertex = PlyElement.describe(elements, "vertex")
        PlyData([vertex], byte_order="<").write(str(path))

    def _write_point_cloud_ply(self, path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
        """Write simple point cloud PLY file (for debug/preview use)."""
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise ImportError("plyfile required. Install with: pip install plyfile")

        # Ensure rgb is uint8
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Create structured array
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        normals = np.zeros_like(xyz)
        elements = np.empty(xyz.shape[0], dtype=dtype)

        for i in range(xyz.shape[0]):
            elements[i] = (
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                normals[i, 0],
                normals[i, 1],
                normals[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )

        vertex = PlyElement.describe(elements, "vertex")
        PlyData([vertex]).write(str(path))
