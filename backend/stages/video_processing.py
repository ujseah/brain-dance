"""Stage 1: Video Processing - Frame extraction and camera pose estimation."""

import subprocess
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Path to Mega-SAM submodule
MEGASAM_PATH = Path(__file__).parent.parent.parent / "instant4d" / "SLAM" / "mega-sam"


class MegaSamError(RuntimeError):
    """Exception indicating Mega-SAM pipeline failed and should fall back to hloc."""
    pass


@dataclass
class MegaSamResult:
    """Result from Mega-SAM pose estimation pipeline."""

    poses: np.ndarray
    """Camera-to-world matrices (N, 4, 4)."""

    intrinsics: np.ndarray
    """Camera intrinsics matrix (3, 3)."""

    depth_maps_dir: Optional[str] = None
    """Directory containing per-frame depth NPZ files."""

    motion_prob_path: Optional[str] = None
    """Path to motion probability NPY file."""

    num_keyframes: int = 0
    """Number of keyframes tracked by DROID SLAM."""

    metrics: dict = field(default_factory=dict)
    """Quality metrics for validation."""


class MegaSamPoseEstimator:
    """
    Mega-SAM pose estimation pipeline wrapper.

    Mega-SAM is a SLAM-based camera tracking system that produces:
    1. Camera poses (SE3 format, converted to 4x4 c2w matrices)
    2. Dense depth maps per frame
    3. Camera intrinsics from UniDepth FOV estimation
    4. Motion probability masks (dynamic/static regions)

    The pipeline consists of:
    1. UniDepth - Metric depth + FOV estimation per frame
    2. Depth-Anything - Mono disparity estimation
    3. Depth Alignment - Scale/shift calibration between depths
    4. DROID SLAM - Camera tracking with bundle adjustment
    5. Post-processing - SE3 to 4x4, coordinate convention conversion

    Output format is compatible with Instant4D's filtered_cvd.npz.
    """

    # Required checkpoints
    DROID_WEIGHTS = MEGASAM_PATH / "checkpoints" / "megasam_final.pth"
    DEPTH_ANYTHING_WEIGHTS = MEGASAM_PATH / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"

    def __init__(self, config: dict = None):
        """
        Initialize Mega-SAM pose estimator.

        Args:
            config: Configuration dict with optional keys:
                - opt_focal: Whether to optimize focal length (default: True)
                - max_frames: Max frames before subsampling (default: 300)
                - disable_vis: Disable DROID visualization (default: True)
        """
        self.config = config or {}
        self.opt_focal = self.config.get("opt_focal", True)
        self.max_frames = self.config.get("max_frames", 300)
        self.disable_vis = self.config.get("disable_vis", True)
        self._path_added = False
        self._available = None

    def is_available(self) -> bool:
        """
        Check if Mega-SAM dependencies are available.

        Returns:
            True if all checkpoints and imports are available.
        """
        if self._available is not None:
            return self._available

        # Check checkpoints exist
        if not self.DROID_WEIGHTS.exists():
            logger.warning(
                f"Mega-SAM DROID weights not found: {self.DROID_WEIGHTS}\n"
                "Run: bash scripts/setup_megasam.sh"
            )
            self._available = False
            return False

        # Check if lietorch can be imported (requires CUDA compilation)
        try:
            self._add_megasam_to_path()
            import lietorch  # noqa: F401
            self._available = True
        except ImportError as e:
            logger.warning(
                f"lietorch not available: {e}\n"
                "Mega-SAM requires CUDA-compiled lietorch.\n"
                "Run: cd instant4d/SLAM/mega-sam/base && python setup.py install"
            )
            self._available = False

        return self._available

    def _add_megasam_to_path(self):
        """Add Mega-SAM directories to Python path."""
        if self._path_added:
            return

        paths_to_add = [
            str(MEGASAM_PATH / "base" / "droid_slam"),
            str(MEGASAM_PATH / "base"),
            str(MEGASAM_PATH),
        ]

        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

        self._path_added = True

    def estimate_poses(
        self,
        frames_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MegaSamResult:
        """
        Run full Mega-SAM pipeline for pose estimation.

        Args:
            frames_dir: Directory containing extracted frames (*.jpg)
            output_dir: Working directory for intermediate outputs
            progress_callback: Optional callback(progress, message)

        Returns:
            MegaSamResult with poses, intrinsics, depth, motion probability

        Raises:
            MegaSamError: If pipeline fails (caller should fall back to hloc)
        """
        import torch
        import cv2

        def report(pct: float, msg: str):
            logger.info(f"[Mega-SAM {pct:.0%}] {msg}")
            if progress_callback:
                progress_callback(pct, msg)

        if not self.is_available():
            raise MegaSamError("Mega-SAM dependencies not available")

        self._add_megasam_to_path()

        # Get list of frames
        image_list = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        if not image_list:
            raise MegaSamError(f"No images found in {frames_dir}")

        num_frames = len(image_list)
        logger.info(f"Processing {num_frames} frames with Mega-SAM")

        # Subsample if too many frames
        if num_frames > self.max_frames:
            subsample_rate = num_frames // self.max_frames + 1
            image_list = image_list[::subsample_rate]
            logger.info(f"Subsampled to {len(image_list)} frames (rate: 1/{subsample_rate})")

        # Create output directories
        unidepth_dir = output_dir / "megasam" / "unidepth"
        depth_anything_dir = output_dir / "megasam" / "depth_anything"
        reconstructions_dir = output_dir / "megasam" / "reconstructions"
        outputs_dir = output_dir / "megasam" / "outputs"

        for d in [unidepth_dir, depth_anything_dir, reconstructions_dir, outputs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Run UniDepth for metric depth + FOV
            report(0.0, "Running UniDepth for metric depth estimation")
            fovs = self._run_unidepth(image_list, unidepth_dir)
            report(0.2, f"UniDepth complete, median FOV: {np.median(fovs):.1f}°")

            # Step 2: Run Depth-Anything for mono disparity
            report(0.2, "Running Depth-Anything for mono disparity")
            self._run_depth_anything(image_list, depth_anything_dir)
            report(0.4, "Depth-Anything complete")

            # Step 3: Align depths and compute intrinsics
            report(0.4, "Aligning depth estimates")
            aligns, intrinsics_K = self._align_depths(
                image_list, unidepth_dir, depth_anything_dir
            )
            report(0.5, "Depth alignment complete")

            # Step 4: Run DROID SLAM
            report(0.5, "Running DROID SLAM camera tracking")
            poses_se3, depth_est, motion_prob, num_keyframes = self._run_droid_slam(
                image_list, unidepth_dir, depth_anything_dir, aligns, intrinsics_K,
                reconstructions_dir
            )
            report(0.85, f"DROID SLAM complete, {num_keyframes} keyframes")

            # Step 5: Convert SE3 to c2w matrices
            report(0.85, "Converting poses to camera-to-world matrices")
            cam_c2w = self._se3_to_c2w(poses_se3)

            # Step 6: Save outputs
            report(0.9, "Saving outputs")
            scene_name = "scene"

            # Save motion probability
            motion_prob_path = reconstructions_dir / "motion_prob.npy"
            np.save(motion_prob_path, motion_prob)

            # Save depth maps
            depth_output_path = outputs_dir / f"{scene_name}_droid.npz"
            img_0 = cv2.imread(str(image_list[0]))
            np.savez(
                depth_output_path,
                images=np.zeros((len(image_list), img_0.shape[0], img_0.shape[1], 3), dtype=np.uint8),  # Placeholder
                depths=depth_est,
                intrinsic=intrinsics_K,
                cam_c2w=cam_c2w,
            )

            report(1.0, "Mega-SAM pipeline complete")

            return MegaSamResult(
                poses=cam_c2w,
                intrinsics=intrinsics_K,
                depth_maps_dir=str(outputs_dir),
                motion_prob_path=str(motion_prob_path),
                num_keyframes=num_keyframes,
                metrics={
                    "num_frames": len(image_list),
                    "median_fov": float(np.median(fovs)),
                }
            )

        except Exception as e:
            logger.error(f"Mega-SAM pipeline failed: {e}")
            raise MegaSamError(f"Mega-SAM pipeline failed: {e}") from e
        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_unidepth(self, image_list: List[Path], output_dir: Path) -> List[float]:
        """
        Run UniDepth for metric depth and FOV estimation.

        Args:
            image_list: List of image paths
            output_dir: Directory to save depth NPZ files

        Returns:
            List of FOV estimates per frame
        """
        import torch
        import cv2

        try:
            # UniDepth auto-downloads from HuggingFace
            from unidepth.models import UniDepthV2
        except ImportError:
            raise MegaSamError(
                "UniDepth not installed. Install with: pip install unidepth"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        model = model.to(device).eval()

        fovs = []

        try:
            with torch.no_grad():
                for i, img_path in enumerate(image_list):
                    # Load and preprocess image
                    image = cv2.imread(str(img_path))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Run inference — UniDepthV2 expects a CHW float tensor
                    rgb_torch = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(device)
                    predictions = model.infer(rgb_torch)

                    depth = predictions["depth"].cpu().numpy().squeeze()
                    # UniDepthV2 returns intrinsics, not fov — compute FOV from focal length
                    intrinsics = predictions["intrinsics"]
                    fx = intrinsics[0, 0, 0].cpu().item()
                    w = predictions["depth"].shape[-1]
                    fov = float(np.rad2deg(2 * np.arctan(w / (2 * fx))))

                    fovs.append(fov)

                    # Save output
                    output_path = output_dir / f"frame_{i:05d}.npz"
                    np.savez(output_path, depth=depth, fov=fov)

                    if (i + 1) % 10 == 0:
                        logger.debug(f"UniDepth: processed {i + 1}/{len(image_list)} frames")

        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return fovs

    def _run_depth_anything(self, image_list: List[Path], output_dir: Path):
        """
        Run Depth-Anything for mono disparity estimation.

        Args:
            image_list: List of image paths
            output_dir: Directory to save disparity NPY files
        """
        import torch
        import cv2

        # Check if checkpoint exists
        if not self.DEPTH_ANYTHING_WEIGHTS.exists():
            raise MegaSamError(
                f"Depth-Anything weights not found: {self.DEPTH_ANYTHING_WEIGHTS}\n"
                "Download from: https://huggingface.co/spaces/LiheYoung/Depth-Anything"
            )

        try:
            # Add Depth-Anything to path
            da_path = MEGASAM_PATH / "Depth-Anything"
            if str(da_path) not in sys.path:
                sys.path.insert(0, str(da_path))

            from depth_anything.dpt import DepthAnything
            from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
            from torchvision.transforms import Compose
        except ImportError:
            raise MegaSamError(
                "Depth-Anything dependencies not available. Check installation."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        model = DepthAnything.from_pretrained(
            "LiheYoung/depth-anything-large-hf"
        ).to(device).eval()

        # Setup transforms
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        try:
            with torch.no_grad():
                for i, img_path in enumerate(image_list):
                    # Load image
                    image = cv2.imread(str(img_path))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

                    # Transform
                    h, w = image.shape[:2]
                    image_tensor = transform({'image': image_rgb})['image']
                    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)

                    # Inference
                    depth = model(image_tensor)
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(1),
                        size=(h, w),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze().cpu().numpy()

                    # Depth-Anything outputs disparity (inverse depth)
                    disp = depth

                    # Save
                    output_path = output_dir / f"frame_{i:05d}.npy"
                    np.save(output_path, disp.astype(np.float32))

                    if (i + 1) % 10 == 0:
                        logger.debug(f"Depth-Anything: processed {i + 1}/{len(image_list)} frames")

        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _align_depths(
        self,
        image_list: List[Path],
        unidepth_dir: Path,
        depth_anything_dir: Path,
    ) -> Tuple[Tuple[float, float, float], np.ndarray]:
        """
        Align mono disparity with metric depth using scale/shift calibration.

        Follows Mega-SAM's alignment strategy from test_demo.py:
        - Handles sky regions specially (when sky_ratio > 0.5)
        - Uses median-based scale/shift estimation

        Args:
            image_list: List of image paths
            unidepth_dir: Directory with UniDepth outputs
            depth_anything_dir: Directory with Depth-Anything outputs

        Returns:
            Tuple of:
                - aligns: (scale, shift, normalize_scale)
                - intrinsics_K: 3x3 camera intrinsic matrix
        """
        import cv2

        # Load first image for dimensions
        img_0 = cv2.imread(str(image_list[0]))
        h, w = img_0.shape[:2]

        scales = []
        shifts = []
        fovs = []
        mono_disps = []

        for i in range(len(image_list)):
            # Load UniDepth output
            unidepth_path = unidepth_dir / f"frame_{i:05d}.npz"
            uni_data = np.load(unidepth_path)
            metric_depth = uni_data["depth"]
            fovs.append(uni_data["fov"])

            # Load Depth-Anything disparity
            da_path = depth_anything_dir / f"frame_{i:05d}.npy"
            da_disp = np.float32(np.load(da_path))

            # Resize disparity to match metric depth
            da_disp = cv2.resize(
                da_disp,
                (metric_depth.shape[1], metric_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            mono_disps.append(da_disp)

            # Convert metric depth to disparity
            gt_disp = 1.0 / (metric_depth + 1e-8)

            # Handle UniDepth bugs
            valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
            gt_disp[valid_mask] = 1e-2

            # Compute scale/shift with sky handling
            sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])

            if sky_ratio > 0.5:
                # Sky-dominated: use non-sky regions only
                non_sky_mask = da_disp > 0.01
                gt_disp_ms = gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
                da_disp_ms = da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
                scale = np.median(gt_disp_ms / da_disp_ms)
                shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
            else:
                gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
                da_disp_ms = da_disp - np.median(da_disp) + 1e-8
                scale = np.median(gt_disp_ms / da_disp_ms)
                shift = np.median(gt_disp - scale * da_disp)

            scales.append(scale)
            shifts.append(shift)

        # Find median scale/shift
        ss_product = np.array(scales) * np.array(shifts)
        med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))

        align_scale = scales[med_idx]
        align_shift = shifts[med_idx]
        normalize_scale = np.percentile(
            align_scale * np.array(mono_disps[med_idx]) + align_shift, 98
        ) / 2.0

        aligns = (align_scale, align_shift, normalize_scale)

        # Compute intrinsics from median FOV
        median_fov = np.median(fovs)
        focal_length = w / (2 * np.tan(np.radians(median_fov) / 2.0))

        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0

        logger.info(f"Depth alignment: scale={align_scale:.4f}, shift={align_shift:.4f}")
        logger.info(f"Intrinsics from FOV {median_fov:.1f}°: fx=fy={focal_length:.1f}")

        return aligns, K

    def _run_droid_slam(
        self,
        image_list: List[Path],
        unidepth_dir: Path,
        depth_anything_dir: Path,
        aligns: Tuple[float, float, float],
        intrinsics_K: np.ndarray,
        reconstructions_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Run DROID SLAM for camera tracking.

        Args:
            image_list: List of image paths
            unidepth_dir: Directory with UniDepth outputs
            depth_anything_dir: Directory with Depth-Anything outputs
            aligns: (scale, shift, normalize_scale) from depth alignment
            intrinsics_K: 3x3 camera intrinsic matrix
            reconstructions_dir: Directory for SLAM outputs

        Returns:
            Tuple of:
                - poses_se3: (N, 7) SE3 poses [x, y, z, qx, qy, qz, qw]
                - depth_est: (N, H, W) estimated depths
                - motion_prob: (N, H/8, W/8) motion probability
                - num_keyframes: Number of keyframes
        """
        import torch
        import cv2
        from argparse import Namespace

        # Import DROID components
        self._add_megasam_to_path()
        from droid import Droid

        align_scale, align_shift, normalize_scale = aligns

        # Setup DROID arguments
        args = Namespace(
            weights=str(self.DROID_WEIGHTS),
            buffer=1024,
            image_size=[240, 320],  # Will be updated from first frame
            disable_vis=self.disable_vis,
            beta=0.3,
            filter_thresh=2.0,
            warmup=8,
            keyframe_thresh=2.0,
            frontend_thresh=12.0,
            frontend_window=25,
            frontend_radius=2,
            frontend_nms=1,
            stereo=False,
            depth=True,
            upsample=False,
            backend_thresh=16.0,
            backend_radius=2,
            backend_nms=3,
        )

        # Initialize storage
        rgb_list = []
        sensor_depth_list = []
        droid = None

        try:
            for t, img_path in enumerate(image_list):
                # Load image
                image = cv2.imread(str(img_path))
                h0, w0 = image.shape[:2]

                # Resize to ~384x512 area
                h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
                w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
                image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
                image = image[: h1 - h1 % 8, : w1 - w1 % 8]

                # Load mono disparity
                da_path = depth_anything_dir / f"frame_{t:05d}.npy"
                mono_disp = np.float32(np.load(da_path))

                # Align to metric depth
                depth = np.clip(
                    1.0 / ((1.0 / normalize_scale) * (align_scale * mono_disp + align_shift)),
                    1e-4,
                    1e4,
                )
                depth[depth < 1e-2] = 0.0

                # Convert to tensors
                image_tensor = torch.as_tensor(image).permute(2, 0, 1)
                depth_tensor = torch.as_tensor(depth)
                depth_tensor = torch.nn.functional.interpolate(
                    depth_tensor[None, None], (h1, w1), mode="nearest-exact"
                ).squeeze()
                depth_tensor = depth_tensor[: h1 - h1 % 8, : w1 - w1 % 8]

                mask = torch.ones_like(depth_tensor)

                # Scale intrinsics for resized image
                intrinsics = torch.as_tensor([
                    intrinsics_K[0, 0], intrinsics_K[1, 1],
                    intrinsics_K[0, 2], intrinsics_K[1, 2]
                ])
                intrinsics[0::2] *= w1 / w0
                intrinsics[1::2] *= h1 / h0

                # Initialize DROID on first frame
                if t == 0:
                    args.image_size = [image_tensor.shape[1], image_tensor.shape[2]]
                    droid = Droid(args)

                rgb_list.append(image_tensor[None])
                sensor_depth_list.append(depth_tensor)

                # Track frame
                if t < len(image_list) - 1:
                    droid.track(t, image_tensor[None], depth_tensor, intrinsics=intrinsics, mask=mask)
                else:
                    droid.track_final(t, image_tensor[None], depth_tensor, intrinsics=intrinsics, mask=mask)

            # Terminate and get final poses
            traj_est, depth_est, motion_prob = droid.terminate(
                stream=self._make_image_stream(
                    image_list, depth_anything_dir, aligns, intrinsics_K
                ),
                _opt_intr=self.opt_focal,
                full_ba=True,
                scene_name="scene",
            )

            num_keyframes = droid.video.counter.value

            return traj_est, depth_est, motion_prob, num_keyframes

        finally:
            if droid is not None:
                del droid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _make_image_stream(
        self,
        image_list: List[Path],
        depth_anything_dir: Path,
        aligns: Tuple[float, float, float],
        intrinsics_K: np.ndarray,
    ):
        """Generator for DROID's terminate() stream parameter."""
        import torch
        import cv2

        align_scale, align_shift, normalize_scale = aligns

        for t, img_path in enumerate(image_list):
            image = cv2.imread(str(img_path))
            h0, w0 = image.shape[:2]

            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
            image = image[: h1 - h1 % 8, : w1 - w1 % 8]

            da_path = depth_anything_dir / f"frame_{t:05d}.npy"
            mono_disp = np.float32(np.load(da_path))

            depth = np.clip(
                1.0 / ((1.0 / normalize_scale) * (align_scale * mono_disp + align_shift)),
                1e-4,
                1e4,
            )
            depth[depth < 1e-2] = 0.0

            image_tensor = torch.as_tensor(image).permute(2, 0, 1)
            depth_tensor = torch.as_tensor(depth)
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor[None, None], (h1, w1), mode="nearest-exact"
            ).squeeze()
            depth_tensor = depth_tensor[: h1 - h1 % 8, : w1 - w1 % 8]

            mask = torch.ones_like(depth_tensor)

            intrinsics = torch.as_tensor([
                intrinsics_K[0, 0], intrinsics_K[1, 1],
                intrinsics_K[0, 2], intrinsics_K[1, 2]
            ])
            intrinsics[0::2] *= w1 / w0
            intrinsics[1::2] *= h1 / h0

            yield t, image_tensor[None], depth_tensor, intrinsics, mask

    def _se3_to_c2w(self, poses_se3: np.ndarray) -> np.ndarray:
        """
        Convert SE3 poses to 4x4 camera-to-world matrices.

        DROID stores poses as SE3: [x, y, z, qx, qy, qz, qw] (7D)
        We convert using lietorch: SE3(poses).inv().matrix()

        Args:
            poses_se3: (N, 7) SE3 poses

        Returns:
            cam_c2w: (N, 4, 4) camera-to-world matrices
        """
        import torch
        from lietorch import SE3

        poses_th = torch.as_tensor(poses_se3, device="cpu")
        cam_c2w = SE3(poses_th).inv().matrix().numpy()

        return cam_c2w

    def to_transforms_json(
        self,
        result: MegaSamResult,
        frame_paths: List[Path],
        output_path: Path,
        image_size: Tuple[int, int],
    ):
        """
        Export Mega-SAM result to Nerfstudio transforms.json format.

        Converts from COLMAP/Mega-SAM convention to OpenGL convention
        by flipping Y and Z columns.

        Args:
            result: MegaSamResult from estimate_poses()
            frame_paths: List of frame paths
            output_path: Path to save transforms.json
            image_size: (width, height) of original images
        """
        width, height = image_size
        K = result.intrinsics

        # Convert to OpenGL convention (flip Y and Z)
        cam_c2w = result.poses.copy()
        for i in range(cam_c2w.shape[0]):
            cam_c2w[i, :3, 1:3] *= -1

        transforms = {
            "camera_model": "OPENCV",
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": width,
            "h": height,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "frames": []
        }

        for i, frame_path in enumerate(frame_paths):
            if i < len(cam_c2w):
                frame = {
                    "file_path": f"./frames/{frame_path.name}",
                    "transform_matrix": cam_c2w[i].tolist()
                }
                transforms["frames"].append(frame)

        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Exported {len(transforms['frames'])} poses to {output_path}")


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
    2. Runs camera pose estimation with cascading fallback:
       - megasam (primary) -> hloc/GLOMAP -> DUSt3R
    3. Generates sparse point cloud via COLMAP SfM (or dense from Mega-SAM)

    Output: frames/, transforms.json, sparse/points3D.ply (or megasam/ outputs)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        # Default to megasam for Instant4D compatibility
        self.pose_estimator = self.config.get("pose_estimator", "megasam")
        self.frame_interval = self.config.get("frame_interval", 1)  # Extract every Nth frame
        self.max_frames = self.config.get("max_frames", 300)

        # Mega-SAM specific configuration
        self.megasam_config = {
            "opt_focal": self.config.get("megasam_opt_focal", True),
            "max_frames": self.config.get("megasam_max_frames", 300),
            "disable_vis": True,
        }

        # Initialize Mega-SAM estimator (lazy loaded)
        self._megasam_estimator = None

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

        # Store Mega-SAM outputs for later inclusion in metadata
        megasam_outputs = {}

        if self.pose_estimator == "megasam":
            try:
                megasam_result = self._run_megasam_pipeline(
                    frames_dir, output_path, transforms_path,
                    lambda p, m: report(0.3 + p * 0.6, m)  # Scale progress 0.3-0.9
                )
                report(0.9, "Mega-SAM pipeline completed")

                # Store Mega-SAM outputs for downstream stages
                megasam_outputs = {
                    "depth_maps_dir": megasam_result.depth_maps_dir,
                    "motion_prob_path": megasam_result.motion_prob_path,
                    "megasam_metrics": megasam_result.metrics,
                }

            except MegaSamError as e:
                # Cascade fallback to hloc
                logger.warning(f"Mega-SAM pipeline failed: {e}")
                report(0.5, "Falling back to hloc/GLOMAP...")
                try:
                    self._run_hloc_pipeline(frames_dir, transforms_path, sparse_path)
                    report(0.9, "hloc pipeline completed (fallback)")
                except RuntimeError as e2:
                    # Further cascade to DUSt3R
                    logger.warning(f"hloc pipeline also failed: {e2}")
                    report(0.7, "Falling back to DUSt3R...")
                    self._run_dust3r_pipeline(frames_dir, transforms_path)
                    report(0.9, "DUSt3R fallback completed")

        elif self.pose_estimator == "hloc":
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

        # Include Mega-SAM outputs if available
        if megasam_outputs:
            metadata.update(megasam_outputs)

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

    def _get_megasam_estimator(self) -> MegaSamPoseEstimator:
        """Get or create Mega-SAM pose estimator (lazy initialization)."""
        if self._megasam_estimator is None:
            self._megasam_estimator = MegaSamPoseEstimator(self.megasam_config)
        return self._megasam_estimator

    def _run_megasam_pipeline(
        self,
        frames_dir: Path,
        output_dir: Path,
        transforms_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MegaSamResult:
        """
        Run Mega-SAM pipeline for pose estimation.

        Mega-SAM is the preferred pose estimator for Instant4D as it provides:
        - Camera poses via DROID SLAM
        - Dense depth maps per frame
        - Motion probability masks for dynamic/static separation

        Pipeline:
        1. UniDepth for metric depth + FOV estimation
        2. Depth-Anything for mono disparity
        3. Depth alignment (scale/shift calibration)
        4. DROID SLAM for camera tracking
        5. Export to transforms.json

        Args:
            frames_dir: Directory containing extracted frames
            output_dir: Working directory for outputs
            transforms_path: Path to save transforms.json
            progress_callback: Optional callback(progress, message)

        Returns:
            MegaSamResult with poses, depth, motion probability

        Raises:
            MegaSamError: If pipeline fails (caller should fall back to hloc)
        """
        estimator = self._get_megasam_estimator()

        # Check availability before running
        if not estimator.is_available():
            raise MegaSamError(
                "Mega-SAM dependencies not available. "
                "Ensure lietorch is compiled and checkpoints are downloaded. "
                "Run: bash scripts/setup_megasam.sh"
            )

        # Run pose estimation
        result = estimator.estimate_poses(
            frames_dir=frames_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        # Validate result quality
        self._validate_megasam_result(result, len(list(frames_dir.glob("*.jpg"))))

        # Export to transforms.json
        frame_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        image_size = (self._video_metadata["width"], self._video_metadata["height"])
        estimator.to_transforms_json(result, frame_paths, transforms_path, image_size)

        logger.info(
            f"Mega-SAM completed: {result.num_keyframes} keyframes, "
            f"FOV={result.metrics.get('median_fov', 0):.1f}°"
        )

        return result

    def _validate_megasam_result(self, result: MegaSamResult, total_frames: int):
        """
        Validate Mega-SAM output quality, raise MegaSamError if insufficient.

        Validation criteria:
        - Keyframe count >= 50% of input frames
        - Poses are finite (no NaN/Inf)
        - Intrinsics are reasonable (100 < focal < 5000)
        """
        # Criterion 1: Sufficient keyframes
        if result.num_keyframes < total_frames * 0.5:
            raise MegaSamError(
                f"Too few keyframes: {result.num_keyframes}/{total_frames} "
                f"({result.num_keyframes/total_frames:.0%} < 50%)"
            )

        # Criterion 2: Valid poses
        if not np.isfinite(result.poses).all():
            raise MegaSamError("Invalid poses (NaN/Inf detected)")

        # Criterion 3: Reasonable intrinsics
        fx = result.intrinsics[0, 0]
        fy = result.intrinsics[1, 1]
        if fx < 100 or fy < 100 or fx > 5000 or fy > 5000:
            raise MegaSamError(
                f"Unreasonable focal length: fx={fx:.1f}, fy={fy:.1f} "
                "(expected 100-5000)"
            )

        logger.info(
            f"Mega-SAM validation passed: {result.num_keyframes} keyframes, "
            f"fx={fx:.1f}, fy={fy:.1f}"
        )

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
