"""
Utilities for creating test fixtures for Brain Dance integration tests.

These functions generate synthetic data that mimics Stage 1 outputs,
allowing Stage 3 (Instant4D) to be tested without running the full pipeline.
"""

import numpy as np
import json
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class VideoProcessingResult:
    """Mock of the actual VideoProcessingResult from video_processing.py."""

    frames_dir: str
    transforms_path: str
    sparse_points_path: Optional[str] = None
    num_frames: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def create_gradient_image(
    frame_idx: int,
    num_frames: int,
    width: int = 640,
    height: int = 480
) -> np.ndarray:
    """
    Create a synthetic gradient image that changes over time.

    Args:
        frame_idx: Current frame index
        frame_idx: Total number of frames
        width: Image width
        height: Image height

    Returns:
        BGR image array (H, W, 3)
    """
    # Create base gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Time-varying color
    t = frame_idx / max(num_frames - 1, 1)

    # RGB channels that change over time
    r = ((np.sin(xx * 4 + t * 2 * np.pi) + 1) / 2 * 255).astype(np.uint8)
    g = ((np.sin(yy * 4 + t * 2 * np.pi) + 1) / 2 * 255).astype(np.uint8)
    b = ((np.sin((xx + yy) * 2 + t * np.pi) + 1) / 2 * 255).astype(np.uint8)

    # Stack to BGR (OpenCV format)
    img = np.stack([b, g, r], axis=-1)

    return img


def create_circular_camera_path(
    num_frames: int,
    radius: float = 2.0,
    height: float = 1.0,
    focal_length: float = 500.0,
    image_width: int = 640,
    image_height: int = 480
) -> Dict[str, Any]:
    """
    Create synthetic camera poses in a circular path around the origin.

    Args:
        num_frames: Number of camera poses to generate
        radius: Radius of the circular path
        height: Height of the camera above the ground plane
        focal_length: Camera focal length in pixels
        image_width: Image width
        image_height: Image height

    Returns:
        Dictionary in Nerfstudio transforms.json format
    """
    frames = []

    for i in range(num_frames):
        # Angle around the circle
        theta = 2 * np.pi * i / num_frames

        # Camera position (looking at origin)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height

        # Camera looks at origin
        forward = np.array([-x, -y, -z])
        forward = forward / np.linalg.norm(forward)

        # Up vector (world up)
        up = np.array([0, 0, 1])

        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # Recompute up
        up = np.cross(right, forward)

        # Build rotation matrix (camera-to-world)
        R = np.stack([right, up, -forward], axis=1)

        # Build 4x4 transform
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [x, y, z]

        frames.append({
            "file_path": f"frames/{i:04d}.jpg",
            "transform_matrix": c2w.tolist(),
        })

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": image_width / 2,
        "cy": image_height / 2,
        "w": image_width,
        "h": image_height,
        "frames": frames,
    }

    return transforms


def create_random_point_cloud(
    n_points: int = 10000,
    bounds: Tuple[float, float, float] = (2.0, 2.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random point cloud within bounds.

    Args:
        n_points: Number of points
        bounds: (x_range, y_range, z_range) centered at origin

    Returns:
        Tuple of (xyz, rgb) arrays
    """
    xyz = np.random.uniform(
        low=[-bounds[0], -bounds[1], 0],
        high=[bounds[0], bounds[1], bounds[2]],
        size=(n_points, 3)
    ).astype(np.float32)

    # Random colors
    rgb = np.random.randint(0, 255, size=(n_points, 3), dtype=np.uint8)

    return xyz, rgb


def write_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray
) -> None:
    """
    Write a simple PLY file with points and colors.

    Args:
        path: Output PLY path
        xyz: Point positions (N, 3)
        rgb: Point colors (N, 3) as uint8
    """
    n_points = len(xyz)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))

        # Write vertex data
        for i in range(n_points):
            f.write(np.array(xyz[i], dtype=np.float32).tobytes())
            f.write(np.array(rgb[i], dtype=np.uint8).tobytes())


def create_synthetic_video_result(
    output_dir: Path,
    num_frames: int = 30,
    image_width: int = 640,
    image_height: int = 480
) -> VideoProcessingResult:
    """
    Create a minimal VideoProcessingResult for testing Instant4D
    without running Stage 1.

    Args:
        output_dir: Directory to write synthetic data
        num_frames: Number of frames to generate
        image_width: Frame width
        image_height: Frame height

    Returns:
        VideoProcessingResult with synthetic frames and transforms
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic frames
    for i in range(num_frames):
        img = create_gradient_image(i, num_frames, image_width, image_height)
        cv2.imwrite(str(frames_dir / f"{i:04d}.jpg"), img)

    # Create synthetic camera poses
    transforms = create_circular_camera_path(
        num_frames,
        image_width=image_width,
        image_height=image_height
    )
    transforms_path = output_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    # Create sparse point cloud
    xyz, rgb = create_random_point_cloud(n_points=10000)
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    sparse_path = sparse_dir / "points3D.ply"
    write_ply(sparse_path, xyz, rgb)

    return VideoProcessingResult(
        frames_dir=str(frames_dir),
        transforms_path=str(transforms_path),
        sparse_points_path=str(sparse_path),
        num_frames=num_frames,
        metadata={
            "fps": 30.0,
            "width": image_width,
            "height": image_height,
        }
    )


def create_megasam_video_result(
    output_dir: Path,
    num_frames: int = 30,
    image_width: int = 640,
    image_height: int = 480
) -> VideoProcessingResult:
    """
    Create a VideoProcessingResult with synthetic Mega-SAM outputs.

    Includes depth_maps_dir and motion_prob_path in metadata.

    Args:
        output_dir: Directory to write synthetic data
        num_frames: Number of frames to generate
        image_width: Frame width
        image_height: Frame height

    Returns:
        VideoProcessingResult with Mega-SAM-style outputs
    """
    # First create base result
    result = create_synthetic_video_result(
        output_dir, num_frames, image_width, image_height
    )

    output_dir = Path(output_dir)

    # Create synthetic depth maps
    depth_dir = output_dir / "megasam" / "outputs"
    depth_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        # Synthetic depth: gradient from near to far
        depth = np.linspace(0.5, 5.0, image_width).reshape(1, -1)
        depth = np.tile(depth, (image_height, 1)).astype(np.float32)

        # Add some noise
        depth += np.random.normal(0, 0.1, depth.shape).astype(np.float32)
        depth = np.clip(depth, 0.1, 10.0)

        np.savez(
            depth_dir / f"{i:04d}_droid.npz",
            depth=depth,
            confidence=np.ones_like(depth)
        )

    # Create synthetic motion probability
    # Shape: (num_frames, H/8, W/8) at 1/8 resolution
    motion_shape = (num_frames, image_height // 8, image_width // 8)

    # Create motion probability with some "moving objects" in the center
    motion_prob = np.zeros(motion_shape, dtype=np.float32)

    # Add a moving region in the center
    center_h = image_height // 8 // 2
    center_w = image_width // 8 // 2
    radius = min(center_h, center_w) // 3

    for t in range(num_frames):
        # Moving region shifts over time
        offset = int((t / num_frames) * center_w / 2)
        h_start = max(0, center_h - radius)
        h_end = min(motion_shape[1], center_h + radius)
        w_start = max(0, center_w - radius + offset)
        w_end = min(motion_shape[2], center_w + radius + offset)

        motion_prob[t, h_start:h_end, w_start:w_end] = 0.8

    motion_path = output_dir / "megasam" / "reconstructions" / "motion_prob.npy"
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(motion_path, motion_prob)

    # Update metadata
    result.metadata["depth_maps_dir"] = str(depth_dir)
    result.metadata["motion_prob_path"] = str(motion_path)
    result.metadata["megasam_metrics"] = {
        "num_keyframes": num_frames,
        "median_fov": 60.0,
    }

    return result


def create_segmentation_masks(
    output_dir: Path,
    num_frames: int = 30,
    num_objects: int = 3,
    image_width: int = 640,
    image_height: int = 480
) -> Dict[str, Any]:
    """
    Create synthetic segmentation masks for testing.

    Args:
        output_dir: Directory to write masks
        num_frames: Number of frames
        num_objects: Number of objects to create masks for
        image_width: Frame width
        image_height: Frame height

    Returns:
        Dictionary with mask paths and object metadata
    """
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    objects = []

    for obj_id in range(num_objects):
        obj_dir = masks_dir / str(obj_id)
        obj_dir.mkdir(exist_ok=True)

        # Create object with random position and size
        cx = np.random.randint(image_width // 4, 3 * image_width // 4)
        cy = np.random.randint(image_height // 4, 3 * image_height // 4)
        radius = np.random.randint(30, 80)

        mask_paths = []

        for frame_idx in range(num_frames):
            # Object moves slightly between frames
            offset_x = int(10 * np.sin(frame_idx / 5))
            offset_y = int(5 * np.cos(frame_idx / 5))

            # Create circular mask
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.circle(
                mask,
                (cx + offset_x, cy + offset_y),
                radius,
                255,
                -1
            )

            mask_path = obj_dir / f"{frame_idx:04d}.png"
            cv2.imwrite(str(mask_path), mask)
            mask_paths.append(str(mask_path))

        objects.append({
            "object_id": obj_id,
            "label": f"object_{obj_id}",
            "mask_paths": mask_paths,
            "frame_indices": list(range(num_frames)),
        })

    # Write metadata
    metadata = {
        "num_objects": num_objects,
        "num_frames": num_frames,
        "objects": objects,
    }

    metadata_path = output_dir / "object_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "masks_dir": str(masks_dir),
        "metadata_path": str(metadata_path),
        "objects": objects,
    }
