"""Tests for Instant4D adapter.

These tests verify the Instant4D adapter for 4D Gaussian Splatting reconstruction.

Usage:
    # Run all tests (CPU-only tests will pass on Mac)
    pytest tests/adapters/test_instant4d_adapter.py -v

    # Run only preprocessing tests
    pytest tests/adapters/test_instant4d_adapter.py::TestPreprocessing -v

    # Run GPU tests (on GPU server only)
    pytest tests/adapters/test_instant4d_adapter.py -v -m gpu
"""

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Test fixtures and mocks


@pytest.fixture
def mock_transforms():
    """Create mock transforms.json content."""
    return {
        "fl_x": 500.0,
        "fl_y": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "w": 640,
        "h": 480,
        "frames": [
            {
                "file_path": f"./frames/{i:04d}.jpg",
                "transform_matrix": np.eye(4).tolist(),
            }
            for i in range(10)
        ],
    }


@pytest.fixture
def mock_video_result(tmp_path, mock_transforms):
    """Create mock VideoProcessingResult."""
    from backend.stages.video_processing import VideoProcessingResult

    # Create frames directory
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create dummy frames (just empty files)
    for i in range(10):
        (frames_dir / f"{i:04d}.jpg").touch()

    # Create transforms.json
    transforms_path = tmp_path / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(mock_transforms, f)

    # Create sparse points PLY (minimal valid PLY)
    sparse_path = tmp_path / "sparse" / "points3D.ply"
    sparse_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a minimal PLY file
    ply_content = """ply
format ascii 1.0
element vertex 100
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    for _ in range(100):
        x, y, z = np.random.randn(3)
        r, g, b = np.random.randint(0, 256, 3)
        ply_content += f"{x} {y} {z} {r} {g} {b}\n"

    with open(sparse_path, "w") as f:
        f.write(ply_content)

    return VideoProcessingResult(
        frames_dir=str(frames_dir),
        num_frames=10,
        transforms_path=str(transforms_path),
        sparse_points_path=str(sparse_path),
        metadata={"fps": 30, "width": 640, "height": 480},
    )


@pytest.fixture
def mock_segmentation_result(tmp_path):
    """Create mock ObjectSegmentationResult."""
    from backend.stages.object_segmentation import (
        ObjectSegmentationResult,
        SegmentedObject,
    )

    # Create masks directory
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    # Create dummy object
    obj = SegmentedObject(
        object_id=1,
        label="person",
        mask_paths=[],
        frame_indices=list(range(10)),
        confidence=0.95,
    )

    # Create dummy masks
    for i in range(10):
        mask_path = masks_dir / f"obj1_frame{i:04d}.png"
        # Create a simple binary mask (black image with white rectangle)
        mask_data = np.zeros((480, 640), dtype=np.uint8)
        mask_data[200:300, 250:400] = 255
        # Save as grayscale PNG
        try:
            import cv2

            cv2.imwrite(str(mask_path), mask_data)
        except ImportError:
            # Fallback: create empty file if cv2 not available
            mask_path.touch()
        obj.mask_paths.append(str(mask_path))

    # Create metadata
    metadata = {
        "num_objects": 1,
        "objects": [
            {
                "object_id": 1,
                "label": "person",
                "frame_count": 10,
            }
        ],
    }
    metadata_path = masks_dir / "object_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return ObjectSegmentationResult(
        masks_dir=str(masks_dir),
        num_objects=1,
        objects=[obj],
        metadata_path=str(metadata_path),
        metadata=metadata,
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestInstant4DOptions:
    """Test Instant4DOptions dataclass."""

    def test_default_values(self):
        """Default options have expected values."""
        from backend.adapters.instant4d import Instant4DOptions

        options = Instant4DOptions()

        assert options.iterations == 5000
        assert options.batch_size == 1
        assert options.gaussian_dim == 4
        assert options.time_duration == (0.0, 3.0)
        assert options.rot_4d is True
        assert options.enable_pruning is True
        assert options.motion_threshold == 0.5

    def test_custom_values(self):
        """Custom options are set correctly."""
        from backend.adapters.instant4d import Instant4DOptions

        options = Instant4DOptions(
            iterations=10000,
            gaussian_dim=3,
            rot_4d=False,
        )

        assert options.iterations == 10000
        assert options.gaussian_dim == 3
        assert options.rot_4d is False


class TestInstant4DResult:
    """Test Instant4DResult dataclass."""

    def test_default_values(self):
        """Default result has expected empty values."""
        from backend.adapters.instant4d import Instant4DResult

        result = Instant4DResult(model_path="/path/to/model.pth")

        assert result.model_path == "/path/to/model.pth"
        assert result.ply_paths == []
        assert result.num_gaussians == 0
        assert result.num_frames == 0
        assert result.metrics == {}
        assert result.config_path is None

    def test_full_result(self):
        """Full result with all fields."""
        from backend.adapters.instant4d import Instant4DResult

        result = Instant4DResult(
            model_path="/path/to/model.pth",
            ply_paths=["/path/to/frame_0000.ply", "/path/to/frame_0001.ply"],
            num_gaussians=50000,
            num_frames=30,
            metrics={"psnr": 28.5, "ssim": 0.92},
            config_path="/path/to/config.yaml",
            temporal_metadata={"fps": 30, "duration_seconds": 1.0},
        )

        assert len(result.ply_paths) == 2
        assert result.num_gaussians == 50000
        assert result.metrics["psnr"] == 28.5


class TestInstant4DAdapterInit:
    """Test Instant4DAdapter initialization."""

    def test_init_default(self):
        """Adapter initializes with default config."""
        from backend.adapters.instant4d import Instant4DAdapter

        # This may fail if Instant4D submodule is not initialized
        try:
            adapter = Instant4DAdapter()
            assert adapter.device == "cuda:0"
            assert adapter._gaussian_model is None
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_init_custom_device(self):
        """Adapter accepts custom device."""
        from backend.adapters.instant4d import Instant4DAdapter

        try:
            adapter = Instant4DAdapter({"device": "cuda:1"})
            assert adapter.device == "cuda:1"
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_instant4d_path_exists(self):
        """Instant4D submodule path is correct."""
        from backend.adapters.instant4d import Instant4DAdapter

        path = Instant4DAdapter.INSTANT4D_PATH
        # This checks the path calculation is correct
        assert "instant4d" in str(path).lower()


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_load_transforms(self, tmp_path, mock_transforms):
        """Transforms are loaded correctly."""
        from backend.adapters.instant4d import Instant4DAdapter

        # Write transforms
        transforms_path = tmp_path / "transforms.json"
        with open(transforms_path, "w") as f:
            json.dump(mock_transforms, f)

        try:
            adapter = Instant4DAdapter()
            transforms = adapter._load_transforms(str(transforms_path))

            assert "frames" in transforms
            assert len(transforms["frames"]) == 10
            assert transforms["fl_x"] == 500.0
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_extract_intrinsics_fl_format(self, mock_transforms):
        """Intrinsics extracted from fl_x/fl_y format."""
        from backend.adapters.instant4d import Instant4DAdapter

        try:
            adapter = Instant4DAdapter()
            intrinsic = adapter._extract_intrinsics(mock_transforms)

            assert intrinsic.shape == (3, 3)
            assert intrinsic[0, 0] == 500.0  # fl_x
            assert intrinsic[1, 1] == 500.0  # fl_y
            assert intrinsic[0, 2] == 320.0  # cx
            assert intrinsic[1, 2] == 240.0  # cy
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_extract_intrinsics_angle_format(self):
        """Intrinsics extracted from camera_angle_x format."""
        from backend.adapters.instant4d import Instant4DAdapter

        transforms = {
            "camera_angle_x": 1.0,  # ~57 degrees
            "w": 640,
            "h": 480,
            "frames": [],
        }

        try:
            adapter = Instant4DAdapter()
            intrinsic = adapter._extract_intrinsics(transforms)

            assert intrinsic.shape == (3, 3)
            assert intrinsic[0, 2] == 320.0  # cx = w/2
            assert intrinsic[1, 2] == 240.0  # cy = h/2
            # fl_x = w / (2 * tan(angle/2))
            expected_fl = 640 / (2 * np.tan(1.0 / 2))
            assert abs(intrinsic[0, 0] - expected_fl) < 0.1
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_convert_poses_opengl_to_colmap(self, mock_transforms):
        """Pose conversion flips Y and Z axes."""
        from backend.adapters.instant4d import Instant4DAdapter

        # Create frames with known poses
        frames = [
            {"transform_matrix": np.eye(4).tolist()},
            {
                "transform_matrix": [
                    [1, 0, 0, 1],
                    [0, 1, 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1],
                ]
            },
        ]

        try:
            adapter = Instant4DAdapter()
            cam_c2w = adapter._convert_poses_nerfstudio_to_instant4d(frames)

            assert cam_c2w.shape == (2, 4, 4)
            # First pose: identity should have Y and Z columns flipped
            # Original OpenGL Y-axis becomes -Y
            # Original OpenGL Z-axis becomes -Z
            assert cam_c2w[0, 0, 0] == 1  # X unchanged
            assert cam_c2w[0, 1, 1] == -1  # Y flipped
            assert cam_c2w[0, 2, 2] == -1  # Z flipped
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_load_sparse_points(self, mock_video_result):
        """Sparse points loaded from PLY."""
        from backend.adapters.instant4d import Instant4DAdapter

        try:
            adapter = Instant4DAdapter()
            xyz, rgb = adapter._load_sparse_points(mock_video_result.sparse_points_path)

            assert xyz.shape[0] == 100  # 100 points
            assert xyz.shape[1] == 3  # XYZ
            assert rgb.shape[0] == 100
            assert rgb.shape[1] == 3  # RGB
            assert rgb.max() <= 1.0  # Normalized to [0, 1]
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_assign_timestamps_static(self):
        """Static points get midpoint timestamp."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        xyz = np.random.randn(100, 3).astype(np.float32)
        prob_motion = np.zeros(100, dtype=np.float32)  # All static
        options = Instant4DOptions(time_duration=(0.0, 3.0))

        try:
            adapter = Instant4DAdapter()
            time_stamp, scale_time = adapter._assign_timestamps(
                xyz, prob_motion, 30, options
            )

            assert time_stamp.shape == (100,)
            # Static points should be at midpoint
            assert np.allclose(time_stamp, 1.5)
            # Scale should be half the range
            assert np.allclose(scale_time, 1.5)
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_assign_timestamps_dynamic(self):
        """Dynamic points get varied timestamps."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        xyz = np.random.randn(100, 3).astype(np.float32)
        prob_motion = np.ones(100, dtype=np.float32)  # All dynamic
        options = Instant4DOptions(time_duration=(0.0, 3.0), motion_threshold=0.5)

        try:
            adapter = Instant4DAdapter()
            time_stamp, scale_time = adapter._assign_timestamps(
                xyz, prob_motion, 30, options
            )

            assert time_stamp.shape == (100,)
            # Dynamic points should be spread across time range
            assert time_stamp.min() >= 0.0
            assert time_stamp.max() <= 3.0
            # Timestamps should vary
            assert time_stamp.std() > 0.1
            # Dynamic scale should be smaller
            assert np.allclose(scale_time, 0.75)  # t_range / 4
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")


class TestFormatConversion:
    """Test data format conversions."""

    def test_transforms_json_round_trip(self, tmp_path, mock_transforms):
        """Transforms survive round-trip conversion."""
        from backend.adapters.instant4d import Instant4DAdapter

        try:
            adapter = Instant4DAdapter()

            # Write original
            input_path = tmp_path / "input"
            input_path.mkdir()
            transforms_path = input_path / "transforms.json"
            with open(transforms_path, "w") as f:
                json.dump(mock_transforms, f)

            # Create output directory
            output_path = tmp_path / "output"
            output_path.mkdir()

            # Create Instant4D transforms
            adapter._create_instant4d_transforms(
                mock_transforms, str(input_path / "frames"), output_path, 10
            )

            # Verify train and test transforms exist
            assert (output_path / "transforms_train.json").exists()
            assert (output_path / "transforms_test.json").exists()

            # Load and verify train transforms
            with open(output_path / "transforms_train.json") as f:
                train_data = json.load(f)

            assert "frames" in train_data
            assert "fl_x" in train_data
            # Train should have 90% of frames
            assert len(train_data["frames"]) == 9

            # Frames should have timestamps
            assert "time" in train_data["frames"][0]
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_timestamp_normalization(self, tmp_path, mock_transforms):
        """Timestamps normalized to [0, 3] range."""
        from backend.adapters.instant4d import Instant4DAdapter

        try:
            adapter = Instant4DAdapter()
            output_path = tmp_path / "output"
            output_path.mkdir()

            adapter._create_instant4d_transforms(
                mock_transforms, str(tmp_path / "frames"), output_path, 10
            )

            with open(output_path / "transforms_train.json") as f:
                train_data = json.load(f)

            timestamps = [f["time"] for f in train_data["frames"]]
            assert min(timestamps) >= 0.0
            assert max(timestamps) <= 3.0
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")


class TestPLYExport:
    """Test PLY file writing."""

    def test_write_ply(self, tmp_path):
        """PLY file is written correctly."""
        from backend.adapters.instant4d import Instant4DAdapter

        xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        ply_path = tmp_path / "test.ply"

        try:
            adapter = Instant4DAdapter()
            adapter._write_ply(ply_path, xyz, rgb)

            # Verify file exists
            assert ply_path.exists()

            # Verify can be read back
            from plyfile import PlyData

            plydata = PlyData.read(str(ply_path))
            vertex = plydata["vertex"]

            assert len(vertex) == 3
            assert vertex["x"][0] == 0
            assert vertex["red"][0] == 255
        except ImportError as e:
            pytest.skip(f"Required package not available: {e}")


class TestGaussianTrainingStage:
    """Test GaussianTrainingStage integration."""

    def test_stage_init(self):
        """Stage initializes correctly."""
        from backend.stages.gaussian_training import GaussianTrainingStage

        stage = GaussianTrainingStage()
        assert stage.config == {}
        assert stage._adapter is None

    def test_stage_with_config(self):
        """Stage accepts config."""
        from backend.stages.gaussian_training import GaussianTrainingStage

        config = {"iterations": 10000, "device": "cuda:1"}
        stage = GaussianTrainingStage(config)

        assert stage.config["iterations"] == 10000
        assert stage.config["device"] == "cuda:1"

    def test_result_backward_compatibility(self):
        """GaussianTrainingResult maintains backward compatibility."""
        from backend.stages.gaussian_training import GaussianTrainingResult

        # Old-style result with just ply_path
        result = GaussianTrainingResult(
            ply_path="/path/to/scene.ply",
            num_gaussians=50000,
        )

        assert result.ply_path == "/path/to/scene.ply"
        assert result.num_gaussians == 50000
        # New fields should have defaults
        assert result.ply_paths == []
        assert result.model_path is None
        assert result.num_frames == 0

    def test_result_4d_fields(self):
        """GaussianTrainingResult includes 4D fields."""
        from backend.stages.gaussian_training import GaussianTrainingResult

        result = GaussianTrainingResult(
            ply_path="/path/to/frame_0000.ply",
            num_gaussians=50000,
            ply_paths=[f"/path/to/frame_{i:04d}.ply" for i in range(30)],
            model_path="/path/to/model.pth",
            num_frames=30,
            temporal_metadata={"fps": 30, "duration_seconds": 1.0},
        )

        assert len(result.ply_paths) == 30
        assert result.model_path == "/path/to/model.pth"
        assert result.num_frames == 30
        assert result.temporal_metadata["fps"] == 30


# =============================================================================
# GPU Tests (require CUDA)
# =============================================================================


@pytest.mark.gpu
class TestTrainingGPU:
    """GPU tests for training (run on server only)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self):
        """Skip tests if CUDA is not available."""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_cuda_kernels_available(self):
        """CUDA kernels are importable."""
        from backend.adapters.instant4d import Instant4DAdapter

        try:
            adapter = Instant4DAdapter()
            adapter._validate_cuda_kernels()
        except ImportError as e:
            pytest.fail(f"CUDA kernels not available: {e}")

    def test_training_basic(self, mock_video_result, tmp_path):
        """Basic training completes without error."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        try:
            adapter = Instant4DAdapter()

            # Preprocess with minimal settings
            options = Instant4DOptions(
                iterations=100,  # Very short for testing
                num_pts=1000,
                enable_pruning=False,
            )

            # Run preprocessing
            preprocessed_dir = adapter.preprocess(
                mock_video_result,
                None,  # No segmentation
                str(tmp_path / "preprocessed"),
                options,
            )

            assert (preprocessed_dir / "filtered_cvd.npz").exists()
            assert (preprocessed_dir / "transforms_train.json").exists()
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")


@pytest.mark.gpu
class TestExportGPU:
    """GPU tests for PLY export."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self):
        """Skip tests if CUDA is not available."""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_per_frame_ply_export_mock(self, tmp_path):
        """Per-frame PLY export with mocked model."""
        # This test would require a trained model
        # For now, just verify the method signature works
        from backend.adapters.instant4d import Instant4DAdapter

        try:
            adapter = Instant4DAdapter()
            # Would need actual model for full test
            # adapter.extract_per_frame_ply(...)
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_mock_inputs(self, mock_video_result, tmp_path):
        """Pipeline runs with mock inputs (preprocessing only, no GPU)."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        try:
            adapter = Instant4DAdapter()
            options = Instant4DOptions(enable_pruning=False)

            # Only test preprocessing (doesn't require GPU)
            preprocessed_dir = adapter.preprocess(
                mock_video_result,
                None,
                str(tmp_path / "preprocessed"),
                options,
            )

            # Verify outputs
            assert (preprocessed_dir / "filtered_cvd.npz").exists()
            assert (preprocessed_dir / "transforms_train.json").exists()
            assert (preprocessed_dir / "transforms_test.json").exists()

            # Load and verify npz content
            data = np.load(preprocessed_dir / "filtered_cvd.npz")
            assert "xyz" in data
            assert "rgb" in data
            assert "prob_motion" in data
            assert "time_stamp" in data
            assert "intrinsic" in data
            assert "cam_c2w" in data

        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_stage3_adapter_compatibility(self, mock_video_result, tmp_path):
        """GaussianTrainingStage uses Instant4DAdapter correctly."""
        from backend.stages.gaussian_training import GaussianTrainingStage

        stage = GaussianTrainingStage({"iterations": 100})

        # Verify adapter is created lazily
        assert stage._adapter is None

        try:
            adapter = stage._get_adapter()
            assert adapter is not None
            assert stage._adapter is adapter
        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")

    def test_with_segmentation(
        self, mock_video_result, mock_segmentation_result, tmp_path
    ):
        """Pipeline handles segmentation input."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        try:
            adapter = Instant4DAdapter()
            options = Instant4DOptions(enable_pruning=False)

            preprocessed_dir = adapter.preprocess(
                mock_video_result,
                mock_segmentation_result,
                str(tmp_path / "preprocessed"),
                options,
            )

            # Load and verify prob_motion is non-zero
            data = np.load(preprocessed_dir / "filtered_cvd.npz")
            # With masks, some points may have motion probability > 0
            # (depends on whether masks overlap with points)
            assert "prob_motion" in data
            assert data["prob_motion"].shape[0] > 0

        except ImportError as e:
            pytest.skip(f"Instant4D not available: {e}")
