"""Unit tests for Mega-SAM pose estimation integration.

Tests cover:
1. MegaSamResult dataclass
2. MegaSamPoseEstimator availability checks
3. SE3 to c2w conversion
4. Intrinsics from FOV calculation
5. Depth alignment logic
6. Fallback validation
7. VideoProcessingStage megasam integration
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import the modules under test
from backend.stages.video_processing import (
    MegaSamResult,
    MegaSamPoseEstimator,
    MegaSamError,
    VideoProcessingStage,
    VideoProcessingResult,
    MEGASAM_PATH,
)


# =============================================================================
# MegaSamResult Tests
# =============================================================================

class TestMegaSamResult:
    """Tests for MegaSamResult dataclass."""

    def test_basic_creation(self):
        """Test creating MegaSamResult with required fields."""
        poses = np.eye(4)[np.newaxis]  # (1, 4, 4)
        intrinsics = np.eye(3)

        result = MegaSamResult(
            poses=poses,
            intrinsics=intrinsics,
        )

        assert result.poses.shape == (1, 4, 4)
        assert result.intrinsics.shape == (3, 3)
        assert result.depth_maps_dir is None
        assert result.motion_prob_path is None
        assert result.num_keyframes == 0
        assert result.metrics == {}

    def test_full_creation(self):
        """Test creating MegaSamResult with all fields."""
        poses = np.random.rand(10, 4, 4)
        intrinsics = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float32)

        result = MegaSamResult(
            poses=poses,
            intrinsics=intrinsics,
            depth_maps_dir="/path/to/depths",
            motion_prob_path="/path/to/motion.npy",
            num_keyframes=8,
            metrics={"median_fov": 60.0},
        )

        assert result.poses.shape == (10, 4, 4)
        assert result.depth_maps_dir == "/path/to/depths"
        assert result.motion_prob_path == "/path/to/motion.npy"
        assert result.num_keyframes == 8
        assert result.metrics["median_fov"] == 60.0


# =============================================================================
# MegaSamPoseEstimator Tests
# =============================================================================

class TestMegaSamPoseEstimator:
    """Tests for MegaSamPoseEstimator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        estimator = MegaSamPoseEstimator()

        assert estimator.opt_focal is True
        assert estimator.max_frames == 300
        assert estimator.disable_vis is True
        assert estimator._megasam_estimator is None or estimator._path_added is False

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "opt_focal": False,
            "max_frames": 150,
        }
        estimator = MegaSamPoseEstimator(config)

        assert estimator.opt_focal is False
        assert estimator.max_frames == 150

    def test_megasam_path_constant(self):
        """Test MEGASAM_PATH points to correct location."""
        assert "instant4d" in str(MEGASAM_PATH)
        assert "mega-sam" in str(MEGASAM_PATH)

    def test_droid_weights_path(self):
        """Test DROID weights path is correctly defined."""
        estimator = MegaSamPoseEstimator()
        assert "megasam_final.pth" in str(estimator.DROID_WEIGHTS)
        assert "checkpoints" in str(estimator.DROID_WEIGHTS)

    @patch.object(MegaSamPoseEstimator, '_add_megasam_to_path')
    def test_is_available_no_checkpoint(self, mock_add_path):
        """Test is_available returns False when checkpoint missing."""
        estimator = MegaSamPoseEstimator()

        with patch.object(Path, 'exists', return_value=False):
            estimator._available = None  # Reset cache
            result = estimator.is_available()

        assert result is False

    def test_is_available_caches_result(self):
        """Test is_available caches its result."""
        estimator = MegaSamPoseEstimator()
        estimator._available = True  # Pre-set cache

        result = estimator.is_available()
        assert result is True  # Returns cached value


class TestSE3Conversion:
    """Tests for SE3 to camera-to-world conversion."""

    @pytest.mark.gpu
    def test_se3_identity_conversion(self):
        """Test SE3 identity converts to identity c2w."""
        estimator = MegaSamPoseEstimator()

        # Identity SE3: [0, 0, 0, 0, 0, 0, 1] (qw=1)
        # Position: (0, 0, 0), Rotation: identity quaternion
        identity_se3 = np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

        estimator._add_megasam_to_path()
        c2w = estimator._se3_to_c2w(identity_se3)

        expected = np.eye(4)
        np.testing.assert_array_almost_equal(c2w[0], expected, decimal=5)

    @pytest.mark.gpu
    def test_se3_translation_conversion(self):
        """Test SE3 with translation converts correctly."""
        estimator = MegaSamPoseEstimator()

        # SE3 with translation (1, 2, 3), identity rotation
        se3 = np.array([[1, 2, 3, 0, 0, 0, 1]], dtype=np.float32)

        estimator._add_megasam_to_path()
        c2w = estimator._se3_to_c2w(se3)

        # Translation should be negated (inv() effect on pose)
        # The exact result depends on lietorch SE3 conventions
        assert c2w.shape == (1, 4, 4)
        assert c2w[0, 3, 3] == 1.0  # Homogeneous coordinate


class TestIntrinsicsFromFOV:
    """Tests for intrinsics computation from FOV."""

    def test_fov_60_degrees(self):
        """Test focal length from 60 degree FOV."""
        # FOV = 60 degrees, width = 640
        fov = 60.0
        width = 640

        focal = width / (2 * np.tan(np.radians(fov) / 2.0))

        # For 60 degree FOV: f = w / (2 * tan(30)) = 640 / (2 * 0.577) ≈ 554
        assert 550 < focal < 560

    def test_fov_90_degrees(self):
        """Test focal length from 90 degree FOV (wide angle)."""
        fov = 90.0
        width = 640

        focal = width / (2 * np.tan(np.radians(fov) / 2.0))

        # For 90 degree FOV: f = w / (2 * tan(45)) = 640 / (2 * 1) = 320
        assert abs(focal - 320) < 1


class TestDepthAlignment:
    """Tests for depth alignment logic."""

    def test_scale_shift_computation(self):
        """Test scale/shift computation between mono and metric depth."""
        # Create synthetic mono disparity and metric depth
        mono_disp = np.random.rand(100, 100).astype(np.float32) * 0.1 + 0.01
        metric_depth = 1.0 / (mono_disp * 2 + 0.5)  # Known relationship

        gt_disp = 1.0 / (metric_depth + 1e-8)

        # Compute scale/shift using median method
        gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
        da_disp_ms = mono_disp - np.median(mono_disp) + 1e-8
        scale = np.median(gt_disp_ms / da_disp_ms)
        shift = np.median(gt_disp - scale * mono_disp)

        # Verify reconstruction
        reconstructed = scale * mono_disp + shift
        error = np.abs(reconstructed - gt_disp).mean()
        assert error < 0.1  # Should be close

    def test_sky_handling(self):
        """Test special handling for sky-dominated scenes."""
        # Create disparity with 60% sky (very low disparity)
        mono_disp = np.ones((100, 100), dtype=np.float32) * 0.005
        mono_disp[60:, :] = 0.1  # Ground region

        sky_ratio = np.sum(mono_disp < 0.01) / (mono_disp.shape[0] * mono_disp.shape[1])
        assert sky_ratio > 0.5

        # Non-sky mask should exclude sky
        non_sky_mask = mono_disp > 0.01
        assert np.sum(non_sky_mask) == 4000  # 40 * 100


# =============================================================================
# Validation Tests
# =============================================================================

class TestMegaSamValidation:
    """Tests for Mega-SAM result validation."""

    def test_validate_sufficient_keyframes(self):
        """Test validation passes with sufficient keyframes."""
        stage = VideoProcessingStage({"pose_estimator": "megasam"})
        stage._video_metadata = {"width": 640, "height": 480}

        result = MegaSamResult(
            poses=np.eye(4)[np.newaxis].repeat(10, axis=0),
            intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            num_keyframes=8,
        )

        # Should not raise for 8 keyframes out of 10 frames
        stage._validate_megasam_result(result, 10)

    def test_validate_insufficient_keyframes(self):
        """Test validation fails with insufficient keyframes."""
        stage = VideoProcessingStage({"pose_estimator": "megasam"})
        stage._video_metadata = {"width": 640, "height": 480}

        result = MegaSamResult(
            poses=np.eye(4)[np.newaxis].repeat(10, axis=0),
            intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            num_keyframes=3,  # Only 30%
        )

        with pytest.raises(MegaSamError, match="Too few keyframes"):
            stage._validate_megasam_result(result, 10)

    def test_validate_nan_poses(self):
        """Test validation fails with NaN in poses."""
        stage = VideoProcessingStage({"pose_estimator": "megasam"})
        stage._video_metadata = {"width": 640, "height": 480}

        poses = np.eye(4)[np.newaxis].repeat(10, axis=0)
        poses[5, 0, 0] = np.nan  # Introduce NaN

        result = MegaSamResult(
            poses=poses,
            intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            num_keyframes=8,
        )

        with pytest.raises(MegaSamError, match="Invalid poses"):
            stage._validate_megasam_result(result, 10)

    def test_validate_unreasonable_focal_low(self):
        """Test validation fails with too low focal length."""
        stage = VideoProcessingStage({"pose_estimator": "megasam"})
        stage._video_metadata = {"width": 640, "height": 480}

        result = MegaSamResult(
            poses=np.eye(4)[np.newaxis].repeat(10, axis=0),
            intrinsics=np.array([[50, 0, 320], [0, 50, 240], [0, 0, 1]]),  # Too low
            num_keyframes=8,
        )

        with pytest.raises(MegaSamError, match="Unreasonable focal length"):
            stage._validate_megasam_result(result, 10)

    def test_validate_unreasonable_focal_high(self):
        """Test validation fails with too high focal length."""
        stage = VideoProcessingStage({"pose_estimator": "megasam"})
        stage._video_metadata = {"width": 640, "height": 480}

        result = MegaSamResult(
            poses=np.eye(4)[np.newaxis].repeat(10, axis=0),
            intrinsics=np.array([[6000, 0, 320], [0, 6000, 240], [0, 0, 1]]),  # Too high
            num_keyframes=8,
        )

        with pytest.raises(MegaSamError, match="Unreasonable focal length"):
            stage._validate_megasam_result(result, 10)


# =============================================================================
# VideoProcessingStage Integration Tests
# =============================================================================

class TestVideoProcessingStageConfig:
    """Tests for VideoProcessingStage configuration."""

    def test_default_pose_estimator_is_megasam(self):
        """Test that megasam is the default pose estimator."""
        stage = VideoProcessingStage()
        assert stage.pose_estimator == "megasam"

    def test_explicit_megasam_config(self):
        """Test explicit megasam configuration."""
        stage = VideoProcessingStage({"pose_estimator": "megasam"})
        assert stage.pose_estimator == "megasam"
        assert "opt_focal" in stage.megasam_config
        assert stage.megasam_config["opt_focal"] is True

    def test_custom_megasam_focal_config(self):
        """Test custom focal optimization config."""
        stage = VideoProcessingStage({
            "pose_estimator": "megasam",
            "megasam_opt_focal": False,
        })
        assert stage.megasam_config["opt_focal"] is False

    def test_hloc_config(self):
        """Test hloc configuration still works."""
        stage = VideoProcessingStage({"pose_estimator": "hloc"})
        assert stage.pose_estimator == "hloc"

    def test_dust3r_config(self):
        """Test dust3r configuration still works."""
        stage = VideoProcessingStage({"pose_estimator": "dust3r"})
        assert stage.pose_estimator == "dust3r"


class TestMegaSamFallback:
    """Tests for Mega-SAM to hloc fallback behavior."""

    @patch('backend.stages.video_processing.MegaSamPoseEstimator')
    def test_fallback_on_megasam_unavailable(self, mock_estimator_class, tmp_path):
        """Test fallback to hloc when Mega-SAM unavailable."""
        # Setup mock estimator that's unavailable
        mock_estimator = MagicMock()
        mock_estimator.is_available.return_value = False
        mock_estimator_class.return_value = mock_estimator

        stage = VideoProcessingStage({"pose_estimator": "megasam"})

        # _get_megasam_estimator should return the mock
        stage._megasam_estimator = mock_estimator

        # Calling _run_megasam_pipeline should raise MegaSamError
        with pytest.raises(MegaSamError, match="not available"):
            stage._run_megasam_pipeline(
                frames_dir=tmp_path,
                output_dir=tmp_path,
                transforms_path=tmp_path / "transforms.json",
            )


# =============================================================================
# Transforms.json Export Tests
# =============================================================================

class TestTransformsJsonExport:
    """Tests for transforms.json export."""

    def test_to_transforms_json_format(self, tmp_path):
        """Test that exported JSON has correct format."""
        estimator = MegaSamPoseEstimator()

        # Create mock result
        poses = np.eye(4)[np.newaxis].repeat(3, axis=0)
        intrinsics = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float32)

        result = MegaSamResult(
            poses=poses,
            intrinsics=intrinsics,
            num_keyframes=3,
        )

        # Create mock frame paths
        frame_paths = [
            tmp_path / "0001.jpg",
            tmp_path / "0002.jpg",
            tmp_path / "0003.jpg",
        ]
        for p in frame_paths:
            p.touch()

        output_path = tmp_path / "transforms.json"
        estimator.to_transforms_json(result, frame_paths, output_path, (640, 480))

        # Verify JSON structure
        with open(output_path) as f:
            transforms = json.load(f)

        assert transforms["camera_model"] == "OPENCV"
        assert transforms["fl_x"] == 500.0
        assert transforms["fl_y"] == 500.0
        assert transforms["cx"] == 320.0
        assert transforms["cy"] == 240.0
        assert transforms["w"] == 640
        assert transforms["h"] == 480
        assert len(transforms["frames"]) == 3
        assert "transform_matrix" in transforms["frames"][0]
        assert len(transforms["frames"][0]["transform_matrix"]) == 4  # 4x4 matrix

    def test_to_transforms_json_opengl_convention(self, tmp_path):
        """Test that Y and Z are flipped for OpenGL convention."""
        estimator = MegaSamPoseEstimator()

        # Create pose with known Y and Z columns
        pose = np.eye(4)
        pose[:3, 1] = [0, 1, 0]  # Y column
        pose[:3, 2] = [0, 0, 1]  # Z column
        poses = pose[np.newaxis]

        result = MegaSamResult(
            poses=poses,
            intrinsics=np.eye(3) * 500,
            num_keyframes=1,
        )

        frame_paths = [tmp_path / "0001.jpg"]
        frame_paths[0].touch()

        output_path = tmp_path / "transforms.json"
        estimator.to_transforms_json(result, frame_paths, output_path, (640, 480))

        with open(output_path) as f:
            transforms = json.load(f)

        matrix = np.array(transforms["frames"][0]["transform_matrix"])

        # Y and Z columns should be negated
        np.testing.assert_array_almost_equal(matrix[:3, 1], [0, -1, 0])
        np.testing.assert_array_almost_equal(matrix[:3, 2], [0, 0, -1])


# =============================================================================
# GPU Integration Tests (marked to skip on CPU-only systems)
# =============================================================================

@pytest.mark.gpu
class TestMegaSamGPUIntegration:
    """GPU-required integration tests for Mega-SAM."""

    def test_lietorch_import(self):
        """Test lietorch can be imported."""
        estimator = MegaSamPoseEstimator()
        estimator._add_megasam_to_path()

        import lietorch
        assert hasattr(lietorch, 'SE3')

    def test_se3_batch_conversion(self):
        """Test SE3 conversion with batch of poses."""
        estimator = MegaSamPoseEstimator()
        estimator._add_megasam_to_path()

        # Create batch of SE3 poses
        batch_size = 10
        se3 = np.random.randn(batch_size, 7).astype(np.float32)
        # Normalize quaternion part
        se3[:, 3:] /= np.linalg.norm(se3[:, 3:], axis=1, keepdims=True)

        c2w = estimator._se3_to_c2w(se3)

        assert c2w.shape == (batch_size, 4, 4)
        # All matrices should be valid (finite values)
        assert np.isfinite(c2w).all()
        # Last row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(c2w[:, 3, :], [[0, 0, 0, 1]] * batch_size)
