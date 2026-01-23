"""Unit tests for Phase 2: Keyframe Segmentation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from backend.stages.object_segmentation import (
    ObjectSegmentationStage,
    ObjectSegmentationResult,
    SegmentedObject,
    SEGMENTATION_PRESETS,
    SAM2_MODELS,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda to simulate GPU environment."""
    with patch("backend.stages.object_segmentation.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=24 * 1e9  # 24GB VRAM
        )
        mock_torch.cuda.memory_allocated.return_value = 8 * 1e9
        mock_torch.cuda.memory_reserved.return_value = 10 * 1e9
        mock_torch.cuda.empty_cache = MagicMock()
        mock_torch.cuda.synchronize = MagicMock()
        mock_torch.cuda.amp.autocast = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        yield mock_torch


@pytest.fixture
def stage_config():
    """Default stage configuration for testing."""
    return {
        "keyframe_interval": 10,
        "min_object_size": 100,
        "model_size": "tiny",
        "quality_preset": "balanced",
        "allow_cpu": True,  # Allow CPU for tests without GPU
    }


@pytest.fixture
def temp_frames_dir():
    """Create temporary directory with fake frame files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir()

        # Create 25 fake frame files
        for i in range(25):
            frame_path = frames_dir / f"{i+1:04d}.jpg"
            frame_path.touch()

        yield frames_dir


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "output"


# ============================================================================
# Test: _sample_keyframes
# ============================================================================


class TestSampleKeyframes:
    """Tests for _sample_keyframes method."""

    def test_empty_frame_list(self, stage_config):
        """Empty frame list returns empty keyframes."""
        stage = ObjectSegmentationStage(stage_config)
        result = stage._sample_keyframes([])
        assert result == []

    def test_single_frame(self, stage_config):
        """Single frame returns just that frame."""
        stage = ObjectSegmentationStage(stage_config)
        frames = [Path("frame_0001.jpg")]
        result = stage._sample_keyframes(frames)
        assert result == [0]

    def test_two_frames(self, stage_config):
        """Two frames returns both."""
        stage = ObjectSegmentationStage(stage_config)
        frames = [Path(f"frame_{i:04d}.jpg") for i in range(2)]
        result = stage._sample_keyframes(frames)
        assert result == [0, 1]

    def test_always_includes_first_and_last(self, stage_config):
        """First and last frames are always included."""
        stage = ObjectSegmentationStage(stage_config)
        frames = [Path(f"frame_{i:04d}.jpg") for i in range(100)]
        result = stage._sample_keyframes(frames)

        assert 0 in result, "First frame must be included"
        assert 99 in result, "Last frame must be included"

    def test_interval_respected(self, stage_config):
        """Keyframe interval is respected."""
        stage_config["keyframe_interval"] = 10
        stage = ObjectSegmentationStage(stage_config)
        frames = [Path(f"frame_{i:04d}.jpg") for i in range(50)]
        result = stage._sample_keyframes(frames)

        # Should include: 0, 10, 20, 30, 40, 49 (last)
        assert 0 in result
        assert 10 in result
        assert 20 in result
        assert 30 in result
        assert 40 in result
        assert 49 in result

    def test_interval_larger_than_frames(self, stage_config):
        """When interval > num_frames, returns first and last only."""
        stage_config["keyframe_interval"] = 100
        stage = ObjectSegmentationStage(stage_config)
        frames = [Path(f"frame_{i:04d}.jpg") for i in range(5)]
        result = stage._sample_keyframes(frames)

        # Should return first and last
        assert 0 in result
        assert 4 in result
        assert len(result) == 2

    def test_result_is_sorted(self, stage_config):
        """Result is always sorted."""
        stage = ObjectSegmentationStage(stage_config)
        frames = [Path(f"frame_{i:04d}.jpg") for i in range(30)]
        result = stage._sample_keyframes(frames)

        assert result == sorted(result)


# ============================================================================
# Test: _compute_mask_iou
# ============================================================================


class TestComputeMaskIoU:
    """Tests for _compute_mask_iou method."""

    def test_identical_masks(self, stage_config):
        """Identical masks have IoU of 1.0."""
        stage = ObjectSegmentationStage(stage_config)
        mask = np.ones((100, 100), dtype=bool)
        iou = stage._compute_mask_iou(mask, mask)
        assert iou == 1.0

    def test_no_overlap(self, stage_config):
        """Non-overlapping masks have IoU of 0.0."""
        stage = ObjectSegmentationStage(stage_config)
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[:50, :] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:, :] = True

        iou = stage._compute_mask_iou(mask1, mask2)
        assert iou == 0.0

    def test_partial_overlap(self, stage_config):
        """Partially overlapping masks have IoU between 0 and 1."""
        stage = ObjectSegmentationStage(stage_config)
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[:60, :] = True  # Top 60 rows

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[40:, :] = True  # Bottom 60 rows

        # Intersection: rows 40-60 = 20 rows * 100 cols = 2000 pixels
        # Union: rows 0-100 = 100 rows * 100 cols = 10000 pixels
        iou = stage._compute_mask_iou(mask1, mask2)
        assert 0.15 < iou < 0.25  # 2000/10000 = 0.2

    def test_empty_masks(self, stage_config):
        """Empty masks have IoU of 0.0."""
        stage = ObjectSegmentationStage(stage_config)
        mask1 = np.zeros((100, 100), dtype=bool)
        mask2 = np.zeros((100, 100), dtype=bool)
        iou = stage._compute_mask_iou(mask1, mask2)
        assert iou == 0.0


# ============================================================================
# Test: _merge_keyframe_detections
# ============================================================================


class TestMergeKeyframeDetections:
    """Tests for _merge_keyframe_detections method."""

    def test_empty_detections(self, stage_config):
        """Empty detection list returns empty."""
        stage = ObjectSegmentationStage(stage_config)
        result = stage._merge_keyframe_detections([])
        assert result == []

    def test_single_detection(self, stage_config):
        """Single detection returns unchanged."""
        stage = ObjectSegmentationStage(stage_config)
        mask = np.ones((100, 100), dtype=bool)
        detection = {
            "mask": mask,
            "bbox": [0, 0, 100, 100],
            "area": 10000,
            "confidence": 0.9,
            "frame_idx": 0,
        }
        result = stage._merge_keyframe_detections([detection])
        assert len(result) == 1
        assert result[0] == detection

    def test_keeps_highest_confidence(self, stage_config):
        """Keeps detection with highest confidence when merging."""
        stage = ObjectSegmentationStage(stage_config)
        mask = np.ones((100, 100), dtype=bool)

        det1 = {
            "mask": mask,
            "bbox": [0, 0, 100, 100],
            "area": 10000,
            "confidence": 0.7,
            "frame_idx": 0,
        }
        det2 = {
            "mask": mask,  # Same mask = high IoU
            "bbox": [0, 0, 100, 100],
            "area": 10000,
            "confidence": 0.9,  # Higher confidence
            "frame_idx": 10,
        }

        result = stage._merge_keyframe_detections([det1, det2])
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9  # Kept higher confidence

    def test_no_merge_same_frame(self, stage_config):
        """Detections from same frame are not merged."""
        stage = ObjectSegmentationStage(stage_config)
        mask = np.ones((100, 100), dtype=bool)

        det1 = {
            "mask": mask,
            "bbox": [0, 0, 100, 100],
            "area": 10000,
            "confidence": 0.9,
            "frame_idx": 0,  # Same frame
        }
        det2 = {
            "mask": mask,
            "bbox": [0, 0, 100, 100],
            "area": 10000,
            "confidence": 0.8,
            "frame_idx": 0,  # Same frame
        }

        result = stage._merge_keyframe_detections([det1, det2])
        # Both should be kept since they're from the same frame
        # (could be two different objects in the same location on the same frame)
        assert len(result) == 2

    def test_no_merge_low_iou(self, stage_config):
        """Detections with low IoU are not merged."""
        stage = ObjectSegmentationStage(stage_config)

        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[:50, :50] = True  # Top-left quadrant

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:, 50:] = True  # Bottom-right quadrant (no overlap)

        det1 = {
            "mask": mask1,
            "bbox": [0, 0, 50, 50],
            "area": 2500,
            "confidence": 0.9,
            "frame_idx": 0,
        }
        det2 = {
            "mask": mask2,
            "bbox": [50, 50, 50, 50],
            "area": 2500,
            "confidence": 0.8,
            "frame_idx": 10,
        }

        result = stage._merge_keyframe_detections([det1, det2])
        assert len(result) == 2  # Both kept, different objects


# ============================================================================
# Test: _get_quality_preset
# ============================================================================


class TestGetQualityPreset:
    """Tests for _get_quality_preset method."""

    def test_fast_preset(self):
        """Fast preset returns correct parameters."""
        stage = ObjectSegmentationStage({"quality_preset": "fast", "allow_cpu": True})
        preset = stage._get_quality_preset()

        assert preset["points_per_side"] == 48
        assert preset["pred_iou_thresh"] == 0.6
        assert preset["stability_score_thresh"] == 0.75

    def test_balanced_preset(self):
        """Balanced preset returns correct parameters."""
        stage = ObjectSegmentationStage({"quality_preset": "balanced", "allow_cpu": True})
        preset = stage._get_quality_preset()

        assert preset["points_per_side"] == 32
        assert preset["pred_iou_thresh"] == 0.7
        assert preset["stability_score_thresh"] == 0.85

    def test_thorough_preset(self):
        """Thorough preset returns correct parameters."""
        stage = ObjectSegmentationStage({"quality_preset": "thorough", "allow_cpu": True})
        preset = stage._get_quality_preset()

        assert preset["points_per_side"] == 64
        assert preset["pred_iou_thresh"] == 0.8
        assert preset["stability_score_thresh"] == 0.9

    def test_invalid_preset_defaults_to_balanced(self):
        """Invalid preset falls back to balanced."""
        stage = ObjectSegmentationStage({"quality_preset": "invalid", "allow_cpu": True})
        preset = stage._get_quality_preset()

        assert preset == SEGMENTATION_PRESETS["balanced"]


# ============================================================================
# Test: _validate_path
# ============================================================================


class TestValidatePath:
    """Tests for _validate_path security method."""

    def test_valid_path(self, stage_config):
        """Valid path is resolved correctly."""
        stage = ObjectSegmentationStage(stage_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = stage._validate_path(tmpdir, "test_dir")
            assert result == Path(tmpdir).resolve()

    def test_path_traversal_blocked(self, stage_config):
        """Path traversal attempts are blocked."""
        stage = ObjectSegmentationStage(stage_config)
        with pytest.raises(ValueError) as exc_info:
            stage._validate_path("/tmp/../etc/passwd", "test_dir")

        assert "SEG-ERR-009" in str(exc_info.value)
        assert "path traversal" in str(exc_info.value).lower()


# ============================================================================
# Test: Model VRAM Selection
# ============================================================================


class TestSelectModelForVram:
    """Tests for _select_model_for_vram method."""

    def test_requested_model_fits(self, mock_torch_cuda, stage_config):
        """Returns requested model when it fits in VRAM."""
        stage = ObjectSegmentationStage(stage_config)
        result = stage._select_model_for_vram("tiny")
        assert result == "tiny"

    def test_auto_downgrades_for_vram(self, stage_config):
        """Auto-selects smaller model when requested doesn't fit."""
        with patch("backend.stages.object_segmentation.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = MagicMock(
                total_memory=6 * 1e9  # Only 6GB VRAM
            )

            stage = ObjectSegmentationStage(stage_config)
            result = stage._select_model_for_vram("large")

            # Large needs 16GB, should downgrade to tiny (4GB)
            assert result == "tiny"


# ============================================================================
# Test: Cleanup
# ============================================================================


class TestCleanup:
    """Tests for cleanup method."""

    def test_cleanup_all_models(self, stage_config, mock_torch_cuda):
        """Cleanup removes all model references."""
        stage = ObjectSegmentationStage(stage_config)

        # Simulate loaded models
        stage.image_model = MagicMock()
        stage.mask_generator = MagicMock()
        stage.video_predictor = MagicMock()

        stage.cleanup()

        assert stage.image_model is None
        assert stage.mask_generator is None
        assert stage.video_predictor is None
        mock_torch_cuda.cuda.empty_cache.assert_called()


# ============================================================================
# Test: Integration (with mocked SAM-2)
# ============================================================================


class TestSegmentKeyframesIntegration:
    """Integration tests for _segment_keyframes with mocked SAM-2."""

    def test_segment_keyframes_with_mock(self, stage_config, temp_frames_dir):
        """Test full keyframe segmentation flow with mocked SAM-2."""
        with patch("backend.stages.object_segmentation.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            stage = ObjectSegmentationStage(stage_config)

            # Create mock mask generator
            mock_mask = np.ones((100, 100), dtype=bool)
            mock_masks = [
                {
                    "segmentation": mock_mask,
                    "bbox": [0, 0, 100, 100],
                    "area": 10000,
                    "stability_score": 0.95,
                }
            ]

            stage.mask_generator = MagicMock()
            stage.mask_generator.generate.return_value = mock_masks

            # Create real frame files with actual image content
            from PIL import Image

            frame_files = sorted(temp_frames_dir.glob("*.jpg"))
            for frame_path in frame_files:
                img = Image.new("RGB", (100, 100), color="red")
                img.save(frame_path)

            keyframe_indices = [0, 10, 20]

            # Run segmentation
            result = stage._segment_keyframes(frame_files, keyframe_indices)

            # Verify results
            assert len(result) > 0
            assert stage.mask_generator.generate.call_count == 3  # 3 keyframes

            # Check detection format
            det = result[0]
            assert "mask" in det
            assert "bbox" in det
            assert "area" in det
            assert "confidence" in det
            assert "frame_idx" in det


# ============================================================================
# Test: Context Manager
# ============================================================================


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_cleanup(self, stage_config, mock_torch_cuda):
        """Context manager calls cleanup on exit."""
        with ObjectSegmentationStage(stage_config) as stage:
            stage.image_model = MagicMock()
            stage.video_predictor = MagicMock()

        # After exiting context, models should be cleaned up
        assert stage.image_model is None
        assert stage.video_predictor is None
