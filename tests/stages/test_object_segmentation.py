"""Unit tests for Object Segmentation Stage (Phase 2 & 3).

Note: These tests require torch to be installed.
Run in an environment with torch (e.g., Google Colab) or install torch locally.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

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
    """Create temporary directory with real frame files."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir()

        # Create 25 small test images (100x100 pixels)
        for i in range(25):
            frame_path = frames_dir / f"{i+1:04d}.jpg"
            # Create a simple colored image
            img = Image.new("RGB", (100, 100), color=(i * 10 % 256, 100, 150))
            img.save(frame_path)

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
        """Fast preset returns correct parameters (fewer objects, faster)."""
        stage = ObjectSegmentationStage({"quality_preset": "fast", "allow_cpu": True})
        preset = stage._get_quality_preset()

        # Fast: fewer points + higher thresholds = fewer detected objects
        assert preset["points_per_side"] == 24
        assert preset["pred_iou_thresh"] == 0.8
        assert preset["stability_score_thresh"] == 0.9

    def test_balanced_preset(self):
        """Balanced preset returns correct parameters."""
        stage = ObjectSegmentationStage({"quality_preset": "balanced", "allow_cpu": True})
        preset = stage._get_quality_preset()

        assert preset["points_per_side"] == 32
        assert preset["pred_iou_thresh"] == 0.7
        assert preset["stability_score_thresh"] == 0.85

    def test_thorough_preset(self):
        """Thorough preset (deprecated) maps to detailed preset."""
        stage = ObjectSegmentationStage({"quality_preset": "thorough", "allow_cpu": True})
        preset = stage._get_quality_preset()

        # thorough is deprecated, maps to detailed
        # Detailed: more points + lower thresholds = more objects
        assert preset["points_per_side"] == 48
        assert preset["pred_iou_thresh"] == 0.6
        assert preset["stability_score_thresh"] == 0.75

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


# ============================================================================
# Phase 3: Video Object Tracking Tests
# ============================================================================


class TestMaskToBbox:
    """Tests for _mask_to_bbox helper method."""

    def test_simple_mask(self, stage_config):
        """Extract bbox from simple rectangular mask."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True  # Rectangle from (30,20) to (60,40)

        bbox = stage._mask_to_bbox(mask)

        assert bbox == [30.0, 20.0, 59.0, 39.0]  # x1, y1, x2, y2

    def test_empty_mask(self, stage_config):
        """Empty mask returns zeros."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.zeros((100, 100), dtype=bool)
        bbox = stage._mask_to_bbox(mask)

        assert bbox == [0.0, 0.0, 0.0, 0.0]

    def test_full_mask(self, stage_config):
        """Full mask returns image dimensions."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.ones((50, 80), dtype=bool)
        bbox = stage._mask_to_bbox(mask)

        assert bbox == [0.0, 0.0, 79.0, 49.0]


class TestMaskToCentroid:
    """Tests for _mask_to_centroid helper method."""

    def test_centered_mask(self, stage_config):
        """Centered mask returns center of image."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # 20x20 box centered at (50, 50)

        centroid = stage._mask_to_centroid(mask)

        # Center of the masked region
        assert centroid[0] == pytest.approx(49.5, rel=0.1)
        assert centroid[1] == pytest.approx(49.5, rel=0.1)

    def test_corner_mask(self, stage_config):
        """Corner mask returns corner centroid."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.zeros((100, 100), dtype=bool)
        mask[0:10, 0:10] = True  # Top-left corner

        centroid = stage._mask_to_centroid(mask)

        assert centroid[0] == pytest.approx(4.5, rel=0.1)
        assert centroid[1] == pytest.approx(4.5, rel=0.1)

    def test_empty_mask(self, stage_config):
        """Empty mask returns image center."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.zeros((100, 80), dtype=bool)
        centroid = stage._mask_to_centroid(mask)

        # Returns center of image for empty mask
        assert centroid == [40.0, 50.0]


class TestNormalizeBbox:
    """Tests for _normalize_bbox helper method."""

    def test_normalize_full_image(self, stage_config):
        """Bbox covering full image normalizes to [0,0,1,1]."""
        stage = ObjectSegmentationStage(stage_config)

        bbox = [0.0, 0.0, 100.0, 50.0]
        normalized = stage._normalize_bbox(bbox, img_height=50, img_width=100)

        assert normalized == [0.0, 0.0, 1.0, 1.0]

    def test_normalize_centered_box(self, stage_config):
        """Centered box normalizes correctly."""
        stage = ObjectSegmentationStage(stage_config)

        bbox = [25.0, 12.5, 75.0, 37.5]  # Centered in 100x50 image
        normalized = stage._normalize_bbox(bbox, img_height=50, img_width=100)

        assert normalized == [0.25, 0.25, 0.75, 0.75]


class TestInitVideoState:
    """Tests for _init_video_state method."""

    def test_calls_init_state_with_directory_path(self, stage_config, temp_frames_dir):
        """Verify init_state receives directory path, not numpy arrays."""
        stage = ObjectSegmentationStage(stage_config)

        # Mock video predictor
        mock_predictor = MagicMock()
        mock_predictor.init_state.return_value = {"test": "state"}
        stage.video_predictor = mock_predictor

        result = stage._init_video_state(temp_frames_dir)

        # Verify called with string path
        mock_predictor.init_state.assert_called_once()
        call_args = mock_predictor.init_state.call_args
        assert call_args.kwargs["video_path"] == str(temp_frames_dir)

        assert result == {"test": "state"}

    def test_raises_if_predictor_not_loaded(self, stage_config, temp_frames_dir):
        """Raises RuntimeError if video predictor not initialized."""
        stage = ObjectSegmentationStage(stage_config)
        stage.video_predictor = None

        with pytest.raises(RuntimeError, match="SEG-ERR-013"):
            stage._init_video_state(temp_frames_dir)


class TestAddObjectPrompts:
    """Tests for _add_object_prompts method."""

    def test_converts_bbox_to_xyxy_normalized(self, stage_config):
        """Verify bbox converted from XYWH to XYXY normalized."""
        stage = ObjectSegmentationStage(stage_config)

        mock_predictor = MagicMock()
        mock_predictor.add_new_points_or_box.return_value = (0, [0], np.ones((1, 10, 10)))
        stage.video_predictor = mock_predictor

        # Detection with XYWH bbox
        detections = [{
            "mask": np.ones((100, 100), dtype=bool),
            "bbox": [25, 25, 50, 50],  # XYWH: x=25, y=25, w=50, h=50
            "confidence": 0.9,
            "frame_idx": 0,
        }]

        inference_state = {"test": "state"}
        stage._add_object_prompts(inference_state, detections)

        # Verify call
        call_args = mock_predictor.add_new_points_or_box.call_args
        box_arg = call_args.kwargs["box"]

        # Should be normalized XYXY: [0.25, 0.25, 0.75, 0.75]
        expected = np.array([0.25, 0.25, 0.75, 0.75])
        np.testing.assert_array_almost_equal(box_arg, expected)

    def test_assigns_sequential_object_ids(self, stage_config):
        """Verify object IDs are sequential starting from 0."""
        stage = ObjectSegmentationStage(stage_config)

        mock_predictor = MagicMock()
        mock_predictor.add_new_points_or_box.return_value = (0, [0], np.ones((1, 10, 10)))
        stage.video_predictor = mock_predictor

        detections = [
            {"mask": np.ones((100, 100), dtype=bool), "bbox": [0, 0, 10, 10], "confidence": 0.9, "frame_idx": 0},
            {"mask": np.ones((100, 100), dtype=bool), "bbox": [0, 0, 10, 10], "confidence": 0.8, "frame_idx": 5},
            {"mask": np.ones((100, 100), dtype=bool), "bbox": [0, 0, 10, 10], "confidence": 0.7, "frame_idx": 10},
        ]

        inference_state = {"test": "state"}
        result = stage._add_object_prompts(inference_state, detections)

        # Should have been called 3 times with obj_id 0, 1, 2
        assert mock_predictor.add_new_points_or_box.call_count == 3

        calls = mock_predictor.add_new_points_or_box.call_args_list
        assert calls[0].kwargs["obj_id"] == 0
        assert calls[1].kwargs["obj_id"] == 1
        assert calls[2].kwargs["obj_id"] == 2


class TestPropagateMasks:
    """Tests for _propagate_masks method."""

    def test_iterates_generator_correctly(self, stage_config):
        """Verify generator consumption and mask processing."""
        stage = ObjectSegmentationStage(stage_config)
        # Set video dimensions for mask resizing
        stage._video_height = 50
        stage._video_width = 50

        # Create real tensor for mask_logits (num_objects, 1, H, W)
        test_logits = torch.ones((2, 1, 50, 50))

        # Create mock generator that yields 3 frames
        def mock_propagate(inference_state):
            for frame_idx in range(3):
                obj_ids = [0, 1]
                yield frame_idx, obj_ids, test_logits

        mock_predictor = MagicMock()
        mock_predictor.propagate_in_video.side_effect = mock_propagate
        stage.video_predictor = mock_predictor

        result = stage._propagate_masks({}, num_frames=3)

        # Should have 3 frames
        assert len(result) == 3
        assert 0 in result
        assert 1 in result
        assert 2 in result

    def test_computes_per_frame_metadata(self, stage_config):
        """Verify bbox, centroid, area computed for each frame."""
        stage = ObjectSegmentationStage(stage_config)
        # Set video dimensions for mask resizing
        stage._video_height = 100
        stage._video_width = 100

        # Create a simple mask tensor (num_objects, 1, H, W) with foreground region
        # SAM-2 uses logits where >0 = foreground
        test_logits = torch.zeros((1, 1, 100, 100))
        test_logits[0, 0, 20:40, 30:60] = 1.0  # Set foreground region

        def mock_propagate(inference_state):
            obj_ids = [0]
            yield 0, obj_ids, test_logits

        mock_predictor = MagicMock()
        mock_predictor.propagate_in_video.side_effect = mock_propagate
        stage.video_predictor = mock_predictor

        result = stage._propagate_masks({}, num_frames=1)

        # Check metadata was computed
        frame_data = result[0][0]
        assert "bbox" in frame_data
        assert "centroid" in frame_data
        assert "area" in frame_data
        assert frame_data["area"] == test_mask.sum()


class TestValidateTrackingQuality:
    """Tests for _validate_tracking_quality method."""

    def test_perfect_overlap_high_iou(self, stage_config):
        """Perfect overlap between frames gives IoU of 1.0."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.ones((50, 50), dtype=bool)
        all_masks = {
            0: [{"object_id": 0, "mask": mask}],
            1: [{"object_id": 0, "mask": mask}],
        }

        quality = stage._validate_tracking_quality(all_masks)

        assert quality["mean_iou"] == 1.0
        assert quality["min_iou"] == 1.0
        assert len(quality["warnings"]) == 0

    def test_low_iou_generates_warning(self, stage_config):
        """Low IoU between frames generates warning."""
        stage = ObjectSegmentationStage(stage_config)

        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:30, 0:30] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[70:100, 70:100] = True  # No overlap

        all_masks = {
            0: [{"object_id": 0, "mask": mask1}],
            1: [{"object_id": 0, "mask": mask2}],
        }

        quality = stage._validate_tracking_quality(all_masks)

        assert quality["mean_iou"] < 0.7
        assert len(quality["warnings"]) > 0
        assert "low IoU" in quality["warnings"][0]


class TestExportTracks:
    """Tests for _export_tracks method."""

    def test_creates_directory_structure(self, stage_config, temp_output_dir):
        """Verify masks/{object_id}/ directories created."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.ones((50, 50), dtype=bool)
        all_masks = {
            0: [
                {"object_id": 0, "mask": mask, "bbox": [0, 0, 50, 50], "centroid": [25, 25], "area": 2500},
                {"object_id": 1, "mask": mask, "bbox": [0, 0, 50, 50], "centroid": [25, 25], "area": 2500},
            ],
        }

        objects, metadata = stage._export_tracks(all_masks, temp_output_dir)

        # Check directories created
        masks_dir = temp_output_dir / "masks"
        assert masks_dir.exists()
        assert (masks_dir / "0").exists()
        assert (masks_dir / "1").exists()

    def test_saves_binary_png_masks(self, stage_config, temp_output_dir):
        """Verify masks saved as 0/255 PNG files."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True

        all_masks = {
            0: [{"object_id": 0, "mask": mask, "bbox": [10, 10, 40, 40], "centroid": [25, 25], "area": 900}],
        }

        stage._export_tracks(all_masks, temp_output_dir)

        # Load saved mask and verify
        from PIL import Image

        mask_path = temp_output_dir / "masks" / "0" / "0000.png"
        assert mask_path.exists()

        loaded = np.array(Image.open(mask_path))
        assert loaded.dtype == np.uint8
        assert loaded.max() == 255
        assert loaded.min() == 0

    def test_generates_enhanced_metadata(self, stage_config, temp_output_dir):
        """Verify per_frame_data and frame_to_objects included."""
        stage = ObjectSegmentationStage(stage_config)

        mask = np.ones((50, 50), dtype=bool)
        all_masks = {
            0: [{"object_id": 0, "mask": mask, "bbox": [0, 0, 50, 50], "centroid": [25, 25], "area": 2500}],
            1: [{"object_id": 0, "mask": mask, "bbox": [0, 0, 50, 50], "centroid": [25, 25], "area": 2500}],
        }

        objects, enhanced_metadata = stage._export_tracks(all_masks, temp_output_dir)

        # Check enhanced metadata structure
        assert "frame_to_objects" in enhanced_metadata
        assert "per_object_frame_data" in enhanced_metadata
        assert "0" in enhanced_metadata["frame_to_objects"]
        assert "0" in enhanced_metadata["per_object_frame_data"]


class TestTrackObjectsOrchestration:
    """Tests for _track_objects main orchestration method."""

    def test_full_pipeline_with_mock(self, stage_config, temp_frames_dir, temp_output_dir):
        """Integration test with mocked video predictor."""
        import torch

        stage = ObjectSegmentationStage(stage_config)
        # Set video dimensions for mask resizing (matches test images)
        stage._video_height = 100
        stage._video_width = 100

        # Setup mock video predictor
        mock_predictor = MagicMock()
        mock_predictor.init_state.return_value = {"test": "state"}
        mock_predictor.add_new_points_or_box.return_value = (0, [0], np.ones((1, 50, 50)))

        # Create mask logits tensor (num_objects, 1, H, W)
        test_logits = torch.ones((1, 1, 100, 100))

        def mock_propagate(inference_state):
            """Generator that yields frames with mask logits."""
            for frame_idx in range(3):
                yield frame_idx, [0], test_logits

        mock_predictor.propagate_in_video.side_effect = mock_propagate
        stage.video_predictor = mock_predictor

        # Create frame files
        frame_files = sorted(temp_frames_dir.glob("*.jpg"))[:5]

        # Create detection
        initial_detections = [{
            "mask": np.ones((50, 50), dtype=bool),
            "bbox": [0, 0, 50, 50],
            "confidence": 0.9,
            "frame_idx": 0,
        }]

        # Run tracking
        result = stage._track_objects(frame_files, initial_detections, temp_output_dir)

        # Verify result
        assert len(result) == 1
        assert result[0].object_id == 0
        assert len(result[0].frame_indices) == 3

        # Verify metadata saved
        metadata_path = temp_output_dir / "object_metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["num_objects"] == 1
        assert "frame_to_objects" in metadata
        assert "quality_metrics" in metadata
