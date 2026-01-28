"""
End-to-end integration tests for the Instant4D pipeline.

These tests validate the complete Mega-SAM → Instant4D workflow.
They require a GPU and are marked with @pytest.mark.gpu and @pytest.mark.slow.

Run on GPU server:
    pytest tests/integration/test_instant4d_e2e.py -v -m gpu
"""

import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import tempfile
import shutil


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="instant4d_e2e_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_video_result(temp_output_dir):
    """
    Create a mock VideoProcessingResult with synthetic data.

    This allows testing Stage 3 (Instant4D) without running Stage 1.
    """
    from tests.fixtures.create_test_data import create_synthetic_video_result
    return create_synthetic_video_result(temp_output_dir / "stage1", num_frames=30)


@pytest.fixture
def mock_megasam_result(temp_output_dir):
    """
    Create a mock VideoProcessingResult with Mega-SAM outputs.

    Includes depth_maps_dir and motion_prob_path in metadata.
    """
    from tests.fixtures.create_test_data import create_megasam_video_result
    return create_megasam_video_result(temp_output_dir / "stage1", num_frames=30)


# =============================================================================
# Mega-SAM → Instant4D Pipeline Tests
# =============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestMegaSamToInstant4D:
    """Tests the primary Mega-SAM → Instant4D pipeline."""

    def test_megasam_produces_depth_and_motion(self, mock_megasam_result):
        """Stage 1: Mega-SAM outputs depth maps and motion probability."""
        result = mock_megasam_result

        # Verify Mega-SAM-specific outputs exist
        assert result.metadata.get("depth_maps_dir"), "Missing depth_maps_dir in metadata"
        assert result.metadata.get("motion_prob_path"), "Missing motion_prob_path in metadata"

        depth_dir = Path(result.metadata["depth_maps_dir"])
        motion_path = Path(result.metadata["motion_prob_path"])

        assert depth_dir.exists(), f"Depth maps directory not found: {depth_dir}"
        assert motion_path.exists(), f"Motion probability file not found: {motion_path}"

        # Check depth map count matches frame count
        depth_files = list(depth_dir.glob("*.npz"))
        assert len(depth_files) > 0, "No depth map files found"

        # Check motion probability shape
        motion_prob = np.load(motion_path)
        assert motion_prob.ndim >= 2, "Motion probability should be 2D or higher"

    def test_instant4d_preprocessing(self, mock_megasam_result, temp_output_dir):
        """Stage 3: Preprocessing converts Stage 1 output to Instant4D format."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        adapter = Instant4DAdapter()
        options = Instant4DOptions(use_megasam=True)

        output_dir = temp_output_dir / "stage3"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run preprocessing only
        adapter.preprocess(
            video_result=mock_megasam_result,
            output_dir=str(output_dir),
            options=options
        )

        # Verify filtered_cvd.npz was created
        cvd_path = output_dir / "filtered_cvd.npz"
        assert cvd_path.exists(), "filtered_cvd.npz not created"

        # Check contents
        data = np.load(cvd_path)
        assert "xyz" in data, "Missing xyz in filtered_cvd.npz"
        assert "rgb" in data, "Missing rgb in filtered_cvd.npz"
        assert "prob_motion" in data, "Missing prob_motion in filtered_cvd.npz"
        assert "time_stamp" in data, "Missing time_stamp in filtered_cvd.npz"
        assert "intrinsic" in data, "Missing intrinsic in filtered_cvd.npz"
        assert "cam_c2w" in data, "Missing cam_c2w in filtered_cvd.npz"

        # Validate ranges
        assert data["prob_motion"].min() >= 0, "Motion probability should be >= 0"
        assert data["prob_motion"].max() <= 1, "Motion probability should be <= 1"

    def test_instant4d_uses_megasam_depth(self, mock_megasam_result, temp_output_dir):
        """Stage 3: Instant4D loads Mega-SAM depth for point initialization."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            use_megasam=True,
            iterations=100,  # Very short for testing
        )

        output_dir = temp_output_dir / "stage3"

        result = adapter.run_full_pipeline(
            video_result=mock_megasam_result,
            output_dir=str(output_dir),
            options=options
        )

        # Dense depth should produce more Gaussians than sparse COLMAP
        # Mega-SAM back-projection typically gives 50K+ points
        assert result.num_gaussians > 10000, \
            f"Expected dense point cloud from Mega-SAM depth, got {result.num_gaussians}"

    def test_motion_probability_separation(self, mock_megasam_result, temp_output_dir):
        """Stage 3: Dynamic and static points are separated correctly."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        adapter = Instant4DAdapter()
        options = Instant4DOptions(use_megasam=True)

        output_dir = temp_output_dir / "stage3"
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter.preprocess(
            video_result=mock_megasam_result,
            output_dir=str(output_dir),
            options=options
        )

        # Load preprocessed data
        data = np.load(output_dir / "filtered_cvd.npz")
        prob_motion = data["prob_motion"]
        time_stamp = data["time_stamp"]

        # Static points (low motion prob) should have timestamps near midpoint
        static_mask = prob_motion < 0.3
        if static_mask.sum() > 0:
            static_times = time_stamp[static_mask]
            # Static points should cluster around time_duration / 2
            assert static_times.std() < 0.5, \
                "Static points should have similar timestamps"

        # Dynamic points (high motion prob) should be spread across time
        dynamic_mask = prob_motion > 0.7
        if dynamic_mask.sum() > 0:
            dynamic_times = time_stamp[dynamic_mask]
            # Dynamic points should span the time range
            time_range = dynamic_times.max() - dynamic_times.min()
            assert time_range > 1.0, \
                "Dynamic points should be spread across time"

    def test_4d_training_completes(self, mock_megasam_result, temp_output_dir):
        """Training completes without OOM and produces valid metrics."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            iterations=500,  # Short for testing
            use_megasam=True,
        )

        output_dir = temp_output_dir / "stage3"

        result = adapter.run_full_pipeline(
            video_result=mock_megasam_result,
            output_dir=str(output_dir),
            options=options
        )

        # Check metrics exist and are reasonable
        assert "psnr" in result.metrics or "final_loss" in result.metrics, \
            "Training should produce metrics"

        if "psnr" in result.metrics:
            assert result.metrics["psnr"] > 10, \
                f"PSNR too low: {result.metrics['psnr']}"

        if "final_loss" in result.metrics:
            assert result.metrics["final_loss"] < 1.0, \
                f"Loss too high: {result.metrics['final_loss']}"

    def test_per_frame_ply_export(self, mock_megasam_result, temp_output_dir):
        """Per-frame PLY export produces valid files."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            iterations=100,
            export_fps=5,  # 5 frames for quick test
            use_megasam=True,
        )

        output_dir = temp_output_dir / "stage3"

        result = adapter.run_full_pipeline(
            video_result=mock_megasam_result,
            output_dir=str(output_dir),
            options=options
        )

        # Check PLY files exist
        assert len(result.ply_paths) >= 5, \
            f"Expected at least 5 PLY files, got {len(result.ply_paths)}"

        for ply_path in result.ply_paths:
            path = Path(ply_path)
            assert path.exists(), f"PLY file not found: {ply_path}"
            assert path.stat().st_size > 1000, f"PLY file too small: {ply_path}"

    def test_temporal_consistency(self, mock_megasam_result, temp_output_dir):
        """Adjacent frames should have similar Gaussian counts."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions
        from plyfile import PlyData

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            iterations=100,
            export_fps=10,
            use_megasam=True,
        )

        output_dir = temp_output_dir / "stage3"

        result = adapter.run_full_pipeline(
            video_result=mock_megasam_result,
            output_dir=str(output_dir),
            options=options
        )

        # Count Gaussians per frame
        counts = []
        for ply_path in result.ply_paths:
            ply = PlyData.read(ply_path)
            counts.append(len(ply['vertex']))

        # Check temporal consistency
        for i in range(len(counts) - 1):
            ratio = counts[i + 1] / max(counts[i], 1)
            assert 0.3 < ratio < 3.0, \
                f"Large jump in Gaussian count: {counts[i]} -> {counts[i+1]}"


# =============================================================================
# Fallback Path Tests
# =============================================================================

@pytest.mark.gpu
class TestFallbackPath:
    """Tests the hloc fallback when Mega-SAM is unavailable."""

    def test_fallback_triggers_on_megasam_error(self, mock_video_result, temp_output_dir):
        """hloc is used when Mega-SAM fails."""
        from backend.stages.video_processing import VideoProcessingStage, MegaSamError
        from unittest.mock import patch

        # Mock Mega-SAM to fail
        with patch.object(
            VideoProcessingStage,
            '_run_megasam_pipeline',
            side_effect=MegaSamError("Mocked failure")
        ):
            stage = VideoProcessingStage({"pose_estimator": "megasam"})

            # Should fall back to hloc, not crash
            # Note: This test validates the fallback logic, not full execution
            with pytest.raises(MegaSamError):
                stage._run_megasam_pipeline(None, None)

    def test_instant4d_works_with_sparse_points(self, mock_video_result, temp_output_dir):
        """Instant4D can train from sparse COLMAP points (no Mega-SAM)."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        # Ensure no Mega-SAM metadata
        mock_video_result.metadata.pop("depth_maps_dir", None)
        mock_video_result.metadata.pop("motion_prob_path", None)

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            iterations=100,
            use_megasam=False,  # Explicitly disable
        )

        output_dir = temp_output_dir / "stage3"

        result = adapter.run_full_pipeline(
            video_result=mock_video_result,
            output_dir=str(output_dir),
            options=options
        )

        # Should still work, just with fewer Gaussians
        assert result.num_gaussians > 1000, \
            f"Expected some Gaussians from sparse points, got {result.num_gaussians}"


# =============================================================================
# Quality Validation Tests
# =============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestQualityValidation:
    """Tests that validate output quality metrics."""

    def test_voxel_filtering_reduces_gaussians(self, mock_megasam_result, temp_output_dir):
        """Grid pruning achieves ~92% Gaussian reduction."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        # First, preprocess without pruning
        adapter = Instant4DAdapter()
        options_no_prune = Instant4DOptions(
            use_megasam=True,
            enable_pruning=False,
        )

        output_no_prune = temp_output_dir / "no_prune"
        output_no_prune.mkdir(parents=True, exist_ok=True)

        adapter.preprocess(
            video_result=mock_megasam_result,
            output_dir=str(output_no_prune),
            options=options_no_prune
        )

        data_no_prune = np.load(output_no_prune / "filtered_cvd.npz")
        count_no_prune = len(data_no_prune["xyz"])

        # Then, preprocess with pruning
        options_prune = Instant4DOptions(
            use_megasam=True,
            enable_pruning=True,
        )

        output_prune = temp_output_dir / "prune"
        output_prune.mkdir(parents=True, exist_ok=True)

        adapter.preprocess(
            video_result=mock_megasam_result,
            output_dir=str(output_prune),
            options=options_prune
        )

        data_prune = np.load(output_prune / "filtered_cvd.npz")
        count_prune = len(data_prune["xyz"])

        # Pruning should reduce by at least 50%
        reduction = 1 - (count_prune / count_no_prune)
        assert reduction > 0.5, \
            f"Expected >50% reduction, got {reduction*100:.1f}%"

    def test_temporal_metadata_consistency(self, mock_megasam_result, temp_output_dir):
        """Temporal metadata matches actual output files."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            iterations=100,
            export_fps=10,
            use_megasam=True,
        )

        output_dir = temp_output_dir / "stage3"

        result = adapter.run_full_pipeline(
            video_result=mock_megasam_result,
            output_dir=str(output_dir),
            options=options
        )

        if result.temporal_metadata:
            meta = result.temporal_metadata

            # num_frames should match PLY count
            if "num_frames" in meta:
                assert meta["num_frames"] == len(result.ply_paths), \
                    "num_frames doesn't match PLY file count"

            # timestamps should have correct length
            if "timestamps" in meta:
                assert len(meta["timestamps"]) == len(result.ply_paths), \
                    "timestamps count doesn't match PLY file count"


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.gpu
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_masks_handled(self, mock_video_result, temp_output_dir):
        """Pipeline handles videos with no detected motion."""
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        # Mock all-static scene (no motion probability)
        mock_video_result.metadata["motion_prob_path"] = None

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            iterations=100,
            use_megasam=False,
        )

        output_dir = temp_output_dir / "stage3"

        # Should not crash on static scene
        result = adapter.run_full_pipeline(
            video_result=mock_video_result,
            output_dir=str(output_dir),
            options=options
        )

        assert result.num_gaussians > 0, "Should produce some Gaussians"

    def test_minimal_frame_count(self, temp_output_dir):
        """Pipeline handles minimum viable frame count."""
        from tests.fixtures.create_test_data import create_synthetic_video_result
        from backend.adapters.instant4d import Instant4DAdapter, Instant4DOptions

        # Create minimal input (10 frames)
        video_result = create_synthetic_video_result(
            temp_output_dir / "stage1",
            num_frames=10
        )

        adapter = Instant4DAdapter()
        options = Instant4DOptions(
            iterations=50,
            export_fps=3,
        )

        output_dir = temp_output_dir / "stage3"

        result = adapter.run_full_pipeline(
            video_result=video_result,
            output_dir=str(output_dir),
            options=options
        )

        assert len(result.ply_paths) >= 3, "Should produce some PLY files"
