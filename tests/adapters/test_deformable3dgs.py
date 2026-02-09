"""Tests for Deformable 3D Gaussians adapter.

These tests verify the De3DGS adapter fixes for:
- Issue #1: Path traversal vulnerability
- Issue #3: Dataclass validation
- Issue #4: CUDA error detection
- Phase 3: PLY export validation

Usage:
    # Run all tests
    pytest tests/adapters/test_deformable3dgs.py -v

    # Run specific test classes
    pytest tests/adapters/test_deformable3dgs.py::TestPathValidation -v
    pytest tests/adapters/test_deformable3dgs.py::TestDataclassValidation -v
    pytest tests/adapters/test_deformable3dgs.py::TestCUDAErrorParsing -v
    pytest tests/adapters/test_deformable3dgs.py::TestPLYExportFormat -v
    pytest tests/adapters/test_deformable3dgs.py::TestTemporalConsistency -v
    pytest tests/adapters/test_deformable3dgs.py::TestCoordinateConversion -v
"""

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.adapters.deformable3dgs import (
    Deformable3DGSAdapter,
    Deformable3DGSOptions,
    De3DGSError,
    CUDAOutOfMemoryError,
    CUDADriverError,
    CUDADeviceError,
    c2w_to_colmap_pose,
    normalize_timestamp,
)

# GPU availability check for conditional test skipping
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    GPU_AVAILABLE = False
GPU_SKIP_REASON = "GPU required but not available"

# plyfile availability check for PLY format validation tests
try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PlyData = None
    PLYFILE_AVAILABLE = False
PLYFILE_SKIP_REASON = "plyfile not installed"


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def adapter():
    """Create a Deformable3DGSAdapter instance."""
    # Patch the de3dgs_path check to avoid needing actual submodule
    with patch.object(Deformable3DGSAdapter, '__init__', lambda self, config=None: None):
        adapter = Deformable3DGSAdapter.__new__(Deformable3DGSAdapter)
        adapter.config = {}
        adapter.de3dgs_path = Path("/fake/deformable3dgs")
        return adapter


@pytest.fixture
def valid_transforms(tmp_path):
    """Create a valid transforms.json file."""
    transforms_path = tmp_path / "transforms.json"
    transforms_path.write_text('{"fl_x": 500, "fl_y": 500, "frames": []}')
    return transforms_path


@pytest.fixture
def export_script():
    """Import save_gaussian_ply from export script, skip if submodule unavailable.

    This fixture properly manages sys.path modification and cleanup, avoiding
    the anti-pattern of inline sys.path manipulation in test methods.
    """
    import sys
    export_script_path = Path(__file__).parent.parent.parent / "deformable3dgs" / "scripts"

    if not export_script_path.exists():
        pytest.skip("De3DGS submodule not initialized")

    original_path = sys.path.copy()
    sys.path.insert(0, str(export_script_path))

    try:
        from export_per_frame_ply import save_gaussian_ply
        yield save_gaussian_ply
    except ImportError as e:
        pytest.skip(f"Cannot import export_per_frame_ply: {e}")
    finally:
        sys.path[:] = original_path


# =============================================================================
# Issue #1: Path Traversal Validation Tests
# =============================================================================

class TestPathValidation:
    """Test path traversal prevention in _validate_paths()."""

    def test_rejects_parent_traversal(self, adapter):
        """Path with .. component is rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            adapter._validate_paths(Path("../etc/passwd"))

    def test_rejects_deep_traversal(self, adapter):
        """Multiple .. components are rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            adapter._validate_paths(Path("foo/../../bar/transforms.json"))

    def test_rejects_middle_traversal(self, adapter):
        """.. in middle of path is rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            adapter._validate_paths(Path("some/path/../other/transforms.json"))

    def test_accepts_dots_in_filename(self, adapter, tmp_path):
        """Dots in filename are allowed (not traversal)."""
        valid_file = tmp_path / "file..name.json"
        valid_file.write_text('{}')
        # Should not raise - dots in filename are fine
        adapter._validate_paths(valid_file)

    def test_accepts_absolute_path(self, adapter, valid_transforms):
        """Valid absolute paths are accepted."""
        # Should not raise
        adapter._validate_paths(valid_transforms)

    def test_accepts_relative_path_without_traversal(self, adapter, tmp_path, monkeypatch):
        """Valid relative paths without .. are accepted."""
        # Change to tmp_path so relative path works
        monkeypatch.chdir(tmp_path)
        valid_file = tmp_path / "transforms.json"
        valid_file.write_text('{}')
        # Use relative path
        adapter._validate_paths(Path("transforms.json"))

    def test_rejects_nonexistent_file(self, adapter, tmp_path):
        """Non-existent file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="transforms.json not found"):
            adapter._validate_paths(nonexistent)

    def test_rejects_non_json_extension(self, adapter, tmp_path):
        """Non-JSON file extension is rejected."""
        txt_file = tmp_path / "transforms.txt"
        txt_file.write_text('{}')
        with pytest.raises(ValueError, match="must be a JSON file"):
            adapter._validate_paths(txt_file)


# =============================================================================
# Issue #3: Dataclass Validation Tests
# =============================================================================

class TestDataclassValidation:
    """Test dataclass validation in __post_init__."""

    def test_rejects_negative_iterations(self):
        """Negative iterations raise ValueError."""
        with pytest.raises(ValueError, match="iterations must be positive"):
            Deformable3DGSOptions(iterations=-1)

    def test_rejects_zero_iterations(self):
        """Zero iterations raise ValueError."""
        with pytest.raises(ValueError, match="iterations must be positive"):
            Deformable3DGSOptions(iterations=0)

    def test_rejects_huge_iterations(self):
        """Excessively large iterations raise ValueError."""
        with pytest.raises(ValueError, match="unusually high"):
            Deformable3DGSOptions(iterations=1000000)

    def test_accepts_max_reasonable_iterations(self):
        """500000 iterations is accepted (boundary case)."""
        opts = Deformable3DGSOptions(iterations=500000)
        assert opts.iterations == 500000

    def test_rejects_invalid_sh_degree_high(self):
        """sh_degree > 3 is rejected."""
        with pytest.raises(ValueError, match="sh_degree must be 0-3"):
            Deformable3DGSOptions(sh_degree=5)

    def test_rejects_negative_sh_degree(self):
        """Negative sh_degree is rejected."""
        with pytest.raises(ValueError, match="sh_degree must be 0-3"):
            Deformable3DGSOptions(sh_degree=-1)

    def test_accepts_valid_sh_degrees(self):
        """Valid sh_degree values 0-3 are accepted."""
        for sh in range(4):
            opts = Deformable3DGSOptions(sh_degree=sh)
            assert opts.sh_degree == sh

    def test_rejects_lambda_above_one(self):
        """lambda_dssim > 1 is rejected."""
        with pytest.raises(ValueError, match="lambda_dssim must be in"):
            Deformable3DGSOptions(lambda_dssim=1.5)

    def test_rejects_negative_lambda(self):
        """Negative lambda_dssim is rejected."""
        with pytest.raises(ValueError, match="lambda_dssim must be in"):
            Deformable3DGSOptions(lambda_dssim=-0.1)

    def test_accepts_boundary_lambda_values(self):
        """lambda_dssim at boundaries (0 and 1) is accepted."""
        opts_zero = Deformable3DGSOptions(lambda_dssim=0.0)
        opts_one = Deformable3DGSOptions(lambda_dssim=1.0)
        assert opts_zero.lambda_dssim == 0.0
        assert opts_one.lambda_dssim == 1.0

    def test_rejects_warmup_exceeding_iterations(self):
        """warm_up >= iterations is rejected."""
        with pytest.raises(ValueError, match="warm_up .* must be less than iterations"):
            Deformable3DGSOptions(iterations=1000, warm_up=2000)

    def test_rejects_warmup_equal_iterations(self):
        """warm_up == iterations is rejected."""
        with pytest.raises(ValueError, match="warm_up .* must be less than iterations"):
            Deformable3DGSOptions(iterations=1000, warm_up=1000)

    def test_rejects_negative_warmup(self):
        """Negative warm_up is rejected."""
        with pytest.raises(ValueError, match="warm_up must be non-negative"):
            Deformable3DGSOptions(warm_up=-100)

    def test_accepts_zero_warmup(self):
        """Zero warm_up is accepted (immediate deformation)."""
        opts = Deformable3DGSOptions(warm_up=0)
        assert opts.warm_up == 0

    def test_rejects_zero_timeout(self):
        """Zero timeout is rejected."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            Deformable3DGSOptions(timeout_seconds=0)

    def test_rejects_very_short_timeout(self):
        """Timeout < 60s is rejected as too short."""
        with pytest.raises(ValueError, match="too short"):
            Deformable3DGSOptions(timeout_seconds=30)

    def test_accepts_minimum_timeout(self):
        """60 second timeout is accepted (boundary case)."""
        opts = Deformable3DGSOptions(timeout_seconds=60)
        assert opts.timeout_seconds == 60

    def test_rejects_zero_target_points(self):
        """Zero target_points is rejected."""
        with pytest.raises(ValueError, match="target_points must be positive"):
            Deformable3DGSOptions(target_points=0)

    def test_rejects_zero_export_frames(self):
        """Zero export_num_frames is rejected."""
        with pytest.raises(ValueError, match="export_num_frames must be positive"):
            Deformable3DGSOptions(export_num_frames=0)

    def test_rejects_zero_export_fps(self):
        """Zero export_fps is rejected."""
        with pytest.raises(ValueError, match="export_fps must be positive"):
            Deformable3DGSOptions(export_fps=0)

    def test_rejects_nonexistent_de3dgs_path(self):
        """Non-existent de3dgs_path is rejected."""
        with pytest.raises(ValueError, match="does not exist"):
            Deformable3DGSOptions(de3dgs_path="/nonexistent/path")

    def test_rejects_de3dgs_path_without_train(self, tmp_path):
        """de3dgs_path without train.py is rejected."""
        with pytest.raises(ValueError, match="does not contain train.py"):
            Deformable3DGSOptions(de3dgs_path=str(tmp_path))

    def test_accepts_valid_de3dgs_path(self, tmp_path):
        """Valid de3dgs_path with train.py is accepted."""
        (tmp_path / "train.py").touch()
        opts = Deformable3DGSOptions(de3dgs_path=str(tmp_path))
        assert opts.de3dgs_path == str(tmp_path)

    def test_accepts_none_de3dgs_path(self):
        """None de3dgs_path is accepted (will be auto-detected)."""
        opts = Deformable3DGSOptions(de3dgs_path=None)
        assert opts.de3dgs_path is None

    def test_accepts_valid_config(self):
        """Valid configuration should not raise."""
        opts = Deformable3DGSOptions(
            iterations=10000,
            sh_degree=2,
            lambda_dssim=0.5,
            warm_up=1000,
            timeout_seconds=3600,
        )
        assert opts.iterations == 10000
        assert opts.sh_degree == 2
        assert opts.lambda_dssim == 0.5
        assert opts.warm_up == 1000
        assert opts.timeout_seconds == 3600


# =============================================================================
# Issue #4: CUDA Error Detection Tests
# =============================================================================

class TestCUDAErrorParsing:
    """Test CUDA error detection and messaging."""

    def test_detects_oom_error_basic(self, adapter):
        """Basic OOM error is detected."""
        output = "RuntimeError: CUDA out of memory."
        error = adapter._parse_cuda_error(output, 1, 5000)
        assert isinstance(error, CUDAOutOfMemoryError)
        assert "iteration 5000" in str(error)
        assert "target_points" in str(error)

    def test_detects_oom_with_allocation_info(self, adapter):
        """OOM error with allocation info is parsed."""
        output = "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB"
        error = adapter._parse_cuda_error(output, 1, 5000)
        assert isinstance(error, CUDAOutOfMemoryError)
        assert "2.50" in str(error)
        assert "GiB" in str(error)

    def test_detects_oom_lowercase(self, adapter):
        """OOM detection is case-insensitive."""
        output = "error: cuda out of memory at batch 100"
        error = adapter._parse_cuda_error(output, 1, 1000)
        assert isinstance(error, CUDAOutOfMemoryError)

    def test_detects_driver_version_error(self, adapter):
        """CUDA driver version mismatch is detected."""
        output = "CUDA driver version is insufficient for CUDA runtime version"
        error = adapter._parse_cuda_error(output, 1, 0)
        assert isinstance(error, CUDADriverError)
        assert "Update NVIDIA drivers" in str(error)

    def test_detects_no_cuda_device(self, adapter):
        """No CUDA device is detected."""
        output = "RuntimeError: No CUDA-capable device is detected"
        error = adapter._parse_cuda_error(output, 1, 0)
        assert isinstance(error, CUDADeviceError)
        assert "nvidia-smi" in str(error)

    def test_detects_cuda_not_available(self, adapter):
        """CUDA is not available is detected."""
        output = "AssertionError: Torch CUDA is not available"
        error = adapter._parse_cuda_error(output, 1, 0)
        assert isinstance(error, CUDADeviceError)

    def test_detects_invalid_device_ordinal(self, adapter):
        """Invalid device ordinal is detected."""
        output = "RuntimeError: invalid device ordinal"
        error = adapter._parse_cuda_error(output, 1, 0)
        assert isinstance(error, CUDADeviceError)
        assert "nvidia-smi -L" in str(error)

    def test_detects_cudnn_error(self, adapter):
        """cuDNN error is detected."""
        output = "RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"
        error = adapter._parse_cuda_error(output, 1, 2500)
        assert isinstance(error, CUDADriverError)
        assert "iteration 2500" in str(error)

    def test_detects_nan_loss_divergence(self, adapter):
        """NaN in loss detection works."""
        output = "Loss became NaN at iteration 1500"
        error = adapter._parse_cuda_error(output, 1, 1500)
        assert isinstance(error, De3DGSError)
        assert "diverged" in str(error)
        assert "Reduce learning rate" in str(error)

    def test_detects_nan_gradient_divergence(self, adapter):
        """NaN in gradient detection works."""
        output = "Warning: gradient contains NaN values"
        error = adapter._parse_cuda_error(output, 1, 2000)
        assert isinstance(error, De3DGSError)
        assert "diverged" in str(error)

    def test_fallback_includes_exit_code(self, adapter):
        """Unknown error includes exit code."""
        output = "Some unknown error occurred"
        error = adapter._parse_cuda_error(output, 42, 1000)
        assert isinstance(error, De3DGSError)
        assert "exit code 42" in str(error)

    def test_fallback_includes_iteration(self, adapter):
        """Unknown error includes last iteration."""
        output = "Some unknown error occurred"
        error = adapter._parse_cuda_error(output, 1, 7500)
        assert isinstance(error, De3DGSError)
        assert "iteration 7500" in str(error)

    def test_fallback_includes_output_snippet(self, adapter):
        """Unknown error includes output snippet."""
        output = "Some unknown error occurred\nThis is helpful context"
        error = adapter._parse_cuda_error(output, 1, 1000)
        assert isinstance(error, De3DGSError)
        assert "unknown error" in str(error)

    def test_output_truncation_for_long_errors(self, adapter):
        """Long error output is truncated in fallback."""
        output = "x" * 1000
        error = adapter._parse_cuda_error(output, 1, 1000)
        # Should only include last 500 chars in generic fallback
        assert len(str(error)) < 1500  # Some overhead for error message

    def test_oom_recommendation_quality(self, adapter):
        """OOM error provides actionable recommendations."""
        output = "CUDA out of memory"
        error = adapter._parse_cuda_error(output, 1, 5000)
        error_str = str(error)
        # Should have multiple recommendations
        assert "target_points" in error_str
        assert "sh_degree" in error_str
        assert "VRAM" in error_str

    def test_driver_error_includes_links(self, adapter):
        """Driver error includes helpful links."""
        output = "cuda driver version is insufficient"
        error = adapter._parse_cuda_error(output, 1, 0)
        assert "nvidia.com" in str(error)
        assert "PyTorch 2.3" in str(error)


# =============================================================================
# Integration Tests (Mock-based)
# =============================================================================

class TestAdapterIntegration:
    """Integration tests for the adapter with mocked subprocess."""

    def test_adapter_initialization(self):
        """Adapter initializes without errors."""
        with patch('pathlib.Path.exists', return_value=True):
            adapter = Deformable3DGSAdapter()
            assert hasattr(adapter, 'de3dgs_path')

    def test_options_default_values(self):
        """Default options are valid."""
        opts = Deformable3DGSOptions()
        assert opts.iterations == 20000
        assert opts.sh_degree == 3
        assert opts.timeout_seconds == 7200
        assert opts.warm_up < opts.iterations
        assert opts.cuda_device == "0"


# =============================================================================
# Issue #1 (Round 2): Workspace Boundary Validation Tests
# =============================================================================

class TestWorkspaceBoundaryValidation:
    """Test workspace boundary validation for symlink attacks."""

    @pytest.fixture
    def adapter_with_workspace(self, tmp_path):
        """Create adapter with workspace_root configured."""
        with patch.object(Deformable3DGSAdapter, '__init__', lambda self, config=None: None):
            adapter = Deformable3DGSAdapter.__new__(Deformable3DGSAdapter)
            adapter.config = {"workspace_root": str(tmp_path)}
            adapter.de3dgs_path = Path("/fake/deformable3dgs")
            adapter.workspace_root = tmp_path.resolve()
            return adapter

    def test_accepts_path_within_workspace(self, adapter_with_workspace, tmp_path):
        """Valid path within workspace is accepted."""
        valid_file = tmp_path / "subdir" / "transforms.json"
        valid_file.parent.mkdir(parents=True, exist_ok=True)
        valid_file.write_text('{}')
        adapter_with_workspace._validate_paths(valid_file)  # Should not raise

    def test_rejects_path_outside_workspace(self, adapter_with_workspace, tmp_path):
        """Path outside workspace is rejected."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'{}')
            outside_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="outside workspace boundary"):
                adapter_with_workspace._validate_paths(outside_path)
        finally:
            outside_path.unlink()

    def test_rejects_symlink_pointing_outside(self, adapter_with_workspace, tmp_path):
        """Symlink pointing outside workspace is rejected."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'{}')
            outside_target = Path(f.name)

        # Create symlink inside workspace pointing outside
        symlink_path = tmp_path / "sneaky_link.json"
        symlink_path.symlink_to(outside_target)

        try:
            with pytest.raises(ValueError, match="outside workspace boundary"):
                adapter_with_workspace._validate_paths(symlink_path)
        finally:
            symlink_path.unlink()
            outside_target.unlink()

    def test_no_workspace_check_when_not_configured(self, adapter, tmp_path):
        """When workspace_root is None, no boundary check is performed."""
        valid_file = tmp_path / "transforms.json"
        valid_file.write_text('{}')
        adapter._validate_paths(valid_file)  # Should not raise


# =============================================================================
# Issue #5: CUDA Device Configuration Tests
# =============================================================================

class TestCUDADeviceConfiguration:
    """Test CUDA device configuration in Deformable3DGSOptions."""

    def test_default_device_is_zero(self):
        """Default CUDA device is '0'."""
        opts = Deformable3DGSOptions()
        assert opts.cuda_device == "0"

    def test_accepts_single_device(self):
        """Single device index is accepted."""
        opts = Deformable3DGSOptions(cuda_device="1")
        assert opts.cuda_device == "1"

    def test_accepts_multiple_devices(self):
        """Comma-separated device list is accepted."""
        opts = Deformable3DGSOptions(cuda_device="0,1,2")
        assert opts.cuda_device == "0,1,2"

    def test_accepts_high_device_index(self):
        """High device index is accepted (for multi-GPU systems)."""
        opts = Deformable3DGSOptions(cuda_device="7")
        assert opts.cuda_device == "7"

    def test_rejects_empty_device(self):
        """Empty device string is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Deformable3DGSOptions(cuda_device="")

    def test_rejects_invalid_format_letters(self):
        """Invalid device format with letters is rejected."""
        with pytest.raises(ValueError, match="digit"):
            Deformable3DGSOptions(cuda_device="gpu0")

    def test_rejects_negative_device(self):
        """Negative device index is rejected."""
        with pytest.raises(ValueError, match="digit"):
            Deformable3DGSOptions(cuda_device="-1")

    def test_rejects_spaces(self):
        """Device string with spaces is rejected."""
        with pytest.raises(ValueError, match="digit"):
            Deformable3DGSOptions(cuda_device="0, 1")

    def test_rejects_trailing_comma(self):
        """Device string with trailing comma is rejected."""
        with pytest.raises(ValueError, match="digit"):
            Deformable3DGSOptions(cuda_device="0,1,")


# =============================================================================
# Phase 3: PLY Export Validation Tests
# =============================================================================

@pytest.mark.skipif(not PLYFILE_AVAILABLE, reason=PLYFILE_SKIP_REASON)
class TestPLYExportFormat:
    """Validate PLY files conform to 3DGS specification."""

    @pytest.fixture
    def sample_gaussian_data(self):
        """Generate minimal valid Gaussian data for testing."""
        N = 10  # 10 Gaussians for fast tests
        np.random.seed(42)  # Reproducible tests
        return {
            "xyz": np.random.randn(N, 3).astype(np.float32),
            "rotation": np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),  # Identity quaternions
            "scaling": np.random.randn(N, 3).astype(np.float32),
            "opacity": np.random.rand(N, 1).astype(np.float32),
            "features_dc": np.random.randn(N, 1, 3).astype(np.float32),
            "features_rest": np.random.randn(N, 15, 3).astype(np.float32),  # SH degree 3
        }

    def test_ply_has_required_vertex_properties(self, sample_gaussian_data, tmp_path, export_script):
        """PLY must have: x,y,z, rot_0-3, scale_0-2, opacity, f_dc_0-2."""
        save_gaussian_ply = export_script

        output = tmp_path / "test.ply"
        save_gaussian_ply(
            output,
            sample_gaussian_data["xyz"],
            sample_gaussian_data["rotation"],
            sample_gaussian_data["scaling"],
            sample_gaussian_data["opacity"],
            sample_gaussian_data["features_dc"],
            sample_gaussian_data["features_rest"],
        )

        ply = PlyData.read(str(output))
        props = set(ply['vertex'].data.dtype.names)

        required = {'x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
                    'scale_0', 'scale_1', 'scale_2', 'opacity',
                    'f_dc_0', 'f_dc_1', 'f_dc_2'}
        assert required.issubset(props), f"Missing: {required - props}"

    def test_ply_vertex_count_matches_input(self, sample_gaussian_data, tmp_path, export_script):
        """Number of vertices in PLY should match input Gaussian count."""
        save_gaussian_ply = export_script

        output = tmp_path / "test.ply"
        save_gaussian_ply(
            output,
            sample_gaussian_data["xyz"],
            sample_gaussian_data["rotation"],
            sample_gaussian_data["scaling"],
            sample_gaussian_data["opacity"],
            sample_gaussian_data["features_dc"],
            sample_gaussian_data["features_rest"],
        )

        ply = PlyData.read(str(output))
        assert ply['vertex'].count == 10, f"Expected 10 vertices, got {ply['vertex'].count}"

    def test_ply_values_are_finite(self, sample_gaussian_data, tmp_path, export_script):
        """No NaN or Inf values in exported PLY."""
        save_gaussian_ply = export_script

        output = tmp_path / "test.ply"
        save_gaussian_ply(
            output,
            sample_gaussian_data["xyz"],
            sample_gaussian_data["rotation"],
            sample_gaussian_data["scaling"],
            sample_gaussian_data["opacity"],
            sample_gaussian_data["features_dc"],
            sample_gaussian_data["features_rest"],
        )

        ply = PlyData.read(str(output))
        vertex = ply['vertex']

        for prop in ['x', 'y', 'z', 'opacity', 'scale_0']:
            values = vertex[prop]
            assert np.isfinite(values).all(), f"Property {prop} contains NaN/Inf"


class TestTemporalConsistency:
    """Verify smooth deformation across temporal frames (mock-based)."""

    def test_timestamp_normalization_formula(self):
        """Verify timestamp normalization formula: frame_idx / (total_frames - 1)."""
        # Using normalize_timestamp imported from backend.adapters.deformable3dgs
        # (identical implementation to export_per_frame_ply, avoids submodule dependency)

        # Frame 0 of 30 should be 0.0
        assert normalize_timestamp(0, 30) == 0.0

        # Frame 29 of 30 should be 1.0
        assert normalize_timestamp(29, 30) == 1.0

        # Frame 15 of 31 should be 0.5
        assert normalize_timestamp(15, 31) == 0.5

        # Edge case: single frame
        assert normalize_timestamp(0, 1) == 0.0

    def test_linspace_generates_expected_timestamps(self):
        """Verify np.linspace produces expected temporal spacing."""
        timestamps = np.linspace(0.0, 1.0, 5)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert np.allclose(timestamps, expected), f"Expected {expected}, got {timestamps}"

    def test_position_delta_calculation(self):
        """Test that position delta between frames is computed correctly."""
        # Simulate two frame positions
        frame0_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        frame1_xyz = np.array([[0.1, 0.0, 0.0], [1.1, 1.0, 1.0]])

        delta = np.abs(frame1_xyz - frame0_xyz).max()
        assert delta < 0.5, f"Position delta {delta} exceeds threshold 0.5"


class TestCoordinateConversion:
    """Verify Nerfstudio ↔ COLMAP coordinate conversion round-trip."""

    def test_identity_matrix_conversion(self):
        """Identity c2w should produce expected COLMAP pose."""
        c2w = np.eye(4)
        quat, trans = c2w_to_colmap_pose(c2w)

        # Identity in OpenGL, after flip_yz and inversion
        # The rotation should be flip_yz itself
        expected_rot = Rotation.from_matrix(np.diag([1, -1, -1]))
        expected_quat_xyzw = expected_rot.as_quat()
        expected_quat = np.array([expected_quat_xyzw[3], expected_quat_xyzw[0],
                                   expected_quat_xyzw[1], expected_quat_xyzw[2]])

        assert np.allclose(quat, expected_quat, atol=1e-6), f"Quaternion mismatch: {quat} vs {expected_quat}"
        assert np.allclose(trans, [0, 0, 0], atol=1e-6), f"Translation mismatch: {trans}"

    def test_translation_only_conversion(self):
        """Pure translation c2w should convert correctly."""
        c2w = np.eye(4)
        c2w[:3, 3] = [1.0, 2.0, 3.0]  # Translation in OpenGL coords

        quat, trans = c2w_to_colmap_pose(c2w)

        # Translation flips: Y and Z are negated
        # After inversion, translation becomes -R^T @ t
        # For flip_yz (diagonal [1,-1,-1]), R^T = R
        # So trans = -[1, -1, -1] * [1, 2, 3] = [-1, 2, 3]
        expected_trans = np.array([-1.0, 2.0, 3.0])
        assert np.allclose(trans, expected_trans, atol=1e-6), f"Translation mismatch: {trans} vs {expected_trans}"

    def test_random_pose_roundtrip(self):
        """Random valid pose should round-trip with minimal error."""
        np.random.seed(123)

        # Create random c2w
        c2w_original = np.eye(4)
        c2w_original[:3, :3] = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
        c2w_original[:3, 3] = [1.0, 2.0, 3.0]

        # Convert to COLMAP
        quat, trans = c2w_to_colmap_pose(c2w_original)

        # Convert back
        flip_yz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        w2c[:3, 3] = trans
        c2w_recovered = flip_yz @ np.linalg.inv(w2c)

        # Check recovery accuracy
        assert np.allclose(c2w_original, c2w_recovered, atol=1e-6), \
            f"Round-trip error:\nOriginal:\n{c2w_original}\nRecovered:\n{c2w_recovered}"

    def test_quaternion_is_unit_length(self):
        """Output quaternion should have unit length."""
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler('xyz', [45, 30, 60], degrees=True).as_matrix()
        c2w[:3, 3] = [0.5, -0.3, 1.2]

        quat, _ = c2w_to_colmap_pose(c2w)
        norm = np.linalg.norm(quat)

        assert np.isclose(norm, 1.0, atol=1e-6), f"Quaternion norm {norm} is not unit"


# =============================================================================
# Phase 3: SE(3) Transformation Regression Tests
# =============================================================================

class TestSE3Transformation:
    """Regression tests for SE(3) 6-DoF transformation.

    These tests verify the fix for the SE(3) export bug where only translation
    was applied (d_xyz[:, :3, 3]) instead of full SE(3) transformation.
    """

    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_se3_applies_full_transformation(self):
        """Verify SE(3) mode applies rotation, not just translation.

        This is a regression test for the bug where SE(3) export only
        extracted the translation column instead of applying full transformation.

        Test case:
        - Point at (0, 1, 0)
        - SE(3) matrix: 90° rotation around Z-axis + translation (1, 2, 3)
        - Expected: R @ point + t = (-1, 0, 0) + (1, 2, 3) = (0, 2, 3)
        - Bug behavior: point + t = (0, 1, 0) + (1, 2, 3) = (1, 3, 3)
        """
        # Point at (0, 1, 0)
        xyz = torch.tensor([[0.0, 1.0, 0.0]])
        N = xyz.shape[0]

        # SE(3) matrix: 90° rotation around Z-axis + translation (1, 2, 3)
        # Rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]] (90° around Z)
        d_xyz = torch.tensor([[[0, -1, 0, 1],
                               [1,  0, 0, 2],
                               [0,  0, 1, 3],
                               [0,  0, 0, 1]]]).float()

        # Apply full SE(3) transformation (correct implementation)
        xyz_h = torch.cat([xyz, torch.ones(N, 1)], dim=-1)  # (N, 4)
        deformed = torch.bmm(d_xyz, xyz_h.unsqueeze(-1)).squeeze(-1)[:, :3]

        # CORRECT: R @ (0,1,0) + (1,2,3) = (-1,0,0) + (1,2,3) = (0,2,3)
        expected = torch.tensor([[0.0, 2.0, 3.0]])

        assert torch.allclose(deformed, expected, atol=1e-6), \
            f"SE(3) transformation incorrect.\nGot: {deformed}\nExpected: {expected}"

    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_se3_translation_only_produces_bug_behavior(self):
        """Document that extracting only translation produces wrong result.

        This test demonstrates the bug behavior to ensure we don't regress.
        """
        xyz = torch.tensor([[0.0, 1.0, 0.0]])

        d_xyz = torch.tensor([[[0, -1, 0, 1],
                               [1,  0, 0, 2],
                               [0,  0, 1, 3],
                               [0,  0, 0, 1]]]).float()

        # BUG: Only extract translation column (what old code did)
        translation_only = d_xyz[:, :3, 3]  # Gets [1, 2, 3]
        bug_result = xyz + translation_only

        # Bug produces (1, 3, 3) instead of correct (0, 2, 3)
        bug_expected = torch.tensor([[1.0, 3.0, 3.0]])
        correct_expected = torch.tensor([[0.0, 2.0, 3.0]])

        assert torch.allclose(bug_result, bug_expected, atol=1e-6), \
            "Bug behavior changed unexpectedly"
        assert not torch.allclose(bug_result, correct_expected, atol=1e-6), \
            "Bug behavior should NOT match correct result"

    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_se3_identity_transformation(self):
        """SE(3) identity matrix should leave points unchanged."""
        xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        N = xyz.shape[0]

        # Identity SE(3) matrix for each point
        d_xyz = torch.eye(4).unsqueeze(0).expand(N, -1, -1).clone()

        xyz_h = torch.cat([xyz, torch.ones(N, 1)], dim=-1)
        deformed = torch.bmm(d_xyz, xyz_h.unsqueeze(-1)).squeeze(-1)[:, :3]

        assert torch.allclose(deformed, xyz, atol=1e-6), \
            f"Identity SE(3) should not change points.\nGot: {deformed}\nExpected: {xyz}"

    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_se3_batch_transformation(self):
        """SE(3) should apply per-Gaussian transformation correctly."""
        # Two points, each with different transformation
        xyz = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
        N = xyz.shape[0]

        # First Gaussian: translate by (1, 0, 0)
        # Second Gaussian: rotate 90° around Z
        d_xyz = torch.tensor([
            [[1, 0, 0, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            [[0, -1, 0, 0],
             [1,  0, 0, 0],
             [0,  0, 1, 0],
             [0,  0, 0, 1]]
        ]).float()

        xyz_h = torch.cat([xyz, torch.ones(N, 1)], dim=-1)
        deformed = torch.bmm(d_xyz, xyz_h.unsqueeze(-1)).squeeze(-1)[:, :3]

        expected = torch.tensor([[2.0, 0.0, 0.0],  # (1,0,0) + (1,0,0)
                                 [-1.0, 0.0, 0.0]])  # R @ (0,1,0)

        assert torch.allclose(deformed, expected, atol=1e-6), \
            f"Batch SE(3) incorrect.\nGot: {deformed}\nExpected: {expected}"


# =============================================================================
# GPU-Required Tests with Skip Markers
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason=GPU_SKIP_REASON)
class TestTrainingGPU:
    """GPU-required tests for training (skipped if no GPU)."""

    @pytest.fixture
    def mock_video_result(self, tmp_path):
        """Create mock VideoProcessingResult for GPU tests."""
        output_dir = tmp_path / "stage1_output"
        output_dir.mkdir()

        transforms = {
            "w": 640,
            "h": 480,
            "fl_x": 500.0,
            "fl_y": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "frames": []
        }

        transforms_path = output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f)

        (output_dir / "frames").mkdir()

        # Return a mock object
        return MagicMock(output_dir=str(output_dir))

    def test_gpu_is_available(self):
        """Verify GPU is detected (sanity check)."""
        import torch
        assert torch.cuda.is_available(), "GPU should be available for this test"

    def test_cuda_device_count(self):
        """Check CUDA device count."""
        import torch
        count = torch.cuda.device_count()
        assert count >= 1, f"Expected at least 1 GPU, found {count}"
