"""Tests for Deformable 3D Gaussians adapter.

These tests verify the De3DGS adapter fixes for:
- Issue #1: Path traversal vulnerability
- Issue #3: Dataclass validation
- Issue #4: CUDA error detection

Usage:
    # Run all tests
    pytest tests/adapters/test_deformable3dgs.py -v

    # Run specific test classes
    pytest tests/adapters/test_deformable3dgs.py::TestPathValidation -v
    pytest tests/adapters/test_deformable3dgs.py::TestDataclassValidation -v
    pytest tests/adapters/test_deformable3dgs.py::TestCUDAErrorParsing -v
"""

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from backend.adapters.deformable3dgs import (
    Deformable3DGSAdapter,
    Deformable3DGSOptions,
    De3DGSError,
    CUDAOutOfMemoryError,
    CUDADriverError,
    CUDADeviceError,
)


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
