"""Instant4D environment verification tests.

These tests verify that the Instant4D submodule and CUDA kernels are properly
installed. Run on GPU server for full verification.

Usage:
    # On Mac (submodule tests only)
    pytest tests/test_instant4d_setup.py::TestSubmodule -v

    # On GPU server (full suite)
    pytest tests/test_instant4d_setup.py -v

    # Just CUDA kernel tests
    pytest tests/test_instant4d_setup.py -v -m gpu
"""

import sys
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestSubmodule:
    """Verify Instant4D submodule is properly installed."""

    def test_instant4d_directory_exists(self):
        """Instant4D submodule directory exists."""
        instant4d_path = PROJECT_ROOT / "instant4d"
        assert instant4d_path.exists(), (
            "Instant4D submodule not found. "
            "Run: git submodule update --init --recursive"
        )

    def test_instant4d_not_empty(self):
        """Instant4D directory contains files (not empty submodule)."""
        instant4d_path = PROJECT_ROOT / "instant4d"
        readme = instant4d_path / "README.md"
        assert readme.exists(), (
            "Instant4D submodule appears empty. "
            "Run: git submodule update --init --recursive"
        )

    def test_mega_sam_submodule_initialized(self):
        """Mega-SAM nested submodule is initialized."""
        mega_sam_path = PROJECT_ROOT / "instant4d" / "SLAM" / "mega-sam"
        assert mega_sam_path.exists(), "Mega-SAM directory not found"
        # Check it's not empty
        contents = list(mega_sam_path.iterdir()) if mega_sam_path.exists() else []
        assert len(contents) > 0, (
            "Mega-SAM submodule is empty. "
            "Run: git submodule update --init --recursive"
        )

    def test_core_scripts_exist(self):
        """Core Instant4D scripts are present."""
        instant4d_path = PROJECT_ROOT / "instant4d"

        required_files = [
            "script/prune.py",
            "gaussian_renderer/__init__.py",
            "scene/__init__.py",
            "requirement.txt",
        ]

        for f in required_files:
            file_path = instant4d_path / f
            assert file_path.exists(), f"Missing required file: {f}"

    def test_cuda_kernel_directories_exist(self):
        """CUDA kernel source directories exist."""
        instant4d_path = PROJECT_ROOT / "instant4d"

        kernel_dirs = [
            "diff-gaussian-rasterization",
            "submodule/pointops2",
            "submodule/simple-knn",
            "submodule/fussed-ssim",
        ]

        for d in kernel_dirs:
            dir_path = instant4d_path / d
            assert dir_path.exists(), f"Missing CUDA kernel directory: {d}"


@pytest.mark.gpu
class TestCUDAKernels:
    """Verify CUDA kernels are compiled and importable.

    These tests require CUDA and will be skipped on Mac/CPU-only systems.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self):
        """Skip tests if CUDA is not available."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_diff_gaussian_rasterization_import(self):
        """diff-gaussian-rasterization is importable."""
        try:
            from diff_gaussian_rasterization import GaussianRasterizer
            assert GaussianRasterizer is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import diff-gaussian-rasterization: {e}\n"
                "Run: bash scripts/setup_instant4d.sh"
            )

    def test_pointops_import(self):
        """pointops2 is importable."""
        try:
            import pointops
            assert hasattr(pointops, 'knn_query') or hasattr(pointops, 'knnquery')
        except ImportError as e:
            pytest.fail(
                f"Failed to import pointops: {e}\n"
                "Run: bash scripts/setup_instant4d.sh"
            )

    def test_simple_knn_import(self):
        """simple-knn is importable."""
        try:
            from simple_knn import distCUDA2
            assert distCUDA2 is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import simple_knn: {e}\n"
                "Run: bash scripts/setup_instant4d.sh"
            )

    def test_fused_ssim_import(self):
        """fused-ssim is importable."""
        try:
            from fused_ssim import fused_ssim
            assert fused_ssim is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import fused_ssim: {e}\n"
                "Run: bash scripts/setup_instant4d.sh"
            )


@pytest.mark.gpu
class TestPyTorchCUDA:
    """Verify PyTorch CUDA configuration meets requirements."""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        """Skip tests if PyTorch is not installed."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_pytorch_version_sufficient(self):
        """PyTorch version is 2.3+."""
        import torch
        version_parts = torch.__version__.split('.')[:2]
        major, minor = int(version_parts[0]), int(version_parts[1].split('+')[0])

        assert major >= 2, f"PyTorch 2.x required, found {torch.__version__}"
        if major == 2:
            assert minor >= 3, f"PyTorch 2.3+ required, found {torch.__version__}"

    def test_cuda_available(self):
        """CUDA is available in PyTorch."""
        import torch
        assert torch.cuda.is_available(), (
            "CUDA not available. Install PyTorch with CUDA support:\n"
            "pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121"
        )

    def test_cuda_version_sufficient(self):
        """CUDA version is 12.1+."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cuda_version = torch.version.cuda
        if cuda_version is None:
            pytest.skip("CUDA version not reported")

        major = int(cuda_version.split('.')[0])
        minor = int(cuda_version.split('.')[1])

        assert major >= 12, f"CUDA 12.x required, found {cuda_version}"
        if major == 12:
            assert minor >= 1, f"CUDA 12.1+ required, found {cuda_version}"

    def test_gpu_detected(self):
        """At least one GPU is detected."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        gpu_count = torch.cuda.device_count()
        assert gpu_count > 0, "No CUDA GPUs detected"

    def test_gpu_memory_sufficient(self):
        """GPU has sufficient VRAM (16GB+ recommended)."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)

        # Warning threshold at 16GB, but don't fail
        if vram_gb < 16:
            pytest.skip(
                f"GPU has {vram_gb:.1f}GB VRAM. "
                "16GB+ recommended for Instant4D training."
            )


@pytest.mark.gpu
class TestInstant4DImports:
    """Verify Instant4D modules are importable."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        """Add Instant4D to Python path for imports."""
        instant4d_path = PROJECT_ROOT / "instant4d"
        if str(instant4d_path) not in sys.path:
            sys.path.insert(0, str(instant4d_path))
        yield
        # Cleanup
        if str(instant4d_path) in sys.path:
            sys.path.remove(str(instant4d_path))

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self):
        """Skip tests if CUDA is not available."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_gaussian_renderer_import(self):
        """gaussian_renderer module is importable."""
        try:
            from gaussian_renderer import render
            assert render is not None
        except ImportError as e:
            pytest.fail(f"Failed to import gaussian_renderer: {e}")

    def test_scene_module_import(self):
        """scene module is importable."""
        try:
            from scene import Scene
            assert Scene is not None
        except ImportError as e:
            # Scene might have additional dependencies
            pytest.skip(f"scene module import skipped: {e}")


class TestDependencies:
    """Verify Python dependencies are installed."""

    def test_unidepth_installed(self):
        """UniDepth depth estimation is installed."""
        try:
            import unidepth
            assert unidepth is not None
        except ImportError:
            pytest.skip(
                "unidepth not installed. "
                "Install with: pip install unidepth"
            )

    def test_xformers_installed(self):
        """xformers is installed (optional but recommended)."""
        try:
            import xformers
            assert xformers is not None
        except ImportError:
            pytest.skip(
                "xformers not installed (optional). "
                "Install with: pip install xformers"
            )
