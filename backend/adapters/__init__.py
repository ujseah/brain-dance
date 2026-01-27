"""World Model Adapters - Abstraction layer for different world model backends."""

from .base import (
    WorldModelAdapter,
    PreprocessResult,
    GenerateResult,
    ModelCapabilities,
    DetectedObject,
    CameraInfo,
)
from .video_to_3dgs import VideoTo3DGSAdapter

# Legacy adapter (kept for MoGe-V2 depth estimation in hole-filling)
from .versecrafter import VerseCrafterAdapter

# Instant4D adapter for 4D Gaussian Splatting (used internally by GaussianTrainingStage)
from .instant4d import Instant4DAdapter, Instant4DOptions, Instant4DResult

# Registry of available adapters
ADAPTERS = {
    "video_to_3dgs": VideoTo3DGSAdapter,
    "versecrafter": VerseCrafterAdapter,  # Legacy, kept for depth estimation
}


def get_adapter(name: str, config: dict = None) -> WorldModelAdapter:
    """
    Get an adapter instance by name.

    Args:
        name: Adapter identifier (e.g., "video_to_3dgs", "versecrafter")
        config: Optional configuration dictionary

    Returns:
        Initialized adapter instance

    Raises:
        ValueError: If adapter name is not found
    """
    if name not in ADAPTERS:
        available = ", ".join(ADAPTERS.keys())
        raise ValueError(f"Unknown adapter: {name}. Available: {available}")
    return ADAPTERS[name](config)


def list_adapters() -> list:
    """
    List all available adapters with their capabilities.

    Returns:
        List of adapter info dictionaries
    """
    result = []
    for name, adapter_class in ADAPTERS.items():
        adapter = adapter_class({})
        result.append({
            "id": name,
            "name": adapter.name,
            "description": adapter.description,
            "capabilities": adapter.get_capabilities().__dict__,
        })
    return result


__all__ = [
    "WorldModelAdapter",
    "PreprocessResult",
    "GenerateResult",
    "ModelCapabilities",
    "DetectedObject",
    "CameraInfo",
    "VideoTo3DGSAdapter",
    "VerseCrafterAdapter",
    "Instant4DAdapter",
    "Instant4DOptions",
    "Instant4DResult",
    "get_adapter",
    "list_adapters",
    "ADAPTERS",
]
