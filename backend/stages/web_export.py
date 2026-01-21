"""Stage 5: Web Export - Convert PLY to web-viewable format."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable


@dataclass
class WebExportResult:
    """Result of web export stage."""

    splat_path: str
    """Path to compressed splat file (SPZ/KSPLAT)."""

    viewer_path: str
    """Path to viewer HTML bundle."""

    file_size_mb: float
    """Size of compressed splat file in MB."""

    format: str
    """Export format used (spz, ksplat, ply)."""

    embed_code: Optional[str] = None
    """HTML embed code for the viewer."""


class WebExportStage:
    """
    Stage 5: Export 3DGS for web viewing.

    This stage:
    1. Converts PLY to compressed format (SPZ or KSPLAT)
    2. Bundles with Spark.js viewer
    3. Generates embeddable viewer package

    Output: scene.spz, viewer.html
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.format = self.config.get("format", "spz")  # spz, ksplat, ply
        self.compression_level = self.config.get("compression_level", "balanced")

    def export(
        self,
        ply_path: str,
        output_dir: str,
        scene_name: str = "scene",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> WebExportResult:
        """
        Export 3DGS to web-viewable format.

        Args:
            ply_path: Path to trained PLY file.
            output_dir: Directory to store outputs.
            scene_name: Name for the scene (used in filenames).
            progress_callback: Optional callback(progress, message).

        Returns:
            WebExportResult with paths to viewer bundle.
        """

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report(0.0, "Starting web export")

        # Step 1: Convert PLY to target format
        report(0.1, f"Converting PLY to {self.format.upper()}")
        splat_path = output_path / f"{scene_name}.{self.format}"
        self._convert_format(ply_path, splat_path)

        file_size_mb = splat_path.stat().st_size / (1024 * 1024)
        report(0.6, f"Converted to {file_size_mb:.1f} MB")

        # Step 2: Generate viewer HTML
        report(0.7, "Generating viewer bundle")
        viewer_path = output_path / "viewer.html"
        self._generate_viewer(splat_path, viewer_path, scene_name)

        # Step 3: Generate embed code
        embed_code = self._generate_embed_code(scene_name)

        report(1.0, "Web export complete")

        return WebExportResult(
            splat_path=str(splat_path),
            viewer_path=str(viewer_path),
            file_size_mb=file_size_mb,
            format=self.format,
            embed_code=embed_code,
        )

    def _convert_format(self, ply_path: str, output_path: Path):
        """Convert PLY to target format using 3dgsconverter."""
        # TODO: Use 3dgsconverter library
        # from dgsconverter import convert
        # convert(ply_path, str(output_path), format=self.format)
        raise NotImplementedError("Format conversion not yet implemented")

    def _generate_viewer(self, splat_path: Path, viewer_path: Path, scene_name: str):
        """Generate standalone HTML viewer with Spark.js."""
        # TODO: Generate HTML with embedded Spark.js viewer
        viewer_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{scene_name} - Brain Dance Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #1a1a2e; overflow: hidden; }}
        #viewer {{ width: 100vw; height: 100vh; }}
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-family: system-ui, sans-serif;
        }}
    </style>
</head>
<body>
    <div id="viewer">
        <div class="loading">Loading 3D scene...</div>
    </div>

    <!-- Spark.js for 3DGS rendering -->
    <script type="module">
        // TODO: Import Spark.js and initialize viewer
        // import {{ Viewer }} from 'spark';
        //
        // const viewer = new Viewer({{
        //     container: document.getElementById('viewer'),
        //     url: '{splat_path.name}',
        // }});
        //
        // viewer.on('load', () => {{
        //     document.querySelector('.loading').remove();
        // }});

        console.log('Brain Dance Viewer - Spark.js integration pending');
    </script>
</body>
</html>
"""
        viewer_path.write_text(viewer_html)

    def _generate_embed_code(self, scene_name: str) -> str:
        """Generate HTML embed code for the viewer."""
        return f'<iframe src="viewer.html" width="100%" height="600" frameborder="0" title="{scene_name}"></iframe>'
