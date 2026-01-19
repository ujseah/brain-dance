# Brain Dance

Transform video footage into explorable 3D worlds.

## Overview

Brain Dance reconstructs 3D scenes from video input, creating Gaussian Splat representations you can explore in your browser. Point your camera at a space, walk through it, and Brain Dance builds a 3D world you can navigate freely — with AI filling in the parts you never saw.

## Features

- **Video to 3D** — Convert video footage into explorable 3D Gaussian Splats
- **AI hole-filling** — Automatically fill unseen regions with plausible geometry
- **Web viewer** — Explore reconstructed scenes in any browser
- **Compressed export** — SPZ format for fast web streaming

## How It Works

```text
Video Input → Frame Extraction → Pose Estimation → 3DGS Training → AI Hole-Fill → Web Export
              (ffmpeg)           (hloc/GLOMAP)     (Splatfacto)     (MVDream)      (SPZ)
```

1. **Video Processing**: Extract frames and estimate camera poses
2. **3DGS Training**: Train 3D Gaussian Splatting model from multi-view images
3. **Hole Filling**: AI fills gaps where the camera never looked
4. **Web Export**: Compress and bundle for browser viewing

## Quick Start

```bash
git clone --recursive https://github.com/YOUR_ORG/brain-dance.git
cd brain-dance
pip install -r backend/requirements.txt
python -m backend.server
```

The API server runs at `http://localhost:8000`. Check `/health` and `/adapters` to verify it's working.

## Requirements

- Python 3.10+
- CUDA 12.1+ for GPU inference (required for 3DGS training)
- 24GB+ VRAM recommended (RTX 3090/4090 or A100)

## Project Structure

```text
brain-dance/
├── backend/
│   ├── adapters/           # Model adapters
│   │   ├── video_to_3dgs.py   # Main pipeline adapter
│   │   └── versecrafter.py    # Legacy (MoGe-V2 depth)
│   ├── stages/             # Pipeline stages
│   │   ├── video_processing.py
│   │   ├── gaussian_training.py
│   │   ├── hole_filling.py
│   │   └── web_export.py
│   └── server.py           # FastAPI server
├── frontend/               # Next.js viewer (WIP)
├── versecrafter/           # Submodule for MoGe-V2
└── docs/                   # Documentation
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and pipeline details
- [API](docs/API.md) — REST API specification
- [Roadmap](docs/ROADMAP.md) — Implementation phases

## Acknowledgments

- [Nerfstudio](https://nerf.studio/) for Splatfacto implementation
- [VerseCrafter](https://github.com/TencentARC/VerseCrafter) for MoGe-V2 depth estimation
- [Spark.js](https://sparkjs.dev/) for web-based 3DGS rendering
- [hloc](https://github.com/cvg/Hierarchical-Localization) for pose estimation

## License

This project is for research and exploration. Dependencies have their own licenses.
