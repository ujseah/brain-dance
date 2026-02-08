# Brain Dance

Transform video footage into explorable **4D worlds**.

## Overview

Brain Dance reconstructs dynamic 4D scenes from monocular video input, creating Gaussian Splat representations you can explore in your browser. Point your camera at a space, walk through it, and Brain Dance builds a 4D world you can navigate freely in space **while time progresses**.

## Features

- **Video to 4D** — Convert video footage into explorable 4D Gaussian Splats
- **Temporal playback** — Navigate in 3D while time progresses
- **Web viewer** — Explore reconstructed scenes in any browser with time controls
- **Compressed export** — SPZ format for fast web streaming

## How It Works

```text
Video Input → Frame Extraction → Pose Estimation → 4D Reconstruction → Web Export
              (ffmpeg)           (hloc/GLOMAP)      (De3DGS)            (SPZ)
```

1. **Video Processing**: Extract frames and estimate camera poses (hloc/GLOMAP)
2. **4D Reconstruction**: Train Deformable 3D Gaussians with temporal deformation MLP
3. **Web Export**: Extract per-frame PLYs, compress to SPZ, bundle with temporal viewer

> **Note**: Object Segmentation (SAM-2) and Scene Completion (MVDream) are available but currently skipped/deferred for the core 4D pipeline.

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
│   │   ├── deformable3dgs.py  # De3DGS adapter (in progress)
│   │   └── instant4d.py       # Legacy (being replaced)
│   ├── stages/             # Pipeline stages
│   │   ├── video_processing.py    # ✅ Complete
│   │   ├── object_segmentation.py # ⏭ Skipped for De3DGS
│   │   ├── gaussian_training.py   # 🔄 In progress
│   │   ├── scene_completion.py    # ⏸ Deferred
│   │   └── web_export.py          # ⏳ Pending
│   └── server.py           # FastAPI server
├── frontend/               # Next.js viewer (WIP)
├── deformable3dgs/         # De3DGS submodule (4D reconstruction)
├── instant4d/              # Instant4D submodule (legacy)
└── docs/                   # Documentation
```

## Documentation

- [Architecture](docs/architecture.md) — System design and pipeline details
- [API](docs/API.md) — REST API specification
- [Setup](docs/setup.md) — Environment setup guide
- [Roadmap](docs/ROADMAP.md) — Implementation phases
- [Adding Adapters](docs/adding_models.md) — How to add new pipeline adapters

## Acknowledgments

- [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) (CVPR 2024) for 4D reconstruction
- [Spark.js](https://sparkjs.dev/) for web-based 3DGS rendering
- [hloc](https://github.com/cvg/Hierarchical-Localization) for pose estimation
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for the foundational representation

## License

This project is for research and exploration. Dependencies have their own licenses.
