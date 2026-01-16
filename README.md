# Brain Dance

Generate controllable videos from static images with precise camera and object motion control.

## Overview

Brain Dance is an interactive world model editor for research and creative exploration. It lets you transform a single image into a dynamic video with explicit control over how the camera moves and how objects in the scene behave.

Most video generation tools treat output as a black box—you describe what you want, and the model decides how it happens. Brain Dance takes a different approach: you define the geometry. Specify camera trajectories, object motion paths, and keyframes, then let the world model bring it to life with physical plausibility.

## Features

- **Image-to-video generation** — Start from a single static image
- **4D geometric control** — Precise camera motion and multi-object trajectories
- **Trajectory editing** — Define keyframes with position, rotation, and interpolation
- **Extensible architecture** — Plug in different world model backends

## How It Works

```
Image → Preprocess → Edit Trajectories → Generate Video
         (depth,       (camera path,
        segments)      object motion)
```

Brain Dance preprocesses your image to extract depth and segment objects, then lets you design motion trajectories in 3D space. These trajectories are rendered as control maps that guide a video diffusion model to generate the final output.

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
- CUDA 12.1+ for GPU inference

## Acknowledgments

- [VerseCrafter](https://github.com/TencentARC/VerseCrafter) by Tencent ARC
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) video diffusion model

## License

This project is for research and exploration. VerseCrafter (included as a submodule) has its own license restricting use to academic purposes.
