# Brain Dance

Interactive world model editor for research and exploration. Generate controllable videos from static images using state-of-the-art world models.

## Overview

Brain Dance provides a unified interface for working with video world models. It abstracts different backends (VerseCrafter, and future models) behind a common API, allowing you to:

- **Preprocess** images to extract depth, segment objects, and build 3D scene representations
- **Edit trajectories** interactively - define camera motion and object movement
- **Generate videos** with precise 4D geometric control

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│    Backend      │────▶│  World Models   │
│  (Next.js/3JS)  │     │   (FastAPI)     │     │ (VerseCrafter)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     Browser              Python API             GPU Inference
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.

## Quick Start

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/YOUR_ORG/brain-dance.git
cd brain-dance
```

### 2. Set Up Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Run in Mock Mode (No GPU)

```bash
MOCK_INFERENCE=true python -m backend.server
```

### 4. Test the API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/adapters
```

See [docs/SETUP.md](docs/SETUP.md) for full setup including GPU configuration.

## Project Structure

```
brain-dance/
├── versecrafter/       # Git submodule - Tencent's VerseCrafter
├── backend/            # Python API server
│   ├── adapters/       # World model adapters
│   │   ├── base.py     # Abstract interface
│   │   └── versecrafter.py
│   └── server.py       # FastAPI application
├── frontend/           # Next.js editor (TBD)
├── docs/               # Documentation
│   ├── ARCHITECTURE.md # System design
│   ├── SETUP.md        # Environment setup
│   ├── API.md          # API specification
│   └── ADDING_MODELS.md# Extensibility guide
└── README.md           # This file
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [Setup Guide](docs/SETUP.md) - Environment and GPU configuration
- [API Reference](docs/API.md) - Backend REST API specification
- [Adding Models](docs/ADDING_MODELS.md) - How to add new world model backends

## World Model Backends

| Model | Status | Description |
|-------|--------|-------------|
| **VerseCrafter** | ✅ Integrated | 4D geometric control (camera + objects) |
| Runway | 🔮 Future | API-based video generation |
| Pika | 🔮 Future | API-based video generation |

## Requirements

- **Python 3.10+** (backend)
- **Node.js 18+** (frontend, when implemented)
- **CUDA 12.1+** (for GPU inference - not needed for development)

## Development

### Running Tests

```bash
pytest backend/
```

### Adding a New World Model

1. Create adapter in `backend/adapters/your_model.py`
2. Implement the `WorldModelAdapter` interface
3. Register in `backend/adapters/__init__.py`

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for details.

## License

This project is for research and exploration. Note that VerseCrafter (submodule) has its own license restricting use to academic purposes.

## Acknowledgments

- [VerseCrafter](https://github.com/TencentARC/VerseCrafter) by Tencent ARC
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) video diffusion model
