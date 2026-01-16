"""Brain Dance API Server - FastAPI backend for world model inference."""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .adapters import get_adapter, list_adapters, PreprocessResult


# Configuration
MOCK_INFERENCE = os.environ.get("MOCK_INFERENCE", "false").lower() == "true"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/brain-dance/uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/brain-dance/outputs"))

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job storage (replace with Redis/DB for production)
jobs: dict = {}


# Pydantic models
class CameraKeyframe(BaseModel):
    time: float
    position: list[float] = Field(..., min_length=3, max_length=3)
    rotation: list[float] = Field(..., min_length=4, max_length=4)
    fov: Optional[float] = None


class CameraTrajectory(BaseModel):
    keyframes: list[CameraKeyframe]
    interpolation: str = "cubic"


class ObjectKeyframe(BaseModel):
    time: float
    position: list[float] = Field(..., min_length=3, max_length=3)
    rotation: list[float] = Field(..., min_length=4, max_length=4)
    scale: Optional[list[float]] = None


class ObjectTrajectory(BaseModel):
    id: str
    keyframes: list[ObjectKeyframe]


class Trajectory(BaseModel):
    duration: float = Field(..., gt=0)
    fps: int = Field(default=24, gt=0)
    camera: CameraTrajectory
    objects: list[ObjectTrajectory] = []


class PreprocessRequest(BaseModel):
    adapter: str = "versecrafter"
    detect_objects: bool = True
    object_prompts: list[str] = []


class GenerateRequest(BaseModel):
    preprocess_job_id: str
    adapter: str = "versecrafter"
    trajectory: Trajectory
    resolution: tuple[int, int] = (1280, 720)
    seed: Optional[int] = None
    guidance_scale: float = 7.5


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = 0.0
    stage: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Brain Dance API starting...")
    print(f"Mock inference: {MOCK_INFERENCE}")
    print(f"Upload dir: {UPLOAD_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    yield
    # Shutdown
    print("Brain Dance API shutting down...")


# Create app
app = FastAPI(
    title="Brain Dance API",
    description="World model inference API for interactive trajectory editing",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/health")
async def health_check():
    """Check server health and GPU availability."""
    gpu_available = False
    gpu_info = None

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
            }
    except ImportError:
        pass

    return {
        "status": "healthy",
        "mock_mode": MOCK_INFERENCE,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
    }


@app.get("/adapters")
async def get_adapters():
    """List available world model adapters."""
    adapters = list_adapters()
    return {
        "adapters": adapters,
        "default": "versecrafter",
    }


@app.post("/preprocess")
async def preprocess_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    adapter: str = "versecrafter",
    detect_objects: bool = True,
):
    """
    Start preprocessing pipeline for an image.

    Returns a job ID to poll for results.
    """
    # Generate job ID
    job_id = f"prep_{uuid.uuid4().hex[:12]}"

    # Save uploaded image
    image_path = UPLOAD_DIR / f"{job_id}_{image.filename}"
    content = await image.read()
    with open(image_path, "wb") as f:
        f.write(content)

    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "stage": None,
        "result": None,
        "error": None,
    }

    # Run preprocessing in background
    background_tasks.add_task(
        run_preprocess,
        job_id,
        str(image_path),
        adapter,
        {"detect_objects": detect_objects},
    )

    return {"job_id": job_id, "status": "queued"}


async def run_preprocess(job_id: str, image_path: str, adapter_name: str, options: dict):
    """Background task for preprocessing."""
    try:
        jobs[job_id]["status"] = "processing"

        def progress_callback(pct, msg):
            jobs[job_id]["progress"] = pct
            jobs[job_id]["stage"] = msg

        # Get adapter
        adapter = get_adapter(adapter_name, {"mock": MOCK_INFERENCE})

        # Run preprocessing (this is sync, but we're in a background task)
        result = adapter.preprocess(
            image_path,
            options=options,
            progress_callback=progress_callback,
        )

        # Store result
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "depth_map_path": result.depth_map_path,
            "objects": [
                {"id": obj.id, "label": obj.label, "center": obj.center}
                for obj in result.objects
            ],
            "camera": {
                "fov": result.camera.fov if result.camera else None,
            },
            "scene_data": result.scene_data,
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/preprocess/{job_id}")
async def get_preprocess_status(job_id: str):
    """Get status of a preprocessing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        error=job["error"],
        result=job["result"],
    )


@app.post("/generate")
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start video generation from preprocessed scene and trajectory.

    Returns a job ID to poll for results.
    """
    # Check preprocess job exists and completed
    if request.preprocess_job_id not in jobs:
        raise HTTPException(status_code=404, detail="Preprocess job not found")

    prep_job = jobs[request.preprocess_job_id]
    if prep_job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Preprocess job not completed: {prep_job['status']}",
        )

    # Generate job ID
    job_id = f"gen_{uuid.uuid4().hex[:12]}"

    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "stage": None,
        "result": None,
        "error": None,
    }

    # Run generation in background
    background_tasks.add_task(
        run_generate,
        job_id,
        prep_job["result"],
        request,
    )

    return {"job_id": job_id, "status": "queued"}


async def run_generate(job_id: str, preprocess_result: dict, request: GenerateRequest):
    """Background task for video generation."""
    try:
        jobs[job_id]["status"] = "processing"

        def progress_callback(pct, msg):
            jobs[job_id]["progress"] = pct
            jobs[job_id]["stage"] = msg

        # Get adapter
        adapter = get_adapter(request.adapter, {"mock": MOCK_INFERENCE})

        # Reconstruct PreprocessResult
        prep = PreprocessResult(
            depth_map_path=preprocess_result.get("depth_map_path"),
            scene_data=preprocess_result.get("scene_data", {}),
        )

        # Convert trajectory to dict
        trajectory = request.trajectory.model_dump()

        # Run generation
        result = adapter.generate(
            prep,
            trajectory,
            options={
                "resolution": request.resolution,
                "seed": request.seed,
                "guidance_scale": request.guidance_scale,
            },
            progress_callback=progress_callback,
        )

        # Store result
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "video_path": result.video_path,
            "video_url": f"/files/{job_id}/output.mp4",
            "control_map_path": result.control_map_path,
            "duration": result.duration,
            "metadata": result.metadata,
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/generate/{job_id}")
async def get_generate_status(job_id: str):
    """Get status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        error=job["error"],
        result=job["result"],
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # TODO: Actually cancel running job
    jobs[job_id]["status"] = "cancelled"

    return {"job_id": job_id, "status": "cancelled"}


@app.get("/files/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    """Download a generated file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    # Get file path from result
    result = job.get("result", {})
    video_path = result.get("video_path")

    if video_path and Path(video_path).exists():
        return FileResponse(video_path, media_type="video/mp4", filename=filename)

    raise HTTPException(status_code=404, detail="File not found")


# Main entry point
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
