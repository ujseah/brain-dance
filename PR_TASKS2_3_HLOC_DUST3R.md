# PR: Implement hloc+GLOMAP and DUSt3R fallback pipelines (Stage 1, Tasks 2-3)

## Summary

Implements complete camera pose estimation pipeline with automatic quality-based fallback for Brain Dance Stage 1 video processing. Primary path uses hloc+GLOMAP for fast, accurate pose estimation. Falls back to DUSt3R when quality thresholds are not met (low texture, sparse views, motion blur).

## Changes

### Task 2: hloc + GLOMAP Pipeline

- Implement `_run_hloc_pipeline()` method (209-326)
  - SuperPoint feature extraction (2048 keypoints, NMS radius 3)
  - Sequential frame pairing (match with next 10 frames)
  - LightGlue feature matching
  - GLOMAP global SfM (10-100x faster than COLMAP)
  - COLMAP fallback if GLOMAP fails
  - Quality metric validation
  - Nerfstudio transforms.json export
  - Sparse point cloud PLY export

### Task 3: DUSt3R Fallback

- Implement `_run_dust3r_pipeline()` method (512-588)
  - Load DUSt3R ViT-Large model from HuggingFace (2.4GB)
  - Adaptive pairing strategy based on frame count:
    - <30 frames: complete graph (best quality)
    - 30-100 frames: sliding window (balanced)
    - >100 frames: one-ref (memory efficient)
  - Inference with batch_size=1 for memory efficiency
  - Global alignment (MST init, 300 iterations, cosine schedule)
  - Export to transforms.json (same format as hloc)

### Helper Methods

- `_compute_quality_metrics()` (328-370) - Validates reconstruction quality
  - Reprojection error (mean across all 2D points)
  - Track length (mean and max views per 3D point)
  - Pose coverage (% of frames with valid poses)
  - Number of registered frames and 3D points

- `_should_use_dust3r_fallback()` (372-400) - Automatic fallback decision
  - Criterion 1: Coverage < 60%
  - Criterion 2: Reprojection error > 3px
  - Criterion 3: Track length < 3 views
  - Criterion 4: < 100 3D points

- `_export_transforms_json()` (402-419) - Uses nerfstudio converter
- `_manual_export_transforms_json()` (421-473) - Manual COLMAP→Nerfstudio conversion
- `_export_sparse_points()` (475-510) - Export PLY point cloud
- `_dust3r_to_transforms_json()` (590-636) - DUSt3R→Nerfstudio conversion

### Automatic Fallback Integration

- Updated `process()` method (84-100) with try-catch fallback
- hloc pipeline runs first
- On quality failure or exception → automatic DUSt3R fallback
- Progress reporting at each stage

## Implementation Details

### Pipeline Flow

```
Input: frames/*.jpg (from Task 1)
    ↓
[Task 2] hloc + GLOMAP
    ├─→ Extract SuperPoint features
    ├─→ Match with LightGlue
    ├─→ Run GLOMAP (fallback to COLMAP if needed)
    ├─→ Validate quality metrics
    └─→ [PASS] Export transforms.json + points3D.ply
        [FAIL] ↓
[Task 3] DUSt3R fallback
    ├─→ Load ViT-Large model
    ├─→ Create image pairs (adaptive strategy)
    ├─→ Run inference
    ├─→ Global alignment
    └─→ Export transforms.json
        ↓
Output:
- transforms.json (Nerfstudio format, OpenGL convention)
- sparse/points3D.ply (from hloc only)
```

### Quality Thresholds

From ROADMAP.md:

| Metric | hloc Success | Trigger Fallback |
|--------|--------------|------------------|
| Pose coverage | > 80% | < 60% |
| Reprojection error | < 2px mean | > 3px mean |
| Track length | > 100 max | < 3 mean |
| 3D points | - | < 100 |

### Coordinate System Conversion

Both pipelines convert from OpenCV to OpenGL convention:

```python
c2w[:3, 1:3] *= -1  # Flip Y and Z axes
```

- COLMAP/DUSt3R: +X right, +Y down, +Z forward (OpenCV)
- Nerfstudio: +X right, +Y up, +Z backward (OpenGL)

### Output Format

transforms.json (identical for both pipelines):

```json
{
  "camera_model": "OPENCV",
  "fl_x": 1234.5,
  "fl_y": 1234.5,
  "cx": 960.0,
  "cy": 540.0,
  "w": 1920,
  "h": 1080,
  "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
  "frames": [
    {
      "file_path": "./frames/0001.jpg",
      "transform_matrix": [[4x4 camera-to-world matrix]]
    }
  ]
}
```

## Dependencies

### Required for hloc + GLOMAP

```bash
pip install hloc pycolmap nerfstudio
```

### Required for DUSt3R fallback

```bash
pip install git+https://github.com/naver/dust3r.git roma trimesh torch torchvision
```

### Hardware Requirements

- **hloc + GLOMAP**: 8-16GB VRAM (CUDA required)
- **DUSt3R**: 8-24GB VRAM (depending on pairing strategy)
- **Minimum**: RTX 3080 (12GB)
- **Recommended**: RTX 3090/4090 (24GB) or A100

## Error Handling

- Helpful ImportError messages with installation instructions
- GLOMAP → COLMAP fallback on SfM failure
- hloc → DUSt3R fallback on quality failure
- Logging at each pipeline stage
- Empty point cloud handling

## Test Plan

- [ ] Run hloc pipeline on outdoor video (good texture)
  - [ ] Verify features.h5 created
  - [ ] Verify pairs.txt generated
  - [ ] Verify matches.h5 created
  - [ ] Verify GLOMAP reconstruction succeeds
  - [ ] Verify quality metrics meet thresholds (coverage > 80%, error < 2px)
  - [ ] Verify transforms.json in Nerfstudio format
  - [ ] Verify sparse/points3D.ply exported

- [ ] Run hloc pipeline on indoor video (less texture)
  - [ ] Verify reconstruction succeeds or fallback triggers appropriately

- [ ] Test DUSt3R fallback on challenging video
  - [ ] Verify fallback triggers on low coverage (<60%)
  - [ ] Verify DUSt3R model auto-downloads from HuggingFace
  - [ ] Verify adaptive pairing strategy (complete/swin/one-ref)
  - [ ] Verify global alignment converges
  - [ ] Verify transforms.json output matches hloc format

- [ ] Verify coordinate system conversion (OpenCV → OpenGL)
- [ ] Test with different frame counts (10, 50, 150 frames)
- [ ] Verify camera intrinsics are reasonable (focal length ~width)

## Files Changed

| File | Lines Changed | Description |
|------|---------------|-------------|
| backend/stages/video_processing.py | +429 lines | Implement hloc, DUSt3R, and helper methods |

## Related

- Part of Phase 1: Core Pipeline (per ROADMAP.md)
- Builds on Task 1: ffmpeg frame extraction
- Prerequisite for Stage 2: Gaussian Training (3DGS)
- Outputs transforms.json for Nerfstudio/Splatfacto

## Performance Notes

- **GLOMAP**: 10-100x faster than COLMAP incremental SfM
- **DUSt3R**: Single forward pass (no iterative bundle adjustment)
- **Memory optimization**: Adaptive pairing strategy reduces VRAM usage for large sequences
- **Quality-first**: Automatic fallback ensures robust pose estimation

🤖 Generated with [Claude Code](https://claude.com/claude-code)
