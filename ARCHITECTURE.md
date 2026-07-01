# Architecture

## Pipeline Overview

```
                         mono3d-recon Pipeline
 ┌─────────────────────────────────────────────────────────────────┐
 │                                                                 │
 │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
 │  │  Video   │──▶│  Frame   │──▶│  Depth   │──▶│ Feature  │    │
 │  │  Input   │   │ Extract  │   │ Estimate │   │ Matching │    │
 │  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
 │                                                     │          │
 │  ┌──────────┐   ┌──────────┐   ┌──────────┐        │          │
 │  │   Web    │◀──│   Mesh   │◀──│  Point   │◀──┌────┴────┐    │
 │  │  Viewer  │   │  Recon   │   │  Cloud   │   │  Pose   │    │
 │  └──────────┘   └──────────┘   └──────────┘   │ Estimate│    │
 │                                                └─────────┘    │
 └─────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `src/video_utils.py`
- Extract frames from video at configurable FPS
- Blur detection (Laplacian variance) to skip low-quality frames
- Resize frames for processing efficiency

### `src/depth_estimator.py`
- Monocular depth estimation using MiDaS v3.1 or Depth Anything V2
- GPU/CPU auto-detection
- Bilateral filtering and normalization of depth maps
- Batch processing with colorized visualization output

### `src/camera.py`
- Camera intrinsic matrix estimation from image dimensions
- FOV-based focal length computation
- Calibration file I/O (JSON)

### `src/feature_matching.py`
- SIFT/ORB keypoint detection
- FLANN-based matching with Lowe's ratio test
- Sequential pair matching for video frames

### `src/pose_estimation.py`
- Essential matrix estimation with RANSAC
- Relative pose recovery (R, t)
- Pose chaining into global camera trajectory

### `src/point_cloud.py`
- Depth map back-projection to 3D using camera intrinsics
- Per-frame point cloud generation with RGB colors
- Multi-frame fusion with pose transforms
- ICP refinement (coarse-to-fine)
- Statistical outlier removal + voxel downsampling

### `src/mesh.py`
- Poisson surface reconstruction
- Ball Pivoting Algorithm (alternative)
- Mesh cleaning, simplification, smoothing
- Vertex color transfer from point cloud
- Export to PLY, GLB, OBJ formats

### `src/pipeline.py`
- Orchestrates all stages end-to-end
- Progress callbacks for UI integration
- Statistics collection and output

### `src/server.py`
- Flask REST API for video upload and reconstruction
- Background thread processing
- Job status polling
- Static file serving for the web viewer

### `web/`
- Three.js-based 3D viewer
- GLB and PLY model loading
- Mesh / Point Cloud / Wireframe display modes
- Camera trajectory visualization
- Screenshot and download functionality

## Data Flow

```
video.mp4
  │
  ├──▶ data/frames/frame_00000.jpg ... frame_00099.jpg
  │
  ├──▶ data/depth/depth_00000.npy  ... depth_00099.npy
  │         └── depth_00000.png (colorized visualization)
  │
  ├──▶ outputs/intrinsics.json
  ├──▶ outputs/trajectory.npy
  ├──▶ outputs/scene.ply         (point cloud)
  ├──▶ outputs/scene_mesh.ply    (mesh in PLY)
  ├──▶ outputs/scene.glb         (mesh for web viewer)
  └──▶ outputs/stats.json        (reconstruction statistics)
```

## Key Algorithms

### Depth Estimation
MiDaS uses a DPT (Dense Prediction Transformer) architecture to predict
relative depth from a single image. The depth maps are normalized and
filtered before back-projection.

### Pose Estimation
1. Detect SIFT features in consecutive frames
2. Match features with FLANN + ratio test
3. Compute Essential Matrix with RANSAC
4. Decompose E into R, t with `cv2.recoverPose`
5. Chain relative poses into global trajectory

### Point Cloud Fusion
1. Back-project each depth map to 3D using camera intrinsics
2. Transform each per-frame cloud into world coordinates using estimated pose
3. Refine alignment between consecutive clouds with ICP
4. Merge all clouds and remove outliers

### Mesh Reconstruction
Poisson reconstruction takes the oriented point cloud and solves a
Poisson equation to extract an implicit surface, which is then
converted to a triangle mesh.
