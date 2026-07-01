# mono3d-recon

**3D Scene Reconstruction from Monocular Video**

Transform a single video into an interactive 3D model — point cloud and textured mesh — viewable in your browser.

```
Video  ──▶  Depth Maps  ──▶  Camera Poses  ──▶  Point Cloud  ──▶  Mesh  ──▶  Web Viewer
```

## Features

- **Monocular depth estimation** using MiDaS v3.1 or Depth Anything V2
- **Automatic camera pose estimation** via SIFT feature matching + Essential Matrix decomposition
- **Dense point cloud reconstruction** with ICP refinement
- **Surface mesh generation** using Poisson or Ball Pivoting algorithms
- **Interactive web viewer** built with Three.js (mesh, point cloud, wireframe modes)
- **One-command CLI** — video in, 3D model out
- **Web UI** — drag-and-drop video upload with live progress tracking
- **Quality presets** — low/medium/high for speed vs. detail tradeoff
- **GPU acceleration** — automatic CUDA detection, CPU fallback

## Quick Start

### Installation

```bash
git clone https://github.com/Eshaan0110/mono3d-recon.git
cd mono3d-recon

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### CLI Usage

```bash
# Basic reconstruction
python -m src.cli reconstruct --input video.mp4

# High quality on GPU
python -m src.cli reconstruct --input video.mp4 --quality high --device cuda

# Custom output directory
python -m src.cli reconstruct --input video.mp4 --output my_scene

# Just extract frames
python -m src.cli extract --input video.mp4 --fps 3

# Video info
python -m src.cli info --input video.mp4
```

### Web Interface

```bash
python run_server.py
# Open http://localhost:5000
```

Upload a video through the web UI, select quality, and watch the reconstruction happen in real time. The result loads directly into the 3D viewer.

## CLI Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--quality` | low, medium, high | medium | Quality preset |
| `--depth-model` | midas, dav2 | midas | Depth estimation backend |
| `--mesh-method` | poisson, bpa | poisson | Mesh reconstruction method |
| `--device` | auto, cuda, cpu | auto | Compute device |
| `--fps` | float | 2.0 | Frame extraction rate |
| `--config` | path | — | YAML config file |

## Project Structure

```
mono3d-recon/
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration system (dataclasses + YAML)
│   ├── video_utils.py       # Frame extraction & filtering
│   ├── depth_estimator.py   # MiDaS / Depth Anything V2
│   ├── camera.py            # Camera intrinsics
│   ├── feature_matching.py  # SIFT/ORB detection & matching
│   ├── pose_estimation.py   # Essential matrix → camera poses
│   ├── point_cloud.py       # Depth → 3D points, fusion, ICP
│   ├── mesh.py              # Surface reconstruction & export
│   ├── pipeline.py          # End-to-end orchestration
│   ├── cli.py               # Command-line interface
│   └── server.py            # Flask web server + API
├── web/
│   ├── index.html           # Web viewer page
│   ├── css/style.css        # Viewer styles
│   └── js/
│       ├── viewer.js         # Three.js 3D engine
│       └── app.js            # UI controller
├── tests/
│   └── test_core.py         # Unit tests
├── data/                     # Extracted frames & depth maps
├── outputs/                  # Reconstruction outputs
├── models/                   # Model weights (auto-downloaded)
├── requirements.txt
├── run_server.py
├── ARCHITECTURE.md
└── README.md
```

## Outputs

After reconstruction, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `scene.ply` | Dense colored point cloud |
| `scene_mesh.ply` | Triangle mesh (PLY format) |
| `scene.glb` | Triangle mesh (glTF binary, for web) |
| `intrinsics.json` | Estimated camera parameters |
| `trajectory.npy` | Camera positions over time |
| `stats.json` | Reconstruction statistics |
| `frames/` | Extracted video frames |
| `depth/` | Depth maps (.npy) and visualizations (.png) |

## Recording Tips

For best results:

1. **Move slowly and steadily** — avoid fast rotations or shaky movement
2. **Keep 15–60 seconds** — too short gives few frames, too long is slow
3. **Record at 1080p** — higher resolution helps feature matching
4. **Ensure good lighting** — dark scenes produce poor depth estimates
5. **Maintain overlap** — each frame should share ~60% with the previous one
6. **Walk around the subject** — for objects, do a 360° orbit; for rooms, walk through

## Testing

```bash
pytest tests/ -v
```

## Tech Stack

- **PyTorch** — depth model inference
- **OpenCV** — frame extraction, feature matching, pose estimation
- **Open3D** — point cloud processing, ICP, mesh reconstruction
- **trimesh** — GLB/glTF export
- **Flask** — web server
- **Three.js** — browser-based 3D rendering

## License

MIT
