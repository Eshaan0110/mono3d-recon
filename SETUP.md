# Setup Guide

## Prerequisites

- Python 3.10 or higher
- pip
- (Optional) NVIDIA GPU with CUDA for faster depth estimation
- (Optional) ffmpeg for broader video format support

## Step-by-Step Installation

### 1. Clone the repository

```bash
git clone https://github.com/Eshaan0110/mono3d-recon.git
cd mono3d-recon
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (recommended):**
```bash
# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import open3d; print('Open3D:', open3d.__version__)"
```

### 5. Run on a sample video

```bash
# Record or download a 15-30 second video
python -m src.cli reconstruct --input your_video.mp4 --quality medium
```

### 6. View results

```bash
# Option A: Web viewer
python run_server.py
# Open http://localhost:5000 and load outputs/scene.glb

# Option B: Open outputs/scene.glb in any 3D viewer (Blender, online glTF viewer)
```

## Troubleshooting

### "No module named 'src'"
Make sure you're running from the project root directory (`mono3d-recon/`).

### "CUDA out of memory"
Use a smaller depth model or lower quality:
```bash
python -m src.cli reconstruct --input video.mp4 --quality low --device cpu
```

### "Only N frames extracted"
Your video might be too short or too blurry. Try:
```bash
python -m src.cli reconstruct --input video.mp4 --fps 5
```

### Poor reconstruction quality
- Record with more overlap between frames
- Move more slowly
- Ensure good lighting
- Try `--quality high`

### Open3D visualization not working (headless server)
The web viewer doesn't need a display. For CLI use on a headless server,
Open3D's visualization windows won't open, but file exports work fine.

### MiDaS model download fails
The first run downloads model weights (~400MB). If your network blocks
`torch.hub`, download manually:
```bash
mkdir -p ~/.cache/torch/hub/checkpoints/
# Download from https://github.com/isl-org/MiDaS/releases
```
