"""Video frame extraction and filtering utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

from .config import VideoConfig


def compute_sharpness(frame: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance.

    Args:
        frame: BGR image as numpy array.

    Returns:
        Sharpness score (higher = sharper).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(
    video_path: str,
    output_dir: str,
    config: Optional[VideoConfig] = None,
) -> List[str]:
    """Extract frames from a video at a specified FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        config: Video extraction configuration.

    Returns:
        List of paths to extracted frames.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If video cannot be opened.
    """
    if config is None:
        config = VideoConfig()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / config.fps))

    print(f"Video: {video_fps:.1f} FPS, {total_frames} frames")
    print(f"Extracting every {frame_interval} frames ({config.fps} target FPS)")

    extracted_paths = []
    frame_idx = 0
    saved_count = 0

    pbar = tqdm(total=min(total_frames, config.max_frames * frame_interval),
                desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        if saved_count >= config.max_frames:
            break

        # Resize if configured
        if config.resize_width and frame.shape[1] > config.resize_width:
            scale = config.resize_width / frame.shape[1]
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (config.resize_width, new_h))

        # Check sharpness
        sharpness = compute_sharpness(frame)
        if sharpness < config.min_sharpness:
            frame_idx += 1
            continue

        # Save frame
        frame_path = output_dir / f"frame_{saved_count:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        extracted_paths.append(str(frame_path))

        saved_count += 1
        frame_idx += 1

    pbar.close()
    cap.release()

    print(f"Extracted {len(extracted_paths)} frames to {output_dir}")
    return extracted_paths


def get_video_info(video_path: str) -> dict:
    """Get basic video information.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with fps, frame_count, width, height, duration.
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / max(info["fps"], 1)
    cap.release()
    return info
