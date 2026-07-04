"""Monocular depth estimation using MiDaS or Depth Anything V2."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

from .config import DepthConfig


class DepthEstimator:
    """Wrapper for monocular depth estimation models.

    Supports MiDaS v3.1 (default) and Depth Anything V2.

    Args:
        config: Depth estimation configuration.
    """

    def __init__(self, config: Optional[DepthConfig] = None):
        self.config = config or DepthConfig()
        self.device = self._resolve_device()
        self.model = None
        self.transform = None
        self._load_model()

    def _resolve_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_model(self):
        """Load the depth estimation model."""
        print(f"Loading depth model: {self.config.model} ({self.config.model_type})")
        print(f"Device: {self.device}")

        if self.config.model == "midas":
            self._load_midas()
        elif self.config.model == "dav2":
            self._load_depth_anything_v2()
        else:
            raise ValueError(f"Unknown depth model: {self.config.model}")

    def _load_midas(self):
        """Load MiDaS model from torch hub."""
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            self.config.model_type,
            trust_repo=True
        )
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True
        )

        if self.config.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def _load_depth_anything_v2(self):
        """Load Depth Anything V2 model."""
        try:
            from transformers import pipeline
            self.model = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=self.device,
            )
            self.transform = None  # Handled by the pipeline
        except ImportError:
            print("transformers not installed, falling back to MiDaS")
            self.config.model = "midas"
            self._load_midas()

    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from a single BGR image.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            Depth map as numpy array (H, W), higher values = farther.
        """
        if self.config.model == "midas":
            return self._estimate_midas(image)
        elif self.config.model == "dav2":
            return self._estimate_dav2(image)

    def _estimate_midas(self, image: np.ndarray) -> np.ndarray:
        """Run MiDaS depth estimation."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()

        # MiDaS outputs inverse depth (close = high values)
        # Invert so that close = low values (metric-like)
        depth = depth.max() - depth + 1e-6

        return self._postprocess(depth)

    def _estimate_dav2(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2 estimation."""
        from PIL import Image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        result = self.model(pil_img)
        depth = np.array(result["depth"])

        if depth.shape[:2] != image.shape[:2]:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        return self._postprocess(depth)

    def _postprocess(self, depth: np.ndarray) -> np.ndarray:
        """Apply normalization and filtering to depth map."""
        if self.config.normalize:
            d_min, d_max = np.percentile(depth, [2, 98])
            depth = np.clip(depth, d_min, d_max)
            depth = (depth - d_min) / (d_max - d_min + 1e-8)

        if self.config.bilateral_filter:
            depth_uint8 = (depth * 255).astype(np.uint8)
            depth_uint8 = cv2.bilateralFilter(
                depth_uint8,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )
            depth = depth_uint8.astype(np.float32) / 255.0

        return depth

    def estimate_batch(
        self,
        frame_paths: List[str],
        output_dir: str,
    ) -> List[str]:
        """Estimate depth for a batch of frames.

        Args:
            frame_paths: List of paths to frame images.
            output_dir: Directory to save depth maps.

        Returns:
            List of paths to saved depth maps (.npy).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        depth_paths = []
        for i, fp in enumerate(tqdm(frame_paths, desc="Estimating depth")):
            image = cv2.imread(fp)
            if image is None:
                print(f"Warning: Could not read {fp}, skipping")
                continue

            depth = self.estimate(image)

            # Save as numpy
            npy_path = output_dir / f"depth_{i:05d}.npy"
            np.save(str(npy_path), depth)

            # Save colorized visualization
            depth_vis = self.colorize(depth)
            vis_path = output_dir / f"depth_{i:05d}.png"
            cv2.imwrite(str(vis_path), depth_vis)

            depth_paths.append(str(npy_path))

        print(f"Saved {len(depth_paths)} depth maps to {output_dir}")
        return depth_paths

    @staticmethod
    def colorize(depth: np.ndarray) -> np.ndarray:
        """Create a colorized visualization of a depth map.

        Args:
            depth: Normalized depth map (0-1).

        Returns:
            BGR colorized depth image.
        """
        depth_normalized = (depth * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        return colored
