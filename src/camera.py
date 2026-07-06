"""Camera intrinsics estimation and manipulation."""

import numpy as np
from typing import Optional, Tuple
import json


class CameraIntrinsics:
    """Camera intrinsic parameters.

    Args:
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    @classmethod
    def estimate_from_image(cls, width: int, height: int,
                            fov_deg: float = 60.0) -> "CameraIntrinsics":
        """Estimate intrinsics from image dimensions assuming a field of view.

        Uses the heuristic: f = max(W, H) for ~53° FOV,
        or computes from a specified FOV.

        Args:
            width: Image width.
            height: Image height.
            fov_deg: Assumed horizontal field of view in degrees.

        Returns:
            CameraIntrinsics with estimated parameters.
        """
        f = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        return cls(
            fx=f, fy=f,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )

    @classmethod
    def from_file(cls, path: str) -> "CameraIntrinsics":
        """Load intrinsics from a JSON calibration file.

        Expected format: {"fx", "fy", "cx", "cy", "width", "height"}
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str):
        """Save intrinsics to a JSON file."""
        data = {
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def scale(self, factor: float) -> "CameraIntrinsics":
        """Return new intrinsics scaled by a factor."""
        return CameraIntrinsics(
            fx=self.fx * factor, fy=self.fy * factor,
            cx=self.cx * factor, cy=self.cy * factor,
            width=int(self.width * factor),
            height=int(self.height * factor),
        )

    def __repr__(self) -> str:
        return (f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}x{self.height})")
