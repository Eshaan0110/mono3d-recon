"""Centralized configuration for the reconstruction pipeline."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class VideoConfig:
    """Settings for video frame extraction."""
    fps: float = 2.0
    max_frames: int = 150
    min_sharpness: float = 50.0
    resize_width: Optional[int] = 640


@dataclass
class DepthConfig:
    """Settings for monocular depth estimation."""
    model: str = "midas"  # "midas" or "dav2"
    model_type: str = "DPT_Large"  # MiDaS model type
    device: str = "auto"  # "auto", "cuda", "cpu"
    normalize: bool = True
    bilateral_filter: bool = True
    bilateral_d: int = 5
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0


@dataclass
class FeatureConfig:
    """Settings for feature detection and matching."""
    detector: str = "sift"  # "sift" or "orb"
    max_features: int = 3000
    ratio_threshold: float = 0.75  # Lowe's ratio test
    min_matches: int = 30


@dataclass
class PoseConfig:
    """Settings for camera pose estimation."""
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.999
    min_inliers: int = 15


@dataclass
class PointCloudConfig:
    """Settings for point cloud processing."""
    voxel_size: float = 0.02
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    max_depth: float = 10.0
    min_depth: float = 0.1


@dataclass
class MeshConfig:
    """Settings for mesh reconstruction."""
    method: str = "poisson"  # "poisson" or "bpa"
    poisson_depth: int = 9
    poisson_scale: float = 1.1
    target_faces: int = 100000
    smooth_iterations: int = 5


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    video: VideoConfig = field(default_factory=VideoConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    point_cloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "models"

    # Quality presets override individual settings
    quality: str = "medium"  # "low", "medium", "high"

    def apply_quality_preset(self):
        """Apply quality preset to override individual settings."""
        if self.quality == "low":
            self.video.fps = 1.0
            self.video.resize_width = 480
            self.depth.model_type = "MiDaS_small"
            self.point_cloud.voxel_size = 0.05
            self.mesh.poisson_depth = 7
            self.mesh.target_faces = 50000
        elif self.quality == "high":
            self.video.fps = 3.0
            self.video.resize_width = 960
            self.depth.model_type = "DPT_Large"
            self.point_cloud.voxel_size = 0.01
            self.mesh.poisson_depth = 11
            self.mesh.target_faces = 200000

    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()
        if "video" in data:
            config.video = VideoConfig(**data["video"])
        if "depth" in data:
            config.depth = DepthConfig(**data["depth"])
        if "features" in data:
            config.features = FeatureConfig(**data["features"])
        if "pose" in data:
            config.pose = PoseConfig(**data["pose"])
        if "point_cloud" in data:
            config.point_cloud = PointCloudConfig(**data["point_cloud"])
        if "mesh" in data:
            config.mesh = MeshConfig(**data["mesh"])
        if "quality" in data:
            config.quality = data["quality"]
        if "data_dir" in data:
            config.data_dir = data["data_dir"]
        if "output_dir" in data:
            config.output_dir = data["output_dir"]
        return config
