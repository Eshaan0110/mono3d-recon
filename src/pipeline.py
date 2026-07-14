"""Main reconstruction pipeline: video -> 3D model."""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Callable

from .config import PipelineConfig
from .video_utils import extract_frames, get_video_info
from .depth_estimator import DepthEstimator
from .camera import CameraIntrinsics
from .feature_matching import FeatureMatcher
from .pose_estimation import PoseEstimator
from .point_cloud import (
    build_reconstruction, save_pointcloud, clean_pointcloud
)
from .mesh import (
    MeshReconstructor, save_mesh_ply, save_mesh_glb,
    save_mesh_obj
)


class ReconstructionPipeline:
    """End-to-end 3D reconstruction from monocular video.

    Orchestrates: frame extraction -> depth estimation -> feature matching ->
    pose estimation -> point cloud fusion -> mesh reconstruction.

    Args:
        config: Pipeline configuration.
        progress_callback: Optional callback(stage, progress, message)
            for reporting progress to a UI.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.config = config or PipelineConfig()
        self.config.apply_quality_preset()
        self.progress_cb = progress_callback or (lambda *a: None)

        # Pipeline state
        self.frame_paths = []
        self.depth_paths = []
        self.intrinsics = None
        self.poses = []
        self.point_cloud = None
        self.mesh = None
        self.stats = {}

    def _report(self, stage: str, progress: float, message: str):
        """Report progress."""
        print(f"[{stage}] {message}")
        self.progress_cb(stage, progress, message)

    def run(self, video_path: str, output_dir: str = "outputs") -> dict:
        """Run the full reconstruction pipeline.

        Args:
            video_path: Path to input video.
            output_dir: Directory for all outputs.

        Returns:
            Dictionary with output paths and statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Stage 1: Extract frames
        self._report("extract", 0.0, "Extracting video frames...")
        frames_dir = output_dir / "frames"
        self.frame_paths = extract_frames(
            video_path, str(frames_dir), self.config.video
        )

        if len(self.frame_paths) < 3:
            raise RuntimeError(
                f"Only {len(self.frame_paths)} frames extracted. "
                "Need at least 3. Try lowering min_sharpness or fps."
            )

        # Get image dimensions for intrinsics
        sample = cv2.imread(self.frame_paths[0])
        h, w = sample.shape[:2]
        self.intrinsics = CameraIntrinsics.estimate_from_image(w, h)
        self.intrinsics.save(str(output_dir / "intrinsics.json"))

        self._report("extract", 1.0, f"Extracted {len(self.frame_paths)} frames")

        # Stage 2: Depth estimation
        self._report("depth", 0.0, "Running depth estimation...")
        depth_estimator = DepthEstimator(self.config.depth)
        depth_dir = output_dir / "depth"
        self.depth_paths = depth_estimator.estimate_batch(
            self.frame_paths, str(depth_dir)
        )
        self._report("depth", 1.0, f"Estimated {len(self.depth_paths)} depth maps")

        # Stage 3: Feature matching
        self._report("features", 0.0, "Matching features...")
        matcher = FeatureMatcher(self.config.features)

        images = []
        for fp in self.frame_paths:
            img = cv2.imread(fp)
            if img is not None:
                images.append(img)

        matches = matcher.match_sequence(images)
        successful = sum(1 for m in matches if m is not None)
        self._report(
            "features", 1.0,
            f"Matched {successful}/{len(matches)} consecutive pairs"
        )

        # Stage 4: Pose estimation
        self._report("poses", 0.0, "Estimating camera poses...")
        pose_estimator = PoseEstimator(self.intrinsics, self.config.pose)
        self.poses = pose_estimator.estimate_trajectory(matches)

        trajectory_length = PoseEstimator.compute_trajectory_length(self.poses)
        self._report(
            "poses", 1.0,
            f"Trajectory: {len(self.poses)} poses, "
            f"length={trajectory_length:.2f}"
        )

        # Save trajectory
        trajectory_pts = PoseEstimator.get_trajectory_points(self.poses)
        np.save(str(output_dir / "trajectory.npy"), trajectory_pts)

        # Stage 5: Point cloud reconstruction
        self._report("pointcloud", 0.0, "Building 3D point cloud...")
        self.point_cloud = build_reconstruction(
            self.frame_paths,
            self.depth_paths,
            self.poses,
            self.intrinsics,
            self.config.point_cloud,
            use_icp=True,
        )

        ply_path = output_dir / "scene.ply"
        save_pointcloud(self.point_cloud, str(ply_path))
        self._report(
            "pointcloud", 1.0,
            f"Point cloud: {len(self.point_cloud.points)} points"
        )

        # Stage 6: Mesh reconstruction
        self._report("mesh", 0.0, "Reconstructing surface mesh...")
        mesh_builder = MeshReconstructor(self.config.mesh)
        self.mesh = mesh_builder.reconstruct(self.point_cloud)

        mesh_stats = MeshReconstructor.get_mesh_stats(self.mesh)

        # Save in multiple formats
        mesh_ply_path = output_dir / "scene_mesh.ply"
        mesh_glb_path = output_dir / "scene.glb"
        save_mesh_ply(self.mesh, str(mesh_ply_path))
        save_mesh_glb(self.mesh, str(mesh_glb_path))
        self._report(
            "mesh", 1.0,
            f"Mesh: {mesh_stats['num_vertices']} vertices, "
            f"{mesh_stats['num_triangles']} triangles"
        )

        # Collect stats
        elapsed = time.time() - start_time
        self.stats = {
            "elapsed_seconds": round(elapsed, 1),
            "num_frames": len(self.frame_paths),
            "num_depth_maps": len(self.depth_paths),
            "num_poses": len(self.poses),
            "trajectory_length": round(trajectory_length, 3),
            "pointcloud_size": len(self.point_cloud.points),
            "mesh": mesh_stats,
            "outputs": {
                "pointcloud": str(ply_path),
                "mesh_ply": str(mesh_ply_path),
                "mesh_glb": str(mesh_glb_path),
                "trajectory": str(output_dir / "trajectory.npy"),
                "intrinsics": str(output_dir / "intrinsics.json"),
            },
        }

        # Save stats
        with open(output_dir / "stats.json", "w") as f:
            json.dump(self.stats, f, indent=2, default=str)

        self._report("done", 1.0, f"Reconstruction complete in {elapsed:.1f}s")
        return self.stats
