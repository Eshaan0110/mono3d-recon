"""Point cloud generation and processing from depth maps."""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from .config import PointCloudConfig
from .camera import CameraIntrinsics
from .pose_estimation import CameraPose


def depth_to_pointcloud(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth: float = 10.0,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map into 3D points with colors.

    Args:
        depth: Depth map (H, W), normalized 0-1.
        image: BGR image (H, W, 3).
        intrinsics: Camera intrinsic parameters.
        max_depth: Maximum depth value to include.
        min_depth: Minimum depth value to include.

    Returns:
        Tuple of (points, colors):
            points: (N, 3) array of 3D coordinates.
            colors: (N, 3) array of RGB colors normalized to [0, 1].
    """
    H, W = depth.shape[:2]

    # Scale depth to a reasonable metric range
    depth_metric = depth * max_depth

    # Create pixel coordinate grid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Back-project to 3D
    Z = depth_metric
    X = (u - intrinsics.cx) * Z / intrinsics.fx
    Y = (v - intrinsics.cy) * Z / intrinsics.fy

    # Stack into (H*W, 3)
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Get colors (BGR -> RGB, normalize to 0-1)
    colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    # Filter by depth range
    valid = (Z.flatten() > min_depth) & (Z.flatten() < max_depth)
    points = points[valid]
    colors = colors[valid]

    return points, colors


def create_open3d_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Create an Open3D PointCloud from numpy arrays.

    Args:
        points: (N, 3) array of 3D coordinates.
        colors: (N, 3) array of RGB colors in [0, 1].

    Returns:
        Open3D PointCloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def transform_pointcloud(
    pcd: o3d.geometry.PointCloud,
    pose: CameraPose,
) -> o3d.geometry.PointCloud:
    """Transform a point cloud from camera frame to world frame.

    Args:
        pcd: Point cloud in camera coordinates.
        pose: Camera pose (world-to-camera transform).

    Returns:
        Transformed point cloud in world coordinates.
    """
    # Invert the pose to get camera-to-world
    R_inv = pose.R.T
    t_inv = -R_inv @ pose.t.flatten()

    T = np.eye(4)
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv

    pcd_transformed = o3d.geometry.PointCloud(pcd)
    pcd_transformed.transform(T)
    return pcd_transformed


def merge_pointclouds(
    pcds: List[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds into one.

    Args:
        pcds: List of Open3D PointCloud objects.

    Returns:
        Combined PointCloud.
    """
    merged = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged += pcd
    return merged


def clean_pointcloud(
    pcd: o3d.geometry.PointCloud,
    config: Optional[PointCloudConfig] = None,
) -> o3d.geometry.PointCloud:
    """Clean a point cloud by removing outliers and downsampling.

    Args:
        pcd: Input point cloud.
        config: Point cloud processing configuration.

    Returns:
        Cleaned point cloud.
    """
    if config is None:
        config = PointCloudConfig()

    original_count = len(pcd.points)

    # Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=config.voxel_size)
    after_voxel = len(pcd.points)

    # Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.nb_neighbors,
        std_ratio=config.std_ratio,
    )
    after_outlier = len(pcd.points)

    print(f"  Point cloud cleaned: {original_count} -> "
          f"{after_voxel} (voxel) -> {after_outlier} (outlier removal)")

    return pcd


def refine_alignment_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.02,
    max_iterations: int = 50,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Refine alignment between two point clouds using ICP.

    Args:
        source: Source point cloud to align.
        target: Target (reference) point cloud.
        voxel_size: Voxel size for downsampling during ICP.
        max_iterations: Maximum ICP iterations.

    Returns:
        Tuple of (aligned source, 4x4 transformation matrix).
    """
    # Estimate normals if not present
    for pcd in [source, target]:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )

    threshold = voxel_size * 3

    # Coarse ICP
    result_coarse = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=threshold * 2,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    # Fine ICP
    result_fine = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=threshold,
        init=result_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    source_aligned = o3d.geometry.PointCloud(source)
    source_aligned.transform(result_fine.transformation)

    return source_aligned, result_fine.transformation


def build_reconstruction(
    frame_paths: List[str],
    depth_paths: List[str],
    poses: List[CameraPose],
    intrinsics: CameraIntrinsics,
    config: Optional[PointCloudConfig] = None,
    use_icp: bool = True,
) -> o3d.geometry.PointCloud:
    """Build a complete 3D reconstruction from frames, depths, and poses.

    Args:
        frame_paths: Paths to frame images.
        depth_paths: Paths to depth maps (.npy).
        poses: Camera poses for each frame.
        intrinsics: Camera intrinsic parameters.
        config: Point cloud processing configuration.
        use_icp: Whether to refine alignment with ICP.

    Returns:
        Merged and cleaned point cloud.
    """
    if config is None:
        config = PointCloudConfig()

    n = min(len(frame_paths), len(depth_paths), len(poses))
    print(f"Building reconstruction from {n} frames...")

    per_frame_pcds = []

    for i in tqdm(range(n), desc="Building point clouds"):
        image = cv2.imread(frame_paths[i])
        depth = np.load(depth_paths[i])

        if image is None:
            continue

        # Resize depth to match image if needed
        if depth.shape[:2] != image.shape[:2]:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        # Back-project to 3D
        points, colors = depth_to_pointcloud(
            depth, image, intrinsics,
            max_depth=config.max_depth,
            min_depth=config.min_depth,
        )

        pcd = create_open3d_pointcloud(points, colors)

        # Downsample per-frame cloud for efficiency
        pcd = pcd.voxel_down_sample(voxel_size=config.voxel_size)

        # Transform to world frame
        pcd = transform_pointcloud(pcd, poses[i])

        per_frame_pcds.append(pcd)

    # Optional ICP refinement between consecutive frames
    if use_icp and len(per_frame_pcds) > 1:
        print("Refining alignment with ICP...")
        aligned_pcds = [per_frame_pcds[0]]
        for i in tqdm(range(1, len(per_frame_pcds)), desc="ICP alignment"):
            aligned, _ = refine_alignment_icp(
                per_frame_pcds[i],
                aligned_pcds[-1],
                voxel_size=config.voxel_size,
            )
            aligned_pcds.append(aligned)
        per_frame_pcds = aligned_pcds

    # Merge all
    merged = merge_pointclouds(per_frame_pcds)
    print(f"Merged cloud: {len(merged.points)} points")

    # Clean
    cleaned = clean_pointcloud(merged, config)

    return cleaned


def save_pointcloud(pcd: o3d.geometry.PointCloud, path: str):
    """Save point cloud to file (PLY format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"Saved point cloud: {path} ({len(pcd.points)} points)")


def load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from file."""
    return o3d.io.read_point_cloud(path)
