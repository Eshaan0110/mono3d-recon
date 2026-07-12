"""Camera pose estimation from feature correspondences."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .config import PoseConfig
from .camera import CameraIntrinsics
from .feature_matching import MatchResult


@dataclass
class CameraPose:
    """Camera pose in world coordinates."""
    R: np.ndarray       # 3x3 rotation matrix
    t: np.ndarray       # 3x1 translation vector
    frame_idx: int      # Which frame this corresponds to

    @property
    def T(self) -> np.ndarray:
        """4x4 transformation matrix (world-to-camera)."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.flatten()
        return T

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates."""
        return -self.R.T @ self.t.flatten()


class PoseEstimator:
    """Estimate camera poses from pairwise feature matches.

    Args:
        intrinsics: Camera intrinsic parameters.
        config: Pose estimation configuration.
    """

    def __init__(self, intrinsics: CameraIntrinsics,
                 config: Optional[PoseConfig] = None):
        self.intrinsics = intrinsics
        self.config = config or PoseConfig()
        self.K = intrinsics.K

    def estimate_relative_pose(
        self, match: MatchResult
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Estimate relative rotation and translation from matched points.

        Args:
            match: Feature matching result.

        Returns:
            Tuple of (R, t, mask) or None if estimation fails.
            R: 3x3 rotation matrix.
            t: 3x1 translation vector (unit length).
            mask: Inlier mask from RANSAC.
        """
        E, mask_e = cv2.findEssentialMat(
            match.pts1, match.pts2, self.K,
            method=cv2.RANSAC,
            prob=self.config.ransac_confidence,
            threshold=self.config.ransac_threshold,
        )

        if E is None or mask_e is None:
            return None

        inlier_count = mask_e.sum()
        if inlier_count < self.config.min_inliers:
            return None

        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(
            E, match.pts1, match.pts2, self.K, mask=mask_e
        )

        return R, t, mask_pose

    def estimate_trajectory(
        self,
        matches: List[Optional[MatchResult]],
    ) -> List[CameraPose]:
        """Chain pairwise poses into a global trajectory.

        Args:
            matches: List of MatchResults between consecutive frames.

        Returns:
            List of CameraPose objects in world coordinates.
        """
        poses = []

        # First camera is at the origin
        R_global = np.eye(3)
        t_global = np.zeros((3, 1))
        poses.append(CameraPose(R=R_global.copy(), t=t_global.copy(), frame_idx=0))

        scale = 1.0

        for i, match in enumerate(matches):
            if match is None:
                # Propagate previous pose for missing matches
                poses.append(CameraPose(
                    R=R_global.copy(), t=t_global.copy(), frame_idx=i + 1
                ))
                print(f"  Frame {i+1}: No match, using previous pose")
                continue

            result = self.estimate_relative_pose(match)
            if result is None:
                poses.append(CameraPose(
                    R=R_global.copy(), t=t_global.copy(), frame_idx=i + 1
                ))
                print(f"  Frame {i+1}: Pose estimation failed, using previous")
                continue

            R_rel, t_rel, _ = result

            # Chain: T_global = T_global * T_relative
            t_global = t_global + scale * R_global @ t_rel
            R_global = R_rel @ R_global

            poses.append(CameraPose(
                R=R_global.copy(), t=t_global.copy(), frame_idx=i + 1
            ))

        print(f"Estimated {len(poses)} camera poses")
        return poses

    @staticmethod
    def get_trajectory_points(poses: List[CameraPose]) -> np.ndarray:
        """Extract camera center positions as an (N, 3) array."""
        return np.array([p.center for p in poses])

    @staticmethod
    def compute_trajectory_length(poses: List[CameraPose]) -> float:
        """Compute total trajectory length."""
        centers = PoseEstimator.get_trajectory_points(poses)
        if len(centers) < 2:
            return 0.0
        diffs = np.diff(centers, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
