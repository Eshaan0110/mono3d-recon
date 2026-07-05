"""Tests for core modules."""

import numpy as np
import cv2
import tempfile
import os
import pytest
from pathlib import Path

# Adjust import path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PipelineConfig, VideoConfig, DepthConfig
from src.camera import CameraIntrinsics
from src.video_utils import compute_sharpness


class TestConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.quality == "medium"
        assert config.video.fps == 2.0
        assert config.depth.model == "midas"

    def test_quality_presets(self):
        config = PipelineConfig(quality="low")
        config.apply_quality_preset()
        assert config.video.fps == 1.0
        assert config.video.resize_width == 480

        config = PipelineConfig(quality="high")
        config.apply_quality_preset()
        assert config.video.fps == 3.0

    def test_save_load_config(self):
        config = PipelineConfig()
        config.video.fps = 5.0

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.save(f.name)
            loaded = PipelineConfig.load(f.name)
            assert loaded.video.fps == 5.0
            os.unlink(f.name)


class TestCamera:
    def test_intrinsics_from_image(self):
        K = CameraIntrinsics.estimate_from_image(640, 480)
        assert K.width == 640
        assert K.height == 480
        assert K.cx == 320.0
        assert K.cy == 240.0
        assert K.fx > 0

    def test_intrinsic_matrix_shape(self):
        cam = CameraIntrinsics.estimate_from_image(640, 480)
        K = cam.K
        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0

    def test_scale(self):
        cam = CameraIntrinsics.estimate_from_image(640, 480)
        scaled = cam.scale(0.5)
        assert scaled.width == 320
        assert scaled.height == 240
        assert scaled.fx == cam.fx * 0.5

    def test_save_load(self):
        cam = CameraIntrinsics.estimate_from_image(1920, 1080)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cam.save(f.name)
            loaded = CameraIntrinsics.from_file(f.name)
            assert loaded.fx == cam.fx
            assert loaded.width == cam.width
            os.unlink(f.name)


class TestVideoUtils:
    def test_compute_sharpness(self):
        # Sharp image (edges)
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[40:60, :] = 255
        score_sharp = compute_sharpness(sharp)

        # Blurry image (uniform)
        blurry = np.ones((100, 100, 3), dtype=np.uint8) * 128
        score_blurry = compute_sharpness(blurry)

        assert score_sharp > score_blurry

    def test_extract_frames_missing_video(self):
        from src.video_utils import extract_frames
        with pytest.raises(FileNotFoundError):
            extract_frames("nonexistent.mp4", "/tmp/frames")


class TestPointCloud:
    def test_depth_to_pointcloud(self):
        from src.point_cloud import depth_to_pointcloud

        depth = np.random.rand(100, 100).astype(np.float32)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cam = CameraIntrinsics.estimate_from_image(100, 100)

        points, colors = depth_to_pointcloud(depth, image, cam)

        assert points.ndim == 2
        assert points.shape[1] == 3
        assert colors.shape[1] == 3
        assert len(points) == len(colors)
        assert len(points) > 0

    def test_create_open3d_pointcloud(self):
        from src.point_cloud import create_open3d_pointcloud

        points = np.random.rand(500, 3)
        colors = np.random.rand(500, 3)

        pcd = create_open3d_pointcloud(points, colors)
        assert len(pcd.points) == 500
        assert len(pcd.colors) == 500


class TestPoseEstimation:
    def test_camera_pose_properties(self):
        from src.pose_estimation import CameraPose

        R = np.eye(3)
        t = np.array([[1], [2], [3]])
        pose = CameraPose(R=R, t=t, frame_idx=0)

        assert pose.T.shape == (4, 4)
        assert pose.center.shape == (3,)
        np.testing.assert_array_almost_equal(
            pose.center, [-1, -2, -3]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
