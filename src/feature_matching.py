"""Feature detection and matching between frame pairs."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .config import FeatureConfig


@dataclass
class MatchResult:
    """Result of feature matching between two frames."""
    pts1: np.ndarray          # (N, 2) matched points in frame 1
    pts2: np.ndarray          # (N, 2) matched points in frame 2
    num_matches: int
    inlier_ratio: float


class FeatureMatcher:
    """Detect and match features between image pairs.

    Args:
        config: Feature detection/matching configuration.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()

    def _create_detector(self):
        """Create the feature detector."""
        if self.config.detector == "sift":
            return cv2.SIFT_create(nfeatures=self.config.max_features)
        elif self.config.detector == "orb":
            return cv2.ORB_create(nfeatures=self.config.max_features)
        else:
            raise ValueError(f"Unknown detector: {self.config.detector}")

    def _create_matcher(self):
        """Create the feature matcher."""
        if self.config.detector == "sift":
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """Detect keypoints and compute descriptors.

        Args:
            image: BGR image.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_pair(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> Optional[MatchResult]:
        """Match features between two images.

        Args:
            image1: First BGR image.
            image2: Second BGR image.

        Returns:
            MatchResult or None if insufficient matches.
        """
        kp1, desc1 = self.detect(image1)
        kp2, desc2 = self.detect(image2)

        if desc1 is None or desc2 is None:
            return None
        if len(desc1) < 2 or len(desc2) < 2:
            return None

        # Ensure float32 for FLANN
        if self.config.detector == "sift":
            desc1 = desc1.astype(np.float32)
            desc2 = desc2.astype(np.float32)

        # KNN match
        try:
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return None

        # Lowe's ratio test
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.config.ratio_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.config.min_matches:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        inlier_ratio = len(good_matches) / max(len(raw_matches), 1)

        return MatchResult(
            pts1=pts1,
            pts2=pts2,
            num_matches=len(good_matches),
            inlier_ratio=inlier_ratio,
        )

    def match_sequence(
        self,
        images: List[np.ndarray],
    ) -> List[Optional[MatchResult]]:
        """Match features between consecutive image pairs.

        Args:
            images: List of BGR images.

        Returns:
            List of MatchResults for consecutive pairs.
            Length is len(images) - 1.
        """
        results = []
        for i in range(len(images) - 1):
            result = self.match_pair(images[i], images[i + 1])
            results.append(result)
            status = f"{result.num_matches} matches" if result else "FAILED"
            print(f"  Pair {i}-{i+1}: {status}")
        return results

    @staticmethod
    def draw_matches(
        img1: np.ndarray,
        img2: np.ndarray,
        match_result: MatchResult,
        max_draw: int = 50,
    ) -> np.ndarray:
        """Draw matches between two images for visualization.

        Args:
            img1: First image.
            img2: Second image.
            match_result: Matching result.
            max_draw: Maximum number of matches to draw.

        Returns:
            Combined image with drawn matches.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2

        n = min(len(match_result.pts1), max_draw)
        indices = np.random.choice(len(match_result.pts1), n, replace=False)

        for idx in indices:
            pt1 = tuple(match_result.pts1[idx].astype(int))
            pt2 = tuple((match_result.pts2[idx] + [w1, 0]).astype(int))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)
            cv2.line(canvas, pt1, pt2, color, 1)

        return canvas
