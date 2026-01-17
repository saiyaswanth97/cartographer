"""Feature matching module using FLANN and RANSAC."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Container for matching results."""
    transform: Optional[np.ndarray]  # 2x3 affine or 3x3 homography
    inliers: List[cv2.DMatch]
    all_matches: List[cv2.DMatch]
    inlier_ratio: float
    success: bool

    @property
    def num_inliers(self) -> int:
        return len(self.inliers)

    @property
    def num_matches(self) -> int:
        return len(self.all_matches)


class SIFTMatcher2D:
    """
    2D feature matcher for SIFT descriptors.

    Uses FLANN for efficient matching and RANSAC for robust transform estimation.
    """

    def __init__(
        self,
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 2.5,
        confidence: float = 0.99,
        min_inliers: int = 10,
        model: str = "similarity"
    ):
        """
        Initialize SIFT matcher.

        Args:
            ratio_thresh: Lowe's ratio test threshold.
            ransac_thresh: RANSAC reprojection threshold in pixels.
            confidence: RANSAC confidence level.
            min_inliers: Minimum inliers for valid match.
            model: Transform model - "similarity", "affine", or "homography".
        """
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.confidence = confidence
        self.min_inliers = min_inliers
        self.model = model

        # FLANN parameters for SIFT/SURF/Superpoint (L2 distance)
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),  # KDTree
            dict(checks=50)
        )

    def match_descriptors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match descriptors using FLANN with Lowe's ratio test.

        Args:
            desc1: Descriptors from first image.
            desc2: Descriptors from second image.

        Returns:
            List of good matches passing ratio test.
        """
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        knn = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for match in knn:
            if len(match) < 2:
                continue
            m, n = match
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)
        
        # matcher = cv2.BFMatcher(
        #     cv2.NORM_HAMMING,
        #     crossCheck=False
        # )
        # matches = matcher.knnMatch(desc1, desc2, k=2)
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.8 * n.distance:
        #         good.append(m)

        return good

    def estimate_transform(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Estimate 2D transform using RANSAC.

        Args:
            kp1: Keypoints from first image.
            kp2: Keypoints from second image.
            matches: List of matches.

        Returns:
            Tuple of (transform matrix, inlier mask, source points).
        """
        if len(matches) < self.min_inliers:
            return None, None, np.array([])

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        w1 = np.max(pts1[:, 0]) - np.min(pts1[:, 0])
        h1 = np.max(pts1[:, 1]) - np.min(pts1[:, 1])
        w2 = np.max(pts2[:, 0]) - np.min(pts2[:, 0])
        h2 = np.max(pts2[:, 1]) - np.min(pts2[:, 1])
        
        pts1_norm = pts1 / np.array([[w1, h1]])
        pts2_norm = pts2 / np.array([[w2, h2]])

        if self.model == "homography":
            M, mask = cv2.findHomography(
                pts1, pts2,
                cv2.RANSAC,
                self.ransac_thresh,
                confidence=self.confidence
            )
        elif self.model == "affine":
            M, mask = cv2.estimateAffine2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=self.confidence
            )
        elif self.model == "similarity":
            M, mask = cv2.estimateAffinePartial2D(
                pts1_norm, pts2_norm,
                # pts1, pts2,
                method=cv2.RANSAC,
                # ransacReprojThreshold=self.ransac_thresh,
                ransacReprojThreshold=0.025,
                confidence=self.confidence
            )
            T1 = np.array([[w1, 0, 0],
                           [0, h1, 0],
                           [0, 0, 1]])
            T2 = np.array([[w2, 0, 0],
                           [0, h2, 0],
                           [0, 0, 1]])
            
            M_norm = np.vstack([M, [0, 0, 1]])
            M_pixel = T2 @ M_norm @ np.linalg.inv(T1)
            M_pixel = M_pixel[:2, :]
            M = M_pixel
            
        else:
            raise ValueError(f"Unknown model type: {self.model}")

        return M, mask, pts1

    def match_and_estimate(
        self,
        kp1: List[cv2.KeyPoint],
        desc1: np.ndarray,
        kp2: List[cv2.KeyPoint],
        desc2: np.ndarray
    ) -> MatchResult:
        """
        Full matching pipeline with RANSAC validation.

        Args:
            kp1: Keypoints from first image.
            desc1: Descriptors from first image.
            kp2: Keypoints from second image.
            desc2: Descriptors from second image.

        Returns:
            MatchResult object containing transform and match information.
        """
        matches = self.match_descriptors(desc1, desc2)

        if len(matches) < self.min_inliers:
            return MatchResult(
                transform=None,
                inliers=[],
                all_matches=matches,
                inlier_ratio=0.0,
                success=False
            )

        M, inlier_mask, _ = self.estimate_transform(kp1, kp2, matches)

        if M is None:
            return MatchResult(
                transform=None,
                inliers=[],
                all_matches=matches,
                inlier_ratio=0.0,
                success=False
            )

        # Flatten inlier_mask if needed
        if inlier_mask is not None:
            inlier_mask = inlier_mask.ravel()

        inliers = [
            matches[i] for i in range(len(matches))
            if inlier_mask is not None and inlier_mask[i]
        ]

        # Post-RANSAC validation
        if len(inliers) < self.min_inliers:
            return MatchResult(
                transform=None,
                inliers=inliers,
                all_matches=matches,
                inlier_ratio=len(inliers) / max(len(matches), 1),
                success=False
            )

        return MatchResult(
            transform=M,
            inliers=inliers,
            all_matches=matches,
            inlier_ratio=len(inliers) / max(len(matches), 1),
            success=True
        )

    @staticmethod
    def draw_matches(
        img1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_draw: int = 50,
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Draw matches between two images.

        Args:
            img1: First image.
            kp1: Keypoints from first image.
            img2: Second image.
            kp2: Keypoints from second image.
            matches: List of matches to draw.
            max_draw: Maximum matches to draw.
            color: Match line color (BGR). None for random colors.

        Returns:
            Image with matches drawn.
        """
        return cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            matches[:max_draw],
            None,
            matchColor=color,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    @staticmethod
    def decompose_similarity(M: np.ndarray) -> dict:
        """
        Decompose a similarity/affine transform into components.

        Args:
            M: 2x3 transform matrix.

        Returns:
            Dictionary with 'scale', 'rotation' (degrees), 'translation'.
        """
        if M is None or M.shape[0] < 2:
            return {'scale': None, 'rotation': None, 'translation': None}

        a, b = M[0, 0], M[0, 1]
        scale = np.sqrt(a**2 + b**2)
        rotation = np.degrees(np.arctan2(b, a))
        translation = (M[0, 2], M[1, 2])

        return {
            'scale': scale,
            'rotation': rotation,
            'translation': translation
        }
