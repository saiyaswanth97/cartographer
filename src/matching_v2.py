"""Feature matching module using FLANN and RANSAC."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import onnxruntime as ort

from src import transform_utils


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
    

class LightGlueMatcher:
    """
    Deep learning based matcher 2D feature matcher.
    """
    # def __init__(self, model_path: str = "weight/lightglue/disk_lightglue.onnx"):
    def __init__(self, model_path: str = "weight/lightglue/superpoint_lightglue.onnx"):
        """
        Initialize LightGlue matcher.

        Args:
            model_path: Path to the LightGlue ONNX model.
        """
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)

    def match_descriptors(
        self,
        kp1: List[cv2.KeyPoint],
        desc1: np.ndarray,
        kp2: List[cv2.KeyPoint],
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match descriptors using LightGlue model.

        Args:
            kp1: Keypoints from first image.
            desc1: Descriptors from first image.
            kp2: Keypoints from second image.
            desc2: Descriptors from second image.

        Returns:
            List of matches.
        """
        input_kp1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1], dtype=np.float32)/512.0
        input_kp1 = input_kp1[np.newaxis, :, :]  # [1, N1, 2]
        input_desc1 = desc1[np.newaxis, :, :]    # [1, N1, D]
        input_kp2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2], dtype=np.float32)/512.0
        input_kp2 = input_kp2[np.newaxis, :, :]  # [1, N2, 2]
        input_desc2 = desc2[np.newaxis, :, :]    # [1, N2, D]

        out = self.session.run(None, {
            "kpts0": input_kp1,
            "desc0": input_desc1,
            "kpts1": input_kp2,
            "desc1": input_desc2
        })

        matches0 = out[0][0]
        matches1 = out[1][0]
        scores0 = out[2][0]
        scores1 = out[3][0]
        
        matches_scores = []
        for i in range(len(matches0)):
            if matches0[i] >= 0:
                j = matches0[i]
                matches_scores.append( (i, j, scores0[i]) )

        matches_cv2 = [
            cv2.DMatch(
                _queryIdx=int(i1),
                _trainIdx=int(i2),
                _distance=float(1.0 - s)
            )
            for (i1, i2, s) in matches_scores
        ]

        return matches_cv2


class OpenCVMatcher:
    """
    Class for OpenCV-based 2D feature matchers.
    """
    def __init__(self, matcher: str = "flann", ratio_thresh: float = 0.9):
        self.matcher_type = matcher
        self.ratio_thresh = ratio_thresh
        # FLANN parameters for SIFT/SURF/Superpoint (L2 distance)
        if matcher == "flann":
            self.matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5),  # KDTree
                dict(checks=50)
            )
        elif matcher == "bf" or matcher == "MNN":
            self.matcher = cv2.BFMatcher(
                cv2.NORM_L2,
                crossCheck=False
            )
        elif matcher == "bf_hamming":
            self.matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=False
            )
        else:
            raise ValueError(f"Unknown matcher type: {matcher}")
        
    def match_descriptors(
        self,
        kp1: List[cv2.KeyPoint],
        desc1: np.ndarray,
        kp2: List[cv2.KeyPoint],
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match descriptors using FLANN with Lowe's ratio test.

        Args:
            kp1: Keypoints from first image - This is not used here.
            desc1: Descriptors from first image.
            kp2: Keypoints from second image - This is not used here.
            desc2: Descriptors from second image.

        Returns:
            List of good matches passing ratio test.
        """
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        if self.matcher_type == "MNN":
            return self.mutual_nn_matches(desc1, desc2)
        
        knn = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for match in knn:
            if len(match) < 2:
                continue
            m, n = match
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        return good

    @staticmethod
    def mutual_nn_matches(desc1: np.ndarray, desc2: np.ndarray) -> list:
        """
        Mutual Nearest Neighbor matching for float descriptors.

        Args:
            desc1: (N1, D) descriptors
            desc2: (N2, D) descriptors

        Returns:
            List of cv2.DMatch
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches_12 = bf.match(desc1, desc2)
        matches_21 = bf.match(desc2, desc1)

        # Build reverse lookup
        reverse = {m.trainIdx: m.queryIdx for m in matches_21}

        mutual = []
        for m in matches_12:
            if reverse.get(m.queryIdx, -1) == m.trainIdx:
                mutual.append(m)

        return mutual
        

class HomographyEstimator:
    """
    Estimate 2D transforms using RANSAC.
    """

    def __init__(
        self,
        ratio_thresh: float = 0.9,
        ransac_thresh: float = 0.025,
        confidence: float = 0.99,
        min_inliers: int = 20, 
        model: str = "similarity"
    ):
        """
        Initialize estimator.

        Args:
            ratio_thresh: Lowe's ratio test threshold.
            ransac_thresh: RANSAC reprojection threshold in pixels.
            confidence: RANSAC confidence level.
            min_inliers: Minimum inliers for valid match.
        """
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.confidence = confidence
        self.min_inliers = min_inliers
        if model not in ["homography", "affine", "similarity"]:
            raise ValueError(f"Unknown model type: {model}")
        self.model = model

    def estimate_transform(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
            return None, None

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
                pts1_norm, pts2_norm,
                cv2.RANSAC,
                self.ransac_thresh,
                confidence=self.confidence
            )
        elif self.model == "affine":
            M, mask = cv2.estimateAffine2D(
                pts1_norm, pts2_norm,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=self.confidence
            )
        elif self.model == "similarity":
            M, mask = cv2.estimateAffinePartial2D(
                pts1_norm, pts2_norm,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=self.confidence
            )
        else:
            raise ValueError(f"Unknown model type: {self.model}")
        
        T1 = np.array([[1/w1, 0, 0],
                        [0, 1/h1, 0],
                        [0, 0, 1]])
        T2 = np.array([[w2, 0, 0],
                        [0, h2, 0],
                        [0, 0, 1]])
        
        M_norm = np.vstack([M, [0, 0, 1]])
        M_pixel = T2 @ M_norm @ T1
        M = M_pixel[:2, :]

        return M, mask
    

class EstimationPipeline:
    """
    Full matching and estimation pipeline.
    """
    def __init__(
        self,
        matcher: OpenCVMatcher,
        estimator: HomographyEstimator
    ):
        """
        Initialize pipeline with matcher and estimator.

        Args:
            matcher: OpenCVMatcher instance for descriptor matching.
            estimator: HomographyEstimator instance for transform estimation.
        """
        self.matcher = matcher
        self.estimator = estimator

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
        matches = self.matcher.match_descriptors(kp1, desc1, kp2, desc2)

        if len(matches) < self.estimator.min_inliers:
            return MatchResult(
                transform=None,
                inliers=[],
                all_matches=matches,
                inlier_ratio=0.0,
                success=False
            )

        M, inlier_mask = self.estimator.estimate_transform(kp1, kp2, matches)

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
        if len(inliers) < self.estimator.min_inliers:
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
