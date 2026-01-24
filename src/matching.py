"""Feature matching module using FLANN and RANSAC."""

import cv2
import numpy as np
from typing import List
import onnxruntime as ort
from abc import ABC, abstractmethod


class BaseMatcher(ABC):
    """
    Base class for feature matching.
    """
    @abstractmethod
    def match_descriptors(
        self,
        kp1:np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match features between two sets of keypoints and descriptors.

        Args:
            kp1: Keypoints from the first image.
            desc1: Descriptors from the first image.
            kp2: Keypoints from the second image.
            desc2: Descriptors from the second image.

        Returns:
            matches: List of matches between keypoints along with their distances.
        """
        pass

    def plot_matchings(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_matches: int = 50
    ) -> np.ndarray:
        """
        Visualize matches between two images.
        Args:
            img1: First input image.
            img2: Second input image.
            kp1: Keypoints from the first image.
            kp2: Keypoints from the second image.
            matches: List of matches between keypoints.
            max_matches: Maximum number of matches to draw.
        Returns:
            Image with matches drawn.
        """
        matched_img = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            matches[:max_matches],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return matched_img


class LightGlueMatcher(BaseMatcher):
    """
    Deep learning based matcher 2D feature matcher.
    """
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
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
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
        input_kp1 = kp1.astype(np.float32)[np.newaxis, :, :]  # [1, N1, 2]
        input_desc1 = desc1[np.newaxis, :, :]    # [1, N1, D]
        input_kp2 = kp2.astype(np.float32)[np.newaxis, :, :]  # [1, N2, 2]
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

        matches_scores.sort(key=lambda x: x[2], reverse=True)

        matches_cv2 = [
            cv2.DMatch(
                _queryIdx=int(i1),
                _trainIdx=int(i2),
                _distance=float(1.0 - s)
            )
            for (i1, i2, s) in matches_scores
        ]

        return matches_cv2
    

class OpenCVMatcher(BaseMatcher):
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
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
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
        good = sorted(good, key=lambda x: x.distance)

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
        mutual.sort(key=lambda x: x.distance)

        return mutual
    

if __name__ == "__main__":
    # Example usage
    image_size = 512
    img1 = cv2.imread("data/train_data/drone_images/1522.723948864.png")
    img1 = cv2.resize(img1, (image_size, image_size))
    img2 = cv2.imread("data/train_data/drone_images/1564.747154496.png")
    img2 = cv2.resize(img2, (image_size, image_size))

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 5))

    print("SuperPoint Descriptor Extraction")
    from src.feature_extraction import SuperPointDescriptor
    extractor = SuperPointDescriptor()
    
    kp1_norm, kp1, desc1 = extractor.compute(img1)
    kp2_norm, kp2, desc2 = extractor.compute(img2)

    matcher = OpenCVMatcher(matcher="MNN", ratio_thresh=0.85)
    matches = matcher.match_descriptors(kp1_norm, desc1, kp2_norm, desc2)
    print(f"Found {len(matches)} matches using OpenCV MNN matcher")
    img = matcher.plot_matchings(img1, img2, kp1, kp2, matches, max_matches=50)
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
    ax1.set_title("OpenCV MNN Matches with SuperPoint Features")

    matcher = OpenCVMatcher(matcher="flann", ratio_thresh=0.85)
    matches = matcher.match_descriptors(kp1_norm, desc1, kp2_norm, desc2)
    print(f"Found {len(matches)} matches using OpenCV FLANN matcher")
    img = matcher.plot_matchings(img1, img2, kp1, kp2, matches, max_matches=50)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    ax2.set_title("OpenCV FLANN Matches with SuperPoint Features")

    matcher = LightGlueMatcher("weight/lightglue/superpoint_lightglue.onnx")
    matches = matcher.match_descriptors(kp1_norm, desc1, kp2_norm, desc2)
    print(f"Found {len(matches)} matches using LightGlue matcher")
    img = matcher.plot_matchings(img1, img2, kp1, kp2, matches, max_matches=50)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax3.axis("off")
    ax3.set_title("LightGlue Matches with SuperPoint Features")

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 5))

    print("SuperPoint Descriptor Extraction")
    from src.feature_extraction import SIFTDescriptor
    extractor = SIFTDescriptor()
    
    kp1_norm, kp1, desc1 = extractor.compute(img1)
    kp2_norm, kp2, desc2 = extractor.compute(img2)

    matcher = OpenCVMatcher(matcher="MNN", ratio_thresh=0.85)
    matches = matcher.match_descriptors(kp1_norm, desc1, kp2_norm, desc2)
    print(f"Found {len(matches)} matches using OpenCV MNN matcher")
    img = matcher.plot_matchings(img1, img2, kp1, kp2, matches, max_matches=50)
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
    ax1.set_title("OpenCV MNN Matches with SIFT Features")

    matcher = OpenCVMatcher(matcher="flann", ratio_thresh=0.85)
    matches = matcher.match_descriptors(kp1_norm, desc1, kp2_norm, desc2)
    print(f"Found {len(matches)} matches using OpenCV FLANN matcher")
    img = matcher.plot_matchings(img1, img2, kp1, kp2, matches, max_matches=50)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    ax2.set_title("OpenCV FLANN Matches with SIFT Features")

    plt.tight_layout()
    plt.show()

