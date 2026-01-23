"""Module to give final output with kp and matches from ONNX model."""

import cv2
import onnxruntime as ort
import numpy as np
from typing import Tuple, List
from matplotlib import pyplot as plt
from src.matching_v2 import LightGlueMatcher
from src.feature_extraction import SuperPointDescriptor, DISKDescriptor


class SuperPointLightGlue:
    """
    SuperPoint + LightGlue feature extractor and matcher.
    """

    def __init__(self, model_path: str = "weight/superpoint_lightglue_pipeline.ort.onnx"):
        self.model_path = model_path
        
        self.session = ort.InferenceSession(model_path)
        # Load your SuperPoint + LightGlue model here
    
    def compute_and_match(
            self,
            img1: np.ndarray, 
            img2: np.ndarray
        ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray]:
        """
        Detect, compute, and match features between two images.
        This method should be overridden to implement the full pipeline.
        Args:
            img1: First input image.
            img2: Second input image.
        Returns:
            kp1: Keypoints from first image.
            kp2: Keypoints from second image.
            matches: List of matches between keypoints.
            scores: Matching scores.
        """
        # Implement SuperPoint + LightGlue feature extraction and matching logic here
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        gray_img1 = cv2.resize(gray_img1, (512, 512))
        input_1 = gray_img1[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0

        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        gray_img2 = cv2.resize(gray_img2, (512, 512))
        input_2 = gray_img2[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0

        input = np.concatenate([input_1, input_2], axis=0)
        kp, matches, scores = self.session.run(None, {"images": input})
        matches = matches[:, 1:]

        kp1 = [
            cv2.KeyPoint(float(p[0]*img1.shape[1]/512), float(p[1]*img1.shape[0]/512), 1)
            for p in kp[0]
        ]
            
        kp2 = [
            cv2.KeyPoint(float(p[0]*img2.shape[1]/512), float(p[1]*img2.shape[0]/512), 1)
            for p in kp[1]
        ]
        
        matches_cv2 = [
            cv2.DMatch(
                _queryIdx=int(i1),
                _trainIdx=int(i2),
                _distance=float(1.0 - s)
            )
            for (i1, i2), s in zip(matches, scores)
            if i2 >= 0 and i1 < len(kp1) and i2 < len(kp2)
        ]

        for index, s in zip(matches, scores):
            kp1[index[0]].size = float(s) * 200
            kp2[index[1]].size = float(s) * 200

        return kp1, kp2, matches_cv2, scores
    

class DiskLightGlue:
    """
    SuperPoint + LightGlue feature extractor and matcher.
    """
    def __init__(self, model_path: str = "weight/disk_lightglue_end2end_fused_cpu.onnx"):
        self.model_path = model_path
        
        self.session = ort.InferenceSession(model_path)
        
    def compute_and_match(
            self,
            img1: np.ndarray, 
            img2: np.ndarray
        ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], np.ndarray]:
        """
        Detect, compute, and match features between two images.
        This method should be overridden to implement the full pipeline.
        Args:
            img1: First input image.
            img2: Second input image.
        Returns:
            kp1: Keypoints from first image.
            kp2: Keypoints from second image.
            matches: List of matches between keypoints.
            scores: Matching scores.
        """
        # Implement DISK + LightGlue feature extraction and matching logic here
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB) if img1.ndim == 2 else img1
        img1_rgb = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.resize(img1_rgb, (512, 512))
        input_1 = img1_rgb.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB) if img2.ndim == 2 else img2
        img2_rgb = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.resize(img2_rgb, (512, 512))
        input_2 = img2_rgb.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
        
        out = self.session.run(None, {"image0": input_1, "image1": input_2})

        kp1 = [cv2.KeyPoint(float(p[0]*img1.shape[1]/512), float(p[1]*img1.shape[0]/512), 1) for p in out[0][0]]
        kp2 = [cv2.KeyPoint(float(p[0]*img2.shape[1]/512), float(p[1]*img2.shape[0]/512), 1) for p in out[1][0]]

        matches = out[2]
        scores = out[3]
        matches_cv2 = [
            cv2.DMatch(
                _queryIdx=int(i1),
                _trainIdx=int(i2),
                _distance=float(1.0 - s)
            )
            for (i1, i2), s in zip(matches, scores)
            if i2 >= 0 and i1 < len(kp1) and i2 < len(kp2)
        ]

        for index, s in zip(matches, scores):
            kp1[index[0]].size = float(s) * 20
            kp2[index[1]].size = float(s) * 20

        return kp1, kp2, matches_cv2, scores
    

if __name__ == "__main__":
    # Example usage
    image_size = 512
    img1 = cv2.imread("data/train_data/drone_images/1522.723948864.png")
    img1 = cv2.resize(img1, (image_size, image_size))
    img2 = cv2.imread("data/train_data/drone_images/1564.747154496.png")
    img2 = cv2.resize(img2, (image_size, image_size))
    # img2 = cv2.rotate(img2, cv2.ROTATE_180)

    model = SuperPointLightGlue()
    # model = DiskLightGlue()
    kp1, kp2, matches, scores = model.compute_and_match(img1, img2)
    matches1to2 = {i: m for i, m in enumerate(matches)}

    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")
    print(f"Matches found: {len(matches)}")

    image = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(image)
    plt.show()

    superpoint = SuperPointDescriptor()
    lightglue = LightGlueMatcher("weight/lightglue/superpoint_lightglue.onnx")

    kp1, desc1 = superpoint.compute(img1)
    kp2, desc2 = superpoint.compute(img2)
    matches = lightglue.match_descriptors(kp1, desc1, kp2, desc2)

    print(f"LightGlue Matches found: {len(matches)}")

    image = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        matchColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(image)
    plt.show()

    
    disk = DISKDescriptor()
    lightglue = LightGlueMatcher("weight/lightglue/disk_lightglue.onnx")

    kp1, desc1 = disk.compute(img1)
    kp2, desc2 = disk.compute(img2)
    matches = lightglue.match_descriptors(kp1, desc1, kp2, desc2)

    print(f"LightGlue Matches found: {len(matches)}")

    image = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        matchColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(image)
    plt.show()
    # model.compute_and_match(img1, img2)
