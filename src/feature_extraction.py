"""Feature extraction module using SIFT."""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class SparseFeatureDescriptor:
    """
    Base class for sparse feature descriptors.

    Provides a common interface and default OpenCV-based implementation.
    """

    def __init__(
        self,
        detector=None,
        descriptor_dim: int = 0,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize descriptor.

        Args:
            detector: OpenCV detector instance (None for non-OpenCV backends).
            descriptor_dim: Descriptor dimensionality.
            dtype: Descriptor data type.
        """
        self.detector = detector
        self.descriptor_dim = descriptor_dim
        self.dtype = dtype

    def compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[cv2.KeyPoint], np.ndarray]:
        """
        Detect and compute sparse features.

        Args:
            image: Input image (BGR or grayscale).
            mask: Optional mask.

        Returns:
            Tuple of (kp_norm, keypoints_cv2, descriptors).
            - kp_norm: Normalized keypoint coordinates in [0,1] range (Nx2 array).
            - keypoints_cv2: OpenCV KeyPoint objects with pixel coordinates.
            - descriptors: Feature descriptors array.
        """
        if self.detector is None:
            raise NotImplementedError(
                "compute() must be overridden for non-OpenCV backends"
            )

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else image
        )

        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)

        if descriptors is None:
            empty_kp_norm = np.empty((0, 2), dtype=np.float32)
            empty_desc = np.empty((0, self.descriptor_dim), dtype=self.dtype)
            return empty_kp_norm, [], empty_desc

        keypoints, descriptors = self.sort_by_strength(keypoints, descriptors)

        # Extract pixel coordinates and normalize by image dimensions
        h, w = gray.shape
        kp_pixels = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        kp_norm = kp_pixels / np.array([w, h], dtype=np.float32)

        return kp_norm, keypoints, descriptors

    @staticmethod
    def sort_by_strength(
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Sort keypoints by response strength (descending).
        """
        idx = sorted(
            range(len(keypoints)),
            key=lambda i: keypoints[i].response,
            reverse=True
        )
        return [keypoints[i] for i in idx], descriptors[idx]

    @staticmethod
    def draw_keypoints(
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        max_kp: int = 200
    ) -> np.ndarray:
        """
        Draw keypoints on image.
        """
        return cv2.drawKeypoints(
            image,
            keypoints[:max_kp],
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    

class SIFTDescriptor(SparseFeatureDescriptor):
    """
    SIFT feature extractor.
    """

    def __init__(
        self,
        nfeatures: int = 0,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: int = 15,
        sigma: float = 1.6
    ):
        detector = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        super().__init__(detector, descriptor_dim=128, dtype=np.float32)


class SURFDescriptor(SparseFeatureDescriptor):
    """
    SURF feature extractor.
    """

    def __init__(
        self,
        hessian_threshold: float = 400.0,
        n_octaves: int = 4,
        n_octave_layers: int = 3,
        extended: bool = True,
        upright: bool = False
    ):
        detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            extended=extended,
            upright=upright
        )
        descriptor_dim = 128 if extended else 64
        super().__init__(detector, descriptor_dim=descriptor_dim, dtype=np.float32)
        
        
class ORBDescriptor(SparseFeatureDescriptor):
    """
    ORB feature extractor.
    """

    def __init__(
        self,
        nfeatures: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        first_level: int = 0,
        WTA_K: int = 2,
        score_type: int = cv2.ORB_HARRIS_SCORE,
        patch_size: int = 31,
        fast_threshold: int = 20
    ):
        detector = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=WTA_K,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold
        )
        super().__init__(detector, descriptor_dim=32, dtype=np.uint8)


class SuperPointDescriptor(SparseFeatureDescriptor):
    """
    SuperPoint feature extractor.
    Note: This is a placeholder for a non-OpenCV backend implementation.
    """

    def __init__(self, model_path: str = "weight/superpoint_2048.onnx"):
        super().__init__(detector=None, descriptor_dim=256, dtype=np.float32)
        self.model_path = model_path
        
        self.session = ort.InferenceSession(model_path)
        # Load your SuperPoint model here

    def compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[cv2.KeyPoint], np.ndarray]:
        """
        Detect and compute SuperPoint features.
        This method should be overridden to implement SuperPoint feature extraction.
        """
        # Implement SuperPoint feature extraction logic here

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        gray = gray.astype(np.float32) / 255.0
        inp = gray[np.newaxis, np.newaxis, :, :]  # [1,1,H,W]
        # TODO remove hardcoded input name
        out = self.session.run(None, {"image": inp})
        score = out[1][0]
        kp_pixels = out[0][0]  # Pixel coordinates from model
        desc = out[2][0]

        # Create cv2.KeyPoint objects with pixel coordinates
        kps_cv2 = [
            cv2.KeyPoint(float(p[0]), float(p[1]), int(s*50))
            for p, s in zip(kp_pixels, score)
        ]

        # Normalize coordinates to [0,1] range
        kp_norm = kp_pixels / np.array([w, h], dtype=np.float32)

        return kp_norm.astype(np.float32), kps_cv2, desc.astype(np.float32)
    

class DISKDescriptor(SparseFeatureDescriptor):
    """
    DISK feature extractor.
    Note: This is a placeholder for a non-OpenCV backend implementation.
    """

    def __init__(self, model_path: str = "weight/disk.onnx"):
        super().__init__(detector=None, descriptor_dim=128, dtype=np.float32)
        self.model_path = model_path
        
        self.session = ort.InferenceSession(model_path)
        # Load your DISK model here

    def compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[cv2.KeyPoint], np.ndarray]:
        """
        Detect and compute DISK features.
        This method should be overridden to implement DISK feature extraction.
        """
        # Implement DISK feature extraction logic here
        # It takes 13HW input
        h, w = image.shape[:2]
        input = image.astype(np.float32) / 255.0
        input = input.transpose(2, 0, 1)  # HWC to CHW
        input = input[np.newaxis, :, :, :]  # 1CHW

        # TODO remove hardcoded input name
        out = self.session.run(None, {"image": input})
        score = out[1][0]
        kp_pixels = out[0][0]  # Pixel coordinates from model
        desc = out[2][0]

        # Create cv2.KeyPoint objects with pixel coordinates
        kps_cv2 = [
            cv2.KeyPoint(float(p[0]), float(p[1]), int(s))
            for p, s in zip(kp_pixels, score)
        ]

        # Normalize coordinates to [0,1] range
        kp_norm = kp_pixels / np.array([w, h], dtype=np.float32)

        return kp_norm.astype(np.float32), kps_cv2, desc.astype(np.float32)
    

def plot_keypoints(
    image: np.ndarray,
    kp_cv2: List[cv2.KeyPoint],
    title: str
) -> np.ndarray:
    """Plot keypoints on image using OpenCV drawing."""
    img_kp = cv2.drawKeypoints(
        image,
        kp_cv2,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_kp
    
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))

    img = cv2.imread("data/train_data/drone_images/1522.723948864.png")
    
    sift = SIFTDescriptor(nfeatures=500)
    kp_norm, kp_cv2, desc = sift.compute(img)
    print(f"SIFT: {len(kp_cv2)} keypoints, descriptor shape:{desc.shape}")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(plot_keypoints(img, kp_cv2, "SIFT Keypoints"), cv2.COLOR_BGR2RGB))
    ax1.axis("off")

    # TODO: This is not supported on my machine
    # surf = SURFDescriptor(hessian_threshold=400.0)
    # kp_norm, kp_cv2, desc = surf.compute(img)
    # print(f"SURF: {len(kp_cv2)} keypoints, descriptor shape:{desc.shape}")
    # ax2 = fig.add_subplot(3, 2, 2)
    # ax2.imshow(cv2.cvtColor(plot_keypoints(img, kp_cv2, "SURF Keypoints"), cv2.COLOR_BGR2RGB))
    # ax2.axis("off")

    orb = ORBDescriptor(nfeatures=500)
    kp_norm, kp_cv2, desc = orb.compute(img)
    print(f"ORB: {len(kp_cv2)} keypoints, descriptor shape:{desc.shape}")
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.imshow(cv2.cvtColor(plot_keypoints(img, kp_cv2, "ORB Keypoints"), cv2.COLOR_BGR2RGB))
    ax3.axis("off")   

    superpoint = SuperPointDescriptor(model_path="weight/superpoint_512.onnx")
    kp_norm, kp_cv2, desc = superpoint.compute(img)
    print(f"SuperPoint: {len(kp_cv2)} keypoints, descriptor shape:{desc.shape}")
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.imshow(cv2.cvtColor(plot_keypoints(img, kp_cv2, "SuperPoint Keypoints"), cv2.COLOR_BGR2RGB))
    ax4.axis("off")

    disk = DISKDescriptor(model_path="weight/disk.onnx")
    kp_norm, kp_cv2, desc = disk.compute(img)
    print(f"DISK: {len(kp_cv2)} keypoints, descriptor shape:{desc.shape}")
    ax5 = fig.add_subplot(2, 2, 4)
    ax5.imshow(cv2.cvtColor(plot_keypoints(img, kp_cv2, "DISK Keypoints"), cv2.COLOR_BGR2RGB))
    ax5.axis("off")

    plt.tight_layout()
    plt.show()
