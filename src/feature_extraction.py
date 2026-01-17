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
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect and compute sparse features.

        Args:
            image: Input image (BGR or grayscale).
            mask: Optional mask.

        Returns:
            Tuple of (keypoints list, descriptors array).
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
            return [], np.empty((0, self.descriptor_dim), dtype=self.dtype)

        return self.sort_by_strength(keypoints, descriptors)

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
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect and compute SuperPoint features.
        This method should be overridden to implement SuperPoint feature extraction.
        """
        # Implement SuperPoint feature extraction logic here
        
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = gray.astype(np.float32) / 255.0
        inp = gray[np.newaxis, np.newaxis, :, :]  # [1,1,H,W]
        # TODO remove hardcoded input name
        out = self.session.run(None, {"image": inp})
        score = out[1][0]
        kp = out[0][0]
        desc = out[2][0]
        
        kps = [
            cv2.KeyPoint(float(p[0]), float(p[1]), int(s*50))
            for p, s in zip(kp, score)
        ]
        
        return kps, desc.astype(np.float32)
    

if __name__ == "__main__":
    # Example usage
    # superpoint = SuperPointDescriptor(model_path="weight/superpoint_v6_from_tf.pth")
    superpoint = SuperPointDescriptor(model_path="weight/superpoint_512.onnx")
    image = np.zeros((480, 640), dtype=np.uint8)
    out = superpoint.compute(image)
    print(out)