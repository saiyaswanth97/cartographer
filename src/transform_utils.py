"""2D Transform utilities for image matching."""

import cv2
import numpy as np
from typing import List, Tuple


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


def compose_similarity(
    scale: float,
    rotation: float,
    image_width: float,
    image_height: float
) -> np.ndarray:
    """
    Construct a 2D similarity transform matrix that rotates around image center.

    Args:
        scale: Scaling factor.
        rotation: Rotation angle in degrees.
        image_width: Width of the image.
        image_height: Height of the image.

    Returns:
        2x3 similarity transform matrix.
    """
    theta = np.radians(rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    cx = image_width / 2.0
    cy = image_height / 2.0

    a = scale * cos_theta
    b = scale * sin_theta

    tx = cx - (a * cx - b * cy)
    ty = cy - (b * cx + a * cy)

    return np.array([
        [a, -b, tx],
        [b, a, ty]
    ], dtype=np.float32)


# def transform_keypoints(
def transform_points(
    keypoints: List[cv2.KeyPoint],
    M: np.ndarray
) -> List[cv2.KeyPoint]:
    """
    Apply 2D transform to cv2.KeyPoint list.

    Args:
        keypoints: List of cv2.KeyPoint objects.
        M: 2x3 transform matrix.

    Returns:
        Transformed keypoints as list of cv2.KeyPoint.
    """
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    pts_t = (M @ pts_h.T).T

    return [
        cv2.KeyPoint(x=pt[0], y=pt[1], size=kp.size)
        for pt, kp in zip(pts_t, keypoints)
    ]


def invert_transform(M: np.ndarray) -> np.ndarray:
    """
    Invert a 2x3 affine transform.

    Args:
        M: 2x3 transform matrix.

    Returns:
        2x3 inverted transform matrix.
    """
    M_h = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(M_h)[:2, :]


def compose_transforms(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
    Compose two 2x3 transforms: result = M2 @ M1.

    Args:
        M1: First 2x3 transform (applied first).
        M2: Second 2x3 transform (applied second).

    Returns:
        Combined 2x3 transform matrix.
    """
    M1_h = np.vstack([M1, [0, 0, 1]])
    M2_h = np.vstack([M2, [0, 0, 1]])
    return (M2_h @ M1_h)[:2, :]


def to_homogeneous(M: np.ndarray) -> np.ndarray:
    """Convert 2x3 affine to 3x3 homogeneous matrix."""
    return np.vstack([M, [0, 0, 1]])


def from_homogeneous(M: np.ndarray) -> np.ndarray:
    """Convert 3x3 homogeneous to 2x3 affine matrix."""
    return M[:2, :]
