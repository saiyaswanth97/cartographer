"""2D Transform utilities for image matching."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


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
    # TODO: deprecate this function
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
    # TODO: deprecate this function
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


def homography_estimate(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_thresh: float = 0.025,
    confidence: float = 0.95,
    max_iters: int = 2000,
    refine_iters: int = 10,
    type: str = "similarity"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate 2D transform using RANSAC.

    Args:
        src_pts: Source points (Nx2) Normalized.
        dst_pts: Destination points (Nx2) Normalized.
        ransac_thresh: RANSAC reprojection threshold in pixels.
        confidence: RANSAC confidence level.
        max_iters: Maximum RANSAC iterations.
        refine_iters: Number of refinement iterations after RANSAC.
        type: Type of model to estimate ("homography", "affine", "similarity").

    Returns:
        Tuple of (transform matrix, inlier mask).
    """
    if type == "homography":
        M, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters,
            confidence=confidence,
            refineIters=refine_iters
        )
    elif type == "affine":
        M, mask = cv2.estimateAffine2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters,
            confidence=confidence,
            refineIters=refine_iters,
            refineIters=refine_iters
        )
    elif type == "similarity":
        M, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters,
            confidence=confidence,
            refineIters=refine_iters,
            refineIters=refine_iters
        )
    else:
        raise ValueError(f"Unknown model type: {type}")

    return M, mask


def renormalize_transform(
    M: np.ndarray,
    query_size: Tuple[int, int],
    map_size: Tuple[int, int]
) -> np.ndarray:
    """
    Renormalize transform matrix from normalized coords to pixel coords.

    Args:
        M: 2x3 transform matrix in normalized coords.
        query_size: (width, height) of source image in pixels.
        map_size: (width, height) of destination image in pixels.
    Returns:
        2x3 transform matrix in pixel coords.
    """
    qw, qh = query_size
    mw, mh = map_size

    # TODO: verify correctness
    S_query = np.array([
        [1.0/qw, 0, 0],
        [0, 1.0/qh, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    S_map_inv = np.array([
        [mw, 0, 0],
        [0, mh, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    M_h = np.vstack([M, [0, 0, 1]])
    M_renorm = S_map_inv @ M_h @ S_query

    return M_renorm[:2, :]

