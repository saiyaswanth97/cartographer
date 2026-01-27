"""Visualization module for matching results."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path

from .feature_extraction import SIFTDescriptor
from .transform_utils import decompose_similarity


def visualize_matching_results(
    drone_img: np.ndarray,
    map_img: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    inliers: List[cv2.DMatch],
    all_matches: List[cv2.DMatch],
    transform_matrix: Optional[np.ndarray],
    save_path: Optional[str] = None,
    show: bool = True,
    max_keypoints: int = 100,
    max_matches: int = 50
) -> plt.Figure:
    """
    Comprehensive visualization of matching results.

    Args:
        drone_img: Drone image (BGR).
        map_img: Map patch (BGR).
        kp1: Keypoints from drone image.
        kp2: Keypoints from map image.
        inliers: List of inlier matches.
        all_matches: All matches before RANSAC.
        transform_matrix: Estimated transform matrix.
        save_path: Path to save figure (optional).
        show: Whether to display the figure.
        max_keypoints: Maximum keypoints to draw.
        max_matches: Maximum matches to draw.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Drone image with keypoints
    drone_kp_img = SIFTDescriptor.draw_keypoints(drone_img, kp1, max_kp=max_keypoints)
    axes[0, 0].imshow(cv2.cvtColor(drone_kp_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Drone Image ({len(kp1)} keypoints)")
    axes[0, 0].axis("off")

    # 2. Map image with keypoints
    map_kp_img = SIFTDescriptor.draw_keypoints(map_img, kp2, max_kp=max_keypoints)
    axes[0, 1].imshow(cv2.cvtColor(map_kp_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Map Patch ({len(kp2)} keypoints)")
    axes[0, 1].axis("off")

    # 3. Inlier matches
    if len(inliers) > 0:
        match_img = cv2.drawMatches(
            drone_img, kp1,
            map_img, kp2,
            inliers[:max_matches],
            None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        axes[0, 2].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"Inlier Matches ({len(inliers)} inliers)")
    else:
        axes[0, 2].text(0.5, 0.5, "No inliers found", ha='center', va='center', fontsize=14)
    axes[0, 2].axis("off")

    # 4. All matches (before RANSAC)
    if len(all_matches) > 0:
        all_match_img = cv2.drawMatches(
            drone_img, kp1,
            map_img, kp2,
            all_matches[:max_matches],
            None,
            matchColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        axes[1, 0].imshow(cv2.cvtColor(all_match_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"All Matches Before RANSAC ({len(all_matches)})")
    else:
        axes[1, 0].text(0.5, 0.5, "No matches found", ha='center', va='center', fontsize=14)
    axes[1, 0].axis("off")

    # 5. Warped drone image overlay
    if transform_matrix is not None:
        h, w = map_img.shape[:2]

        # Warp drone image to map coordinate space
        if transform_matrix.shape[0] == 2:
            warped = cv2.warpAffine(drone_img, transform_matrix, (w, h))
        else:
            warped = cv2.warpPerspective(drone_img, transform_matrix, (w, h))

        # Create overlay
        overlay = cv2.addWeighted(map_img, 0.5, warped, 0.5, 0)
        axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Warped Drone Image Overlay")
    else:
        axes[1, 1].text(0.5, 0.5, "Transform not available", ha='center', va='center', fontsize=14)
    axes[1, 1].axis("off")

    # 6. Statistics panel
    axes[1, 2].axis("off")
    stats_text = f"""
    Matching Statistics
    -------------------
    Drone keypoints: {len(kp1)}
    Map keypoints: {len(kp2)}

    Matches (ratio test): {len(all_matches)}
    Inliers (RANSAC): {len(inliers)}
    Inlier ratio: {len(inliers)/max(len(all_matches), 1)*100:.1f}%
    """

    if transform_matrix is not None and transform_matrix.shape[0] == 2:
        params = decompose_similarity(transform_matrix)
        if params['scale'] is not None:
            stats_text += f"""
    Transform Parameters
    -------------------
    Scale: {params['scale']:.3f}
    Rotation: {params['rotation']:.1f} deg
    Translation: ({params['translation'][0]:.1f}, {params['translation'][1]:.1f})
    """

    axes[1, 2].text(
        0.1, 0.9, stats_text,
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    title: str = "Comparison",
    labels: Tuple[str, str] = ("Image 1", "Image 2"),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Simple side-by-side visualization of two images.

    Args:
        img1: First image (BGR).
        img2: Second image (BGR).
        title: Figure title.
        labels: Tuple of labels for each image.
        save_path: Path to save figure (optional).
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(labels[0])
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(labels[1])
    axes[1].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_location_on_map(
    map_img: np.ndarray,
    location: Tuple[float, float],
    window_offset: Tuple[float, float] = (0, 0),
    marker_size: int = 20,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize estimated location on map patch.

    Args:
        map_img: Map image (BGR).
        location: (x, y) location in patch coordinates.
        window_offset: Offset of patch window from full map origin.
        marker_size: Size of location marker.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))

    # Draw crosshair at location
    x, y = location
    ax.plot(x, y, 'r+', markersize=marker_size, markeredgewidth=3)
    ax.plot(x, y, 'ro', markersize=marker_size//2, fillstyle='none', markeredgewidth=2)

    ax.set_title(f"Location: ({x:.1f}, {y:.1f}) in patch\n"
                 f"Global: ({x + window_offset[0]:.1f}, {y + window_offset[1]:.1f})")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig
