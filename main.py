#!/usr/bin/env python3
"""
Cartographer - Drone to Map Matching Pipeline

Main entry point for the drone image localization system.
Uses feature matching to locate drone images on satellite maps.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List

import cv2
import numpy as np

from src.config import load_config, Config
from src.map_query import MapQuery
from src.drone_image import DroneImage
from src.feature_extraction import SuperPointDescriptor, DISKDescriptor, SIFTDescriptor, ORBDescriptor
from src.feature_matching import LightGlueMatcher, OpenCVMatcher
from src.transform_utils import homography_estimate, renormalize_transform, transform_points, decompose_similarity
from src.visualization import visualize_matching_results


logger = logging.getLogger(__name__)


@dataclass
class MatchingContext:
    """Container for matching pipeline data."""
    map_query: MapQuery
    map_img: np.ndarray
    window: object
    drone_img_raw: np.ndarray
    config: Config


def setup_logging(config: Config) -> logging.Logger:
    """Configure logging based on config."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format
    )
    return logging.getLogger(__name__)


def load_map_patch(config: Config) -> Tuple[MapQuery, np.ndarray, object]:
    """Load map and extract patch at configured location."""
    px, py = config.input.map_location

    map_query = MapQuery(config.paths.map_tif)
    map_patch, window = map_query.extract_patch(
        px, py,
        config.map_extraction.default_patch_size,
        allow_clip=config.map_extraction.allow_clip
    )
    map_img = map_query.patch_to_opencv(map_patch)
    logger.info(f"Extracted map patch at ({px}, {py}): {map_img.shape}")

    return map_query, map_img, window


def load_drone_image(config: Config) -> np.ndarray:
    """Load drone image from configured path."""
    drone_handler = DroneImage(
        img_data_path=config.paths.drone_images_dir,
        calibration_file_path=config.paths.camera_calibration
    )

    drone_img = drone_handler.get_image(
        img_filename=config.input.drone_image,
        undistort=config.preprocessing.undistort
    )
    logger.info(f"Loaded drone image: {drone_img.shape}")

    return drone_img


def compute_location(
    transform: np.ndarray,
    drone_img: np.ndarray,
    window: object,
    map_query: MapQuery
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute global pixel and lat/lon coordinates from transform."""
    h, w = drone_img.shape[:2]
    center = np.array([[w / 2, h / 2, 1]], dtype=np.float32)
    map_center = (transform @ center.T).T[0]

    global_x = window.col_off + map_center[0]
    global_y = window.row_off + map_center[1]
    lat, lon = map_query.convert_pixel_to_latlon(global_x, global_y)

    return (global_x, global_y), (lat, lon)


def refine_transform_for_visualization(
    kp1: np.ndarray,
    T_pre: np.ndarray,
    result_transform: np.ndarray,
    transform_fn: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform keypoints back to raw image coordinates and update transform."""
    T_pre_h = np.vstack([T_pre, [0, 0, 1]])
    T_pre_inv = np.linalg.inv(T_pre_h)[:2, :]
    kp1_viz = transform_fn(kp1, T_pre_inv)
    transform_viz = (result_transform @ T_pre_h)[:2, :]

    return kp1_viz, transform_viz


class EstimatePose:
    """Pose estimation using feature matching and RANSAC."""

    def __init__(self, config: Config):
        
        if config.pipeline.feature_extractor == "disk":
            self.feature_extractor = DISKDescriptor()
        elif config.pipeline.feature_extractor == "superpoint":
            self.feature_extractor = SuperPointDescriptor()
        elif config.pipeline.feature_extractor == "sift":
            self.feature_extractor = SIFTDescriptor()
        elif config.pipeline.feature_extractor == "orb":
            self.feature_extractor = ORBDescriptor()
        else:
            raise ValueError(f"Unknown feature extractor: {config.pipeline.feature_extractor}")
        
        if config.pipeline.feature_matcher == "lightglue":
            if config.pipeline.feature_extractor == "disk":
                self.matcher = LightGlueMatcher(config.weights.disk_lightglue)
            elif config.pipeline.feature_extractor == "superpoint":
                self.matcher = LightGlueMatcher(config.weights.superpoint_lightglue)
            else:
                raise ValueError(f"LightGlue matcher not supported for extractor: {config.pipeline.feature_extractor}")
        elif config.pipeline.feature_matcher == "opencv":
            self.matcher = OpenCVMatcher(config.pipeline.matcher, ratio_thresh=config.pipeline.ratio_thresh)
        else:
            raise ValueError(f"Unknown feature matcher: {config.pipeline.feature_matcher}")
        
        self.image_size = config.pipeline.image_size
        self.ransac_thresh = config.matching.ransac_thresh
        self.confidence = config.matching.confidence
        self.refine_iters = config.matching.refine_iters
        
    def estimate(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Estimate pose between two images.
        Args:
            img1: First image (drone image).
            img2: Second image (map image).
        Returns:
            Tuple of (transform matrix, inlier mask, matches, (keypoints1, keypoints2))
        """
        input1 = cv2.resize(img1, (self.image_size, self.image_size))
        input2 = cv2.resize(img2, (self.image_size, self.image_size))

        kp1_norm, kp1, desc1 = self.feature_extractor.compute(input1)
        kp2_norm, kp2, desc2 = self.feature_extractor.compute(input2)

        matches = self.matcher.match_descriptors(kp1_norm, desc1, kp2_norm, desc2)

        if len(matches) < 4:
            logger.warning("Not enough matches to estimate pose")
            return None, None, matches, (kp1, kp2)

        M, mask = homography_estimate(
            kp1_norm, kp2_norm,
            matches,
            ransac_thresh=self.ransac_thresh,
            confidence=self.confidence,
            max_iters=2000,
            refine_iters=self.refine_iters,
            type="similarity"
        )

        if M is None:
            logger.warning("Pose estimation failed, not enough inliers")
            return None, None, matches, (kp1, kp2)
        else:
            logger.info(f"Total matches: {len(matches)} / {len(kp1)} keypoints in img1, {len(kp2)} keypoints in img2")
            logger.info(f"Pose Estimation successful: {np.sum(mask)} inliers out of {len(matches)} matches")
            # TODO: combine mask with matches to return only inlier matches
            # for i in range(len(matches)):
            #     if mask[i] == 0:
            #         matches[i] = None
            # matches = [m for m in matches if m is not None]

        M = renormalize_transform(
            M,
            (img1.shape[1], img1.shape[0]),
            (img2.shape[1], img2.shape[0])
        )

        return M, mask, matches, (kp1, kp2)
    

def estimate_pose(config: Config, drone_img: np.ndarray, map_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Search for best matching angle using pose estimation.
    Args:
        config: Configuration object.
        drone_img: Drone image.
        map_img: Map image.
    Returns:
        Tuple of (transform matrix, inlier mask, matches, (keypoints1, keypoints2))
        """
    pose_estimator = EstimatePose(config)

    if not config.angle_search.enabled:
        # Single angle from preprocessing config
        drone_img_iter, _ = DroneImage.preprocess(
            drone_img,
            scale=config.preprocessing.scale,
            rotation=config.preprocessing.rotation
        )
        return pose_estimator.estimate(drone_img_iter, map_img)

    for angle in range(*config.angle_search.range):
        logger.debug(f"Trying angle: {angle} degrees")

        drone_img_iter, _ = DroneImage.preprocess(
            drone_img,
            scale=config.preprocessing.scale,
            rotation=angle
        )

        M, mask, matches, (kp1, kp2) = pose_estimator.estimate(drone_img_iter, map_img)
        if mask is not None and np.sum(mask) >= 60:
            logger.info(f"Angle search successful at angle {angle} degrees with {np.sum(mask)} inliers.")
        else:
            M = None

        if M is not None:
            if config.angle_search.refine:
                params = decompose_similarity(M)
                scale_ = params['scale']
                angle_ = params['rotation'] + angle
                print(f"Refined angle: {angle_} degrees, scale: {scale_}")

                drone_img_final, drone_transform = DroneImage.preprocess(
                    drone_img,
                    scale=scale_,
                    rotation=angle_
                )
                M, mask, matches, (kp1, kp2) = pose_estimator.estimate(drone_img_final, map_img)

                drone_transform = np.vstack([drone_transform, [0, 0, 1]])
                M = M @ drone_transform
                kp1 = transform_points(kp1, np.linalg.inv(drone_transform))  # Transform keypoints back to raw image coords

            return M, mask, matches, (kp1, kp2)

    return None, None, None, None


def build_output(result, ctx: MatchingContext, params: dict) -> dict:
    """Build output dictionary from matching result."""
    if result is None or not result.success:
        logger.warning("Matching failed - not enough inliers")
        return {
            'success': False,
            'num_matches': result.num_matches if result else 0,
            'num_inliers': result.num_inliers if result else 0
        }

    pixel_loc, lat_lon = compute_location(
        result.transform, ctx.drone_img_raw, ctx.window, ctx.map_query
    )

    if params.get('scale'):
        logger.info(f"Transform: scale={params['scale']:.3f}, "
                    f"rotation={params['rotation']:.1f}deg")

    logger.info(f"Estimated location: Pixel=({pixel_loc[0]:.1f}, {pixel_loc[1]:.1f}), "
                f"Lat/Lon=({lat_lon[0]:.6f}, {lat_lon[1]:.6f})")

    return {
        'success': True,
        'pixel_location': pixel_loc,
        'lat_lon': lat_lon,
        'num_inliers': result.num_inliers,
        'inlier_ratio': result.inlier_ratio,
        'transform': result.transform,
        'transform_params': params
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Drone to Map Matching Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Enable visualization display'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving visualization'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config()

    # Override from command line
    if args.visualize:
        config.visualization.enabled = True
    if args.no_save:
        config.visualization.save_results = False

    drone_image = load_drone_image(config)
    map_query, map_image, window = load_map_patch(config)

    M, mask, matches, (kp1, kp2) = estimate_pose(config, drone_image, map_image)
    if M is None:
        logger.error("Pose estimation failed.")
        return 1
    else:
        logger.info("Pose estimation succeeded.")
        visualize_matching_results(
            drone_image,
            map_image,
            kp1,
            kp2,
            [m for i, m in enumerate(matches) if mask[i]],
            matches,
            M,
            save_path=config.paths.output_file if config.visualization.save_results else None,
            show=config.visualization.enabled
        )


if __name__ == '__main__':
    exit(main())
