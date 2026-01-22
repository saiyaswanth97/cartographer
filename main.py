#!/usr/bin/env python3
"""
Cartographer - Drone to Map Matching Pipeline

Main entry point for the drone image localization system.
Uses feature matching to locate drone images on satellite maps.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np

from src.config import load_config, Config
from src.map_query import MapQuery
from src.drone_image import DroneImage
from src.feature_extraction import SuperPointDescriptor, DISKDescriptor
from src.dl_e2e_matching import SuperPointLightGlue, DiskLightGlue
from src.matching import SIFTMatcher2D
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

    logger.info("Matching succeeded!")
    logger.info(f"  - Matches: {result.num_matches}")
    logger.info(f"  - Inliers after RANSAC: {result.num_inliers}")
    logger.info(f"  - Inlier ratio: {result.inlier_ratio:.1%}")

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


def run_visualization(
    config: Config,
    drone_img: np.ndarray,
    map_img: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    result
):
    """Run visualization if enabled in config."""
    if not (config.visualization.enabled or config.visualization.save_results):
        return

    save_path = config.visualization.output_file if config.visualization.save_results else None
    visualize_matching_results(
        drone_img, map_img,
        kp1, kp2,
        result.inliers, result.all_matches,
        result.transform,
        save_path=save_path,
        show=config.visualization.enabled,
        max_keypoints=config.visualization.max_keypoints_draw,
        max_matches=config.visualization.max_matches_draw
    )
    if save_path:
        logger.info(f"Saved visualization to {save_path}")


def get_feature_extractor(config: Config):
    """Get feature extractor based on config."""
    if config.pipeline.feature_extractor == "disk":
        return DISKDescriptor()
    return SuperPointDescriptor()


def get_e2e_matcher(config: Config):
    """Get end-to-end matcher based on config."""
    if config.pipeline.feature_extractor == "disk":
        return DiskLightGlue()
    return SuperPointLightGlue()


def search_best_angle_descriptor(
    ctx: MatchingContext,
    feature_extractor,
    matcher: SIFTMatcher2D,
    kp2: np.ndarray,
    desc2: np.ndarray
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Search for best matching angle using descriptor-based matching."""
    config = ctx.config
    result = None
    kp1 = None

    if not config.angle_search.enabled:
        # Single angle from preprocessing config
        drone_img_iter, T_pre = DroneImage.preprocess(
            ctx.drone_img_raw,
            scale=config.preprocessing.scale,
            rotation=config.preprocessing.rotation
        )
        kp1, desc1 = feature_extractor.compute(drone_img_iter)
        result = matcher.match_and_estimate(kp1, desc1, kp2, desc2)

        if result.success:
            kp1, result.transform = refine_transform_for_visualization(
                kp1, T_pre, result.transform, matcher.transform_points
            )
        return result, kp1, kp2

    for angle in range(*config.angle_search.range):
        drone_img_iter, _ = DroneImage.preprocess(
            ctx.drone_img_raw,
            scale=config.preprocessing.scale,
            rotation=angle
        )

        kp1_iter, desc1 = feature_extractor.compute(drone_img_iter)
        result = matcher.match_and_estimate(kp1_iter, desc1, kp2, desc2)

        if result.success:
            if config.angle_search.refine:
                params = matcher.decompose_similarity(result.transform)
                scale_ = params['scale']
                angle_ = params['rotation'] + angle

                drone_img_final, T_pre = DroneImage.preprocess(
                    ctx.drone_img_raw,
                    scale=scale_,
                    rotation=angle_
                )
                kp1_refined, desc1 = feature_extractor.compute(drone_img_final)
                result = matcher.match_and_estimate(kp1_refined, desc1, kp2, desc2)

                kp1, result.transform = refine_transform_for_visualization(
                    kp1_refined, T_pre, result.transform, matcher.transform_points
                )
            else:
                kp1 = kp1_iter
            break

    return result, kp1, kp2


def search_best_angle_e2e(
    ctx: MatchingContext,
    matcher,
    pose_estimator: SIFTMatcher2D
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Search for best matching angle using end-to-end matching."""
    config = ctx.config
    result = None
    kp1 = None
    kp2 = None

    if not config.angle_search.enabled:
        # Single angle from preprocessing config
        drone_img_iter, T_pre = DroneImage.preprocess(
            ctx.drone_img_raw,
            scale=config.preprocessing.scale,
            rotation=config.preprocessing.rotation
        )
        kp1, kp2, matches, scores = matcher.compute_and_match(drone_img_iter, ctx.map_img)

        if len(matches) >= config.matching.min_inliers:
            result = pose_estimator.match_and_estimate(kp1, None, kp2, None, matches=matches)
            if result.success:
                kp1, result.transform = refine_transform_for_visualization(
                    kp1, T_pre, result.transform, pose_estimator.transform_points
                )
        return result, kp1, kp2

    for angle in range(*config.angle_search.range):
        logger.debug(f"Trying angle: {angle} degrees")

        drone_img_iter, _ = DroneImage.preprocess(
            ctx.drone_img_raw,
            scale=config.preprocessing.scale,
            rotation=angle
        )

        kp1_iter, kp2, matches, scores = matcher.compute_and_match(drone_img_iter, ctx.map_img)
        if len(matches) < config.matching.min_inliers:
            continue

        result = pose_estimator.match_and_estimate(kp1_iter, None, kp2, None, matches=matches)

        if result.success:
            if config.angle_search.refine:
                params = pose_estimator.decompose_similarity(result.transform)
                scale_ = params['scale']
                angle_ = params['rotation'] + angle

                drone_img_final, T_pre = DroneImage.preprocess(
                    ctx.drone_img_raw,
                    scale=scale_,
                    rotation=angle_
                )
                kp1_refined, kp2, matches, scores = matcher.compute_and_match(
                    drone_img_final, ctx.map_img
                )
                if len(matches) < config.matching.min_inliers:
                    continue

                result = pose_estimator.match_and_estimate(
                    kp1_refined, None, kp2, None, matches=matches
                )
                kp1, result.transform = refine_transform_for_visualization(
                    kp1_refined, T_pre, result.transform, pose_estimator.transform_points
                )
            else:
                kp1 = kp1_iter
            break

    return result, kp1, kp2


def run_matching_pipeline(config: Config) -> dict:
    """Run the descriptor-based matching pipeline."""
    global logger
    logger = setup_logging(config)

    # Load data
    map_query, map_img, window = load_map_patch(config)
    drone_img_raw = load_drone_image(config)

    ctx = MatchingContext(
        map_query=map_query,
        map_img=map_img,
        window=window,
        drone_img_raw=drone_img_raw,
        config=config
    )

    # Setup feature extraction and matching
    feature_extractor = get_feature_extractor(config)
    kp2, desc2 = feature_extractor.compute(map_img)

    matcher = SIFTMatcher2D(
        model=config.matching.model,
        ratio_thresh=config.matching.ratio_thresh,
        ransac_thresh=config.matching.ransac_thresh,
        confidence=config.matching.confidence,
        min_inliers=config.matching.min_inliers
    )

    # Search for best angle
    result, kp1, kp2 = search_best_angle_descriptor(
        ctx, feature_extractor, matcher, kp2, desc2
    )

    # Build output
    params = matcher.decompose_similarity(result.transform) if result and result.success else {}
    output = build_output(result, ctx, params)

    # Visualization
    if result and kp1 is not None:
        run_visualization(config, drone_img_raw, map_img, kp1, kp2, result)

    return output


def run_matching_e2e_pipeline(config: Config) -> dict:
    """Run the end-to-end matching pipeline (SuperPoint/DISK + LightGlue)."""
    global logger
    logger = setup_logging(config)

    # Load data
    map_query, map_img, window = load_map_patch(config)
    drone_img_raw = load_drone_image(config)

    ctx = MatchingContext(
        map_query=map_query,
        map_img=map_img,
        window=window,
        drone_img_raw=drone_img_raw,
        config=config
    )

    # Setup matchers
    matcher = get_e2e_matcher(config)
    pose_estimator = SIFTMatcher2D(
        model=config.matching.model,
        ratio_thresh=config.matching.ratio_thresh,
        ransac_thresh=config.matching.ransac_thresh,
        confidence=config.matching.confidence,
        min_inliers=config.matching.min_inliers
    )

    # Search for best angle
    result, kp1, kp2 = search_best_angle_e2e(ctx, matcher, pose_estimator)

    # Build output
    params = pose_estimator.decompose_similarity(result.transform) if result and result.success else {}
    output = build_output(result, ctx, params)

    # Visualization
    if result and kp1 is not None:
        run_visualization(config, drone_img_raw, map_img, kp1, kp2, result)

    return output


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


def print_summary(result: dict):
    """Print matching result summary."""
    print("=" * 60)
    if result['success']:
        print("MATCHING SUCCESSFUL")
        print(f"  Location: {result['lat_lon'][0]:.6f}, {result['lat_lon'][1]:.6f}")
        print(f"  Inliers: {result['num_inliers']} ({result['inlier_ratio']:.1%})")
    else:
        print("MATCHING FAILED")
        print(f"  Matches: {result.get('num_matches', 0)}")
        print(f"  Inliers: {result.get('num_inliers', 0)}")
    print("=" * 60)


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

    # Select pipeline based on config
    if config.pipeline.type == "e2e":
        result = run_matching_e2e_pipeline(config)
    else:
        result = run_matching_pipeline(config)

    print_summary(result)

    return 0 if result['success'] else 1


if __name__ == '__main__':
    exit(main())
