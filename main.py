#!/usr/bin/env python3
"""
Cartographer - Drone to Map Matching Pipeline

Main entry point for the drone image localization system.
Uses SIFT feature matching to locate drone images on satellite maps.
"""

import argparse
import logging
import numpy as np
from pathlib import Path

from src.config import load_config
from src.map_query import MapQuery
from src.drone_image import DroneImage
from src.feature_extraction import SIFTDescriptor, SURFDescriptor, ORBDescriptor, SuperPointDescriptor
from src.matching import SIFTMatcher2D
from src.visualization import visualize_matching_results, visualize_side_by_side


def setup_logging(config):
    """Configure logging based on config."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format
    )
    return logging.getLogger(__name__)


def run_matching_pipeline_v2(config, drone_image_file: str = None, map_location: tuple = None):
    """
    Run the complete matching pipeline.

    Args:
        config: Configuration object.
        drone_image_file: Override drone image filename.
        map_location: Override map extraction location (x, y).

    Returns:
        Dictionary with matching results.
    """
    logger = setup_logging(config)

    # Use provided location or default from ground truth
    if map_location:
        px, py = map_location
    else:
        # Default: ground truth location for 1485.025665088.png
        px, py = 2811, 3651
    map_query = MapQuery(config.paths.map_tif)
    map_patch, window = map_query.extract_patch(
        px, py,
        config.map_extraction.default_patch_size,
        allow_clip=config.map_extraction.allow_clip
    )
    map_img = map_query.patch_to_opencv(map_patch)
    logger.info(f"Extracted map patch at ({px}, {py}): {map_img.shape}")

    # Compute features
    feature_extraction = SuperPointDescriptor()
    kp2, desc2 = feature_extraction.compute(map_img)
    
    # Load drone image
    logger.info("Loading drone image...")
    drone_handler = DroneImage(
        img_data_path=config.paths.drone_images_dir,
        calibration_file_path=config.paths.camera_calibration
    )

    drone_img_raw = drone_handler.get_image(
        # img_filename=drone_image_file or '1485.025665088.png',
        img_filename=drone_image_file or '1537.017421696.png',
        undistort=config.preprocessing.undistort
    )
    logger.info(f"Loaded drone image: {drone_img_raw.shape}")

    matcher = SIFTMatcher2D(
        model=config.matching.model,
        ratio_thresh=config.matching.ratio_thresh,
        ransac_thresh=config.matching.ransac_thresh,
        confidence=config.matching.confidence,
        min_inliers=config.matching.min_inliers
    )
    
    # Preprocess drone image
    for angle in range(-180, 181, 20):
        drone_img_iter = DroneImage.preprocess(
            drone_img_raw,
            scale=config.preprocessing.scale,
            rotation=angle
        )

        kp1, desc1 = feature_extraction.compute(drone_img_iter)
        
        result = matcher.match_and_estimate(kp1, desc1, kp2, desc2)
        if result.success:
            scale_ = matcher.decompose_similarity(result.transform)['scale']
            angle_ = matcher.decompose_similarity(result.transform)['rotation'] + angle

            drone_img_final = DroneImage.preprocess(
                drone_img_raw,
                scale=scale_,
                rotation=angle_
            )
            kp1, desc1 = feature_extraction.compute(drone_img_final)
            result = matcher.match_and_estimate(kp1, desc1, kp2, desc2)
            # T_pre = matcher.get_transform(scale_, -angle_, drone_img_raw.shape[1], drone_img_raw.shape[0])
            # result.transform = T_pre @ np.vstack([result.transform, [0, 0, 1]])

            drone_img = drone_img_final
            break

    # Process results
    if result.success:
        logger.info(f"Matching succeeded!")
        logger.info(f"  - Matches after ratio test: {result.num_matches}")
        logger.info(f"  - Inliers after RANSAC: {result.num_inliers}")
        logger.info(f"  - Inlier ratio: {result.inlier_ratio:.1%}")

        # Decompose transform
        params = matcher.decompose_similarity(result.transform)
        if params['scale']:
            logger.info(f"Transform: scale={params['scale']:.3f}, "
                       f"rotation={params['rotation']:.1f}deg")

        # Estimate location
        h, w = drone_img.shape[:2]
        center = np.array([[w/2, h/2]], dtype=np.float32)

        if result.transform.shape[0] == 2:
            center_homog = np.hstack([center, np.ones((1, 1))])
            map_center = (result.transform @ center_homog.T).T[0]
        else:
            center_homog = np.hstack([center, np.ones((1, 1))])
            map_center_homog = (result.transform @ center_homog.T).T[0]
            map_center = map_center_homog[:2] / map_center_homog[2]

        # Convert to global coordinates
        global_x = window.col_off + map_center[0]
        global_y = window.row_off + map_center[1]

        # Convert to lat/lon
        lat, lon = map_query.convert_pixel_to_latlon(global_x, global_y)

        logger.info(f"Estimated location:")
        logger.info(f"  Pixel: ({global_x:.1f}, {global_y:.1f})")
        logger.info(f"  Lat/Lon: ({lat:.6f}, {lon:.6f})")

        output = {
            'success': True,
            'pixel_location': (global_x, global_y),
            'lat_lon': (lat, lon),
            'num_inliers': result.num_inliers,
            'inlier_ratio': result.inlier_ratio,
            'transform': result.transform,
            'transform_params': params
        }
    else:
        logger.warning("Matching failed - not enough inliers")
        output = {
            'success': False,
            'num_matches': result.num_matches,
            'num_inliers': result.num_inliers
        }

    # Visualization
    if config.visualization.enabled or config.visualization.save_results:
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

    return output


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Drone to Map Matching Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '-i', '--image',
        default=None,
        help='Drone image filename to process'
    )
    parser.add_argument(
        '-x', '--map-x',
        type=int,
        default=None,
        help='Map X coordinate for patch extraction'
    )
    parser.add_argument(
        '-y', '--map-y',
        type=int,
        default=None,
        help='Map Y coordinate for patch extraction'
    )
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Enable visualization'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving visualization'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override from command line
    if args.visualize:
        config.visualization.enabled = True
    if args.no_save:
        config.visualization.save_results = False

    # Determine map location
    map_location = None
    if args.map_x is not None and args.map_y is not None:
        map_location = (args.map_x, args.map_y)

    result = run_matching_pipeline_v2(
        config,
        drone_image_file=args.image,
        map_location=map_location
    )

    # Print summary
    print("="*60)
    if result['success']:
        print("MATCHING SUCCESSFUL")
        print(f"  Location: {result['lat_lon'][0]:.6f}, {result['lat_lon'][1]:.6f}")
        print(f"  Inliers: {result['num_inliers']} ({result['inlier_ratio']:.1%})")
    else:
        print("MATCHING FAILED")
        print(f"  Matches: {result.get('num_matches', 0)}")
        print(f"  Inliers: {result.get('num_inliers', 0)}")
    print("="*60)
    print("\n")

    return 0 if result['success'] else 1


if __name__ == '__main__':
    exit(main())
