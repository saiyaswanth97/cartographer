"""Configuration loader for the cartographer pipeline."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathsConfig:
    map_tif: str = "data/map.tif"
    drone_images_dir: str = "data/train_data/drone_images/"
    camera_calibration: str = "data/train_data/camera_infomation.yaml"
    ground_truth: str = "data/train_data/ground_truth.csv"
    output_dir: str = "output/"


@dataclass
class PreprocessingConfig:
    scale: float = 0.4
    rotation: float = -75.0
    undistort: bool = False


@dataclass
class MapExtractionConfig:
    default_patch_size: int = 1500
    allow_clip: bool = True


@dataclass
class MatchingConfig:
    model: str = "similarity"
    ratio_thresh: float = 0.85
    ransac_thresh: float = 0.025
    confidence: float = 0.99
    min_inliers: int = 6


@dataclass
class VisualizationConfig:
    enabled: bool = True
    save_results: bool = True
    output_file: str = "matching_results.png"
    max_keypoints_draw: int = 100
    max_matches_draw: int = 50


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Main configuration container."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    map_extraction: MapExtractionConfig = field(default_factory=MapExtractionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If None, uses default config.

    Returns:
        Config object with all settings.
    """
    config = Config()

    if config_path is None:
        # Try default locations
        default_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'paths' in data:
            config.paths = PathsConfig(**data['paths'])
        if 'preprocessing' in data:
            config.preprocessing = PreprocessingConfig(**data['preprocessing'])
        if 'map_extraction' in data:
            config.map_extraction = MapExtractionConfig(**data['map_extraction'])
        if 'matching' in data:
            config.matching = MatchingConfig(**data['matching'])
        if 'visualization' in data:
            config.visualization = VisualizationConfig(**data['visualization'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])

    return config
