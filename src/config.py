"""Configuration loader for the cartographer pipeline."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PathsConfig:
    map_tif: str = "data/map.tif"
    drone_images_dir: str = "data/train_data/drone_images/"
    camera_calibration: str = "data/train_data/camera_infomation.yaml"
    ground_truth: str = "data/train_data/ground_truth.csv"
    output_dir: str = "output/"


@dataclass
class WeightsConfig:
    superpoint_lightglue: str = "weight/superpoint_lightglue_pipeline.onnx"
    disk_lightglue: str = "weight/disk_lightglue_end2end_fused_cpu.onnx"
    disk_descriptor: str = "weight/disk_descriptor.onnx"
    superpoint_descriptor: str = "weight/superpoint_descriptor.onnx"


@dataclass
class PipelineConfig:
    type: str = "e2e"  # "e2e" or "descriptor"
    feature_extractor: str = "superpoint"  # "superpoint" or "disk"


@dataclass
class InputConfig:
    drone_image: str = "1537.017421696.png"
    map_center_x: int = 2811
    map_center_y: int = 3651

    @property
    def map_location(self) -> Tuple[int, int]:
        return (self.map_center_x, self.map_center_y)


@dataclass
class PreprocessingConfig:
    scale: float = 1.0
    rotation: float = 0.0
    undistort: bool = False


@dataclass
class AngleSearchConfig:
    enabled: bool = True
    start: int = -180
    end: int = 180
    step: int = 20
    refine: bool = True

    @property
    def range(self) -> Tuple[int, int, int]:
        return (self.start, self.end + 1, self.step)


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
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    input: InputConfig = field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    angle_search: AngleSearchConfig = field(default_factory=AngleSearchConfig)
    map_extraction: MapExtractionConfig = field(default_factory=MapExtractionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _load_section(data: dict, key: str, cls):
    """Load a config section if present in data."""
    if key in data:
        return cls(**data[key])
    return cls()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If None, searches default locations.

    Returns:
        Config object with all settings.
    """
    if config_path is None:
        default_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break

    if not config_path or not Path(config_path).exists():
        return Config()

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return Config(
        paths=_load_section(data, 'paths', PathsConfig),
        weights=_load_section(data, 'weights', WeightsConfig),
        pipeline=_load_section(data, 'pipeline', PipelineConfig),
        input=_load_section(data, 'input', InputConfig),
        preprocessing=_load_section(data, 'preprocessing', PreprocessingConfig),
        angle_search=_load_section(data, 'angle_search', AngleSearchConfig),
        map_extraction=_load_section(data, 'map_extraction', MapExtractionConfig),
        matching=_load_section(data, 'matching', MatchingConfig),
        visualization=_load_section(data, 'visualization', VisualizationConfig),
        logging=_load_section(data, 'logging', LoggingConfig),
    )
