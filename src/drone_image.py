"""Drone image handling module."""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple
from matplotlib import pyplot as plt


class DroneImage:
    """
    Handler for drone images with camera calibration support.

    Supports loading images, undistortion, and preprocessing (scaling/rotation).
    """

    def __init__(
        self,
        img_data_path: str = "data/train_data/drone_images/",
        calibration_file_path: Optional[str] = "data/train_data/camera_infomation.yaml"
    ):
        """
        Initialize DroneImage handler.

        Args:
            img_data_path: Directory containing drone images.
            calibration_file_path: Path to camera calibration YAML file.
        """
        self.img_data_path = Path(img_data_path)
        self.calibration_data = None

        if calibration_file_path and Path(calibration_file_path).exists():
            with open(calibration_file_path, 'r') as f:
                self.calibration_data = yaml.safe_load(f)

    def get_image(
        self,
        img_filename: str = '1485.025665088.png',
        undistort: bool = False
    ) -> np.ndarray:
        """
        Load a drone image.

        Args:
            img_filename: Name of the image file.
            undistort: Whether to apply lens undistortion. The images are undistorted already.

        Returns:
            Image as numpy array in BGR format.

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        img_path = self.img_data_path / img_filename
        drone_img = cv2.imread(str(img_path))

        if drone_img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        if not undistort or self.calibration_data is None:
            return drone_img

        drone_img_undistorted = cv2.undistort(
            drone_img,
            np.array(self.calibration_data['camera_matrix']['data']).reshape((3, 3)),
            np.array(self.calibration_data['distortion_coefficients']['data']).reshape((1, -1))
        )
        return drone_img_undistorted

    def list_images(self) -> list:
        """List all available drone images."""
        return sorted([f.name for f in self.img_data_path.glob("*.png")])

    @staticmethod
    def preprocess(
        img: np.ndarray,
        scale: float = 1.0,
        rotation: float = 0.0,
        expand_canvas: bool = True,
        hist_eq: bool = True
    ) -> np.ndarray:
        """
        Preprocess drone image with scaling and rotation.

        Args:
            img: Input image.
            scale: Scale factor (< 1 to shrink, > 1 to enlarge).
            rotation: Rotation angle in degrees (counter-clockwise).
            expand_canvas: If True, expand canvas to fit rotated image.
            hist_eq: If True, apply histogram equalization.

        Returns:
            Preprocessed image.
        """
        # Scale
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Rotate
        if rotation != 0.0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)

            if expand_canvas:
                # Expand canvas to fit rotated image
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int(h * sin + w * cos)
                new_h = int(h * cos + w * sin)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                img = cv2.warpAffine(img, M, (new_w, new_h))
            else:
                img = cv2.warpAffine(img, M, (w, h))

        if hist_eq:
            # Apply histogram equalization to each channel
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
        return img

    @staticmethod
    def get_image_center(img: np.ndarray) -> Tuple[float, float]:
        """Get the center point of an image."""
        h, w = img.shape[:2]
        return w / 2, h / 2


if __name__ == "__main__":
    # Example usage
    drone_handler = DroneImage()
    img = drone_handler.get_image('1485.025665088.png', undistort=False)
    preprocessed_img = DroneImage.preprocess(img, scale=0.5, rotation=30)
    center = DroneImage.get_image_center(preprocessed_img)
    print(f"Image center: {center}")
    
    img_rotated = DroneImage.preprocess(img, scale=1.0, rotation=15, expand_canvas=True)
    plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    plt.title("Rotated Image with Expanded Canvas")
    plt.axis('off')
    plt.show()
    