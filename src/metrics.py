"""Metrics for evaluation"""

import numpy as np
from typing import Tuple


class LocationErrorMetrics:
    """Class to compute location error metrics."""

    def __init__(self, file_path: str = "data/train_data/ground_truth.csv"):
        self.ground_truth = {}
        with open(file_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                img_name = float(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])
                self.ground_truth[img_name] = (lat, lon)
            print(f"Loaded ground truth for {len(self.ground_truth)} images.")

    def get_lat_lon(self, img_name: str) -> Tuple[float, float]:
        """
        Get ground truth latitude and longitude for an image.
        Args:
            img_name: Image filename.
        Returns:
            Tuple of (latitude, longitude). Returns (None, None) if not found.
        """
        closest_img = min(self.ground_truth.keys(), key=lambda x: abs(x - float(img_name.split('/')[-1].split('.')[0])))
        diff = abs(closest_img - float(img_name.split('/')[-1].split('.')[0]))
        if diff > 1e-1:
            print(f"Warning: No close ground truth found for {img_name}, closest is {closest_img} with diff {diff}")
            return (None, None)
        else:
            print(f"Matched {img_name} to ground truth image {closest_img} with diff {diff}")
        return self.ground_truth.get(closest_img, (None, None))
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two (lat, lon) coordinates in meters.
        Args:
            lat1: Latitude of the first point in degrees.
            lon1: Longitude of the first point in degrees.
            lat2: Latitude of the second point in degrees.
            lon2: Longitude of the second point in degrees.
        Returns:
            Distance in meters between the two points.
        """

        R = 6371000  # Earth radius in meters
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        distance = R * c
        return distance
    
    def compute_location_error(
        self,
        img_name: str,
        est_lat: float,
        est_lon: float
    ) -> float:
        """
        Compute location error in meters for a given image.
        Args:
            img_name: Image filename.
            est_lat: Estimated latitude.
            est_lon: Estimated longitude.
        Returns:
            Location error in meters. Returns None if ground truth is not found.
        """
        gt_lat, gt_lon = self.get_lat_lon(img_name)
        if gt_lat is None or gt_lon is None:
            return None
        
        error = self.haversine_distance(gt_lat, gt_lon, est_lat, est_lon)
        return error
    

if __name__ == "__main__":
    # Example usage
    lem = LocationErrorMetrics()
    img_name = "data/train_data/drone_images/1537.017421696.png"
    # est_lat = 13.026129
    # est_lon = 77.563731
    est_lat = 13.026120
    est_lon = 77.563741
    error = lem.compute_location_error(img_name, est_lat, est_lon)
    if error is not None:
        print(f"Location error for {img_name}: {error:.2f} meters")