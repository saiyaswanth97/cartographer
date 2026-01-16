"""Map query module for extracting patches from GeoTIFF maps."""

import rasterio
import rasterio.warp
import rasterio.windows
import cv2
import numpy as np
from typing import Tuple, Optional


class MapQuery:
    """
    Query interface for GeoTIFF satellite/aerial maps.

    Handles coordinate transformations between pixel, CRS, and lat/lon systems,
    and extracts image patches for matching.
    """

    def __init__(self, tiff_path: str):
        """
        Initialize MapQuery with a GeoTIFF file.

        Args:
            tiff_path: Path to the GeoTIFF file.
        """
        self.tiff_path = tiff_path
        self.ds = rasterio.open(tiff_path)

        # Precompute lat/lon bounds
        left, bottom, right, top = self.ds.bounds
        lons, lats = rasterio.warp.transform(
            self.ds.crs, "EPSG:4326",
            [left, right],
            [bottom, top]
        )
        self.lat_lon_bounds = {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
        }

    @property
    def width(self) -> int:
        """Map width in pixels."""
        return self.ds.width

    @property
    def height(self) -> int:
        """Map height in pixels."""
        return self.ds.height

    @property
    def shape(self) -> Tuple[int, int]:
        """Map shape (height, width) in pixels."""
        return (self.ds.height, self.ds.width)

    def pixel_to_crs(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to CRS coordinates.

        Args:
            pixel_x: X pixel coordinate (column)
            pixel_y: Y pixel coordinate (row)

        Returns:
            Tuple of (crs_x, crs_y) in the map's native CRS.
        """
        crs_x, crs_y = self.ds.transform * (pixel_x, pixel_y)
        return crs_x, crs_y

    def crs_to_pixel(self, crs_x: float, crs_y: float) -> Tuple[int, int]:
        """
        Convert CRS coordinates to pixel coordinates.

        Args:
            crs_x: X coordinate in CRS
            crs_y: Y coordinate in CRS

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        inv_transform = ~self.ds.transform
        pixel_x, pixel_y = inv_transform * (crs_x, crs_y)
        return int(pixel_x), int(pixel_y)

    def convert_pixel_to_latlon(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to latitude/longitude.

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate

        Returns:
            Tuple of (latitude, longitude).
        """
        crs_x, crs_y = self.pixel_to_crs(pixel_x, pixel_y)
        lon, lat = rasterio.warp.transform(self.ds.crs, "EPSG:4326", [crs_x], [crs_y])
        return lat[0], lon[0]

    def convert_latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert latitude/longitude to pixel coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Tuple of (pixel_x, pixel_y).

        Raises:
            ValueError: If coordinates are outside map bounds.
        """
        if not (self.lat_lon_bounds['min_lat'] <= lat <= self.lat_lon_bounds['max_lat'] and
                self.lat_lon_bounds['min_lon'] <= lon <= self.lat_lon_bounds['max_lon']):
            raise ValueError("Latitude or Longitude is out of bounds of the dataset.")
        crs_x, crs_y = rasterio.warp.transform("EPSG:4326", self.ds.crs, [lon], [lat])
        return self.crs_to_pixel(crs_x[0], crs_y[0])

    def check_within_bounds(self, x: int, y: int) -> bool:
        """Check if pixel coordinate is within dataset bounds."""
        return (0 <= x < self.ds.width) and (0 <= y < self.ds.height)

    def check_patch_fits(self, x: int, y: int, patch_size: int) -> bool:
        """Check if the full patch fits within dataset bounds."""
        half = patch_size // 2
        return (half <= x < self.ds.width - half) and (half <= y < self.ds.height - half)

    def extract_patch(
        self,
        x: int,
        y: int,
        patch_size: int,
        allow_clip: bool = True
    ) -> Tuple[np.ndarray, rasterio.windows.Window]:
        """
        Extract a patch centered at (x, y).

        Args:
            x: Center X pixel coordinate
            y: Center Y pixel coordinate
            patch_size: Size of the square patch
            allow_clip: If True, clip patch at boundaries. If False, raise error.

        Returns:
            Tuple of (patch array in rasterio format [bands, h, w], window object).

        Raises:
            ValueError: If coordinates are invalid.
        """
        if not self.check_within_bounds(x, y):
            raise ValueError("Center coordinates are out of bounds of the dataset.")

        if not allow_clip and not self.check_patch_fits(x, y, patch_size):
            raise ValueError("Patch does not fit within dataset bounds.")

        window = rasterio.windows.Window(
            x - patch_size // 2,
            y - patch_size // 2,
            patch_size,
            patch_size
        ).intersection(rasterio.windows.Window(0, 0, self.ds.width, self.ds.height))

        patch = self.ds.read(window=window)
        return patch, window

    def patch_to_opencv(self, patch: np.ndarray, hist_eq: bool = True) -> np.ndarray:
        """
        Convert rasterio patch (bands, h, w) to OpenCV BGR format (h, w, channels).

        Args:
            patch: Rasterio format array [bands, height, width]
            hist_eq: If True, apply histogram equalization to color channels.

        Returns:
            OpenCV format array [height, width, channels] in BGR.
        """
        if patch.shape[0] >= 3:
            # Rasterio stores as RGB, OpenCV expects BGR
            rgb = patch[:3].transpose(1, 2, 0).astype(np.uint8)
            
            if hist_eq:
                # Apply histogram equalization on the Y channel
                yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            return patch[0].astype(np.uint8)

    def close(self):
        """Close the dataset."""
        self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
