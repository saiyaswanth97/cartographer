import rasterio
import rasterio.warp
import cv2
import numpy as np
import matplotlib.pyplot as plt


class MapQuery():
    def __init__(self, tiff_path):
        self.ds = rasterio.open(tiff_path)
        
        # Precompute lat/lon bounds (safety)
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
        
    def convert_xy_to_latlon(self, x, y):
        lon, lat = rasterio.warp.transform(self.ds.crs, "EPSG:4326", [x], [y])
        return lat[0], lon[0]
    
    def convert_latlon_to_xy(self, lat, lon):
        if not (self.lat_lon_bounds['min_lat'] <= lat <= self.lat_lon_bounds['max_lat'] and
                self.lat_lon_bounds['min_lon'] <= lon <= self.lat_lon_bounds['max_lon']):
            raise ValueError("Latitude or Longitude is out of bounds of the dataset.")
        x, y = rasterio.warp.transform("EPSG:4326", self.ds.crs, [lon], [lat])
        return x[0], y[0]
                
    def check_within_patch_bounds(self, x, y):
        return (0 <= x < self.ds.width) and (0 <= y < self.ds.height)
    
    def extract_patch(self, x, y, patch_size):
        if not self.check_within_patch_bounds(x, y):
            raise ValueError("Coordinates are out of bounds of the dataset.")
        
        window = rasterio.windows.Window(
            x - patch_size // 2,
            y - patch_size // 2,
            patch_size,
            patch_size
        ).intersection(rasterio.windows.Window(0, 0, self.ds.width, self.ds.height))
        
        patch = self.ds.read(window=window)
        return patch, window
        
    @staticmethod
    def show_patch(patch, title):
        if patch.shape[0] >= 3:
            img = patch[:3].transpose(1, 2, 0)
        else:
            img = patch[0]

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()


def preprocess_drone_image(img, scale, yaw_deg):
    h, w = img.shape[:2]

    # Scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # Rotate
    M = cv2.getRotationMatrix2D(
        (img.shape[1] // 2, img.shape[0] // 2),
        yaw_deg,
        1.0
    )
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return img


if __name__ == '__main__':
    # Example usage
    img_path = 'data/train_data/drone_images/1485.025665088.png'
    drone_img = cv2.imread(img_path)
    drone_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

    # scale = 0.5  # Example scale factor
    # yaw_deg = 30  # Example yaw angle in degrees

    # processed_img = preprocess_drone_image(img, scale, yaw_deg)

    # plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    # plt.title("Processed Drone Image")
    # plt.axis("off")
    # plt.show()
    
    # map_query = MapQuery('data/map.tif')
    # map_image,_ = map_query.extract_patch(2600, 3600, 1600)
    # plt.imshow(map_image.transpose(1, 2, 0))
    # plt.title("Extracted Map Patch")
    # plt.axis("off")
    # plt.show()
    
    # img_g = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    # map_query_g = cv2.cvtColor(map_image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    # res = cv2.matchTemplate(map_query_g, img_g, cv2.TM_CCOEFF_NORMED)
    # _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    # print(f"Max correlation value: {max_val} at location: {max_loc}")
    
    # transformed_x = max_loc[0] + processed_img.shape[1] // 2
    # transformed_y = max_loc[1] + processed_img.shape[0] // 2
    
    # img_warped = preprocess_drone_image(img, scale, yaw_deg)
    # map_patch, _ = map_query.extract_patch(transformed_x, transformed_y, processed_img.shape[1])
    # plt.imshow(map_patch.transpose(1, 2, 0))
    # plt.title("Matched Map Patch")
    # plt.axis("off")
    # plt.show()
    
    map_query = MapQuery('data/map.tif')
    seed_col, seed_row = 2600, 3600
    search_size = 2000

    map_patch, window = map_query.extract_patch(
        seed_col, seed_row, search_size
    )

    map_rgb = map_patch[:3].transpose(1, 2, 0)
    map_gray = cv2.cvtColor(map_rgb, cv2.COLOR_BGR2GRAY)

    MapQuery.show_patch(map_patch, "Search Map Patch")

    # -----------------------------
    # Template matching (scale + yaw sweep)
    # -----------------------------
    best_score = -1
    best_result = None

    for scale in [0.5, 0.75, 1.0]:
        for yaw in range(-30, 31, 10):

            proc = preprocess_drone_image(drone_img, scale, yaw)
            proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

            if (proc_gray.shape[0] > map_gray.shape[0] or
                proc_gray.shape[1] > map_gray.shape[1]):
                continue

            res = cv2.matchTemplate(
                map_gray,
                proc_gray,
                cv2.TM_CCOEFF_NORMED
            )

            _, score, _, loc = cv2.minMaxLoc(res)

            if score > best_score:
                best_score = score
                best_result = (loc, scale, yaw, proc_gray.shape)

    if best_result is None:
        raise RuntimeError("No valid match found")

    (best_loc, best_scale, best_yaw, (h, w)) = best_result

    print(f"Best NCC score: {best_score:.3f}")
    print(f"Best scale: {best_scale}, yaw: {best_yaw}")

    # -----------------------------
    # Convert match → global pixel
    # -----------------------------
    match_col = window.col_off + best_loc[0] + w // 2
    match_row = window.row_off + best_loc[1] + h // 2

    # -----------------------------
    # Extract final matched patch
    # -----------------------------
    final_patch, _ = map_query.extract_patch(
        match_col, match_row, w
    )

    MapQuery.show_patch(final_patch, "Matched Map Patch")

    # -----------------------------
    # Convert to lat/lon
    # -----------------------------
    lat, lon = map_query.convert_xy_to_latlon(match_col, match_row)
    print(f"Matched location → lat: {lat:.6f}, lon: {lon:.6f}")
    