import os
import cv2
import glob
import shutil
import numpy as np
import geopandas as gpd
import rasterio
from tqdm import tqdm
from shapely.geometry import Polygon
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# === PATHS ===
image_path = r"C:\chatbotApp\tiledAsemaSAHI\segment\dataset\train\images\clippedmisrSat_updatedrgb_0_0.tif"
output_shp = r"C:\chatbotApp\Asema_SAHI.shp"
model_path = r"C:\YOLOV5\weights5\best (3).pt"

from shapely.geometry import Polygon
import numpy as np

def mask_to_polygons_fallback(mask_bool, min_area=10):
    """
    Convert boolean mask (2D numpy array) to a list of shapely Polygons (in pixel coords).
    mask_bool: numpy array of dtype bool or 0/1
    min_area: ignore tiny contours (pixels)
    """
    mask_uint8 = (mask_bool.astype('uint8') * 255)  # 0 or 255
    # findContours expects single-channel; use RETR_EXTERNAL to get outer contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        # optional: approximate contour to reduce vertices
        epsilon = 0.0  # set >0.0 to simplify, e.g., 1.0 or 2.0
        if epsilon > 0:
            cnt = cv2.approxPolyDP(cnt, epsilon, True)
        coords = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]  # (x=col, y=row) pixel coords
        poly = Polygon(coords)
        if poly.is_valid and poly.area >= min_area:
            polygons.append(poly)
    return polygons
# === LOAD MODEL ===
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.1,
    device="cuda:0"  
)

# === OPEN RASTER TO READ CRS & TRANSFORM ===
with rasterio.open(image_path) as src:
    transform = src.transform
    crs = src.crs

# === RUN SAHI PREDICTION ===
result = get_sliced_prediction(
    image=image_path,
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_class_agnostic=True,
    verbose=2
)

print(f"✅ Detected {len(result.object_prediction_list)} objects")

# === EXTRACT POLYGONS IN GEO COORDINATES ===
all_polygons = []
all_confs = []
all_classes = []

for obj in tqdm(result.object_prediction_list):
    if obj.mask is not None:
        mask_array = obj.mask.bool_mask  # binary mask (numpy array)
        polygons = mask_to_polygons_fallback(mask_array)  # list of shapely polygons (in pixel coords)

        for poly in polygons:
            if poly.area > 0:
                # Convert pixel → map coordinates
                coords = np.array(poly.exterior.coords)
                geo_coords = []
                for x, y in coords:
                    X, Y = rasterio.transform.xy(transform, y, x)
                    geo_coords.append((X, Y))
                geo_poly = Polygon(geo_coords)

                all_polygons.append(geo_poly)
                all_confs.append(obj.score.value)
                all_classes.append(obj.category.name)

# === SAVE TO SHAPEFILE ===
if all_polygons:
    gdf = gpd.GeoDataFrame(
        {"class": all_classes, "confidence": all_confs},
        geometry=all_polygons,
        crs=crs if crs else "EPSG:4326"
    )
    gdf.to_file(output_shp)
    print(f"Shapefile saved at: {output_shp}")
else:
    print("No valid polygons detected.")
