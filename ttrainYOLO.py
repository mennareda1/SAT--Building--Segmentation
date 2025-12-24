import geopandas as gpd
from shapely.geometry import box
import os
import rasterio
import numpy as np
import shutil
import random
from ultralytics import YOLO
from shapely.geometry import box, Polygon
import matplotlib.pyplot as plt
import glob
import yaml
model = YOLO("yolo11n-seg.pt")  
model.train(
    data="/kaggle/input/asemaoctoberr/data.yaml",
    epochs=1650,
    imgsz=512,
    batch=16,
    lr0=0.002,
    lrf=0.01,
    patience=100,
    workers=4, 
    cos_lr=True,  
    task="segmentation",
)