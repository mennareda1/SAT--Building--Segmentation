import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import os
import numpy as np
import shutil
import random
from ultralytics import YOLO
from shapely.geometry import box, Polygon
import matplotlib.pyplot as plt

def arrange_points(points):
    """
    yolo pose model need the points to be in a same order
    this order ( top-left, top-right, bottom-right, bottom-left)is fixed across the training and prediction 

    
    """
    points = sorted(points, key=lambda p: p[1])
    top_points = points[:2]
    bottom_points = points[2:]
    top_points = sorted(top_points, key=lambda p: p[0])
    top_left, top_right = top_points
    bottom_points = sorted(bottom_points, key=lambda p: p[0])
    bottom_left, bottom_right = bottom_points
    arranged_points=[top_left, top_right, bottom_right, bottom_left]
    return arranged_points
def preparing_train_data_detection (images,output_train_images_dir,output_train_labels_dir):
    """
    prepare yolo 500x500 8-bit normalized tiles and their corresponding labels with step size 400 for a detection tasks 
    label format (class id ,center x normalized , center y normalized , width normalized , height normalized)
    Args :
         images -> images path list
         output_train_images_dir -> directory to save the training tiles 
         output_train_labels_dir ->directory to save the labels 
    
    """
    os.makedirs(output_train_images_dir, exist_ok=True)
    os.makedirs(output_train_labels_dir, exist_ok=True)
    window_size = 512
    step_size=400
  
    for image_path in images:
      
        image_name=(image_path.split("\\")[-1]).split('.tif')[0]
        shapefile_path=image_path.replace('.tif','.shp')
        shapefile = gpd.read_file(shapefile_path)
    
        with rasterio.open(image_path) as src:
            raster_crs = src.crs
            transform = src.transform
            image_width = src.width
            image_height = src.height
            image_data=src.read()
            image_data = image_data.astype(float)  
            image_data/=2**12 #for misrsat change to 10000 for sentinel normalization
            image_data*=255.0
            image_data = image_data.astype(np.uint8)# Convert to uint8
          
            if shapefile.crs != raster_crs:
                shapefile = shapefile.to_crs(raster_crs)

           
            for i in range(0, image_height, step_size):
                for j in range(0, image_width, step_size):

                    window_width = min(window_size, image_width - j)
                    window_height = min(window_size, image_height - i)
                    window = Window(j, i, window_width, window_height)
                    transform_window = src.window_transform(window)
                    window_bounds = rasterio.windows.bounds(window, transform)
                    bbox = box(*window_bounds)
                    clipped = shapefile.clip(bbox)
                    yolo_labels = []

                    for _, row in clipped.iterrows():
                        if row.geometry.type == 'Polygon':
                           
                            minx, miny, maxx, maxy = row.geometry.bounds
                            min_col, min_row = ~transform * (minx, miny)
                            max_col, max_row = ~transform * (maxx, maxy)
                            
                            # Convert to window-local tile coordinates
                            min_col -= j
                            min_row -= i
                            max_col -= j
                            max_row -= i

                            x_center = (min_col + max_col) / 2 / window_width
                            y_center = (min_row + max_row) / 2 / window_height
                            bbox_width = (max_col - min_col) / window_width
                            bbox_height = ( min_row-max_row ) / window_height
                            yolo_labels.append([0, x_center, y_center, bbox_width, bbox_height])


                    window_data =image_data [:3,i:(i+window_height),j:(j+window_width)]
                    image_output_path = os.path.join(output_train_images_dir, f"{image_name}_{i}_{j}.tif")
                    if np.count_nonzero(window_data) != 0 :
                        with rasterio.open(
                            image_output_path,
                            'w',
                            driver='GTiff',
                            height=window_data.shape[1],
                            width=window_data.shape[2],
                            count=3,
                            dtype='uint8',  
                            crs=src.crs,
                            transform=transform_window
                        ) as dst:
                            dst.write(window_data)
            
                    
                        labels_output_path = os.path.join(output_train_labels_dir, f"{image_name}_{i}_{j}.txt")
                        with open(labels_output_path, 'w') as f:
                            for label in yolo_labels:
                                # Ensure all values are within the [0, 1] range
                                assert all(0 <= v <= 1 for v in label[1:]), f"Label values out of range: {label}"
                                label_str = ' '.join(map(str, label))
                                f.write(label_str + '\n')
def preparing_train_data_pose(images,output_train_images_dir,output_train_labels_dir):

    """
    prepare yolo 500x500 8-bit normalized tiles and their corresponding labels with step size 400 for a pose estimation  tasks 
    label format <class-index> <bbox x center normalized > <bbox y center normalized > < bbox width normalized> <bbox height normalized> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
    Args :
         images -> images path list
         output_train_images_dir -> directory to save the training tiles 
         output_train_labels_dir ->directory to save the labels 
    
    """
    os.makedirs(output_train_images_dir, exist_ok=True)
    os.makedirs(output_train_labels_dir, exist_ok=True)
    window_size = 512
    step_size=400
  
    for image_path in images:
      
        image_name=(image_path.split("\\")[-1]).split('.tif')[0]
        shapefile_path=image_path.replace('.tif','.shp')
     
        with rasterio.open(image_path) as src:
            transform = src.transform
            width, height = src.width, src.height
            crs = src.crs
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
         
            image_data=src.read()
            image_data = image_data.astype(float)  
            
            image_data = (image_data  / 4095) * 255.0
            image_data = image_data.astype(np.uint8)
           
           
            for i in range(0, height, step_size):
                for j in range(0, width, step_size):
                    window_width = min(window_size, width - j)
                    window_height = min(window_size, height - i)
                    window = Window(j, i, window_width, window_height)
                    window_transform = src.window_transform(window)
                    window_bounds = rasterio.windows.bounds(window, transform)
            
                    window_data=image_data [:3,i:(i+window_height),j:(j+window_width)]
               
                    bbox = box(*window_bounds)
                    # print(bbox)
                    clipped = gdf.clip(bbox)
                    yolo_labels=[]
                
                    for _, row in clipped.iterrows():
                            if row.geometry.type == 'Polygon':
                                minx, miny, maxx, maxy = row.geometry.bounds
                                min_col, min_row = ~transform * (minx, miny)
                                max_col, max_row = ~transform * (maxx, maxy)

                            
                                min_col -= j
                                min_row -= i
                                max_col -= j
                                max_row -= i

                                x_center = (min_col + max_col) / 2 / window_width
                                y_center = (min_row + max_row) / 2 / window_height
                                bbox_width = (max_col - min_col) / window_width
                                bbox_height = ( min_row-max_row ) / window_height
                            


                            
                                xx= row.geometry.exterior.coords[:]
                                if(len(xx)!=5):
                                    
                                    continue
                                
                                points=[~transform * (xx[i]) for i in range(len(xx))]
                                
                                coords=[0,x_center,y_center,bbox_width,bbox_height]
                                points=arrange_points(points[:-1])
                                for x,y in points:
                                    normalized_x = max(0, min(1, (x - j) / window_width))
                                    normalized_y = max(0, min(1, (y - i) / window_height))

                                    coords.append(normalized_x)
                                    coords.append(normalized_y)
                                

                                yolo_labels.append(coords)
        
                    tile_label_path =os.path.join(output_train_labels_dir, f"{image_name}_{i}_{j}.txt")
                    if np.count_nonzero(window_data) != 0 :
                        with open(tile_label_path, 'w') as f:
                                for label in yolo_labels:
                                    
                                    assert all(0 <= v <= 1 for v in label[1:]), f"Label values out of range: {label}{tile_id}"
                                    label_str = ' '.join(map(str, label))
                                    f.write(label_str + '\n')   

                        tile_image_path = os.path.join(output_train_images_dir, f"{image_name}_{i}_{j}.tif")       
                        with rasterio.open(
                            tile_image_path,
                            "w",
                            driver="GTiff",
                            height=window_height,
                            width=window_width,
                            count=3,
                            dtype='uint8',
                            crs=src.crs,
                            transform=window_transform,
                            nodata=0
                        ) as dst:
                            dst.write(window_data)
def preparing_train_data_segment(images,output_train_images_dir,output_train_labels_dir):

    """
    
   
    prepare yolo 500x500 8-bit normalized tiles and their corresponding labels with step size 400 for a pose estimation  tasks 
    label format :<class-index> <x1> <y1> <x2> <y2> <x3> <y3> ... <pxn> <pyn>
    The length of each row does not have to be equal.
    Each segmentation label must have a minimum of 3 xy points: <class-index> <x1> <y1> <x2> <y2> <x3> <y3>
    Args :
         images -> images path list
         output_train_images_dir -> directory to save the training tiles 
         output_train_labels_dir ->directory to save the labels 
    """
    os.makedirs(output_train_images_dir, exist_ok=True)
    os.makedirs(output_train_labels_dir, exist_ok=True)
    window_size = 512
    step_size=400
  
    for image_path in images:
 
        image_name=(image_path.split("\\")[-1]).split('.tif')[0]
        shapefile_path=image_path.replace('.tif','.shp')
     
        with rasterio.open(image_path) as src:
            transform = src.transform
            width, height = src.width, src.height
            crs = src.crs
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
         
            image_data=src.read()
            image_data = image_data.astype(float)  
            
            image_data = (image_data  / 4095) * 255.0
            image_data = image_data.astype(np.uint8)
           
           
            for i in range(0, height, step_size):
                for j in range(0, width, step_size):
                    window_width = min(window_size, width - j)
                    window_height = min(window_size, height - i)
                    #window_width = width
                    #window_height = height
                    window = Window(j, i, window_width, window_height)
                    window_transform = src.window_transform(window)
                    window_bounds = rasterio.windows.bounds(window, transform)
            
                    window_data=image_data [0:3,i:(i+window_height),j:(j+window_width)]
                 
                    bbox = box(*window_bounds)
                    # print(bbox)
                    clipped = gdf.clip(bbox)
                    yolo_labels=[]
                
                    for _, row in clipped.iterrows():
                            if row.geometry.type == 'Polygon':
                            
                            
                                xx= row.geometry.exterior.coords[:]
                              
                                
                                points=[~transform * (xx[i]) for i in range(len(xx))]
                                
                                coords=[0]
                             
                                for x,y in points[:-1]:
                                    normalized_x = max(0, min(1, (x - j) / window_width))
                                    normalized_y = max(0, min(1, (y - i) / window_height))

                                    coords.append(normalized_x)
                                    coords.append(normalized_y)
                                

                                yolo_labels.append(coords)
        
                    tile_label_path =os.path.join(output_train_labels_dir, f"{image_name}_{i}_{j}.txt")
                    if np.count_nonzero(window_data) != 0 :
                       with open(tile_label_path, 'w') as f:
                            for label in yolo_labels:
                                
                                assert all(0 <= v <= 1 for v in label[1:]), f"Label values out of range: {label}{tile_id}"
                                label_str = ' '.join(map(str, label))
                                f.write(label_str + '\n')   

                       tile_image_path = os.path.join(output_train_images_dir, f"{image_name}_{i}_{j}.tif")       
                       with rasterio.open(
                        tile_image_path,
                        "w",
                        driver="GTiff",
                        height=window_height,
                        width=window_width,
                        count=3,
                        dtype='uint8',
                        crs=src.crs,
                        transform=window_transform,
                        nodata=0
                    ) as dst:
                        dst.write(window_data)
def preparing_train_data_obb (images,output_train_images_dir,output_train_labels_dir):
    """
    prepare yolo 500x500 8-bit normalized tiles and their corresponding labels with step size 400 for a detection tasks 
    label format (class id ,center x normalized , center y normalized , width normalized , height normalized)
    Args :
         images -> images path list
         output_train_images_dir -> directory to save the training tiles 
         output_train_labels_dir ->directory to save the labels 
    
    """
    os.makedirs(output_train_images_dir, exist_ok=True)
    os.makedirs(output_train_labels_dir, exist_ok=True)
    window_size = 512
    step_size=400
  
    for image_path in images:
      
        image_name=(image_path.split("\\")[-1]).split('.tif')[0]
        shapefile_path=image_path.replace('.tif','.shp')
        shapefile = gpd.read_file(shapefile_path)
    
        with rasterio.open(image_path) as src:
            raster_crs = src.crs
            transform = src.transform
            image_width = src.width
            image_height = src.height
            image_data=src.read()
            image_data = image_data.astype(float)  
            image_data/=2**12
            image_data*=255.0
            image_data = image_data.astype(np.uint8)# Convert to uint8
            # image_data_min = image_data.min()
            # image_data_max = image_data.max()
            # image_data = (image_data - image_data_min) / (image_data_max - image_data_min) * 255.0
            # image_data = image_data.astype(np.uint8)
            
            if shapefile.crs != raster_crs:
                shapefile = shapefile.to_crs(raster_crs)

           
            for i in range(0, image_height, step_size):
                for j in range(0, image_width, step_size):

                    window_width = min(window_size, image_width - j)
                    window_height = min(window_size, image_height - i)
                    window = Window(j, i, window_width, window_height)
                    transform_window = src.window_transform(window)
                    window_bounds = rasterio.windows.bounds(window, transform)
                    bbox = box(*window_bounds)
                    clipped = shapefile.clip(bbox)
                    yolo_labels = []

                    for _, row in clipped.iterrows():
                        if row.geometry.type == 'Polygon':
                           
                             
                             xx= row.geometry.exterior.coords[:]
                             if(len(xx)!=5):
                                 
                                   continue
                          
                             points=[~transform * (xx[z]) for z in range(4)]
                             coords=[0]
                             for x,y in points:
                                normalized_x = max(0, min(1, (x - j) / window_width))
                                normalized_y = max(0, min(1, (y - i) / window_height))

                                coords.append(normalized_x)
                                coords.append(normalized_y)
                             yolo_labels.append(coords)

                    window_data =image_data [:3,i:(i+window_height),j:(j+window_width)]
                    image_output_path = os.path.join(output_train_images_dir, f"{image_name}_{i}_{j}.tif")
                    if np.count_nonzero(window_data) != 0 :
                        with rasterio.open(
                            image_output_path,
                            'w',
                            driver='GTiff',
                            height=window_data.shape[1],
                            width=window_data.shape[2],
                            count=3,
                            dtype='uint8',  
                            crs=src.crs,
                            transform=transform_window
                        ) as dst:
                            dst.write(window_data)
            
                    
                        labels_output_path = os.path.join(output_train_labels_dir, f"{image_name}_{i}_{j}.txt")
                        with open(labels_output_path, 'w') as f:
                            for label in yolo_labels:
                                # Ensure all values are within the [0, 1] range
                                assert all(0 <= v <= 1 for v in label[1:]), f"Label values out of range: {label}"
                                label_str = ' '.join(map(str, label))
                                f.write(label_str + '\n')
def preparing_val_data(output_train_images_dir,output_train_labels_dir,output_val_images_dir,output_val_labels_dir):
    """
    take 20% of training data to be the validation data 
    """
    # output_train_images_dir=r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\FILimage"#->filtered output
    # output_train_labels_dir=r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\FILlabels"

    os.makedirs(output_val_images_dir, exist_ok=True)
    os.makedirs(output_val_labels_dir, exist_ok=True)
    image_files = [f for f in os.listdir(output_train_images_dir) if f.endswith('.tif')]
    val_size = int(0.2 * len(image_files))
    val_images = random.sample(image_files, val_size)
    for image in val_images:
        
        image_path = os.path.join(output_train_images_dir, image)
        shutil.move(image_path, os.path.join(output_val_images_dir, image))
        label_file = image.replace('.tif', '.txt')
        label_path = os.path.join(output_train_labels_dir, label_file)
        
        if os.path.exists(label_path): 
            shutil.move(label_path, os.path.join(output_val_labels_dir, label_file))
if __name__ == "__main__":
    # Paths after data preparation
    # output_dir = r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\output3"
    # train_images = os.path.join(output_dir, "train", "images")
    # val_images   = os.path.join(output_dir, "val", "images")
    preparing_val_data(r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\train\images",r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\train\labels",r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\val\images",r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\val\labels")


