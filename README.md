# SAT--Building--Segmentation
A YOLOv11-based deep learning model for building segmentation from satellite imagery, producing pixel-level masks instead of bounding boxes for accurate building footprint extraction.

Overview

This project implements a deep learning–based building segmentation system for satellite imagery.
The model is designed to extract precise building footprints by generating pixel-level segmentation masks, rather than traditional bounding-box detections.

A YOLOv11 segmentation model was pretrained and fine-tuned on satellite data, with extensive preprocessing to handle variations in scale, illumination, and urban density.
This approach is suitable for large-scale geospatial applications where accurate building boundaries are required.

Key Features

YOLOv11 Segmentation Model
Optimized for satellite imagery
Pixel-level building masks (not bounding boxes)
Pretrained weights with transfer learning
Robust performance across dense and sparse urban areas
Fast and scalable inference

Project Pipeline

1.Data Collection
2.High-resolution satellite images
3.Preprocessing
4.Image resizing and normalization
5.Noise reduction and spatial enhancement
6.Model Training
7.Transfer learning using pretrained YOLOv11
8.Fine-tuning for building segmentation
9.Inference
Input satellite image
Output segmentation mask highlighting buildings
