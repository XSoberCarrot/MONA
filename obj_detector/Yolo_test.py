import json
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt 
from ultralytics import YOLO
# Initialize YOLO11
weights_path = "/home/boxun/work/Project/CV2024_Object_detection/yolo11x.pt"
yolo = YOLO(weights_path)

# Path to the directory containing the video frames
output_video_path = "output_video_with_dynamic_mask.mp4"
frames_dir = "/home/boxun/work/Project/CV2024_Object_detection/leapvo/data/samples/sintel_market_5/frames"
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
test_frame = cv2.imread(frame_files[25])

yolo_results = yolo(test_frame)
yolo_results[0].show()

print("First frame")