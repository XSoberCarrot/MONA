import json
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt 
from ultralytics import YOLO


def point_in_box_count(points, box):
    points = np.array(points)
    x_within = (points[:, 0] > box[0]) & (points[:, 0] < box[2])
    y_within = (points[:, 1] > box[1]) & (points[:, 1] < box[3])
    pt_num_in_box = np.sum(x_within & y_within)
    return pt_num_in_box

def yolo_boxes_extractor(yolo_result, points, threshold_npt=3):
    boxes_raw = yolo_result.boxes.xyxy.cpu().numpy().tolist()
    boxes = np.array(boxes_raw)
    boxes_filtered = []
    for box in boxes:
        pt_num_in_box = point_in_box_count(points, box)
        if pt_num_in_box >= threshold_npt:
            boxes_filtered.append(box)
    boxes_filtered = np.array(boxes_filtered)
    return boxes_filtered

def points_filter(points, boxes):
    points_filtered = np.array([])
    points = np.array(points)
    for box in boxes:
        x_within = (points[:, 0] > box[0]) & (points[:, 0] < box[2])
        y_within = (points[:, 1] > box[1]) & (points[:, 1] < box[3])
        pt_in_box = points[x_within & y_within]
        if len(pt_in_box) > 0:
            if len(points_filtered) == 0:
                points_filtered = pt_in_box
            else:
                points_filtered = np.vstack((points_filtered, pt_in_box))
    return points_filtered

# Path to the JSON file containing dynamic points
json_file_dir = f"{os.getcwd()}/logs/sintel_market_5/sintel_market_5/dynamic_points"
if not os.path.exists(json_file_dir):
    # Show error message
    print("Error: JSON file does not exist")
    exit()
json_files = []
dynamic_points_data = []
for i in range(len(os.listdir(json_file_dir))):
    json_files.append(f"{json_file_dir}/frame_{i}.json")
    with open(json_files[i], "r") as f:
        dynamic_points_data.append(json.load(f))

test_dynamic_points = dynamic_points_data[25]

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
yolo_boxes = yolo_boxes_extractor(yolo_results[0], test_dynamic_points)
filtered_points = points_filter(test_dynamic_points,yolo_boxes)

# Visualize the results.

# Plot the dynamic points on the frame
if len(filtered_points) > 0:
    # Plot the filtered dynamic points
    for pt in filtered_points:
        test_frame = cv2.circle(test_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    # Plot the bounding boxes
    for box in yolo_boxes:
        test_frame = cv2.rectangle(test_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    cv2.imshow("Frame", test_frame)
    cv2.waitKey(0)


print("First frame")