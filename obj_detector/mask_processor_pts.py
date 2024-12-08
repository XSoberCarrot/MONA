import json
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

def point_in_box_count(points, box):
    points = np.array(points)
    x_within = (points[:, 0] > box[0]) & (points[:, 0] < box[2])
    y_within = (points[:, 1] > box[1]) & (points[:, 1] < box[3])
    pt_num_in_box = np.sum(x_within & y_within)
    return pt_num_in_box

def yolo_boxes_extractor(yolo_result, points, threshold_npt=1):
    boxes_raw = yolo_result.boxes.xyxy.cpu().numpy().tolist()
    boxes = np.array(boxes_raw)
    boxes_filtered = []

    if len(boxes) == 0:
        return boxes_filtered
    # Calculate the area of each box
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    min_area = np.min(box_areas)
    
    for box, area in zip(boxes, box_areas):
        # Adjust the threshold based on the area of the box
        adjusted_threshold = threshold_npt * (area / min_area)
        pt_num_in_box = point_in_box_count(points, box)
        if pt_num_in_box >= adjusted_threshold:
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


def mask_img_overlay(img, mask, points=None, alpha=0.5):
    mask_colored = np.zeros_like(img)
    mask_colored[mask == 1] = np.array([0, 255, 0])
    
    # Create an overlay image
    overlay_img = img.copy()
    mask_indices = mask_colored[:, :, 1] == 255  # Check where the mask is applied
    if np.any(mask_indices):
        overlay_img[mask_indices] = cv2.addWeighted(img[mask_indices], 1 - alpha, mask_colored[mask_indices], alpha, 0)
    
    return overlay_img

# SAM settings
sam_checkpoint = "/home/boxun/work/Project/CV2024_Object_detection/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5",
    "market_6", "shaman_3", "sleeping_1", "sleeping_2",
    "temple_2", "temple_3"
]
DATASET = "./data/MPI-Sintel-complete/training"
SAVEDIR = "./logs/sintel"

for scene in tqdm(SCENES):
    # Path to the directory containing the video frames
    frames_dir = os.path.join(DATASET, "final", scene)
    # Path to the JSON file containing dynamic points
    json_file_dir = os.path.join(SAVEDIR, f"sintel-{scene}", "dynamic_points")
    output_frames_dir = os.path.join(DATASET, "pure_pts", scene)
    os.makedirs(output_frames_dir, exist_ok=True)

    # Load the dynamic points data
    if not os.path.exists(json_file_dir):
        print("Error: JSON file does not exist")
        exit()
    json_files = []
    dynamic_points_data = []
    for i in range(len(os.listdir(json_file_dir))):
        json_files.append(f"{json_file_dir}/frame_{i}.json")
        with open(json_files[i], "r") as f:
            dynamic_points_data.append(json.load(f))
    # Load the original video frames
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
    first_frame = cv2.imread(frame_files[1])
    frame_height, frame_width, _ = first_frame.shape

    # Masked the dynamic objects
    for i in range(len(dynamic_points_data)):
        dynamic_points = dynamic_points_data[i]
        frame_file = frame_files[i]
        # print(f"Processing frame {frame_file}")
        frame = cv2.imread(frame_file)

        # Plot the dynamic points on the frame
        if len(dynamic_points) > 0:
            predictor.set_image(frame)
            input_points = np.array(dynamic_points)
            input_labels = np.ones(input_points.shape[0])
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            best_mask = masks[np.argmax(scores)]
            masked_image = mask_img_overlay(frame, best_mask, input_points)
        else:
            masked_image = frame
            
        # Save the frame with dynamic points
        output_frame_path = os.path.join(output_frames_dir, f"frame_{i+1:04d}.png")
        cv2.imwrite(output_frame_path, masked_image)

    print(f"Completed processing scene {scene}")