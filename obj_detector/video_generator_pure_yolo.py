import json
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt 
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
from tqdm import tqdm

def point_in_box_count(points, box):
    points = np.array(points)
    x_within = (points[:, 0] > box[0]) & (points[:, 0] < box[2])
    y_within = (points[:, 1] > box[1]) & (points[:, 1] < box[3])
    pt_num_in_box = np.sum(x_within & y_within)
    return pt_num_in_box

def yolo_boxes_extractor(yolo_result, points, threshold_npt=0):
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

def mask_img_overlay_tensor(img, mask, points=None, boxes = None, alpha=0.5):
    masks = mask.cpu().numpy().astype(np.uint8)
    mask_colored = np.zeros_like(img)
    # if len(masks) >1:
    #     print("More than one mask")
    for i in range(len(masks)):
        mask = masks[i].squeeze(0)
        mask_colored[mask == 1] = np.array([0, 255, 0])
    overlap_img = cv2.addWeighted(img, 1-alpha, mask_colored, alpha, 0)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlap_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return overlap_img


# SAM settings
sam_checkpoint = f"{os.path.dirname(__file__)}/../weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
# Initialize YOLO11
weights_path = f"{os.path.dirname(__file__)}/../weights/yolo11x.pt"
yolo = YOLO(weights_path)

DATASET = "/data/MPI-Sintel-complete/training"
SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5",
    "market_6", "shaman_3", "sleeping_1", "sleeping_2",
    "temple_2", "temple_3"
]
output_video_dir = f"{os.path.dirname(__file__)}/../result_videos/pure_boxs"
os.makedirs(output_video_dir, exist_ok=True)

for scene in SCENES:

    json_file_dir = f"{os.path.dirname(__file__)}/../logs/sintel_pure_pts/sintel-{scene}/dynamic_points"
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
    # Path to the directory containing the video frames
    output_video_path = f"{output_video_dir}/{scene}.mp4"
    print(f"Generating video for scene {scene} at {output_video_path}")
    frames_dir = f"{os.path.dirname(__file__)}/..{DATASET}/final/{scene}"
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
    first_frame = cv2.imread(frame_files[1])
    frame_height, frame_width, _ = first_frame.shape

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

    for i in tqdm(range(len(dynamic_points_data))):
        dynamic_points = dynamic_points_data[i]
        frame_file = frame_files[i]
        frame = cv2.imread(frame_file)

        # Plot the dynamic points on the frame
        if len(dynamic_points) > 0:
            frame_tensor = torch.tensor(frame).to(device=device)
            
            # Apply YOLO11 to the frame, extract bounding boxes, and filter dynamic points
            yolo_results = yolo(frame, verbose=False)
            yolo_boxes = yolo_boxes_extractor(yolo_results[0], dynamic_points)
            if len(yolo_boxes) > 0:
                filtered_points = points_filter(dynamic_points, yolo_boxes)
                predictor.set_image(frame)
                input_points = torch.tensor(filtered_points).to(device=device)
                input_labels = torch.ones(len(filtered_points)).to(device=device)
                input_boxes = torch.tensor(yolo_boxes).to(device=device)
                
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                masked_image = mask_img_overlay_tensor(frame, masks, filtered_points, yolo_boxes)
            else:
                masked_image = frame

        else:
            masked_image = frame
            
        # Display the frame with dynamic points
        cv2.imshow(f'Frame with mask', masked_image)
        cv2.waitKey(1)
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        out.write(masked_image)

    # Release the VideoWriter object
    out.release()
