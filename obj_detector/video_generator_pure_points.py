import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt 
from segment_anything import SamPredictor, sam_model_registry

def mask_img_overlay(img, mask, points=None, alpha=0.5):
    mask_colored = np.zeros_like(img)
    mask_colored[mask] = np.array([0, 255, 0])
    overlap_img = cv2.addWeighted(img, 1-alpha, mask_colored, alpha, 0)
    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(overlap_img, (x, y), 5, (0, 0, 255), -1)
    return overlap_img

# SAM settings
sam_checkpoint = f"{os.path.dirname(__file__)}/../weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Path to the JSON file containing dynamic points
DATASET = "/data/MPI-Sintel-complete/training"
SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5",
    "market_6", "shaman_3", "sleeping_1", "sleeping_2",
    "temple_2", "temple_3"
]
output_video_dir = f"{os.path.dirname(__file__)}/../result_videos/pure_points"
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

    output_video_path = f"{output_video_dir}/{scene}.mp4"
    print(f"Generating video for scene {scene} at {output_video_path}")
    frames_dir = f"{os.path.dirname(__file__)}/..{DATASET}/final/{scene}"
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
    first_frame = cv2.imread(frame_files[1])
    frame_height, frame_width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

    for i in tqdm(range(len(dynamic_points_data))):
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
            
        # Display the frame with dynamic points
        cv2.imshow(f'Frame with mask', masked_image)
        cv2.waitKey(1)
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        out.write(masked_image)

    # Release the VideoWriter object
    out.release()
