import json
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt 
from segment_anything import SamPredictor, sam_model_registry

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

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
sam_checkpoint = "/home/boxun/work/Project/CV2024_Object_detection/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


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

# Path to the directory containing the video frames
frames_dir = "/home/boxun/work/Project/CV2024_Object_detection/leapvo/data/samples/sintel_market_5/frames"
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
first_frame = cv2.imread(frame_files[25])

# for i in range(len(dynamic_points_data)):
#     dynamic_points = dynamic_points_data[i]
#     frame_file = frame_files[i]
#     print(f"Processing frame {frame_file}")
#     frame = cv2.imread(frame_file)

#     # Plot the dynamic points on the frame
#     if len(dynamic_points) > 0:
#         for point in dynamic_points:
#             x, y = point
#             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
#     # Display the frame with dynamic points
#     cv2.imshow('Frame with Dynamic Points', frame)
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
predictor.set_image(first_frame)
input_points = np.array(dynamic_points_data[25])
input_labels = np.ones(input_points.shape[0])
# plt.figure(figsize=(10,10))
# plt.imshow(first_frame)
# show_points(input_points, input_label, plt.gca())
# plt.axis('on')
# plt.show()  
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)
best_mask = masks[np.argmax(scores)]
masked_image = mask_img_overlay(first_frame, best_mask, input_points)
masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
cv2.imshow('Frame with Dynamic Points', masked_image)
cv2.waitKey(0)

# plt.figure(figsize=(10,10))
# plt.imshow(first_frame)
# show_mask(best_mask, plt.gca())
# show_points(input_points, input_labels, plt.gca())
# plt.title(f"Best Mask, Score: {np.max(scores):.3f}", fontsize=18)
# plt.axis('off')
# plt.show()  
# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(first_frame)
#     show_mask(mask, plt.gca())
#     show_points(input_points, input_labels, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()  

print("First frame")