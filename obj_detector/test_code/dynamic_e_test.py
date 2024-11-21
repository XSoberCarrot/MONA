import json
import os
import numpy as np
import cv2

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
first_frame = cv2.imread(frame_files[0])

for i in range(len(dynamic_points_data)):
    dynamic_points = dynamic_points_data[i]
    frame_file = frame_files[i]
    print(f"Processing frame {frame_file}")
    frame = cv2.imread(frame_file)

    # Plot the dynamic points on the frame
    if len(dynamic_points) > 0:
        for point in dynamic_points:
            x, y = point
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
    # Display the frame with dynamic points
    cv2.imshow('Frame with Dynamic Points', frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


print("Finished")