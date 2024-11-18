import json
import os
import numpy as np
import cv2

# Path to the JSON file containing dynamic points
json_file_path = "/home/boxun/work/Project/CV2024_Object_detection/leapvo/logs/sintel_market_5/sintel_market_5/dynamic_points.json"

# Path to the directory containing the video frames
frames_dir = "/home/boxun/work/Project/CV2024_Object_detection/leapvo/data/samples/sintel_market_5/frames"

# Path to save the output video
output_video_path = "output_video_with_dynamic_points.mp4"

# Load the dynamic points from the JSON file
with open(json_file_path, "r") as f:
    dynamic_points_data = json.load(f)

# Get the list of frame files
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])

# Read the first frame to get the frame size
first_frame = cv2.imread(frame_files[0])
frame_height, frame_width, _ = first_frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

# Iterate over the frames and plot the dynamic points
idx = 0
for frame_file, dynamic_points in zip(frame_files, dynamic_points_data):
    print(f"Processing frame {idx}")
    frame = cv2.imread(frame_file)
    frame_id = dynamic_points["frame_id"]
    points = dynamic_points["dynamic_points"]

    # Plot the dynamic points on the frame
    for point in points:
        # x, y = point
        # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        for pt in point:
            x, y = pt
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red color for dynamic points

    # Display the frame with dynamic points
    cv2.imshow('Frame with Dynamic Points', frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    idx += 1
    # Write the frame to the output video
    # out.write(frame)

# Release the VideoWriter object
# out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# print(f"Output video saved to {output_video_path}")