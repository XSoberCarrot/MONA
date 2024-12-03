import json
import os
import cv2


# Path to the directory containing the video frames
output_video_path = "market_4_ori.mp4"
frames_dir = "/home/boxun/work/Project/CV2024_Object_detection/leapvo/data/samples/sintel_market_5/market_4"
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
first_frame = cv2.imread(frame_files[1])
frame_height, frame_width, _ = first_frame.shape

# Initialize the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

for i in range(len(frame_files)):
    frame_file = frame_files[i]
    print(f"Processing frame {frame_file}")
    frame = cv2.imread(frame_file)

    out.write(frame)

# Release the VideoWriter object
out.release()



print("First frame")