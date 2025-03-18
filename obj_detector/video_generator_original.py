import os
import cv2
from tqdm import tqdm


# Path to the directory containing the video frames
DATASET = "/data/MPI-Sintel-complete/training"
SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5",
    "market_6", "shaman_3", "sleeping_1", "sleeping_2",
    "temple_2", "temple_3"
]

output_video_dir = f"{os.path.dirname(__file__)}/../result_videos/original"
os.makedirs(output_video_dir, exist_ok=True)

for scene in SCENES:

    output_video_path = f"{output_video_dir}/{scene}.mp4"
    print(f"Generating video for scene {scene} at {output_video_path}")
    frames_dir = f"{os.path.dirname(__file__)}/..{DATASET}/final/{scene}"
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
    first_frame = cv2.imread(frame_files[1])
    frame_height, frame_width, _ = first_frame.shape

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

    for i in range(len(frame_files)):
        frame_file = frame_files[i]
        # print(f"Processing frame {frame_file}")
        frame = cv2.imread(frame_file)

        out.write(frame)

    # Release the VideoWriter object
    out.release()