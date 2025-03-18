import os
import subprocess
from datetime import datetime

# Define dataset and save directory
DATASET = "./data/TartanAir_shibuya"
SAVEDIR = "logs/shibuya"

# Create the save directory if it doesn't exist
os.makedirs(SAVEDIR, exist_ok=True)

# Log the current date and time into the error summary file
with open(os.path.join(SAVEDIR, "error_sum.txt"), "a") as error_file:
    error_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

# List of scenes to process
SCENES = [
    "Standing01", "Standing02",
    "RoadCrossing03", "RoadCrossing04",
    "RoadCrossing05", "RoadCrossing06", "RoadCrossing07"
]

# Iterate through each scene and execute the Python script with parameters
for scene in SCENES:
    SCENES_PATH = os.path.join(DATASET, scene, "image_0")
    command = [
        "python", "main/eval.py",
        "--config-path=../configs",
        "--config-name=shibuya",
        f"data.imagedir={SCENES_PATH}",
        f"data.gt_traj={os.path.join(DATASET, scene, 'gt_pose.txt')}",
        f"data.savedir={SAVEDIR}",
        "data.calib=calibs/tartan_shibuya.txt",
        f"data.name=replica-{scene}",
        "save_video=true"
    ]
    
    # Run the command as a subprocess
    subprocess.run(command, check=True)
