import os
import subprocess
from datetime import datetime

# Define dataset and save directory
DATASET = "./data/MPI-Sintel-complete/training"
SAVEDIR = "logs/sintel_masked"

# Create the save directory if it doesn't exist
os.makedirs(SAVEDIR, exist_ok=True)

# Log the current date and time into the error summary file
with open(os.path.join(SAVEDIR, "error_sum.txt"), "a") as error_file:
    error_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

# List of scenes to process
SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5",
    "market_6", "shaman_3", "sleeping_1", "sleeping_2",
    "temple_2", "temple_3"
]

# Iterate through each scene and execute the Python script with parameters
for scene in SCENES:
    SCENE_PATH = os.path.join(DATASET, "masked", scene)
    command = [
        "python", "main/eval.py",
        "--config-path=../configs",
        "--config-name=sintel",
        f"data.imagedir={SCENE_PATH}",
        f"data.gt_traj={os.path.join(DATASET, 'camdata_left', scene)}",
        f"data.savedir={SAVEDIR}",
        f"data.calib={os.path.join(DATASET, 'camdata_left', scene)}",
        f"data.name=sintel-{scene}",
        "save_video=true"
    ]
    
    # Run the command as a subprocess
    subprocess.run(command, check=True)
