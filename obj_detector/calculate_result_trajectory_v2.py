import re
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sintel_cam_read(filename):
    """Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    TAG_FLOAT = 202021.25

    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    M = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
    N = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
    return M, N

def load_sintel_traj(gt_file):
    """Read trajectory format. Return in TUM-RGBD format.
    Returns:
        traj_tum (N, 7): camera to world poses in (x,y,z,qx,qy,qz,qw)
        timestamps_mat (N, 1): timestamps
    """
    # Refer to ParticleSfM
    gt_pose_lists = sorted(os.listdir(gt_file))
    gt_pose_lists = [os.path.join(gt_file, x) for x in gt_pose_lists]
    tstamps = [float(x.split("/")[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [sintel_cam_read(f)[1] for f in gt_pose_lists]
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]])], 0)
        gt_pose_inv = np.linalg.inv(gt_pose)  # world2cam -> cam2world
        xyz = gt_pose_inv[:3, -1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose_inv[:3, :3])
        xyzw = R.as_quat()  # scalar-last for scipy
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
        wxyzs.append(wxyz)
        tum_gt_pose = np.concatenate([xyz, wxyz], 0)
        tum_gt_poses.append(tum_gt_pose)

    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:, :3] = tum_gt_poses[:, :3] - np.mean(
        tum_gt_poses[:, :3], 0, keepdims=True
    )
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    return tum_gt_poses, tt

def read_traj_file(file_path):
    columns = ["frame_No", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
    # Read the file into a pandas DataFrame
    trajectory_data = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    # Get rid of the frame_No column
    trajectory_data = trajectory_data.drop(columns=["frame_No"])
    # Convert the DataFrame to a numpy array
    trajectory_data = trajectory_data.to_numpy()

    return trajectory_data

def extract_trajectory(trajectory):
    point_xyz = trajectory[:, :3]

    # Extract the x, y, and z components of the trajectory
    x = point_xyz[:, 0]
    y = point_xyz[:, 1]
    z = point_xyz[:, 2]
    
    return x, y, z

def calculate_dis(pt1,pt2):
    return np.linalg.norm(pt1-pt2)


def offset_calculator(trajectory_poimnts):
    """Calculate the trajectory that relative to the first point"""
    offset_trajectory = trajectory_poimnts - trajectory_poimnts[0]
    return offset_trajectory



def trajectory_alignment(trajectory_gt, trajectory_pred_offset):
    """Align the predicted trajectory origin to the ground truth and translate the predicted trajectory using the offset trajectory"""
    # Move the predicted trajectory to the origin of the ground truth trajectory
    trajectory_aligned = trajectory_pred_offset + trajectory_gt[0]
    # Compute the rotation to align the predicted trajectory to the ground truth using pseudo-inverse
    A = trajectory_aligned.T
    B = trajectory_gt.T
    R = np.linalg.inv(A @ A.T) @ A @ B.T
    R = R.T
    # Apply the rotation to the predicted trajectory
    trajectory_aligned = (R @ trajectory_aligned.T).T

    return trajectory_aligned


SCENES = [
    "ambush_6","temple_3"
]
columns = ["frame_No", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]

scaler = {}

for scene in SCENES:
    print(scene)
    sintel_gt_file = f"{os.path.dirname(__file__)}/../data/MPI-Sintel-complete/training/camdata_left/{scene}"
    sintel_traj_file_masked = f"{os.path.dirname(__file__)}/../logs/sintel_masked/sintel-{scene}/leapvo_traj.txt"
    sintel_traj_file_unmasked = f"{os.path.dirname(__file__)}/../logs/sintel/sintel-{scene}/leapvo_traj.txt"

    traj_gt, timestamps_mat = load_sintel_traj(sintel_gt_file)
    traj_data_masked = read_traj_file(sintel_traj_file_masked)
    traj_data_unmasked = read_traj_file(sintel_traj_file_unmasked)
    traj_gt_xyz = traj_gt[:, :3]  # Extract ground truth translation (x, y, z)
    masked_xyz = traj_data_masked[:, :3]
    unmasked_xyz = traj_data_unmasked[:, :3]

    scaler[scene] = 1
    scaler[scene] = calculate_dis(traj_gt_xyz[0], traj_gt_xyz[-1]) / calculate_dis(masked_xyz[0], masked_xyz[-1])

    # Get the offset trajectory 
    traj_masked_offset = offset_calculator(masked_xyz)*scaler[scene]
    traj_unmasked_offset = offset_calculator(unmasked_xyz)*scaler[scene]

    # Move gt to the origin
    traj_gt_offset = offset_calculator(traj_gt_xyz)

    # Align the trajectory to the ground truth
    traj_masked_aligned = trajectory_alignment(traj_gt_offset, traj_masked_offset)
    traj_unmasked_aligned = trajectory_alignment(traj_gt_offset, traj_unmasked_offset)

    # Extract x, y, z for visualization
    x_gt, y_gt, z_gt = traj_gt_offset[:, 0], traj_gt_offset[:, 1], traj_gt_offset[:, 2]
    x_masked, y_masked, z_masked = traj_masked_aligned[:, 0], traj_masked_aligned[:, 1], traj_masked_aligned[:, 2]
    x_unmasked, y_unmasked, z_unmasked = traj_unmasked_aligned[:, 0], traj_unmasked_aligned[:, 1], traj_unmasked_aligned[:, 2]

    # Create a 3D plot of the trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the trajectories
    ax.plot(x_gt, y_gt, z_gt, label="Ground Truth", color="blue", linewidth=2)
    ax.plot(x_masked, y_masked, z_masked, label="Masked", color="red", linewidth=2)
    ax.plot(x_unmasked, y_unmasked, z_unmasked, label="Unmasked", color="green", linewidth=2)

    # Add labels and legend
    ax.set_title(f"Aligned 3D Trajectory - {scene}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    # ax.view_init(elev=view[scene][0], azim=view[scene][1])
    # Show the plot
plt.show()

    
