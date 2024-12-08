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
    # point_quat = trajectory[:, 3:]
    # # Apply quaternion rotation to the trajectory points inversely
    # rotation = Rotation.from_quat(point_quat)
    # rotation = rotation.inv()
    # point_xyz = rotation.apply(point_xyz)

    # # Extract the x, y, and z components of the trajectory
    x = point_xyz[:, 0]
    y = point_xyz[:, 1]
    z = point_xyz[:, 2]
    
    return x, y, z

def compute_transformation(source_points, target_points):
    """
    Compute the rigid transformation (rotation and translation) to align source points to target points.
    
    Args:
        source_points (numpy.ndarray): Source point cloud of shape (N, 3).
        target_points (numpy.ndarray): Target point cloud of shape (N, 3).

    Returns:
        R (numpy.ndarray): Rotation matrix of shape (3, 3).
        t (numpy.ndarray): Translation vector of shape (3,).
        transformed_points (numpy.ndarray): Transformed source points aligned to target points.
    """
    # Ensure the input shapes match
    assert source_points.shape == target_points.shape, "Source and target must have the same shape."

    # Compute the centroids of both point sets
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Center the points by subtracting the centroids
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Compute the cross-covariance matrix
    H = np.dot(centered_source.T, centered_target)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)  # Rotation matrix

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    t = centroid_target - np.dot(R, centroid_source)

    # Transform the source points
    transformed_points = np.dot(source_points, R.T) + t

    return transformed_points

def calculate_dis(pt1,pt2):
    return np.linalg.norm(pt1-pt2)


SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5",
    "market_6", "shaman_3", "sleeping_1", "sleeping_2",
    "temple_2", "temple_3"
]
columns = ["frame_No", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]

scaler = {}
# view = {"temple_2": [22, -136], "sleeping_1": [22, 136]}

for scene in SCENES:
    print(scene)
    sintel_gt_file = f"/home/boxun/work/Project/CV2024_Object_detection/leapvo/data/MPI-Sintel-complete/training/camdata_left/{scene}"
    traj_gt, timestamps_mat = load_sintel_traj(sintel_gt_file)
    traj_gt_xyz = traj_gt[:, :3]  # Extract ground truth translation (x, y, z)
    print(calculate_dis(traj_gt_xyz[0], traj_gt_xyz[-1]))

    sintel_traj_file_masked = f"/home/boxun/work/Project/CV2024_Object_detection/leapvo/logs/sintel_masked/sintel-{scene}/leapvo_traj.txt"
    sintel_traj_file_unmasked = f"/home/boxun/work/Project/CV2024_Object_detection/leapvo/logs/sintel/sintel-{scene}/leapvo_traj.txt"
    trajectory_data_masked = read_traj_file(sintel_traj_file_masked)
    trajectory_data_unmasked = read_traj_file(sintel_traj_file_unmasked)

    scaler[scene] = 1
    scaler[scene] = calculate_dis(traj_gt_xyz[0], traj_gt_xyz[-1]) / calculate_dis(trajectory_data_masked[0, :3], trajectory_data_masked[-1, :3])

    masked_xyz = trajectory_data_masked[:, :3]*scaler[scene]  # Extract masked translation (x, y, z)
    unmasked_xyz = trajectory_data_unmasked[:, :3]*scaler[scene]  # Extract unmasked translation (x, y, z)
    print(calculate_dis(masked_xyz[0], masked_xyz[-1]))
    print(calculate_dis(unmasked_xyz[0], unmasked_xyz[-1]))

    

    # Compute transformation (align GT to masked)
    transformed_gt = compute_transformation(traj_gt_xyz, masked_xyz)

    # Extract x, y, z for visualization
    x_gt, y_gt, z_gt = transformed_gt[:, 0], transformed_gt[:, 1], transformed_gt[:, 2]
    x_masked, y_masked, z_masked = masked_xyz[:, 0], masked_xyz[:, 1], masked_xyz[:, 2]
    x_unmasked, y_unmasked, z_unmasked = unmasked_xyz[:, 0], unmasked_xyz[:, 1], unmasked_xyz[:, 2]

    # Create a 3D plot of the trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the trajectories
    ax.plot(x_gt, y_gt, z_gt, label="Ground Truth", color="blue", linewidth=2)
    ax.plot(x_masked, y_masked, z_masked, label="Masked", color="red", linewidth=2)
    ax.plot(x_unmasked, y_unmasked, z_unmasked, label="Unmasked", color="green", linewidth=2)

    # # Mark start and end points for both
    # ax.scatter(x_gt[0], y_gt[0], z_gt[0], color="green", label="Start (GT)", s=50)
    # ax.scatter(x_gt[-1], y_gt[-1], z_gt[-1], color="red", label="End (GT)", s=50)
    # ax.scatter(x_masked[0], y_masked[0], z_masked[0], color="green", label="Start (Masked)", s=50)
    # ax.scatter(x_masked[-1], y_masked[-1], z_masked[-1], color="red", label="End (Masked)", s=50)
    # ax.scatter(x_unmasked[0], y_unmasked[0], z_unmasked[0], color="green", label="Start (Unmasked)", s=50)
    # ax.scatter(x_unmasked[-1], y_unmasked[-1], z_unmasked[-1], color="red", label="End (Unmasked)", s=50)

    # Add labels and legend
    ax.set_title(f"Aligned 3D Trajectory - {scene}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    # ax.view_init(elev=view[scene][0], azim=view[scene][1])
    # Show the plot
plt.show()

    
