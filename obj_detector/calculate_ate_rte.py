import math
import os

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

# from main.leapvo import LEAPVO
from leapvo import LEAPVO
from main.stream import dataset_stream, sintel_stream, video_stream
from main.utils import (eval_metrics, load_traj, plot_trajectory,
                        save_trajectory_tum_format, update_timestamps)

import os.path as osp
from tqdm import tqdm
import pickle
# from lib.utils import transforms
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def w2c_matrices_to_tensor(w2c_matrices):
    # Ensure the input is (N, 4, 4)
    assert len(w2c_matrices.shape) == 3 and w2c_matrices.shape[1:] == (4, 4), "Input tensor must be of shape (N, 4, 4)"
    
    N = w2c_matrices.shape[0]
    result = np.zeros((N, 7))
    
    for i in range(N):
        w2c_matrix = w2c_matrices[i]
        
        # Extract the rotation matrix (top-left 3x3)
        rotation_matrix = w2c_matrix[:3, :3]

        # Extract the translation vector (first three elements of the last column)
        translation_vector = w2c_matrix[:3, 3]

        # Convert rotation matrix to quaternion
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        # Combine translation and quaternion into a (1, 7) tensor
        result[i] = np.hstack((translation_vector, quaternion))

    return result

EMDB = "EMDB_2"

EMDB_source_dataroot = "/scratch4/haofrankyang/thrust_2/gwu/Datasets/EMDB"
save_root = f"/scratch4/haofrankyang/thrust_2/gwu/Code/leapvo/outputs/trajectories_plotted/{EMDB}_100_t2"
os.makedirs(save_root, exist_ok=True)

if EMDB == "EMDB_1":
    gvhmr_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/GVHMR/outputs/EMDB_1_ori"
    masked_gvhmr_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/GVHMR/outputs/EMDB_1"
    leapvo_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/leapvo/logs/EMDB_1_ori"
    masked_leapvo_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/leapvo/logs/EMDB_1"
else:
    gvhmr_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/GVHMR/outputs/EMDB_2_ori"
    masked_gvhmr_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/GVHMR/outputs/EMDB_2"
    leapvo_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/leapvo/logs/EMDB_2_ori"
    masked_leapvo_video_folder = "/scratch4/haofrankyang/thrust_2/gwu/Code/leapvo/logs/EMDB_2"

gvhmr_sum = {}
gvhmr_sum['ate'] = []
gvhmr_sum['rpe_trans'] = []
gvhmr_sum['rpe_rot'] = []
masked_gvhmr_sum = {}
masked_gvhmr_sum['ate'] = []
masked_gvhmr_sum['rpe_trans'] = []
masked_gvhmr_sum['rpe_rot'] = []
leapvo_sum = {}
leapvo_sum['ate'] = []
leapvo_sum['rpe_trans'] = []
leapvo_sum['rpe_rot'] = []
masked_leapvo_sum = {}
masked_leapvo_sum['ate'] = []
masked_leapvo_sum['rpe_trans'] = []
masked_leapvo_sum['rpe_rot'] = []

video_counter = 0
frames_counter = 0

for video in sorted(os.listdir(gvhmr_video_folder)):
    print(video)
    # if video == "P6_51_outdoor_dancing":
    #     continue

    if video == "P5_40_indoor_walk_big_circle":
        continue
    clip_name = video
    emdb_parts = video.split('_', 1)
    emdb_folder_name = emdb_parts[0]
    emdb_video_name = emdb_parts[1] if len(emdb_parts) > 1 else ''
    # print(f"emdb_folder_name: {emdb_folder_name}, emdb_video_name: {emdb_video_name}")
    emdb_pickle_path = osp.join(EMDB_source_dataroot, emdb_folder_name, emdb_video_name, f"{video}_data.pkl")
    gt_motion = pickle.load(open(emdb_pickle_path, "rb"))

    leapvo_file_path = osp.join(leapvo_video_folder, video, 'leapvo_traj.txt')
    masked_leapvo_file_path = osp.join(masked_leapvo_video_folder, video, 'leapvo_traj.txt')
    gvhmr_file_path = osp.join(gvhmr_video_folder, video, 'preprocess', 'slam_results.pt')
    masked_gvhmr_file_path = osp.join(masked_gvhmr_video_folder, video, 'preprocess', 'slam_results.pt')
    leapvo_motion = np.loadtxt(leapvo_file_path)
    masked_leapvo_motion = np.loadtxt(masked_leapvo_file_path)
    gvhmr_motion = torch.load(gvhmr_file_path)
    masked_gvhmr_motion = torch.load(masked_gvhmr_file_path)
    print(leapvo_motion.shape)
    print(gvhmr_motion.shape)
    # motion = np.load(clip_trans_npz)
    # masked_motion = np.load(masked_clip_trans_npz)
    # wham_motion = np.load(wham_trans_npz)
    # tram_motion = np.load(tram_trans_npz)

    # gt_trans = gt_motion['camera']['extrinsics'][:, :3, 3]
    # leapvo_trans = leapvo_motion[:, 1:4]
    # gvhmr_trans = gvhmr_motion[:, :3]
    # masked_leapvo_trans = masked_leapvo_motion[:, 1:4]
    # masked_gvhmr_trans = masked_gvhmr_motion[:, :3]

    # gt_traj = gt_motion['camera']['extrinsics']
    gt_traj = w2c_matrices_to_tensor(gt_motion['camera']['extrinsics'][:-1,])
    leapvo_traj = leapvo_motion[:, 1:]
    gvhmr_traj = gvhmr_motion
    masked_leapvo_traj = masked_leapvo_motion[:, 1:]
    masked_gvhmr_traj = masked_gvhmr_motion
    
    print(f"gt_traj shape: {gt_traj.shape}")
    print(f"leapvo_traj shape: {leapvo_traj.shape}")
    print(f"gvhmr_traj shape: {gvhmr_traj.shape}")
    print(f"masked_leapvo_traj shape: {masked_leapvo_traj.shape}")
    print(f"masked_gvhmr_traj shape: {masked_gvhmr_traj.shape}")

    temp_list = [gvhmr_traj, masked_gvhmr_traj, leapvo_traj, masked_leapvo_traj]
    temp_list_name = ["gvhmr_traj", "masked_gvhmr_traj", "leapvo_traj", "masked_leapvo_traj"]

    error_videos = []
    flag = True
    for i in range(len(temp_list)):
        try:
            pred_traj = temp_list[i]
            ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj = (pred_traj, np.arange(gt_traj.shape[0], dtype=np.float64)),
                        gt_traj= (gt_traj, np.arange(gt_traj.shape[0], dtype=np.float64)),
                        seq=video,
                        filename="outputs/eval_metrics.txt",
                    )
            if i == 0:
                gvhmr_sum['ate'].append(ate)
                gvhmr_sum['rpe_trans'].append(rpe_trans)
                gvhmr_sum['rpe_rot'].append(rpe_rot)
            if i == 1:
                masked_gvhmr_sum['ate'].append(ate)
                masked_gvhmr_sum['rpe_trans'].append(rpe_trans)
                masked_gvhmr_sum['rpe_rot'].append(rpe_rot)
            if i == 2:
                leapvo_sum['ate'].append(ate)
                leapvo_sum['rpe_trans'].append(rpe_trans)
                leapvo_sum['rpe_rot'].append(rpe_rot)
            if i == 3:
                masked_leapvo_sum['ate'].append(ate)
                masked_leapvo_sum['rpe_trans'].append(rpe_trans)
                masked_leapvo_sum['rpe_rot'].append(rpe_rot)
        except:
            flag = False
            print(f"Error in {temp_list_name[i]}")
            if video not in error_videos:
                error_videos.append(video)
            continue
    video_counter += 1
    frames_counter += gt_traj.shape[0]

print(f"video_counter: {video_counter}")
print("-------------------ATE-------------------")
print(f"gvhmr_sum: {sum(gvhmr_sum['ate'])}; /video_counter: {sum(gvhmr_sum['ate'])/video_counter}")
print(f"masked_gvhmr_sum: {sum(masked_gvhmr_sum['ate'])}; /video_counter: {sum(masked_gvhmr_sum['ate'])/video_counter}")
print(f"leapvo_sum: {sum(leapvo_sum['ate'])}; /video_counter: {sum(leapvo_sum['ate'])/video_counter}")
print(f"masked_leapvo_sum: {sum(masked_leapvo_sum['ate'])}; /video_counter: {sum(masked_leapvo_sum['ate'])/video_counter}")

print("-------------------RPE Translation-------------------")
print(f"gvhmr_sum: {sum(gvhmr_sum['rpe_trans'])}; /video_counter: {sum(gvhmr_sum['rpe_trans'])/video_counter}")
print(f"masked_gvhmr_sum: {sum(masked_gvhmr_sum['rpe_trans'])}; /video_counter: {sum(masked_gvhmr_sum['rpe_trans'])/video_counter}")
print(f"leapvo_sum: {sum(leapvo_sum['rpe_trans'])}; /video_counter: {sum(leapvo_sum['rpe_trans'])/video_counter}")
print(f"masked_leapvo_sum: {sum(masked_leapvo_sum['rpe_trans'])}; /video_counter: {sum(masked_leapvo_sum['rpe_trans'])/video_counter}")

print("-------------------RPE Rotation-------------------")
print(f"gvhmr_sum: {sum(gvhmr_sum['rpe_rot'])}; /video_counter: {sum(gvhmr_sum['rpe_rot'])/video_counter}")
print(f"masked_rot_sum: {sum(masked_gvhmr_sum['rpe_rot'])}; /video_counter: {sum(masked_gvhmr_sum['rpe_rot'])/video_counter}")
print(f"leapvo_sum: {sum(leapvo_sum['rpe_rot'])}; /video_counter: {sum(leapvo_sum['rpe_rot'])/video_counter}")
print(f"masked_leapvo_sum: {sum(masked_leapvo_sum['rpe_rot'])}; /video_counter: {sum(masked_leapvo_sum['rpe_rot'])/video_counter}")

print(masked_leapvo_sum)
print(error_videos)