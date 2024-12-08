import re
import numpy as np
import os
import json

def extract_metrics(file_path):
    metrics = []
    with open(file_path, 'r') as file:
        for line in file:
            # Match lines with ATE, RPE trans, and RPE rot values
            match = re.search(r'ATE: ([\d.]+), RPE trans: ([\d.]+), RPE rot: ([\d.]+)', line)
            if match:
                metrics.append({
                    'ATE': float(match.group(1)),
                    'RPE_trans': float(match.group(2)),
                    'RPE_rot': float(match.group(3))
                })
    return metrics

def calculate_statistics(metrics):
    ate_values = [item['ATE'] for item in metrics]
    rpe_trans_values = [item['RPE_trans'] for item in metrics]
    rpe_rot_values = [item['RPE_rot'] for item in metrics]

    statistics = {
        'ATE': {
            'mean': np.mean(ate_values),
            'std': np.std(ate_values)
        },
        'RPE_trans': {
            'mean': np.mean(rpe_trans_values),
            'std': np.std(rpe_trans_values)
        },
        'RPE_rot': {
            'mean': np.mean(rpe_rot_values),
            'std': np.std(rpe_rot_values)
        }
    }
    return statistics

# Path to your error_sum.txt file
Datasets = ["sintel", "sintel_masked", "sintel_pure_pts", "sintel_pure_boxes"]
logs_dir = "/home/boxun/work/Project/CV2024_Object_detection/leapvo/logs/"
save_dir = "/home/boxun/work/Project/CV2024_Object_detection/leapvo/logs/"

output_data = {}

for dataset in Datasets:
    file_path = os.path.join(logs_dir, dataset, "error_sum.txt")
    # Extract metrics
    metrics = extract_metrics(file_path)

    # Calculate statistics
    stats = calculate_statistics(metrics)

    # Print the results
    # print("Scene Metrics:")
    # for idx, result in enumerate(metrics, 1):
    #     print(f"Scene {idx}: ATE={result['ATE']}, RPE_trans={result['RPE_trans']}, RPE_rot={result['RPE_rot']}")

    print(f"\nStatistics of {file_path}:")
    print(f"ATE - Mean: {stats['ATE']['mean']:.5f}, Std: {stats['ATE']['std']:.5f}")
    print(f"RPE Trans - Mean: {stats['RPE_trans']['mean']:.5f}, Std: {stats['RPE_trans']['std']:.5f}")
    print(f"RPE Rot - Mean: {stats['RPE_rot']['mean']:.5f}, Std: {stats['RPE_rot']['std']:.5f}")

    output_data[dataset] = stats

# Save the statistics to a JSON file
output_file = os.path.join(save_dir, "statistics.json")
with open(output_file, 'w') as file:
    json.dump(output_data, file, indent=4)

print(f"\nStatistics saved to {output_file}")

