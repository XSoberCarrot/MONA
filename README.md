# CV2024_moving_object_detection

## Installation
### Requirements
The code was tested on Ubuntu 20.04, PyTorch 2.5.1, CUDA 12.2 with 1 NVIDIA GPU (RTX 4090).

### Install the LEAP-VO

1. Follow this official instrution: https://github.com/chiaki530/leapvo

2. Replace eval.py and leapvo.py with eval_modified.py and leapvo_modified.py (rename them into eval.py and leapvo.py )

## Layout (Useing MPI-Sintel as example)
```
leapvo_folder
└── obj_detector
└── weights
└── result_videos
└── logs
└── data
    └── MPI-Sintel-complete
        └── training
            ├── final
            └── camdata_left
```

## Run the code
1. Run dataset_evaluation_ori.py, which will generate dynamic points and raw LEAP-VO result

2. Run mask_processor.py, which will execute the moving object detection, generate masked frame based on dynamic points

3. Run dataset_evaluation_masked.py which will generate trajectory estimation result after applying moving object detection

4. Run calculate_result.py to see the quantantive results, run calculate_result_trajectory_v2.py to see the trajectory visulization results.

5. Run video generator to generate the result video

## Evaluations
We provide evaluation scripts for MPI-Sinel, TartanAir-Shibuya, and Replica.

### MPI-Sintel
Follow [MPI-Sintel](http://sintel.is.tue.mpg.de/) and download it to the `data` folder. For evaluation, we also need to download the [groundtruth camera pose data](http://sintel.is.tue.mpg.de/depth). The folder structure should look like
```
MPI-Sintel-complete
└── training
    ├── final
    └── camdata_left
```
