# CenterFusionDetect3D
 This repository contains the reimplementation of [CenterFusion](https://github.com/mrnabati/CenterFusion)
 <br>
 [CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection](https://arxiv.org/abs/2011.04841).

 ![](figures/pillars.png)
 ### Citing CenterFusion <!-- omit in toc -->
 If you find CenterFusion useful in your research, please consider citing:

> **[CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection](https://arxiv.org/abs/2011.04841)** \
> Ramin Nabati, Hairong Qi

    @article{nabati2020centerfusion,
      title={CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection},
      author={Nabati, Ramin and Qi, Hairong},
      journal={arXiv preprint arXiv:2011.04841},
      year={2020}
    }

## Contents <!-- omit in toc --> 
- [What's New](#whats-new)
- [Introduction](#introduction)
- [Results](#results)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [References](#references)
- [License](#license)

## What's new
### v2.0.0
- Improved traning **Performance** with new AI training framework: PyTorch Lightning
- Added **Demo** script for visualization of the results
- Added **Inference** script for evaluation of the pure vision model
- Fixed unused parameters problem of DLA-34
- Added **DDP** support for multi-GPU training
- More **readable**, **cleaner** and **faster** code

### v1.0.1
- **Compatible with old model** 
- Added tqdm for **better visualization** of training, validation and evaluation
- Improved **readability** of code **significantly**
- Improved **readability** and versatility of parameters
- Improved **file path handling**
- Improved **loading of dataset** (load radar pointcloud during training)
- Improved **performance** when loading dataset
- Removed all code not related to 3D object detection
- Fixed issue of high RAM and CPU usage
- Fixed deformable convolution library installation issue

## Introduction
We focus on the problem of radar and camera sensor fusion and propose a middle-fusion approach to exploit both radar and camera data for 3D object detection. Our method, called CenterFusion, first uses a center point detection network to detect objects by identifying their center points on the image. It then solves the key data association problem using a novel frustum-based method to associate the radar detections to their corresponding object's center point. The associated radar detections are used to generate radar-based feature maps to complement the image features, and regress to object properties such as depth, rotation and velocity. We evaluate CenterFusion on the challenging nuScenes dataset, where it improves the overall nuScenes Detection Score (NDS) of the state-of-the-art camera-based algorithm by more than 12%. We further show that CenterFusion significantly improves the velocity estimation accuracy without using any additional temporal information.

## Results
- #### Overall results:

  | Dataset      |  NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
  |--------------|------|------|------|------|------|------|------|
  |nuScenes Test | 0.449|0.326 |0.631 |0.261 |0.516 |0.614 |0.115 |
  |nuScenes Val  | 0.453|0.332 |0.649 |0.263 |0.535 |0.540 |0.142 |

- #### Per-class mAP:
  
  |  Dataset    |  Car | Truck | Bus | Trailer | Const. | Pedest. | Motor. | Bicycle | Traff. | Barrier |
  |-------------|------|-------|-----|---------|--------|---------|--------|---------|--------|---------|
  |nuScenes Test|0.509 |0.258  |0.234| 0.235   |0.077   |0.370    |0.314   |0.201    |0.575   | 0.484   |
  |nuScenes Val |0.524 |0.265  |0.362| 0.154   |0.055   |0.389    |0.305   |0.229    |0.563   | 0.470   |

- #### Qualitative results:

<p align="center"> <img src='figures/qualitative_results.jpg' align="center"> </p> 

## Installation

The code has been tested on WSL 2 Ubuntu 22.04.2 with Python 3.9.17, CUDA 12.2 and PyTorch 2.0.1. For installation, follow these steps:

1. Clone this repository with the `--recursive` option. We'll call the directory that you cloned this repo into `CF_ROOT`:
    ```bash
    CF_ROOT=</path/to/RepositoryFolder>
    git clone --recursive https://github.com/HengWeiBin/CenterFusionDetect3D $CF_ROOT
    ```

2. Create a new virtual environment (optional):
    ```bash
    conda create --name centerfusion python=3.9
    conda activate centerfusion
    ```

3. Install [PyTorch](https://pytorch.org/):
    ```bash
    conda install lightning pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4. Install other requirements:
   ```bash
   cd $CF_ROOT
   pip install -r requirements.txt
   sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
   
   # Required by nuScenes dataset evaluation
   cd $CF_ROOT/src
   git clone https://github.com/nutonomy/nuscenes-devkit
   
   cd $CF_ROOT
   ```

## Dataset Preparation

1. Download the nuScenes dataset from [nuScenes website](https://www.nuscenes.org/download).


2. Extract the downloaded files in the `${CF_ROOT}\data\nuscenes` directory. You should have the following directory structure after extraction:

    ~~~
    ${CF_ROOT}
    `-- data
        `-- nuscenes
            |-- maps
            |-- samples
            |   |-- CAM_BACK
            |   |   | -- xxx.jpg
            |   |   ` -- ...
            |   |-- CAM_BACK_LEFT
            |   |-- CAM_BACK_RIGHT
            |   |-- CAM_FRONT
            |   |-- CAM_FRONT_LEFT
            |   |-- CAM_FRONT_RIGHT
            |   |-- RADAR_BACK_LEFT
            |   |   | -- xxx.pcd
            |   |   ` -- ...
            |   |-- RADAR_BACK_RIGHT
            |   |-- RADAR_FRON
            |   |-- RADAR_FRONT_LEFT
            |   `-- RADAR_FRONT_RIGHT
            |-- sweeps
            |-- v1.0-mini
            |-- v1.0-test
            `-- v1.0-trainval
    ~~~
   

3. Run the `convert_nuScenes.py` script to convet the nuScenes dataset to COCO format:
    ```bash
    cd $CF_ROOT
    python src/convert_nuScenes.py
    ```

## Pretrained Models
The pre-trained CenterFusion model and the baseline CenterNet model can be downloaded from the links below:
  | model       | epochs  | GPUs  | Backbone | Val NDS | Val mAP | Test  NDS   | Test mAP |
  |-------------|---------|-------|-----|-----|------|--------|-----------|
  | [centerfusion_e60](https://drive.google.com/uc?export=download&id=1XaYx7JJJmQ6TBjCJJster-Z7ERyqu4Ig) | 60 |  2x Nvidia Quadro P5000 | DLA | 0.453 | 0.332 | 0.449 | 0.326 |
  | [centernet_baseline_e170](https://drive.google.com/uc?export=download&id=1iFF7a5oueFfB5GnUoHFDnntFdTst-bVI) | 170 |  2x Nvidia Quadro P5000 | DLA | 0.328 | 0.306 | - | - |
  | [centerfusion_e230](https://github.com/HengWeiBin/CenterFusionDetect3D/releases/download/v2.0.0/centerfusion_e230.pt) | 230 | 4x Nvidia RTX A6000 | DLA | 0.445 | 0.312 | - | - |
  | [centernet_baseline_e170](https://github.com/HengWeiBin/CenterFusionDetect3D/releases/download/v2.0.0/centernet_baseline_e170.pt) | 170 | 4x Nvidia RTX A6000 | DLA | 0.321 | 0.296 | - | - |
  **Notes:**
  - The *centernet_baseline_e170* model is obtained by starting from the original CenterNet 3D detection model ([nuScenes_3Ddetection_e140](https://github.com/xingyizhou/CenterTrack/blob/master/readme/MODEL_ZOO.md)) and training the velocity and attributes heads for 30 epochs. 

## Training

1. Prepare a configuration file for training. You can use the default configuration files in `$CF_ROOT/configs/` or create a new one. All the information about the configuration file can be found in `$CF_ROOT/src/lib/config/default.py`.

2. The `$CF_ROOT/src/main.py` script can be used to train the network:
    ```bash
    cd $CF_ROOT
    # In this example, we use centerfusion_debug.yaml as the configuration file
    python src/main.py --cfg configs/centerfusion_debug.yaml
    ```

## Evaluation

1. To evaluate the model, use its' configuration file and modify the `EVAL` section to `true`.

2. Run the `$CF_ROOT/src/main.py` script same as training:
    ```bash
    cd $CF_ROOT
    # In this example, we use centerfusion_debug.yaml as the configuration file
    python src/main.py --cfg configs/centerfusion_debug.yaml
    ```

## Demo

1. To run the demo of dataset, modify the `DATASET` section in the configuration file to correct dataset, and ensure the demo split is contained or same as the training split.

2. Run the `$CF_ROOT/src/demo.py` script:
    ```bash
    cd $CF_ROOT
    # In this example, we use centerfusion_debug.yaml as the configuration file
    python src/demo.py --cfg configs/centerfusion_debug.yaml --split val
    ```
    It will demo the first sample of the validation split by default.

3. To control the sample of the demo, add the `--min` or `--max` argument:
    ```bash
    cd $CF_ROOT
    # In this example, we assume inference on the first 10 samples of the validation split
    python src/demo.py --cfg configs/centerfusion_debug.yaml --split val --min 0 --max 10
    ```

4. To disable the visualization, add the `--not-show` argument, this will only accumulate the inference time without presenting the results:
    ```bash
    cd $CF_ROOT
    python src/demo.py --cfg configs/centerfusion_debug.yaml --split val --not-show
    ```

5. To save the visualization, add the `--save` argument, this will save the visualization to the `output/Demo` defined in the configuration file:
    ```bash
    cd $CF_ROOT
    python src/demo.py --cfg configs/centerfusion_debug.yaml --split val --save
    ```

6. If you just want to see single cam result, add the `--single` argument:
    ```bash
    cd $CF_ROOT
    python src/demo.py --cfg configs/centerfusion_debug.yaml --single
    ```

## References
The following works have been used by CenterFusion:

  ~~~

  @inproceedings{zhou2019objects,
    title={Objects as Points},
    author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
    booktitle={arXiv preprint arXiv:1904.07850},
    year={2019}
  }

  @article{zhou2020tracking,
    title={Tracking Objects as Points},
    author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
    journal={ECCV},
    year={2020}
  }

  @inproceedings{nuscenes2019,
    title={{nuScenes}: A multimodal dataset for autonomous driving},
    author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
    booktitle={CVPR},
    year={2020}
  }
  ~~~

## License

CenterFusion is based on [CenterNet](https://github.com/xingyizhou/CenterNet) and is released under the MIT License. See [NOTICE](NOTICE) for license information on other libraries used in this project.
