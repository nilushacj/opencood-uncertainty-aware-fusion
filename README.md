# OpenCOOD extension: FeaCo-based Probabilistic Fusion
This repository contains research development work on uncertainty-aware cooperative BEV fusion, built on top of OpenCOOD and FeaCo-style cooperative perception.

The main focus is robust intermediate fusion under localization uncertainty, including probabilistic positioning-error correction and covariance-aware fusion logic.

## Base Framework

This repository is based on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), an open cooperative detection framework for autonomous driving.

The original OpenCOOD README, documentation, license, and citation should be consulted for the base framework, supported datasets, installation assumptions, and benchmark models.

## Dataset
The V2XSet data can be downloaded from [UCLA BOX](https://ucla.app.box.com/v/UCLA-MobilityLab-V2XVIT). 

Since the data for train/validate/test is very large, it is split into small chunks, which can be found in the directory ending with _chunks, such as train_chunks. Thus, after downloading, run the following command to each set to merge those chunks together:
```bash
cat train.zip.part* > train.zip
unzip train.zip
```

Then make the directory structure for the dataset as follows:
```bash
datasets # root
├── v2xset # the downloaded v2xset data
│   ├── train
│   ├── validate
│   ├── test
```

NOTE: Currently, we are only utilizing the validation split.

## Pretrained checkpoints
The pretrained checkpoint for Feaco were downloaded from [GoogleURL](https://drive.google.com/drive/folders/1reQ7I3jNWRosjpEhVGSSKE2JoLwHIHa4). 

Refer the model zoo of the OpenCOOD repository for checkpoints of other fusion models.

## Installation

The environment below describes the setup that was used for this repository. The code was tested with:

- Python 3.7.11
- PyTorch 1.13.1
- CUDA 11.7 PyTorch/spconv wheels
- spconv-cu117 2.3.6
- Open3D 0.16.0

### 1. Clone the repository

```bash
git clone git@github.com:YOUR_USERNAME/opencood-uncertainty-aware-fusion.git
cd opencood-uncertainty-aware-fusion
```

### 2. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate opencood
```
### 3. Install PyTorch

```bash
pip install torch==1.13.1 torchvision==0.14.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" #sanity check
```

### 4. Install spconv

```bash
pip install cumm-cu117==0.4.11 spconv-cu117==2.3.6
python -c "import spconv.pytorch as spconv; import cumm; print('spconv ok', spconv.__version__)" #sanity check
```

### 5. Install this repository in editable mode

```bash
python -m pip install -e . --no-deps
```

### 6. Compile the bounding-box IoU CUDA extension

```bash
python opencood/utils/setup.py build_ext --inplace
```

### 7. Additional tested dependency versions
```bash
pip install \
  easydict==1.13 \
  einops==0.6.1 \
  matplotlib==3.3.4 \
  numba==0.49.0 \
  opencv-python==4.5.1.48 \
  Pillow==9.5.0 \
  PyYAML==6.0.1 \
  scikit-image==0.19.3 \
  scipy==1.5.4 \
  shapely==2.0.0 \
  tensorboardX==2.6.2.2 \
  timm==0.6.13
```

If scikit-image installation through pip causes build issues on Python 3.7, install it from conda-forge instead:
```bash
mamba install -c conda-forge scikit-image==0.19.3
```
If TIFF/Pillow-related runtime errors occur, the following versions were used in the tested environment:
```bash
conda install -c conda-forge "libtiff=4.*"
python -m pip install Pillow==9.5.0
```

### 8. Verify the installation:
```bash
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
python -c "import spconv.pytorch as spconv; print('spconv:', spconv.__version__)"
python -c "import open3d as o3d; print('open3d:', o3d.__version__)"
python -c "import opencood; print('opencood imported')"
```

## Quick Start
TBD

## Generate experiment configs
TBD

## Run standard inference
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder refers to the correct dataset path e.g. `V2XSet/validate`.
```bash
python opencood/tools/inference.py --model_dir opencood/pretrained/feaco  --fusion_method intermediate --save_npy
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate' but we do 'intermediate' (thus set it to this).
- `save_npy`: whether to save detections as numpy files or not.

The evaluation results  will be dumped in the model directory. 
## Run parallel inference (useful for multiple noise/model setting evaluation)
TBD

## Notes on generated files
TBD

## Acknowledgement
TBD

## Citation
TBD

## References
 ```bibtex
@inproceedings{xu2022opencood,
  author = {Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, Jiaqi Ma},
  title = {OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication},
  booktitle = {2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2022}}
```

