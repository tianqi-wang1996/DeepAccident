Modified from the official mmdet3d [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

# Prerequisites
DeepAccident is developed with the following version of modules.
- Linux or macOS (Windows is not currently officially supported)
- Python 3.7
- PyTorch 1.10.2 
- CUDA 10.2
- GCC 7.5.0
- MMCV==1.3.14
- MMDetection==2.14.0
- MMSegmentation==0.14.1
- numba version in requirements.txt, 0.48.0

# Installation

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n deepaccident python=3.7 -y
conda activate deepaccident
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
conda install pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=10.2 -c pytorch
```

**c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and other requirements.**

```shell
pip install -r requirements.txt
```

**f. Clone the DeepAccident repository.**

```shell
git clone https://github.com/tianqi-wang1996/DeepAccident.git
cd DeepAccident
```

**g.Install build requirements and then install DeepAccident.**

```shell
python setup.py develop
```