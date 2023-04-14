# Dataset Preparation

## Dataset structure

It is recommended to symlink the dataset root to `$DeepAccident/data`.
If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```
DeepAccident
├── mmdet3d
├── tools
├── configs
├── projects
├── data
│   ├── $DeepAccident_data
│   │   ├── type1_subtype1_accident
│   │   ├── type1_subtype1_normal
│   │   ├── type1_subtype2_accident
│   │   ├── type1_subtype2_normal
```

## Download and prepare the nuScenes dataset

Download DeepAccident full dataset data [HERE](https://deepaccident.github.io/data.html), including the train, val, test splits.
Extract all the small split zip files into a same directory, e.g. ./data/DeepAccident_data

Prepare DeepAccident data by running

```bash
python tools/create_data.py carla --root-path ./data/DeepAccident_data --out-dir ./data/DeepAccident_data --extra-tag carla
```
