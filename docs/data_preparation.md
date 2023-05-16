# Dataset Preparation

### Dataset structure

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

### Download and prepare the DeepAccident dataset

Download DeepAccident full dataset data [HERE](https://deepaccident.github.io/download.html), including the train, val, test splits.
Extract all the small split zip files into a same directory, e.g. ./data/DeepAccident_data

Prepare DeepAccident data by running

```bash
python tools/create_data.py carla --root-path ./data/DeepAccident_data --out-dir ./data/DeepAccident_data --extra-tag carla
```

### Download the pretrained weights

Download the pretrained weights of Swin-Transformer and BEVerse to initialize training. Put all the pretrained weights into the dataset root, e.g. ./data/DeepAccident_data. 

The pretrained weights can be downloaded from [swin_tiny_patch4_window7_224.pth](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wangtq_connect_hku_hk/EVbP74jZDdlBvvMwDcoXaCcBgFt0vzCgVuteVa5WoKu0zA),
[swin_small_patch4_window7_224.pth](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wangtq_connect_hku_hk/EZo0zaavPplLkJsjvQ94agQBUNEDbnQ9724Vw2FECRfcKA),
[beverse_tiny.pth](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wangtq_connect_hku_hk/Ea0ToSxXuq5Ev8-lco4x7zIBv7ENUf764N8JJxmrXzkEeQ),
[beverse_small.pth](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wangtq_connect_hku_hk/EdSBp8ZTP2hDhxYmjRjr1iUB5sblbWfQyOQM0_lqeJKVZw).


```bash
python tools/create_data.py carla --root-path ./data/DeepAccident_data --out-dir ./data/DeepAccident_data --extra-tag carla
```
