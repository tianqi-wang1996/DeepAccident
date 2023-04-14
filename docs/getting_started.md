# Getting started with DeepAccident
## Training

To train the models on DeepAccident with 8 GPUs, run:
```bash
bash tools/dist_train.sh $CONFIG 8 --work-dir $WORK_DIR
e.g. bash tools/dist_train.sh projects/configs/DeepAccident_tiny.py 8 --work-dir work_dirs/DeepAccident_tiny
```

To train the models on DeepAccident with single GPU, run:
```bash
python tools/train.py $CONFIG
e.g. python tools/train.py projects/configs/DeepAccident_tiny.py --work-dir work_dirs/DeepAccident_tiny
```

## Evaluation

To evaluate the models on DeepAccident, run:
```bash
python tools/test.py $YOUR_CONFIG --checkpoint $YOUR_CKPT --eval=distance_mAP --mtl --tp-save_dir $TP_DIR_FOR_ACCIDENT_PREDICTION
e.g. python tools/test.py projects/configs/DeepAccident_tiny.py --checkpoint work_dirs/DeepAccident_tiny/latest_epoch.pth --eval=distance_mAP --mtl --tp-save-dir work_dirs/DeepAccident_tiny/accident_tp
```

## Visualization

To visualize the predictions, run:
```bash
python tools/test.py $YOUR_CONFIG --checkpoint $YOUR_CKPT --eval=distance_mAP --mtl --tp-save_dir $TP_DIR_FOR_ACCIDENT_PREDICTION --show --show-dir $SHOW_DIR
e.g. python tools/test.py projects/configs/DeepAccident_tiny.py --checkpoint work_dirs/DeepAccident_tiny/latest_epoch.pth --eval=distance_mAP --mtl --tp-save-dir work_dirs/DeepAccident_tiny/accident_tp --show --show-dir work_dirs/DeepAccident_tiny
```

