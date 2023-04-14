# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import pdb
import time

import numpy as np
import os
import matplotlib as mpl

import cv2

# define semantic metrics
from ..metrics import IntersectionOverUnion, PanopticMetric
from ..visualize import Visualizer

from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from .multi_gpu_test import eval_accidents, eval_metrics, traj_mapping
from ..datasets.utils.instance import predict_instance_segmentation_and_trajectories, predict_instance_segmentation_and_trajectories_accident
import pdb
import copy

def single_gpu_gradcam_vis(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    dataset = data_loader.dataset
    # whether for test submission
    test_mode = dataset.test_submission

    # bev coordinate system, LiDAR or ego
    coordinate_system = dataset.coordinate_system

    prog_bar = mmcv.ProgressBar(len(dataset))

    # logging interval
    logging_interval = 500

    det_results = []


    latencies = []


    # V2X_model = False
    ego_agent_idx = 0

    import pdb

    for i, data in enumerate(data_loader):
        if i == 0:
            V2X_model = isinstance(data['img_metas'][0],list)
            not_V2X_model = not V2X_model
        # pdb.set_trace()

        pdb.set_trace()
        data_raw = copy.deepcopy(data)

        data_prefix = data['img_metas'][0].data[0][0]['sample_idx'] if not_V2X_model else \
        data['img_metas'][ego_agent_idx][0].data[0][0]['sample_idx']
        town_name = data_prefix.split('_type')[0].split('_')[-1]
        with torch.no_grad():
            if test_mode:
                motion_distribution_targets = None
            else:
                motion_distribution_targets = {
                    # for motion prediction
                    'motion_segmentation': data['motion_segmentation'][0] if not_V2X_model else data['motion_segmentation'][ego_agent_idx][0],
                    'motion_instance': data['motion_instance'][0] if not_V2X_model else data['motion_instance'][ego_agent_idx][0],
                    'instance_centerness': data['instance_centerness'][0] if not_V2X_model else data['instance_centerness'][ego_agent_idx][0],
                    'instance_offset': data['instance_offset'][0] if not_V2X_model else data['instance_offset'][ego_agent_idx][0],
                    'instance_flow': data['instance_flow'][0] if not_V2X_model else data['instance_flow'][ego_agent_idx][0],
                    'future_egomotion': data['future_egomotions'][0] if not_V2X_model else data['future_egomotions'][ego_agent_idx][0],
                }


            # print(data['img_inputs'][0][0][0].device, data['future_egomotions'][0][0].device)
            # result = model(return_loss=False,rescale=True,img_metas=data['img_metas'],img_inputs=data['img_inputs'],future_egomotions=data['future_egomotions'],motion_targets=motion_distribution_targets,img_is_valid=[img_valid[0] for img_valid in data['img_is_valid']],relative_pose_to_ego=[relative_pose[0] for relative_pose in data['relative_pose_to_ego']])

            class SegmentationModelOutputWrapper(torch.nn.Module):
                def __init__(self, model):
                    super(SegmentationModelOutputWrapper, self).__init__()
                    self.model = model

                def forward(self, img_metas, img_inputs, future_egomotions, motion_targets, img_is_valid,
                            relative_pose_to_ego):
                    return self.model(
                    return_loss=False,
                    rescale=True,
                    img_metas=img_metas,
                    img_inputs=img_inputs,
                    future_egomotions=future_egomotions,
                    motion_targets=motion_targets,
                    img_is_valid=img_is_valid,
                    relative_pose_to_ego=relative_pose_to_ego)['motion_predictions'][-1]['segmentation'][0, 0]

            model_warp = SegmentationModelOutputWrapper(model)
            img_metas = data['img_metas']
            img_inputs = data['img_inputs']
            future_egomotions = data['future_egomotions']
            motion_targets = motion_distribution_targets
            img_is_valid = [img_valid[0] for img_valid in data['img_is_valid']]
            relative_pose_to_ego = [relative_pose[0] for relative_pose in data['relative_pose_to_ego']]

            pdb.set_trace()
            seg_output = model_warp(img_metas, img_inputs, future_egomotions, motion_targets, img_is_valid,
                            relative_pose_to_ego)

            preds_max = torch.argmax(seg_output, dim=0, keepdims=True).squeeze(0).detach().cpu().numpy()
            # preds_max = preds_max[0, 0]
            vehicles_id = 1
            car_mask_uint8 = 255 * np.uint8(preds_max == vehicles_id)
            car_mask_float = np.float32(preds_max == vehicles_id)
            pdb.set_trace()

            # self.v2x_fusion.model[3].aggregation[0].conv

            from pytorch_grad_cam import GradCAM

            class SemanticSegmentationTarget:
                def __init__(self, category, mask):
                    self.category = category
                    self.mask = torch.from_numpy(mask)
                    if torch.cuda.is_available():
                        self.mask = self.mask.cuda()

                def __call__(self, model_output):
                    return (model_output[1, :, :] * self.mask).sum()

            target_layers = [model.module.v2x_fusion.model[3].aggregation[0].conv]
            targets = [SemanticSegmentationTarget(vehicles_id, car_mask_float)]

            img_metas = data_raw['img_metas']
            img_inputs = data_raw['img_inputs']
            future_egomotions = data_raw['future_egomotions']
            motion_targets = motion_distribution_targets
            img_is_valid = [img_valid[0] for img_valid in data_raw['img_is_valid']]
            relative_pose_to_ego = [relative_pose[0] for relative_pose in data_raw['relative_pose_to_ego']]

            pdb.set_trace()
            with GradCAM(model=model_warp,
                         target_layers=target_layers,
                         use_cuda=torch.cuda.is_available()) as cam:
                grayscale_cam = cam(input_tensor=[img_metas, img_inputs, future_egomotions, motion_targets, img_is_valid,
                            relative_pose_to_ego],
                                    targets=targets)[0, :]
                # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Image.fromarray(cam_image)


            time_stats = result['time_stats']
            num_input_frame = data['img_inputs'][0][0].shape[1] if not_V2X_model else data['img_inputs'][ego_agent_idx][0][0].shape[1]
            latency = (time_stats['t_BEV'] - time_stats['t0']) / \
                num_input_frame + time_stats['t_end'] - time_stats['t_BEV']

            latencies.append(latency)


        # update prog_bar
        for _ in range(data_loader.batch_size):
            prog_bar.update()


    return det_results
