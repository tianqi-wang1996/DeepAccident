import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS
from ..dense_heads.base_taskhead import BaseTaskHead
from ..dense_heads.loss_utils import MotionSegmentationLoss, SpatialRegressionLoss, ProbabilisticLoss, GaussianFocalLoss, SpatialProbabilisticLoss
from ...datasets.utils.geometry import cumulative_warp_features_reverse
from ...datasets.utils.instance import predict_instance_segmentation_and_trajectories
from ...datasets.utils.warper import FeatureWarper

from ..motion_modules import ResFuturePrediction, ResFuturePredictionV2
from ._base_motion_head import BaseMotionHead

import pdb


@HEADS.register_module()
class IterativeFlow(BaseMotionHead):
    def __init__(
        self,
        detach_state=True,
        n_gru_blocks=1,
        using_v2=False,
        flow_warp=True,
        **kwargs,
    ):
        super(IterativeFlow, self).__init__(**kwargs)

        if using_v2:
            self.future_prediction = ResFuturePredictionV2(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )
        else:
            self.future_prediction = ResFuturePrediction(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )


    def forward(self, bevfeats, targets=None, noise=None):
        '''
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        '''
        # import pdb
        # pdb.set_trace()
        bevfeats = bevfeats[0]

        # visualize_feature(bevfeats)

        if self.training or self.posterior_with_label:
            self.training_labels, future_distribution_inputs = self.prepare_future_labels(
                targets)
        else:
            future_distribution_inputs = None

        if not self.training:
            res = list()
            if self.n_future > 0:
                present_state = bevfeats.unsqueeze(dim=1).contiguous()

                # sampling probabilistic distribution
                samples, output_distribution = self.distribution_forward(
                    present_state, future_distribution_inputs, noise
                )

                b, _, _, h, w = present_state.shape
                hidden_state = present_state[:, 0]

                for sample in samples:
                    res_single = {}
                    future_states = self.future_prediction(sample, hidden_state)
                    future_states = torch.cat([present_state, future_states], dim=1)
                    # flatten dimensions of (batch, sequence)
                    batch, seq = future_states.shape[:2]
                    flatten_states = future_states.flatten(0, 1)

                    if self.training:
                        res_single.update(output_distribution)

                    for task_key, task_head in self.task_heads.items():
                        res_single[task_key] = task_head(
                            flatten_states).view(batch, seq, -1, h, w)

                    # visualize_feature(res_single['segmentation'][0, :, 1:2])

                    res.append(res_single)
            else:
                b, _, h, w = bevfeats.shape
                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)
        else:
            res = {}
            if self.n_future > 0:
                present_state = bevfeats.unsqueeze(dim=1).contiguous()

                # sampling probabilistic distribution
                sample, output_distribution = self.distribution_forward(
                    present_state, future_distribution_inputs, noise
                )

                b, _, _, h, w = present_state.shape
                hidden_state = present_state[:, 0]

                future_states = self.future_prediction(sample, hidden_state)
                future_states = torch.cat([present_state, future_states], dim=1)
                # flatten dimensions of (batch, sequence)
                batch, seq = future_states.shape[:2]
                flatten_states = future_states.flatten(0, 1)

                if self.training:
                    res.update(output_distribution)

                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(
                        flatten_states).view(batch, seq, -1, h, w)
            else:
                b, _, h, w = bevfeats.shape
                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)


        return res


def heatmap2d(arr_list):
    import matplotlib.pyplot as plt
    import numpy as np
    if torch.is_tensor(arr_list):
        array_tmp = arr_list.detach().clone().cpu().numpy()
        # batch, channel, height, width = array_tmp.shape
        # array_tmp = array_tmp[:, 0, :, :]

    for i, arr in enumerate(array_tmp):
        arr_show = arr.copy()
        arr_show -= arr_show.mean()
        arr_show /= arr_show.std()
        # # feature = np.clip(feature, 0, 255).astype('uint8')
        # arr_show = arr_show.astype('float32')
        arr_show *= 64
        arr_show += 128
        arr_show = np.clip(arr_show, 0, 255).astype('uint8')

        plt.figure('Figure %d' % i)
        plt.imshow(arr_show, cmap='viridis')
        plt.colorbar()
    plt.show()


def visualize_feature(input_feature):
    import numpy as np
    features = input_feature.detach().clone().cpu().numpy()
    avg_feature = []
    for agent_idx in range(features.shape[0]):
        feature = np.mean(features[agent_idx], axis=0)
        avg_feature.append(feature)

    avg_feature = np.array(avg_feature)
    avg_feature = torch.from_numpy(avg_feature)
    heatmap2d(avg_feature)
