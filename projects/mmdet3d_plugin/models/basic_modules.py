import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from functools import partial

from ..datasets.utils.geometry import warp_features
from ..datasets.utils import FeatureWarper


class ConvBlock(nn.Module):
    """2D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        norm='bn',
        activation='relu',
        bias=False,
        transpose=False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(
            nn.ConvTranspose2d, output_padding=1)
        self.conv = self.conv(in_channels, out_channels,
                              kernel_size, stride, padding=padding, bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Bottleneck(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        # Define the main conv operation
        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=1,
                stride=2,
                output_padding=padding_size,
                padding=padding_size,
                groups=groups,
            )
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                stride=2,
                padding=padding_size,
                groups=groups,
            )
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                padding=padding_size,
                groups=groups,
            )

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    # First projection with 1x1 kernel
                    ('conv_down_project', nn.Conv2d(in_channels,
                     bottleneck_channels, kernel_size=1, bias=False)),
                    ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                                                       nn.ReLU(inplace=True))),
                    # Second conv block
                    ('conv', bottleneck_conv),
                    ('abn', nn.Sequential(nn.BatchNorm2d(
                        bottleneck_channels), nn.ReLU(inplace=True))),
                    # Final projection with 1x1 kernel
                    ('conv_up_project', nn.Conv2d(bottleneck_channels,
                     out_channels, kernel_size=1, bias=False)),
                    ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
                                                     nn.ReLU(inplace=True))),
                    # Regulariser
                    ('dropout', nn.Dropout2d(p=dropout)),
                ]
            )
        )

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update(
                    {'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update(
                    {'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    # pylint: disable=arguments-differ
    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(
                    x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x


class Interpolate(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    # pylint: disable=arguments-differ
    def forward(self, x):
        return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=True)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=False, norm=norm, activation=activation
        )

    def forward(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(
            torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class SpatialGRU(nn.Module):
    """A GRU cell that takes an input tensor [BxTxCxHxW] and an optional previous state and passes a
    convolutional gated recurrent unit over the data"""

    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=False, norm=norm, activation=activation
        )

    def forward(self, x, state=None, flow=None, mode='bilinear'):
        # pylint: disable=unused-argument, arguments-differ
        # Check size
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'
        b, timesteps, c, h, w = x.size()
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'

        # 对于第一层 SpatialGRU，输入 x 为 latent distribution, state 为 present bev features
        # 对于后续的 SpatialGRU，x 与 state 具有类似的特征，可以理解为 transformed future bev features

        # recurrent layers
        rnn_output = []
        rnn_state = torch.zeros(b, self.hidden_size, h,
                                w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]
            if flow is not None:
                rnn_state = warp_features(rnn_state, flow[:, t], mode=mode)

            # propagate rnn state
            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(rnn_state)

        # reshape rnn output to batch tensor
        return torch.stack(rnn_output, dim=1)

    def gru_cell(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(
            torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(
            padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              dilation=dilation, stride=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        (x,) = inputs
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalMaxPool3d(nn.Module):
    def __init__(self, kernel_size=(2, 3, 3)):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = kernel_size[0] - 1
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(
            padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.max_pool = nn.MaxPool3d(kernel_size, stride=1)

    def forward(self, *inputs):
        (x,) = inputs
        x = self.pad(x)
        x = self.max_pool(x)
        return x


def conv_1x1x1_norm_activated(in_channels, out_channels):
    """1x1x1 3D convolution, normalization and activation layer."""
    return nn.Sequential(
        OrderedDict(
            [
                ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm3d(out_channels)),
                ('activation', nn.ReLU(inplace=True)),
            ]
        )
    )


class Bottleneck3D(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=(2, 3, 3), dilation=(1, 1, 1)):
        super().__init__()
        bottleneck_channels = in_channels // 2
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    # First projection with 1x1 kernel
                    ('conv_down_project', conv_1x1x1_norm_activated(
                        in_channels, bottleneck_channels)),
                    # Second conv block
                    (
                        'conv',
                        CausalConv3d(
                            bottleneck_channels,
                            bottleneck_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            bias=False,
                        ),
                    ),
                    # Final projection with 1x1 kernel
                    ('conv_up_project', conv_1x1x1_norm_activated(
                        bottleneck_channels, out_channels)),
                ]
            )
        )

        if out_channels != in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.projection = None

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        x_features = self.projection(x) if self.projection is not None else x
        return x_residual + x_features


class PyramidSpatioTemporalPooling(nn.Module):
    """ Spatio-temporal pyramid pooling.
        Performs 3D average pooling followed by 1x1x1 convolution to reduce the number of channels and upsampling.
        Setting contains a list of kernel_size: usually it is [(2, h, w), (2, h//2, w//2), (2, h//4, w//4)]
    """

    def __init__(self, in_channels, reduction_channels, pool_sizes):
        super().__init__()
        self.features = []
        for pool_size in pool_sizes:
            assert pool_size[0] == 2, (
                "Time kernel should be 2 as PyTorch raises an error when" "padding with more than half the kernel size"
            )
            stride = (1, *pool_size[1:])
            padding = (pool_size[0] - 1, 0, 0)
            self.features.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            # Pad the input tensor but do not take into account zero padding into the average.
                            (
                                'avgpool',
                                torch.nn.AvgPool3d(
                                    kernel_size=pool_size, stride=stride, padding=padding, count_include_pad=False
                                ),
                            ),
                            ('conv_bn_relu', conv_1x1x1_norm_activated(
                                in_channels, reduction_channels)),
                        ]
                    )
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, *inputs):
        (x,) = inputs
        b, _, t, h, w = x.shape
        # Do not include current tensor when concatenating
        out = []
        for f in self.features:
            # Remove unnecessary padded values (time dimension) on the right
            x_pool = f(x)[:, :, :-1].contiguous()
            c = x_pool.shape[1]
            x_pool = nn.functional.interpolate(
                x_pool.view(b * t, c, *x_pool.shape[-2:]), (h, w), mode='bilinear', align_corners=True,
            )
            x_pool = x_pool.view(b, c, t, h, w)
            out.append(x_pool)
        out = torch.cat(out, 1)
        return out


class TemporalBlock(nn.Module):
    """ Temporal block with the following layers:
        - 2x3x3, 1x3x3, spatio-temporal pyramid pooling
        - dropout
        - skip connection.
    """

    def __init__(self, in_channels, out_channels=None, use_pyramid_pooling=False, pool_sizes=None):
        super().__init__()
        self.in_channels = in_channels
        self.half_channels = in_channels // 2
        self.out_channels = out_channels or self.in_channels
        self.kernels = [(2, 3, 3), (1, 3, 3)]

        # Flag for spatio-temporal pyramid pooling
        self.use_pyramid_pooling = use_pyramid_pooling

        # 3 convolution paths: 2x3x3, 1x3x3, 1x1x1
        self.convolution_paths = []
        for kernel_size in self.kernels:
            self.convolution_paths.append(
                nn.Sequential(
                    conv_1x1x1_norm_activated(
                        self.in_channels, self.half_channels),
                    CausalConv3d(self.half_channels,
                                 self.half_channels, kernel_size=kernel_size),
                )
            )
        self.convolution_paths.append(conv_1x1x1_norm_activated(
            self.in_channels, self.half_channels))
        self.convolution_paths = nn.ModuleList(self.convolution_paths)

        agg_in_channels = len(self.convolution_paths) * self.half_channels

        if self.use_pyramid_pooling:
            assert pool_sizes is not None, "setting must contain the list of kernel_size, but is None."
            reduction_channels = self.in_channels // 3
            self.pyramid_pooling = PyramidSpatioTemporalPooling(
                self.in_channels, reduction_channels, pool_sizes)
            agg_in_channels += len(pool_sizes) * reduction_channels

        # Feature aggregation
        self.aggregation = nn.Sequential(
            conv_1x1x1_norm_activated(agg_in_channels, self.out_channels),)

        if self.out_channels != self.in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(self.in_channels, self.out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm3d(self.out_channels),
            )
        else:
            self.projection = None

    def forward(self, *inputs):
        (x,) = inputs
        x_paths = []
        for conv in self.convolution_paths:
            x_paths.append(conv(x))
        x_residual = torch.cat(x_paths, dim=1)
        if self.use_pyramid_pooling:
            x_pool = self.pyramid_pooling(x)
            x_residual = torch.cat([x_residual, x_pool], dim=1)
        x_residual = self.aggregation(x_residual)

        if self.out_channels != self.in_channels:
            x = self.projection(x)
        x = x + x_residual
        return x



class V2X_fusion(nn.Module):
    """ Temporal block with the following layers:
        - 2x3x3, 1x3x3, spatio-temporal pyramid pooling
        - dropout
        - skip connection.
    """

    def __init__(
            self,
            in_channels,
            num_agents,
            input_shape,
            grid_conf=None,
            start_out_channels=64,
            extra_in_channels=0,
            n_spatial_layers_between_temporal_layers=0,
            use_pyramid_pooling=True,
            input_relative_pose=False,
            with_skip_connect=False,
            compression_conv=False,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.compression_conv = compression_conv
        self.input_relative_pose = input_relative_pose
        self.warper = FeatureWarper(grid_conf=grid_conf)

        h, w = input_shape
        modules = []

        block_in_channels = in_channels
        block_out_channels = start_out_channels

        if self.input_relative_pose:
            # using 6DoF ego_pose as extra features for input
            block_in_channels += 6

        n_agents_layers = num_agents - 1
        # n_agents_layers = 1

        for _ in range(n_agents_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(block_out_channels,
                             block_out_channels, kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += extra_in_channels



        self.model = nn.Sequential(*modules)
        self.out_channels = block_in_channels

        if self.compression_conv:
            self.compression = nn.Sequential(
                nn.Conv2d(block_out_channels * self.num_agents, block_out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(block_out_channels),
                nn.ReLU(inplace=True),
            )

        # skip connection to stablize the present features
        self.with_skip_connect = with_skip_connect

    def forward(self, x, relative_pose, aug_transform=None):
        # x = self.warper.cumulative_warp_features_v2x(x, relative_pose, mode='bilinear',
        #                                                      bev_transform=aug_transform)
        x = self.warper.cumulative_warp_features_v2x(x, relative_pose, mode='bilinear',
                                                            bev_transform=aug_transform)

        # pdb.set_trace()

        # features.shape [1, 5, 64, 128, 128] wants to convert to [1, 64, 128, 128]
        batch_size, agent_length, channel, output_h, output_w = x.shape
        x = x.view(batch_size * agent_length, channel, output_h, output_w)
        ego_feature = x[0]

        for agent_channel_idx in range(1, x.shape[0]):
            # for agent_channel_idx, agent_feature in enumerate(x[1:]):
            agent_feature = x[agent_channel_idx]
            mask = torch.abs(agent_feature).mean(axis=0) < 0.001
            agent_feature[:, mask] = ego_feature[:, mask]

        x = x.view(batch_size, agent_length, channel, output_h, output_w)


        # if want just avg bev feature
        ego_feature = x[:, 0].clone()
        output_feature = x.mean(axis=1)
        # big impact
        output_feature[:, 31] = ego_feature[:, 31]
        # little impact
        output_feature[:, 41:45] = ego_feature[:, 41:45]
        # little impact
        output_feature[:, 53] = ego_feature[:, 53]
        # big impact
        output_feature[:, 57] = ego_feature[:, 57]
        # little impact
        output_feature[:, 60] = ego_feature[:, 60]

        return output_feature

        # output = x.mean(axis=0, keepdim=True)


        # input_x = x.clone()



        # _, C, imH, imW = output.shape
        # output = output.view(agent_length, batch_size, C, imH, imW)
        # output = output.swapaxes(0, 1)
        #
        # output = output.reshape(batch_size, agent_length * C, imH, imW)
        #
        # x = x.view(batch_size, agent_length, C, imH, imW)
        #
        #
        #
        # # import pdb
        # # pdb.set_trace()
        # # test_array = x[0].detach().clone().cpu().numpy()
        # # # heatmap2d([test_array[0][31].transpose(1, 2, 0), test_array[1][31].transpose(1, 2, 0), test_array[2][31].transpose(1, 2, 0)])
        # # # test_array.max(dim=1)
        # # test_array2 = test_array.max(axis=1)
        # # heatmap2d([test_array2[0], test_array2[1], test_array2[2]])
        #
        # # pdb.set_trace()
        # # when warping features from temporal frames, the bev-transform should be considered
        # x = self.warper.cumulative_warp_features(
        #     x, relative_pose[:, :x.shape[1]],
        #     mode='bilinear', bev_transform=aug_transform,
        # )
        # # if True:
        # #     return x[:, 1, :]

        if self.input_relative_pose:
            b, s, _, h, w = x.shape
            input_relative_pose = relative_pose.contiguous(
            )
            # (2, 3, 6, 128, 128)
            input_relative_pose = input_relative_pose.view(
                b, s, -1, 1, 1).expand(b, s, -1, h, w)

            # input_relative_pose = torch.cat((torch.zeros_like(
            #     input_relative_pose[:, :1]), input_relative_pose[:, :-1]), dim=1)
            input_relative_pose = torch.cat((torch.zeros_like(
                input_relative_pose[:, :1]), input_relative_pose[:, 1:]), dim=1)

            # x feature order t - 1, t - 0.5, t
            x = torch.cat((x, input_relative_pose), dim=2)

        # # x with shape [b, t, c, h, w]
        # x_valid = img_is_valid[:, :self.receptive_field]
        # for i in range(x.shape[0]):
        #     if x_valid[i].all():
        #         continue
        #     pdb.set_trace()
        #     invalid_index = torch.where(~x_valid[i])[0][0]
        #     valid_feat = x[i, invalid_index + 1]
        #     x[i, :(invalid_index + 1)] = valid_feat

        # Reshape input tensor to (batch, C, time, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)

        # x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.permute(0, 2, 1, 3, 4)

        # visualize_feature_no_normalize(x[0])
        # visualize_feature_no_normalize(input_x[0])
        # both x & input_x have the shape of (batch, time, C, H, W)

        # if self.with_skip_connect:
        #     x += input_x

        # visualize_feature_no_normalize(x[0])

        # pdb.set_trace()
        # test_array = x[0].detach().clone().cpu().numpy()
        # test_array2 = test_array.max(axis=1)
        # heatmap2d([test_array2[0], test_array2[1], test_array2[2]])

        # return features of the present frame
        # x = x[:, 0]

        # visualize_feature(x[0])

        # import pdb
        # pdb.set_trace()
        if self.compression_conv:
            batch_size, agent_length, channel, output_h, output_w = x.shape
            x = x.reshape(batch_size, agent_length*channel, output_h, output_w)
            x = self.compression(x)
        else:
            x = x.mean(axis=1)

        # visualize_feature(x)

        return x


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


def heatmap2d_no_normalize(arr_list):
    import matplotlib.pyplot as plt
    import numpy as np
    if torch.is_tensor(arr_list):
        array_tmp = arr_list.detach().clone().cpu().numpy()
        # batch, channel, height, width = array_tmp.shape
        # array_tmp = array_tmp[:, 0, :, :]

    for i, arr in enumerate(array_tmp):
        arr_show = arr.copy()
        # arr_show -= arr_show.mean()
        # arr_show /= arr_show.std()
        # # # feature = np.clip(feature, 0, 255).astype('uint8')
        # # arr_show = arr_show.astype('float32')
        # arr_show *= 64
        # arr_show += 128
        # arr_show = np.clip(arr_show, 0, 255).astype('uint8')

        plt.figure('Figure %d' % i)
        plt.imshow(arr_show, cmap='viridis')
        plt.colorbar()
    plt.show()


def visualize_feature_no_normalize(input_feature):
    import numpy as np
    features = input_feature.detach().clone().cpu().numpy()
    avg_feature = []
    for agent_idx in range(features.shape[0]):
        feature = np.mean(features[agent_idx], axis=0)
        avg_feature.append(feature)

    avg_feature = np.array(avg_feature)
    avg_feature = torch.from_numpy(avg_feature)
    heatmap2d_no_normalize(avg_feature)