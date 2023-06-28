# Modified from: https://github.com/OpenGVLab/InternImage/blob/master/detection/mmdet_custom/models/backbones/intern_image.py
# Copyright (c) chenjie04. All rights reserved.

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
from mmdet.utils import ConfigType
from mmcv.cnn import build_activation_layer, build_norm_layer
from timm.models.layers import DropPath

from ops_dcnv3.functions import DCNv3Function
from ops_dcnv3.modules.dcnv3 import (
    to_channels_first,
    to_channels_last,
    _is_power_of_2,
)

to_channels_first = to_channels_first()
to_channels_last = to_channels_last()


# -------------------------------------------------------------------------
# 分组可变形卷积
# -------------------------------------------------------------------------
class GroupDCNv3Layer(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        groups=4,
        offset_scale=1.0,
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
        act_cfg=dict(type='GELU'),
    ):

        super().__init__()
        if channels % groups != 0:
            raise ValueError(
                f"channels must be divisible by group, but got {channels} and {groups}"
            )
        _d_per_group = channels // groups
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = groups
        self.group_channels = channels // groups
        self.offset_scale = offset_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels,
            ),
            build_norm_layer(
                norm_cfg,
                channels,
            )[1],
            build_activation_layer(act_cfg),
        )
        self.offset = nn.Linear(channels, groups * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, groups * kernel_size * kernel_size)

        self._reset_parameters()


    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        constant_(self.mask.weight.data, 0.0)
        constant_(self.mask.bias.data, 0.0)


    def forward(self, input):
        """
        :param query                       (N, C, H, W)
        :return output                     (N, C, H, W)
        """

        x = to_channels_last(input)
        N, H, W, _ = x.shape
        dtype = x.dtype

        x1 = input
        x1 = self.dw_conv(x1)
        x1 = to_channels_last(x1)
        offset = self.offset(x1).type(dtype)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)

        x = DCNv3Function.apply(
            x.contiguous(),
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            256,
        )

        x = to_channels_first(x)
        return x

# ------------------------------------------------------------------------------
# Feed Forward 模块
# ------------------------------------------------------------------------------
class FeadForward(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        mlp_ratio: float = 1.0,
        drop_rate: float = 0.1,
        act_cfg: ConfigType = dict(type="GELU"),
    ):
        super(FeadForward, self).__init__()
        self.in_channel = in_channel
        self.mlp_ratio = mlp_ratio
        self.hidden_fetures = int(in_channel * mlp_ratio)

        self.input_project = nn.Conv2d(in_channel, self.hidden_fetures , kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(self.hidden_fetures, self.hidden_fetures, kernel_size=3, padding=1, groups=self.hidden_fetures, bias=True)

        self.output_project = nn.Conv2d(self.hidden_fetures, self.in_channel, kernel_size=1, bias=True)  # 1x1 conv

        self.act = build_activation_layer(act_cfg)

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        """

        :param input: [bs, C, H, W]
        :return: [bs, C, H, W]
        """

        # feed forward
        x = self.input_project(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.dwconv(x)
        x = self.output_project(x)
        return x
    
# ------------------------------------------------------------------------------
# 参考Transformer模块的设计，使用DCN-v3替代多头注意力封装一个ModulatedDeformableConvModule
# ------------------------------------------------------------------------------
class GroupDeformableConvModule(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        groups: int = 4,
        kernel_size: int = 3,
        dilation: int = 1,
        mlp_ratio: float = 1.0,
        drop_rate: float = 0.1,
        norm_cfg: ConfigType = dict(type='GN', num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        drop_path: float = 0.,
        layer_scale = None,
    ) -> None:
        super().__init__()
        # Normalization:
        _, self.norm1 = build_norm_layer(norm_cfg, channels)
        _, self.norm2 = build_norm_layer(norm_cfg, channels)

        # DCN-v3
        self.dcn_v3 = GroupDCNv3Layer(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            offset_scale=1.0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )


        self.feed_forward = FeadForward(channels, mlp_ratio=mlp_ratio, drop_rate=drop_rate, act_cfg=act_cfg)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(1, channels, 1, 1), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(1, channels, 1, 1), requires_grad=True
            )


    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.dcn_v3(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.feed_forward(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.dcn_v3(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.feed_forward(self.norm2(x)))
        return x

