# Modified from: https://github.com/OpenGVLab/InternImage/blob/master/detection/mmdet_custom/models/backbones/intern_image.py
#                https://github.com/hunto/LightViT/blob/main/detection/lightvit.py
# Copyright (c) chenjie04. All rights reserved.


import torch
import torch.nn as nn
from mmcv.cnn import (
    build_norm_layer,
    build_activation_layer,
)
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from dcn_v3 import GroupDeformableConvModule
from ops_dcnv3.modules.dcnv3 import (
    to_channels_first,
    to_channels_last,
)

to_channels_first = to_channels_first()
to_channels_last = to_channels_last()


class StemLayer(nn.Module):
    r"""Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        embed_dim (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=128,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
    ):
        super().__init__()

        stem_norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)

        stem_dim = embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3,
                      stride=2, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),

            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),

            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),

            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=2, padding=1, bias=False),
            build_norm_layer(stem_norm_cfg, stem_dim)[1],
            build_activation_layer(act_cfg),
        )
        self.proj = nn.Conv2d(stem_dim, embed_dim,
                              kernel_size=3,
                              stride=2, padding=1)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        stem = self.stem(x)
        x = self.proj(stem)
        x = self.norm(x)
        return x


class DownsampleLayer(nn.Module):
    r"""Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(
        self,
        channels,
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = build_norm_layer(norm_cfg, 2 * channels)[1]

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class InternImageBlock(nn.Module):
    r"""Block of InternImage
    Args:
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
    """

    def __init__(
        self,
        channels,
        depth,
        groups,
        downsample=True,
        mlp_ratio=1.0,
        drop_rate=0.2,    
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
        post_norm=False,
        drop_path=0.0,
        layer_scale=None,
    ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList(
            [
                GroupDeformableConvModule(
                    channels=channels,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )
        if not self.post_norm:
            self.norm = build_norm_layer(norm_cfg, channels)[1]
        self.downsample = (
            DownsampleLayer(channels=channels, norm_cfg=norm_cfg)
            if downsample
            else None
        )

    def forward(self, x, return_wo_downsample=False):
        for blk in self.blocks:
            x = blk(x)
        if not self.post_norm:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


@MODELS.register_module()
class LightInternImage(BaseModule):
    r"""LightInternImage
        A light-weight impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 128
        depths (list): Depth of each block. Default: [8, 8, 4]
        groups (list): Groups of each block. Default: [4, 8, 16]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 1.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
    """

    def __init__(
        self,
        channels=128,
        depths=[8, 8, 1],
        groups=[4, 8, 16],
        mlp_ratios=[1.0, 1.0, 1.0],
        drop_rate=0.0,
        pos_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_type="linear",
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
        layer_scale=None,
        post_norm=False,
        out_indices=(1, 2, 3),
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2 ** (self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratios = mlp_ratios
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        print(f"using activation layer: {act_cfg}")
        print(f"using main norm layer: {norm_cfg}")
        print(f"using dpr: {drop_path_type}, {drop_path_rate}")

        in_chans = 3
        self.patch_embed = StemLayer(
            in_chans=in_chans, embed_dim=channels, act_cfg=act_cfg, norm_cfg=dict(type='GN', num_groups=int(channels/32), requires_grad=True)
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if drop_path_type == "uniform":
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = InternImageBlock(
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratios[i],
                drop_rate=drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                act_cfg=act_cfg,
                norm_cfg=dict(type='GN', num_groups=groups[i], requires_grad=True),
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        self.num_layers = len(depths)


    def forward(self, x):

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, x_ = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                seq_out.append(x_.contiguous())

        return seq_out
