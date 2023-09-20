# Modified from: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/dyhead.py
# Copyright (c) chenjie04. All rights reserved.


import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from dcn_v3 import GroupDeformableConvModule
from mmdet.models.layers import DyReLU
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d


@MODELS.register_module()
class MultiScaleAttentionFusion(BaseModule):
    """A multi-scale features fusion strategy Base on spatial attention and scale attention.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_fusion_block (int): Number of fusion block.
        reduction=4 (int): The reduction ratio of channel in compression of query and key.
        num_heads=2 (int): Number of heads in multi-head attention.
        mlp_ratio=4 (int): The expansion ratio of channel in feed forward.
        interpolate_mode = "bilinear" (str): The mode of interprolation for up-sampling and down-sampling.
        act_cfg=dict(type="Swish") (dict): The config of activation.
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0) (dict): The config of activation in scale attenion module.
        init_cfg=None,
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(MultiScaleAttentionFusion, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                CustomAttnFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class CustomAttnFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            spatial_attn = GroupDeformableConvModule(
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=drop_path,
                layer_scale=layer_scale,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_spatial_attn", spatial_attn)

            channel_activation = DyReLU(channels, act_cfg=(act_cfg, scale_act_cfg))
            self.add_module(f"level_{level}_channel_activation", channel_activation)

            if level > 0:
                channel_activation_low = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_low", channel_activation_low
                )

            if level < num_levels - 1:
                channel_activation_high = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_high", channel_activation_high
                )

        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            # nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
            build_activation_layer(scale_act_cfg),
        )

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            channel_activation = getattr(self, f"level_{level}_channel_activation")
            mid_feat = channel_activation(mid_feat)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_low = getattr(
                    self, f"level_{level}_channel_activation_low"
                )
                low_feat = channel_activation_low(low_feat)
                sum_feat = sum_feat + low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_high = getattr(
                    self, f"level_{level}_channel_activation_high"
                )
                high_feat = channel_activation_high(high_feat)
                sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1

            spatial_attn_module = getattr(self, f"level_{level}_spatial_attn")
            outs.append(spatial_attn_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class MultiScaleAttentionFusion_test(BaseModule):
    """A multi-scale features fusion strategy Base on spatial attention and scale attention.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_fusion_block (int): Number of fusion block.
        reduction=4 (int): The reduction ratio of channel in compression of query and key.
        num_heads=2 (int): Number of heads in multi-head attention.
        mlp_ratio=4 (int): The expansion ratio of channel in feed forward.
        interpolate_mode = "bilinear" (str): The mode of interprolation for up-sampling and down-sampling.
        act_cfg=dict(type="Swish") (dict): The config of activation.
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0) (dict): The config of activation in scale attenion module.
        init_cfg=None,
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(MultiScaleAttentionFusion_test, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        # 初始融合
        for level in range(self.num_levels):
            spatial_attn = nn.Sequential(
                ModulatedDeformConv2d(
                    in_channels[level],
                    out_channel,
                    3,
                    padding=1,
                    deform_groups=in_channels[level] / 32,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, out_channel)[1],
            )
            self.add_module(f"init_{level}_spatial_attn", spatial_attn)

            channel_activation = DyReLU(out_channel, act_cfg=(act_cfg, scale_act_cfg))
            self.add_module(f"init_{level}_channel_activation", channel_activation)

            if level > 0:
                channel_activation_low = DyReLU(
                    out_channel, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"init_{level}_channel_activation_low", channel_activation_low
                )

            if level < self.num_levels - 1:
                channel_activation_high = DyReLU(
                    out_channel, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"init_{level}_channel_activation_high", channel_activation_high
                )

        self.init_scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, 1, 1),
            # nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
            build_activation_layer(scale_act_cfg),
        )

        # 后续融合
        if self.num_fusion_block > 1:
            follow_fusion_blocks = []
            for idx in range(self.num_fusion_block):
                follow_fusion_blocks.append(
                    CustomAttnFusionBlock(
                        channels=out_channel,
                        groups=groups,
                        mlp_ratio=mlp_ratio,
                        drop_rate=drop_rate,
                        drop_path=drop_path,
                        layer_scale=layer_scale,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        scale_act_cfg=scale_act_cfg,
                    )
                )

        self.follow_fusion_block = nn.Sequential(*follow_fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 初始融合
        init_spatial_outs = []
        for level in range(len(inputs)):
            feat = inputs[level]
            spatial_attn_module = getattr(self, f"init_{level}_spatial_attn")
            init_spatial_outs.append(spatial_attn_module(feat))

        init_outs = []
        for level in range(len(init_spatial_outs)):
            mid_feat = init_spatial_outs[level]
            channel_activation = getattr(self, f"init_{level}_channel_activation")
            mid_feat = channel_activation(mid_feat)
            sum_feat = mid_feat * self.init_scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    init_spatial_outs[level - 1],
                    size=init_spatial_outs[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_low = getattr(
                    self, f"init_{level}_channel_activation_low"
                )
                low_feat = channel_activation_low(low_feat)
                sum_feat = sum_feat + low_feat * self.init_scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(init_spatial_outs) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    init_spatial_outs[level + 1],
                    size=init_spatial_outs[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_high = getattr(
                    self, f"init_{level}_channel_activation_high"
                )
                high_feat = channel_activation_high(high_feat)
                sum_feat = sum_feat + high_feat * self.init_scale_attn_module(high_feat)
                summed_levels += 1

            init_outs.append(sum_feat / summed_levels)

        # 后续融合
        outputs = self.follow_fusion_block(init_outs)

        return tuple(outputs)


# --------------------------------------------------------------------------------------------------------
# 消融实验
# --------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class BaselineNeck(BaseModule):
    """A multi-scale features fusion strategy Base on spatial attention and scale attention.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_fusion_block (int): Number of fusion block.
        reduction=4 (int): The reduction ratio of channel in compression of query and key.
        num_heads=2 (int): Number of heads in multi-head attention.
        mlp_ratio=4 (int): The expansion ratio of channel in feed forward.
        interpolate_mode = "bilinear" (str): The mode of interprolation for up-sampling and down-sampling.
        act_cfg=dict(type="Swish") (dict): The config of activation.
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0) (dict): The config of activation in scale attenion module.
        init_cfg=None,
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(BaselineNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                BaselineFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class BaselineFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            convFusion = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_conv_fusion", convFusion)

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            sum_feat = mid_feat
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + low_feat
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + high_feat
                summed_levels += 1

            conv_fusion_module = getattr(self, f"level_{level}_conv_fusion")
            outs.append(conv_fusion_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class ChannelAttnNeck(BaseModule):
    """A multi-scale features fusion strategy Base on spatial attention and scale attention.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_fusion_block (int): Number of fusion block.
        reduction=4 (int): The reduction ratio of channel in compression of query and key.
        num_heads=2 (int): Number of heads in multi-head attention.
        mlp_ratio=4 (int): The expansion ratio of channel in feed forward.
        interpolate_mode = "bilinear" (str): The mode of interprolation for up-sampling and down-sampling.
        act_cfg=dict(type="Swish") (dict): The config of activation.
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0) (dict): The config of activation in scale attenion module.
        init_cfg=None,
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(ChannelAttnNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                ChannelFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class ChannelFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            convFusion = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_conv_fusion", convFusion)

            channel_activation = DyReLU(channels, act_cfg=(act_cfg, scale_act_cfg))
            self.add_module(f"level_{level}_channel_activation", channel_activation)

            if level > 0:
                channel_activation_low = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_low", channel_activation_low
                )

            if level < num_levels - 1:
                channel_activation_high = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_high", channel_activation_high
                )

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            channel_activation = getattr(self, f"level_{level}_channel_activation")
            mid_feat = channel_activation(mid_feat)
            sum_feat = mid_feat
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_low = getattr(
                    self, f"level_{level}_channel_activation_low"
                )
                low_feat = channel_activation_low(low_feat)
                sum_feat = sum_feat + low_feat
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_high = getattr(
                    self, f"level_{level}_channel_activation_high"
                )
                high_feat = channel_activation_high(high_feat)
                sum_feat = sum_feat + high_feat
                summed_levels += 1

            conv_fusion_module = getattr(self, f"level_{level}_conv_fusion")
            outs.append(conv_fusion_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class ScaleAttnNeck(BaseModule):
    """A multi-scale features fusion strategy Base on spatial attention and scale attention.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_fusion_block (int): Number of fusion block.
        reduction=4 (int): The reduction ratio of channel in compression of query and key.
        num_heads=2 (int): Number of heads in multi-head attention.
        mlp_ratio=4 (int): The expansion ratio of channel in feed forward.
        interpolate_mode = "bilinear" (str): The mode of interprolation for up-sampling and down-sampling.
        act_cfg=dict(type="Swish") (dict): The config of activation.
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0) (dict): The config of activation in scale attenion module.
        init_cfg=None,
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(ScaleAttnNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                ScaleFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class ScaleFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            convFusion = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_conv_fusion", convFusion)

        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            # nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
            build_activation_layer(scale_act_cfg),
        )

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1

            conv_fusion_module = getattr(self, f"level_{level}_conv_fusion")
            outs.append(conv_fusion_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class SpatialAttnNeck(BaseModule):
    """A multi-scale features fusion strategy Base on spatial attention and scale attention.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_fusion_block (int): Number of fusion block.
        reduction=4 (int): The reduction ratio of channel in compression of query and key.
        num_heads=2 (int): Number of heads in multi-head attention.
        mlp_ratio=4 (int): The expansion ratio of channel in feed forward.
        interpolate_mode = "bilinear" (str): The mode of interprolation for up-sampling and down-sampling.
        act_cfg=dict(type="Swish") (dict): The config of activation.
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0) (dict): The config of activation in scale attenion module.
        init_cfg=None,
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(SpatialAttnNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                SpatialFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class SpatialFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            spatial_attn = GroupDeformableConvModule(
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=drop_path,
                layer_scale=layer_scale,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_spatial_attn", spatial_attn)

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            sum_feat = mid_feat
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + low_feat
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + high_feat
                summed_levels += 1

            spatial_attn_module = getattr(self, f"level_{level}_spatial_attn")
            outs.append(spatial_attn_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class ChannelScaleAttnNeck(BaseModule):
    """A multi-scale features fusion strategy Base on spatial attention and scale attention.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_fusion_block (int): Number of fusion block.
        reduction=4 (int): The reduction ratio of channel in compression of query and key.
        num_heads=2 (int): Number of heads in multi-head attention.
        mlp_ratio=4 (int): The expansion ratio of channel in feed forward.
        interpolate_mode = "bilinear" (str): The mode of interprolation for up-sampling and down-sampling.
        act_cfg=dict(type="Swish") (dict): The config of activation.
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0) (dict): The config of activation in scale attenion module.
        init_cfg=None,
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(ChannelScaleAttnNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                ChannelScaleFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class ChannelScaleFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            convFusion = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_conv_fusion", convFusion)

            channel_activation = DyReLU(channels, act_cfg=(act_cfg, scale_act_cfg))
            self.add_module(f"level_{level}_channel_activation", channel_activation)

            if level > 0:
                channel_activation_low = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_low", channel_activation_low
                )

            if level < num_levels - 1:
                channel_activation_high = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_high", channel_activation_high
                )

        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            # nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
            build_activation_layer(scale_act_cfg),
        )

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            channel_activation = getattr(self, f"level_{level}_channel_activation")
            mid_feat = channel_activation(mid_feat)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_low = getattr(
                    self, f"level_{level}_channel_activation_low"
                )
                low_feat = channel_activation_low(low_feat)
                sum_feat = sum_feat + low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_high = getattr(
                    self, f"level_{level}_channel_activation_high"
                )
                high_feat = channel_activation_high(high_feat)
                sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1

            conv_fusion_module = getattr(self, f"level_{level}_conv_fusion")
            outs.append(conv_fusion_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class ChannelSpatialAttnNeck(BaseModule):
    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(ChannelSpatialAttnNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                ChannelSpatialFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class ChannelSpatialFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            spatial_attn = GroupDeformableConvModule(
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=drop_path,
                layer_scale=layer_scale,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_spatial_attn", spatial_attn)

            channel_activation = DyReLU(channels, act_cfg=(act_cfg, scale_act_cfg))
            self.add_module(f"level_{level}_channel_activation", channel_activation)

            if level > 0:
                channel_activation_low = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_low", channel_activation_low
                )

            if level < num_levels - 1:
                channel_activation_high = DyReLU(
                    channels, act_cfg=(act_cfg, scale_act_cfg)
                )
                self.add_module(
                    f"level_{level}_channel_activation_high", channel_activation_high
                )

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            channel_activation = getattr(self, f"level_{level}_channel_activation")
            mid_feat = channel_activation(mid_feat)
            sum_feat = mid_feat
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_low = getattr(
                    self, f"level_{level}_channel_activation_low"
                )
                low_feat = channel_activation_low(low_feat)
                sum_feat = sum_feat + low_feat
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                channel_activation_high = getattr(
                    self, f"level_{level}_channel_activation_high"
                )
                high_feat = channel_activation_high(high_feat)
                sum_feat = sum_feat + high_feat
                summed_levels += 1

            spatial_attn_module = getattr(self, f"level_{level}_spatial_attn")
            outs.append(spatial_attn_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class ScaleSpatialAttnNeck(BaseModule):
    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_fusion_block=2,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ):
        super(ScaleSpatialAttnNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.num_fusion_block = num_fusion_block

        channel_mapper_cfg = dict(
            type="ChannelMapper",
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_outs=len(in_channels),
        )

        self.channel_mapper = MODELS.build(channel_mapper_cfg)

        fusion_blocks = []
        for idx in range(self.num_fusion_block):
            fusion_blocks.append(
                ScaleSpatialFusionBlock(
                    channels=out_channel,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path,
                    layer_scale=layer_scale,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    scale_act_cfg=scale_act_cfg,
                )
            )

        self.fusion_block = nn.Sequential(*fusion_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outputs = self.channel_mapper(inputs)

        outputs = self.fusion_block(outputs)

        return tuple(outputs)


class ScaleSpatialFusionBlock(BaseModule):
    def __init__(
        self,
        channels,
        groups,
        mlp_ratio,
        drop_rate: float = 0.2,
        drop_path: float = 0.005,
        layer_scale: float = 1.0,
        num_levels=3,
        norm_cfg: ConfigType = dict(type="GN", num_groups=1, requires_grad=True),
        act_cfg: ConfigType = dict(type="GELU"),
        scale_act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        for level in range(num_levels):
            spatial_attn = GroupDeformableConvModule(
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=drop_path,
                layer_scale=layer_scale,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.add_module(f"level_{level}_spatial_attn", spatial_attn)

        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            # nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
            build_activation_layer(scale_act_cfg),
        )

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = x[level]
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1

            spatial_attn_module = getattr(self, f"level_{level}_spatial_attn")
            outs.append(spatial_attn_module(sum_feat / summed_levels))

        return outs
