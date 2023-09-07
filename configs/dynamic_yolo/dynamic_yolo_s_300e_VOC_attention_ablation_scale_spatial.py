_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../_base_/datasets/coco_detection.py', './dynamic_yolo_tta.py'
]

custom_imports = dict(
    imports=["dynamic_yolo", "backbone", "neck", "head"], allow_failed_imports=False
)

img_scale = (640, 640)  # width, height

# model settings
model = dict(
    type='DynamicYOLO',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='LightInternImage',
        channels=128,
        depths=[8, 8, 4],
        groups=[4, 8, 16],
        mlp_ratios=[1.0, 1.0, 1.0],
        drop_rate=0.1,
        drop_path_rate=0.1,
        layer_scale=1.0,
        post_norm=False,
        out_indices=(0, 1, 2),
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
        act_cfg=dict(type='GELU'),
        ),
    neck=dict(
        type="ScaleSpatialAttnNeck",
        in_channels=[128, 256, 512],
        out_channel=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate=0.1,
        drop_path=0.0,
        layer_scale=1.0,
        num_fusion_block=4,
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
        act_cfg=dict(type='GELU'),
        ),
    bbox_head=dict(
        type='SepDecoupleHead',
        num_classes=20,
        in_channels=128,
        feat_channels=128,
        groups=4,
        mlp_ratio=1.0,
        drop_rate=0.1,
        drop_path=0.0,
        layer_scale=1.0,
        stacked_convs=2,
        norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
        act_cfg=dict(type='GELU'),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        pred_kernel_size=1,),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
    )

# dataset settings
data_root = "../data/VOCdevkit/"
dataset_type = "CocoDataset"

classes = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

metainfo=dict(classes=classes)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=10,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='voc0712_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='voc0712_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='voc07_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'voc0712_val.json',
    metric='bbox',
    proposal_nums=(100, 1, 10))
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'voc07_test.json',
    metric='bbox',
    proposal_nums=(100, 1, 10))

# training settings
max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.001
interval = 10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16', # valid values: ('float16', 'bfloat16', None)
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    clip_grad=dict(max_norm=35, norm_type=2),
    )

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000), 
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),
    dict(
        type='NumClassCheckHook',
    )
]

auto_scale_lr = dict(enable=False, base_batch_size=16)


vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend',  init_kwargs=dict(magic=True, project="Dynamic-YOLO-VOC"))]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', save_dir='result')
