_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/example_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100.py'
]
crop_size = (
    224,
    224,
)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=False,
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size=None,  # Ensure only one of these is set
    size_divisor=32  # Or set this as needed but not both
)

model = dict(
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        # ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        # use_checkpoint=False
    ),
    decode_head=dict(
        # type='CropYieldRegressionHead',
        in_channels=768,  # This should be the last layer's number of features from the backbone
        num_crops=2, 
        train_cfg = {
            'loss': {
                'loss_type': 'mse',  # Choose from 'mse', 'mae', or 'huber'
                'delta': 1.0  # Used only if 'huber' is selected
            }
                    }
    )
    )

# train_cfg = dict()  # train_cfg is just a place holder for now.
# test_cfg = dict(mode='whole')  # The test mode, options are 'whole' and 'sliding'. 'whole': whole image fully-convolutional test. 'sliding': sliding crop window on the image.
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=5,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU


# In your config file
log_config = dict(
    interval=50,  # Log every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),  # Console logging
        dict(type='TensorBoardLoggerHook')  # TensorBoard logging
    ]
)

# evaluator = dict(
#     type='RegressionMetrics'
# )
# val_evaluator = dict(type='RegressionMetrics')
# test_evaluator = dict(type='RegressionMetrics')

data=dict(samples_per_gpu=2)