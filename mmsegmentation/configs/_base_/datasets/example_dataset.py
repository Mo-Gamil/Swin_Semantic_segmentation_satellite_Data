dataset_type = 'ExampleDataset'
data_root = 'D:\swin2\mmsegmentation\data\my_dataset_8bands'
# data_root = 'D:\swin2\mmsegmentation\data'


crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadTifImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(448, 224),  # Adjusted to be 2x of the target crop_size
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTifImageFromFile'),
    dict(type='Resize', scale=(448, 224), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', 
            seg_map_path='ann_dir/train'),
            # img_path='sentinel2_images_downloaded_resized_normalized', 
            # seg_map_path='CDL_patches'),
        pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', 
            seg_map_path='ann_dir/val'),
            #  img_path='sentinel2_images_downloaded_resized_normalized', 
            # seg_map_path='CDL_patches'),
        pipeline=test_pipeline)
)
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/test', 
            seg_map_path='ann_dir/test'),
        pipeline=test_pipeline)
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
