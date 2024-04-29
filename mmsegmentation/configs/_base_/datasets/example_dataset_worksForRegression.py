dataset_type = 'ExampleDataset'
data_root = 'D:\swin2\mmsegmentation\data\my_dataset_8bands'
# data_root = 'D:\swin2\mmsegmentation\data'
# Paths to the CSV files for each dataset
yield_train_file = r'D:\swin2\mmsegmentation\data\my_dataset_8bands\ann_dir\train\crop_yield_Iowa_2021_train_1to1000.csv'
yield_val_file = r"D:\swin2\mmsegmentation\data\my_dataset_8bands\ann_dir\val\crop_yield_Iowa_2021_val_1001_to1446.csv"
yield_test_file = r"D:\swin2\mmsegmentation\data\my_dataset_8bands\ann_dir\test\crop_yield_Illinois_2021_test_1_to300.csv"

crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadTifImageFromFile'),
    # dict(type='LoadAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(448, 224),  # Adjusted to be 2x of the target crop_size
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTifImageFromFile'),
    dict(type='Resize', scale=(448, 224), keep_ratio=True),
    # dict(type='LoadAnnotations'),
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
            yield_file=yield_train_file,
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
        pipeline=test_pipeline,
        yield_file=yield_val_file )
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
        pipeline=test_pipeline,
        yield_file=yield_test_file)
)

val_evaluator = dict(type='RegressionMetrics', metrics=['rmse'])
test_evaluator = val_evaluator

# val_evaluator = {'type': 'RegressionMetrics', 'metrics': ['rmse']}
# test_evaluator = val_evaluator






















# dataset_type = 'ExampleDataset'
# data_root = 'D:\\swin2\\mmsegmentation\\data\\my_dataset'
# # data_root = 'D:\\swin2\\mmsegmentation\\data'

# crop_size = (224, 224)
# train_pipeline = [
#     dict(type='LoadTifImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(
#         type='RandomResize',
#         scale=(448, 224),  # Adjusted to be 2x of the target crop_size
#         ratio_range=(0.5, 2.0),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
# test_pipeline = [
#     dict(type='LoadTifImageFromFile'),
#     dict(type='Resize', scale=(448, 224), keep_ratio=True),
#     dict(type='LoadAnnotations'),
#     dict(type='PackSegInputs')
# ]

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
#         yield_data_path='D:\\swin2\\mmsegmentation\\data\\dual_model_data\\my_dataset\\ann_dir\\val\\val_data.csv',  # Path to the yield data CSV
#         pipeline=train_pipeline)
# )
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
#         yield_data_path='D:\\swin2\\mmsegmentation\\data\\dual_model_data\\my_dataset\\ann_dir\\test\\test_data.csv',  # Path to the validation yield data CSV
#         pipeline=test_pipeline)
# )
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
#         yield_data_path='D:\\swin2\\mmsegmentation\\data\\my_dataset\\ann_dir\\test\\test_yield_data.csv',  # Path to the test yield data CSV
#         pipeline=test_pipeline)
# )

# val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = val_evaluator
















# from mmseg.transforms import DefaultFormatBundle












# dataset_type = 'ExampleDataset'
# data_root = r'D:\swin2\mmsegmentation\data\my_dataset_8bands'
# yield_train_file = r'D:\swin2\mmsegmentation\data\my_dataset_8bands\ann_dir\train\crop_yield_Iowa_2021_train_1to1000.csv'
# yield_val_file = r'D:\swin2\mmsegmentation\data\my_dataset_8bands\ann_dir\val\crop_yield_Iowa_2021_val_1001_to1446.csv'
# yield_test_file = r'D:\swin2\mmsegmentation\data\my_dataset_8bands\ann_dir\test\crop_yield_Illinois_2021_test_1_to300.csv'

# crop_size = (224, 224)
# train_pipeline = [
#     dict(type='LoadTifImageFromFile'),
#     dict(type='RandomResize', scale=(448, 224), ratio_range=(0.5, 2.0), keep_ratio=True),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     # dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#     # dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'corn_yield', 'soy_yield'])
# ]

# # train_pipeline = [
# #     dict(type='LoadTifImageFromFile'),
# #     # dict(type='LoadAnnotations'),
# #     dict(
# #         type='RandomResize',
# #         scale=(448, 224),  # Adjusted to be 2x of the target crop_size
# #         ratio_range=(0.5, 2.0),
# #         keep_ratio=True),
# #     # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
# #     # dict(type='RandomFlip', prob=0.5),
# #     # dict(type='PhotoMetricDistortion'),
# #     dict(type='PackSegInputs')
# #     ]





# test_pipeline = [
#     dict(type='LoadTifImageFromFile'),
#     dict(type='Resize', scale=(448, 224), keep_ratio=True),
#     # dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#     # dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img'])
# ]

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='img_dir/train'),
#         yield_file=yield_train_file,
#         pipeline=train_pipeline)
# )
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='img_dir/val'),
#         pipeline=test_pipeline,
#         yield_file=yield_val_file)
# )
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='img_dir/test'),
#         pipeline=test_pipeline,
#         yield_file=yield_test_file)
# )

# # Replace 'IoUMetric' with appropriate metrics for regression, e.g., MSE
# val_evaluator = dict(type='RegressionMetrics', metrics=['MSE'])
# test_evaluator = val_evaluator
