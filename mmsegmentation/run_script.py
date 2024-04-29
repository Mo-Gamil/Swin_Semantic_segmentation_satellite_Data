from mmseg.datasets import ExampleDataset
from mmengine.registry import init_default_scope
import torchvision.transforms as transforms
import os,rasterio
from torch.utils.data import DataLoader, Dataset
init_default_scope('mmseg')
import pandas as pd

import copy


data_root = 'D:/swin2/mmsegmentation/data/my_dataset_8bands'
yield_train_file = "D:/swin2/mmsegmentation/data/my_dataset_8bands/ann_dir/train/crop_yield_Iowa_2021_train_1to1000.csv"



# corn_yield
data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train')
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

dataset = ExampleDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False,yield_file=yield_train_file, pipeline=train_pipeline)

print(dataset[10])













































# data_root = 'D:\swin2\mmsegmentation\data\my_dataset'
# data_root = 'D:/swin2/mmsegmentation/data/dual_model_data/my_dataset_8bands'



# # corn_yield
# data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val')
# train_pipeline = [
#     dict(type='LoadTifImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='RandomCrop', crop_size=(224, 224), cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackSegInputs')
# ]



# dataset = ExampleDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)

# print(dataset[0])

# yield_csv_file = "D:/swin2/mmsegmentation/data/dual_model_data/my_dataset/ann_dir/val/val_data.csv"
# yield_data = pd.read_csv(yield_csv_file)


# # Assuming you have a way to match indices to yields
# for i in range(len(dataset)):
#     # img_path = dataset.get_img_info(i)['filename']
#     img_path = dataset[i]['data_samples'].img_path

#     index_val = img_path.split('.tif')[0].split('\\')[-1]
#     corn_yield = yield_data[yield_data['Index'] == index_val]['Corn_Crop'].values[0]
#     soybean_yield = yield_data[yield_data['Index'] == index_val]['Soy_crop'].values[0]
#     dataset.add_yield_data(i, corn_yield, soybean_yield)

# # Now each item will include yield data
# data_item = dataset[10]
# print(data_item.keys())  # This should now include 'corn_yield' and 'soybean_yield'
# print(data_item)













# dataset = ExampleDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)
# # print(len(dataset))
# # print(dataset.get_data_info)
# # print(dataset.metainfo)
# # import mmseg
# # print(mmseg.__version__)
# deep_copied_dataset = copy.copy(dataset)
# yield_csv_file = "D:/swin2/mmsegmentation/data/dual_model_data/my_dataset/ann_dir/val/val_data.csv"
# yield_data = pd.read_csv(yield_csv_file)






# Ensure the yield_data DataFrame is properly formatted and available with 'Index', 'Corn_Crop', and 'Soy_crop' columns

# for i in range(len(dataset)):
#     try:
#         # Extract the image path and derive the index value
#         img_path = dataset[i]['data_samples'].img_path
#         index_val = img_path.split('.tif')[0].split('\\')[-1]

#         # Fetch the corresponding crop yields using the index value
#         corn_yield = yield_data[yield_data['Index'] == index_val]['Corn_Crop'].values[0]
#         soybean_yield = yield_data[yield_data['Index'] == index_val]['Soy_crop'].values[0]

#         # Prepare the dictionary with crop yield data
#         yield_dict = {'corn': corn_yield, 'soybean': soybean_yield}

#         # Update the dataset entry with the new data
#         dataset[i].update(yield_dict)

#     except IndexError:
#         # Handle cases where the yields are not found for the index
#         print(f'Yield data not found for index {index_val}')
#     except Exception as e:
#         # General error handling
#         print(f'An error occurred: {e}')

# print(dataset[0].keys())

# print(vars(dataset[0])) 
# print(dataset[0].keys())


# dataset = dataset_original.copy()
# for i in range(len(deep_copied_dataset)):
#     img_path = deep_copied_dataset[i]["data_samples"].img_path       
#     index_val = img_path.split(".tif")[0].split("\\")[-1]
#     # print(index_val)
#     corn_yield = yield_data[yield_data['Index'] == index_val]["Corn_Crop"].values[0]
#     soybean_yield = yield_data[yield_data['Index'] == index_val]["Soy_crop"].values[0]
#     # print(corn_yield)
#     # # print(soybean_yield)
#     yield_dict = {"corn":corn_yield,"soybean": soybean_yield}
#     # print(yield_dict)
#     deep_copied_dataset[i].update(yield_dict)
#     # print(dataset[i].keys())
#     # break

# # dataset[1]["test"] = "Hdfsdfsdf"
# print(deep_copied_dataset[1].keys())