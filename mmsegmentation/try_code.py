from mmseg.datasets import CustomDataset
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

from mmengine.registry import TRANSFORMS

# data_root = 'D:\swin2\mmsegmentation\data\my_dataset'

# data_prefix=dict(img_path='img_dir/train_jpg', seg_map_path='ann_dir/train_jpg')



# from mmengine.registry import TRANSFORMS
# from osgeo import gdal
# import numpy as np


# import torch
# from osgeo import gdal
# import numpy as np
# @TRANSFORMS.register_module()
# class LoadImageFromTIFWithGDAL:
#     """Load a .tif image file using GDAL and convert it to a PyTorch tensor."""

#     def __init__(self, to_float32=False, to_tensor=True):
#         self.to_float32 = to_float32
#         self.to_tensor = to_tensor

#     def __call__(self, results):
#         """Call method to load image from the file path in results and format the output."""
#         filename = results['img_info']['filename']
#         dataset = gdal.Open(filename, gdal.GA_ReadOnly)
#         if dataset is None:
#             raise FileNotFoundError(f"Unable to open file {filename}")

#         img_bands = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
#         img = np.stack(img_bands, axis=-1)  # Stack bands along the last dimension

#         if self.to_float32:
#             img = img.astype(np.float32)
#         if self.to_tensor:
#             img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # Convert to (C, H, W) format

#         # Assuming you need to handle annotations and other meta info similarly
#         # Here we simulate `gt_sem_seg` with dummy data
#         # Normally, you would fetch and process actual annotation data similarly
#         dummy_gt = torch.randint(0, 10, img.shape[1:])  # Random segmentation map
#         gt_data = {
#             'data': dummy_gt
#         }

#         # Prepare the complete structured output as specified
#         structured_output = {
#             'inputs': img,
#             'data_samples': {
#                 'img_path': results['img_info']['filename'],
#                 'seg_map_path': results['img_info'].get('seg_map_path', 'No seg map path'),
#                 'img_shape': img.shape[1:],  # H, W
#                 'flip_direction': None,
#                 'ori_shape': img.shape[1:],  # H, W, simulate original shape
#                 'flip': False,
#                 'gt_sem_seg': gt_data
#             }
#         }
#         return structured_output






# dataset = CustomDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False)



# # load_image_from_tif.py
# import rasterio
# import numpy as np
# @TRANSFORMS.register_module()
# class LoadImageFromTIF:
#     """Load a .tif image file using rasterio."""

#     def __init__(self, to_float32=False):
#         self.to_float32 = to_float32

#     def __call__(self, results):
#         """Call method to load image from the file path in results."""
#         filename = results['img_info']['filename']
#         with rasterio.open(filename) as src:
#             img = src.read()  # Reads all bands by default
#             if self.to_float32:
#                 img = img.astype(np.float32)

#         # Assuming that the image is stored with bands first
#         img = np.transpose(img, (1, 2, 0))  # Reorder dimensions to Height x Width x Channels if needed
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         results['img_fields'] = ['img']
#         return results

crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadImageFromFile'),
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
data_root = 'D:\swin2'

data_prefix=dict(img_path='mmsegmentation', seg_map_path='mmsegmentation')

dataset = CustomDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)
print(len(dataset))

print(dataset.get_data_info(0))
# print(dataset[0])

# # # print(dataset.metainfo)

# print(dataset[0])
# print(dataset.metainfo)