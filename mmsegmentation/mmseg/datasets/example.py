
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ExampleDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('other','corn', 'soy'),
        palette=[ [0,0,0],[255,127,80], [30,144,255]])


    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
























































# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset


# @DATASETS.register_module()
# class ExampleDataset(BaseSegDataset):

#     METAINFO = dict(
#         classes=('other','corn', 'soy'),
#         palette=[ [0,0,0],[255,127,80], [30,144,255]])

    
#     def __init__(self,
#                  img_suffix='.tif',
#                  seg_map_suffix='.tif',
#                  corn_yield= None,
#                  soybean_yield= None,
#                  **kwargs) -> None:
#         super().__init__(
#             img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)




# works for regression
# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset
# import pandas as pd

# @DATASETS.register_module()
# class ExampleDataset(BaseSegDataset):
#     METAINFO = dict(
#         classes=('other', 'corn', 'soy'),
#         palette=[[0, 0, 0], [255, 127, 80], [30, 144, 255]]
#     )

#     def __init__(self, img_suffix='.tif', seg_map_suffix='.tif', yield_file=None, **kwargs):
#         super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
#         self.yield_data = pd.read_csv(yield_file) if yield_file else None

#     def __getitem__(self, idx):
#         result = super().__getitem__(idx)
#         # print(result.keys)
#         if self.yield_data is not None:
#             # Extract the image name by removing the suffix and taking the last part of the path
#             img_path = result['data_samples'].img_path
#             # print("img_path:_______________",img_path)
#             image_name = img_path.split('.tif')[0].split('\\')[-1]
#             # print("img_:_______________",image_name)
#             # print(self.yield_data.head())
#             # Accessing the yield data using the extracted image name
#             if image_name in self.yield_data['Index'].values:
#                 # print('yes____________________________')
#                 # yields = self.yield_data[self.yield_data['Index'] == image_name][['Corn_crop', 'Soy_crop']].values[0]
#                 corn_yield = self.yield_data[self.yield_data['Index'] == image_name]['Corn_crop'].values[0]
#                 soy_yield = self.yield_data[self.yield_data['Index'] == image_name]['Soy_crop'].values[0]
#                 result['corn_yield'] = corn_yield
#                 result['soy_yield'] = soy_yield
#                 # print(result)
#                 # These two predictions "corn_yield" and "soy_yield" needs to be mapped so that the target of the decoder points to them
#                 # Must be changed in the loss function because thats where the mapping is happening

#                 # result["data_samples"] = [corn_yield,soy_yield]
#                 # print(result)
#         return result






# n range(len(dataset)):
#     # img_path = dataset.get_img_info(i)['filename']
#     img_path = dataset[i]['data_samples'].img_path












# use this to read from the run script:

# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset

# @DATASETS.register_module()
# class ExampleDataset(BaseSegDataset):
#     METAINFO = dict(
#         classes=('corn', 'soy', 'other'),
#         palette=[[255,127,80], [30,144,255], [0,0,0]]
#     )

#     def __init__(self, img_suffix='.tif', seg_map_suffix='.tif', **kwargs) -> None:
#         super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
#         self.corn_yield = {}
#         self.soybean_yield = {}

#     def add_yield_data(self, index, corn_yield, soybean_yield):
#         self.corn_yield[index] = corn_yield
#         self.soybean_yield[index] = soybean_yield

#     def __getitem__(self, idx):
#         data = super().__getitem__(idx)
#         data['corn_yield'] = self.corn_yield.get(idx)
#         data['soybean_yield'] = self.soybean_yield.get(idx)
#         return data





# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset
# import pandas as pd

# @DATASETS.register_module()
# class ExampleDataset(BaseSegDataset):
#     METAINFO = dict(
#         classes=('corn', 'soy', 'other'),
#         palette=[[255,127,80], [30,144,255], [0,0,0]]
#     )

#     def __init__(self, yield_data_path=None, **kwargs):
#         super().__init__(**kwargs)
#         self.yield_data = pd.read_csv(yield_data_path) if yield_data_path else None

#     def __getitem__(self, idx):
#         data = super().__getitem__(idx)
#         if self.yield_data is not None:
#             # Assume data_samples is an attribute of the data item containing img_path
#             img_path = data['data_samples'].img_path
#             index_val = img_path.split('.tif')[0].split('\\')[-1]
#             yield_data = self.yield_data[self.yield_data['Index'] == index_val]

#             if not yield_data.empty:
#                 data['corn_yield'] = yield_data['Corn_Crop'].values[0]
#                 data['soybean_yield'] = yield_data['Soy_crop'].values[0]
#         return data


