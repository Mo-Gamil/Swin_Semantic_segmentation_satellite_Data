from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os



@DATASETS.register_module()
class CustomDataset(BaseSegDataset):
    """ Satellite Dataset. RGB. 3 bands
    and crop data layers
    patch size is 224X224
    pixel size is 30m
    3 classes: corn, soy, others
    """

    
    METAINFO = dict(
        classes=('corn', 'soy', 'other'),
        palette=[[255,127,80], [30,144,255], [0,0,0]])

    def __init__(self,
                 img_suffix='inst.png',
                 seg_map_suffix='inst.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)




















# @DATASETS.register_module()
# class CustomDataset(BaseSegDataset):
    # """ Satellite Dataset. RGB. 3 bands
    # and crop data layers
    # patch size is 224X224
    # pixel size is 30m
    # 3 classes: corn, soy, others
    # """
    # METAINFO = dict(
    #     classes=('corn', 'soy', 'other'),
    #     palette=[[255,127,80], [30,144,255], [0,0,0]])

    # def __init__(self,
    #              img_suffix='.tif',
    #              seg_map_suffix='.tif',
    #              **kwargs) -> None:
    #     super().__init__(
    #         img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    # def prepare_data(self, idx):
    #     img_info = self.get_data_info(idx)
    #     results = {
    #         'filename': os.path.join(self.data_root, self.data_prefix['img_path'], img_info['filename']),
    #         # other fields as necessary
    #     }
    #     return results