# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class satelliteDataset(BaseSegDataset):
    """

    RGB sen2
    """
    METAINFO = dict(
        classes=('corn', 'soy','other'),
        palette=[[255,127,80], [30,144,255], [0,0,0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.jpg',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
