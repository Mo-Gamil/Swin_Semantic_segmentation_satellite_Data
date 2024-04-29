import torch.nn as nn
import torch
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule

@MODELS.register_module()
class CropYieldRegressionHead(nn.Module):
    """Regression head for predicting yields of two different crops."""

    def __init__(self, in_channels, num_crops=2,train_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_crops = num_crops
        self.train_cfg = train_cfg
        self.conv1 = ConvModule(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels // 2, num_crops)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        crop_yields = self.fc(x)
        return crop_yields
