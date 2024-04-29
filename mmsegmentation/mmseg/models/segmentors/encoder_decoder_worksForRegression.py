from mmseg.evaluation.metrics.RegressionMetrics import RegressionMetrics
from mmseg.models.losses.Reg_loss import RegLoss
from mmseg.registry import MODELS
import torch
from torch import nn, Tensor
from mmengine.model import BaseModel

# from mmseg.structures import SampleList
# from .reg_head import CropYieldRegressionHead
# from .Reg_loss import RegLoss
# from .regressionMetrics import RegressionMetrics

# @MODELS.register_module()
# class RegressionMetrics():
#     """Class to hold and calculate all regression metrics."""
#     def __init__(self):
#         # self.metrics = {
#         #     'r2_score': self.r2_score,
#         #     'rmse': self.rmse,
#         #     'mae': self.mae
#         # }
#         pass
    
    # def r2_score(self,pred, target):
    #     """Calculate R² score, the coefficient of determination."""
    #     target_mean = torch.mean(target)
    #     ss_tot = torch.sum((target - target_mean) ** 2)
    #     ss_res = torch.sum((target - pred) ** 2)
    #     r2 = 1 - ss_res / ss_tot
    #     return r2

    # def rmse(self,pred, target):
    #     """Calculate Root Mean Squared Error (RMSE)."""
    #     mse = torch.mean((pred - target) ** 2)
    #     return torch.sqrt(mse)

    # def mae(self,pred, target):
    #     """Calculate Mean Absolute Error (MAE)."""
    #     return torch.mean(torch.abs(pred - target))


@MODELS.register_module()
class EncoderDecoder(BaseModel):
    def __init__(self, backbone, decode_head, neck=None, train_cfg=None, test_cfg=None, 
                 data_preprocessor=None, pretrained=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)
        if neck:
            self.neck = MODELS.build(neck)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Check if 'loss' is in train_cfg before trying to create the loss module
        if train_cfg and 'loss' in train_cfg:
            # print('train_cfg____________',train_cfg)
            self.loss_module = RegLoss(**train_cfg['loss'])
        else:
            raise ValueError("Missing 'loss' configuration in train_cfg")

        self.metrics  = RegressionMetrics()

    def r2_score(self,pred, target):
        """Calculate R² score, the coefficient of determination."""
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def rmse(self,pred, target):
        """Calculate Root Mean Squared Error (RMSE)."""
        mse = torch.mean((pred - target) ** 2)
        return torch.sqrt(mse)

    def mae(self,pred, target):
        """Calculate Mean Absolute Error (MAE)."""
        return torch.mean(torch.abs(pred - target))

    def extract_feat(self, inputs: Tensor) -> Tensor:
        x = self.backbone(inputs)
        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, inputs: Tensor, targets: Tensor, mode='tensor'):
        features = self.extract_feat(inputs)
        outputs = self.decode_head(features)
        if mode == 'loss':
            return {'loss': self.loss_module(outputs, targets)}
        elif mode == 'tensor':
            return outputs
        elif mode == 'predict':
            metrics = self.metrics.compute_metrics(outputs, targets)
            return metrics
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def loss(self, inputs: Tensor, targets: Tensor):
        return self.forward(inputs, targets, mode='loss')

    def predict(self, inputs: Tensor, targets: Tensor):
        return self.forward(inputs, targets, mode='predict')
