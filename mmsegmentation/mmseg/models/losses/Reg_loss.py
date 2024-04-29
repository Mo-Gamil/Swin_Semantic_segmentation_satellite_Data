import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def mean_squared_error(pred, target, reduction='mean'):
    """Calculate Mean Squared Error (MSE) loss."""
    loss = F.mse_loss(pred, target, reduction=reduction)
    return loss
@weighted_loss
def mean_absolute_error(pred, target, reduction='mean'):
    """Calculate Mean Absolute Error (MAE) loss."""
    loss = F.l1_loss(pred, target, reduction=reduction)
    return loss
@weighted_loss
def huber_loss(pred, target, delta=1.0, reduction='mean'):
    """Calculate Smooth L1 (Huber) Loss."""
    loss = F.smooth_l1_loss(pred, target, beta=delta, reduction=reduction)
    return loss


@MODELS.register_module()
class RegLoss(nn.Module):
    """Custom Loss for regression tasks such as crop yield estimation."""
    def __init__(self, loss_type='mse', delta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta
    
    def forward(self, pred, target):
        if self.loss_type == 'mse':
            return mean_squared_error(pred, target)
        elif self.loss_type == 'mae':
            return mean_absolute_error(pred, target)
        elif self.loss_type == 'huber':
            return huber_loss(pred, target, delta=self.delta)
        else:
            raise ValueError("Unsupported loss type")
