import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

# from mmseg.registry import METRICS

from mmengine.registry import EVALUATOR, METRICS
import torch
import numpy as np
# from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric
from typing import List, Sequence

@EVALUATOR.register_module()
class RegressionMetrics(BaseMetric):
    """Regression evaluation metric for crop yields.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str, optional): The directory for output prediction.
            Defaults to None.
    """
    def __init__(self, collect_device: str = 'cpu', output_dir: Optional[str] = None, **kwargs) -> None:
        super().__init__(collect_device=collect_device)
        self.output_dir = output_dir
        self.predictions = []
        self.ground_truths = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # Extract predictions and ground truths from data_batch
        self.predictions.append(data_batch['predictions'])
        self.ground_truths.append(data_batch['ground_truths'])

    def compute_metrics(self,output,targets):
        """Compute regression metrics from processed results.

        Returns:
            Dict[str, float]: The computed metrics including MSE, MAE, RMSE for each crop.
        """
        predictions = torch.cat(output)
        ground_truths = torch.cat(targets)

        # Compute metrics for each crop
        mse_corn = torch.mean((predictions[:, 0] - ground_truths[:, 0]) ** 2)
        mae_corn = torch.mean(torch.abs(predictions[:, 0] - ground_truths[:, 0]))
        rmse_corn = torch.sqrt(mse_corn)

        mse_soy = torch.mean((predictions[:, 1] - ground_truths[:, 1]) ** 2)
        mae_soy = torch.mean(torch.abs(predictions[:, 1] - ground_truths[:, 1]))
        rmse_soy = torch.sqrt(mse_soy)

        metrics = {
            'MSE_Corn': mse_corn.item(),
            'MAE_Corn': mae_corn.item(),
            'RMSE_Corn': rmse_corn.item(),
            'MSE_Soy': mse_soy.item(),
            'MAE_Soy': mae_soy.item(),
            'RMSE_Soy': rmse_soy.item()
        }
        return metrics

    @staticmethod
    def update_directory(output_dir):
        """Create or verify existence of output directory."""
        if output_dir and is_main_process():
            mkdir_or_exist(output_dir)