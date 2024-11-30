import torch
import torch.nn as nn
import torch.nn.functional as F

class LogDepthLoss(BaseLoss):
    def __init__(self):
        super(LogDepthLoss, self).__init__()
    
    def _forward(self, pred, target, weight):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        # L1MaskedMSELoss : https://github.com/dusty-nv/pytorch-depth/blob/master/criteria.py

        # Values ​​over 10000mm are treated as outliers
        valid_mask = (target<10000).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class LogSpecLoss(BaseLoss):
    def __init__(self):
        super(LogSpecLoss, self).__init__()
    
    def _forward(self, pred, target, weight):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        self.loss = diff.abs().mean()
        return self.loss
