import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing for regularization
    """
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float32) if isinstance(weight, (list, np.ndarray)) else weight
        else:
            self.weight = None
    
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_preds)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Calculate loss
        loss = -smooth_targets * log_preds
        
        # Apply class weights if provided
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            weight_expanded = self.weight.unsqueeze(0).expand_as(loss)
            loss = loss * weight_expanded
        
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
