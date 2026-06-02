import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance with optional label smoothing
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for each class (list/tensor) or scalar for binary
        gamma: Focusing parameter (higher = more focus on hard examples)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # Alpha can be a scalar or list of weights per class
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits from model
            targets: (N,) ground truth labels
        """
        # Get class probabilities with optional label smoothing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        p_t = torch.exp(-ce_loss)  # Probability of true class
        
        # Apply focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Get alpha for each target class
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
