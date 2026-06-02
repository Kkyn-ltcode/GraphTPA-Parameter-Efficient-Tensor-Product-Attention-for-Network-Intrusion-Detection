import torch
import numpy as np
from typing import Dict

def process_report(d: Dict, digits: int = 4) -> Dict:
    """Process classification report to convert percentages"""
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = process_report(v, digits)
        elif isinstance(v, float):
            new_d[k] = int(v) if k == 'support' else round(v * 100, digits)
        else:
            new_d[k] = v
    return new_d

def set_seed(seed: int = 2305):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
