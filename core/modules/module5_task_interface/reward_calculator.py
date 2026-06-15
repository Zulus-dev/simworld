import torch
from typing import Dict

from utils.fixed_point import get_fixed_point

class RewardCalculator:
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.fp = get_fixed_point(config)
    
    def compute(self, state, action) -> torch.Tensor:
        # Demo reward
        return torch.tensor(0.15, dtype=torch.float32, device=state.device)