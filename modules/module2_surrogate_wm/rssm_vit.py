import torch
from typing import Dict

from utils.fixed_point import get_fixed_point


class RSSM_ViT:
    """Surrogate World Model with Vision Transformer + Relative Spatial Attention"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_environment"].device
        self.fp = get_fixed_point(config)
        self.hidden_dim = config.get("wm_hidden_dim", 256)
        
        # Pre-allocated state (Zero-Allocation)
        self.surrogate_state = torch.zeros(self.hidden_dim, dtype=torch.int32, device=self.device)
        self.attention_scratch = torch.zeros(self.hidden_dim, dtype=torch.int32, device=self.device)
    
    def predict_next(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """RSSM prediction with fixed-point updates"""
        # Simple attention-like update
        self.attention_scratch.copy_(tokens[:self.hidden_dim])
        self.surrogate_state.add_(self.attention_scratch)
        self.surrogate_state.add_(actions.sum().to(torch.int32))
        self.surrogate_state.clamp_(-1<<30, 1<<30)
        
        return self.surrogate_state


# Global
rssm_vit: RSSM_ViT | None = None

def get_rssm(config: dict, views: Dict) -> RSSM_ViT:
    global rssm_vit
    if rssm_vit is None:
        rssm_vit = RSSM_ViT(config, views)
    return rssm_vit