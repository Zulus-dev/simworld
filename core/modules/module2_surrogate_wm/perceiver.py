import torch
from typing import Dict, Tuple

from utils.tensor_ops import get_tensor_ops
from utils.fixed_point import get_fixed_point


class Perceiver:
    """Task-Agnostic Spatial Tokenizer — converts environment to fixed token vector"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_environment"].device
        self.fp = get_fixed_point(config)
        self.token_dim = config.get("perceiver_tokens", 512)
        
        # Pre-allocated token buffer
        self.token_buffer = torch.zeros(self.token_dim, dtype=torch.int32, device=self.device)
    
    def tokenize(self) -> torch.Tensor:
        """Flatten environment + entities into abstract tokens (Zero-Allocation)"""
        T_env = self.views["T_environment"]
        T_ent = self.views["T_entities"]
        
        # Simple demo: pheromone stats + entity density (real version uses patch projection)
        pheromone_a = (T_env & 0x1C) >> 2
        pheromone_b = (T_env & 0xE0) >> 5
        
        token_stats = torch.cat([
            pheromone_a.float().mean().unsqueeze(0),
            pheromone_b.float().mean().unsqueeze(0),
            (T_ent != 0).float().mean().unsqueeze(0)
        ])
        
        # Pad to fixed token size
        self.token_buffer[:len(token_stats)] = (token_stats * self.fp.M).to(torch.int32)
        return self.token_buffer


class RSSM_ViT:
    """Surrogate World Model with Relative Spatial Attention (simplified)"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_environment"].device
        self.hidden_dim = config.get("wm_hidden_dim", 256)
        
        # Pre-allocated state
        self.surrogate_state = torch.zeros(self.hidden_dim, dtype=torch.int32, device=self.device)
    
    def predict_next(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """RSSM-style prediction (demo)"""
        # Fixed-point update
        self.surrogate_state.add_(tokens[:self.hidden_dim].to(torch.int32) + actions.sum().to(torch.int32))
        self.surrogate_state.clamp_(-1<<30, 1<<30)
        return self.surrogate_state


# Global instances
perceiver: Perceiver | None = None
rssm: RSSM_ViT | None = None

def get_perceiver(config: dict, views: Dict) -> Perceiver:
    global perceiver
    if perceiver is None:
        perceiver = Perceiver(config, views)
    return perceiver

def get_rssm(config: dict, views: Dict) -> RSSM_ViT:
    global rssm
    if rssm is None:
        rssm = RSSM_ViT(config, views)
    return rssm