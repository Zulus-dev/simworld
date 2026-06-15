import torch
from typing import Dict, Tuple

class AffordanceDecoder:
    """Decodes abstract action tensor into environment commands"""
    def __init__(self, config: dict):
        self.config = config
    
    def decode(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Demo: return movement + mental actions
        return actions, torch.zeros_like(actions)