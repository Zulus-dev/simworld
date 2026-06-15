import torch
from typing import Dict


class CognitiveDropout:
    """Stochastic memory cleanup (Hippocampus)"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.dropout_rate = config.get("cognitive_dropout_rate", 0.005)
    
    def apply_dropout(self, memory_module):
        """Apply to HDC or WM state"""
        memory_module.cleanup()
        print(f"[HIPPOCAMPUS] Cognitive dropout applied | Rate={self.dropout_rate}")


# Global
dropout: CognitiveDropout | None = None

def get_cognitive_dropout(config: dict, views: Dict) -> CognitiveDropout:
    global dropout
    if dropout is None:
        dropout = CognitiveDropout(config, views)
    return dropout