import torch
from typing import Dict

from utils.fixed_point import get_fixed_point


class HDCMemory:
    """Hyperdimensional Computing Graph Memory (HDC-Graph)"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_dna_pool"].device
        self.max_nodes = config.get("hdc_max_nodes", 50000)
        self.dim = 1024  # hypervector dimension
        
        self.fp = get_fixed_point(config)
        
        # Pre-allocated hypervectors storage
        self.hypervectors = torch.zeros((self.max_nodes, self.dim), dtype=torch.int32, device=self.device)
        self.centrality = torch.zeros(self.max_nodes, dtype=torch.int16, device=self.device)
        self.node_count = 0
    
    def bind(self, token: torch.Tensor) -> int:
        """Bind new hypervector (simplified)"""
        if self.node_count >= self.max_nodes:
            self.node_count = 0  # cyclic reuse
        self.hypervectors[self.node_count] = token[:self.dim].to(torch.int32)
        self.node_count += 1
        return self.node_count - 1
    
    def cleanup(self):
        """Stochastic cognitive dropout"""
        decay = torch.rand(self.node_count, device=self.device) < 0.01
        self.centrality[decay] = 0


# Global
hdc: HDCMemory | None = None

def get_hdc_memory(config: dict, views: Dict) -> HDCMemory:
    global hdc
    if hdc is None:
        hdc = HDCMemory(config, views)
    return hdc