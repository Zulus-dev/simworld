import torch
from typing import Dict

from utils.bitpack import update_pheromone


class EnvironmentUpdater:
    """T_environment updates — pheromones, resources (bit-packed)"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.T_environment = views["T_environment"]
    
    def update_environment(self, agent_positions: torch.Tensor):
        """Update pheromones and resources in-place (Zero-Allocation)"""
        # Demo pheromone deposit
        mask = torch.zeros_like(self.T_environment, dtype=torch.uint8)
        # In full version: scatter based on positions
        if len(agent_positions) > 0:
            mask[:min(100, len(mask))] = 1  # demo activity
        
        # Update Pheromone A (bits 2-4)
        self.T_environment = update_pheromone(
            self.T_environment, mask, level=5, pheromone_type='A'
        )
        
        active = (self.T_environment > 0).sum().item()
        print(f"[ENV] Updated environment | Active cells: {active}")