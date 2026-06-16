import torch
from typing import Dict, Any

from config.schema import BaseConfig

class TaskAdapter:
    """Task-Agnostic Adapter — loads task_definition.json"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = BaseConfig(**config).model_dump()
        self.task_name = self.config.get("task", {}).get("name", "default_resource_collection")
    
    def get_reward_weights(self):
        """Return composable reward weights"""
        return {
            "pheromone_gradient": 0.4,
            "resource_proximity": 0.35,
            "swarm_cohesion": 0.15,
            "exploration": 0.1
        }
    
    def adapt_observation(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Task-specific token processing"""
        return raw_obs.float()


def get_task_adapter(config: Dict) -> TaskAdapter:
    return TaskAdapter(config)