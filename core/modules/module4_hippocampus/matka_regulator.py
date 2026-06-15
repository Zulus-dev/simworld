import torch
from typing import Dict

from utils.fixed_point import get_fixed_point


class MatkaRegulator:
    """Swarm Intelligence Regulator for Matka (Queen) + KL penalty"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.fp = get_fixed_point(config)
        self.beta = config.get("matka_beta_max", 0.1)
    
    def regulate(self, fitness: torch.Tensor):
        """KL regularization + swarm goal metric"""
        # Simplified KL demo
        kl_penalty = torch.abs(fitness.mean() - fitness.std())
        regulated_fitness = fitness - self.beta * kl_penalty
        print(f"[MATKA] Regulated fitness | KL penalty: {kl_penalty.item():.4f}")
        return regulated_fitness


# Global
matka: MatkaRegulator | None = None

def get_matka_regulator(config: dict, views: Dict) -> MatkaRegulator:
    global matka
    if matka is None:
        matka = MatkaRegulator(config, views)
    return matka