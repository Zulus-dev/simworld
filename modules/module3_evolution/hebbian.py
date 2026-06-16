import torch
from typing import Dict

from utils.fixed_point import get_fixed_point


class HebbianUpdater:
    """Hebbian Learning Module with Gradient Clipping — Sparse Local Synapses (Zero-Allocation)"""
    
    def __init__(self, config: Dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_dna_pool"].device
        self.max_agents = config.get("max_agents", 4096)
        
        self.fp = get_fixed_point(config)
        self.eta = int(config.get("hebbian_eta", 0.0001) * self.fp.M)
        self.gamma = int(config.get("hebbian_gamma", 0.0001) * self.fp.M)
        
        # Sparse Hebbian
        self.local_synapses_per_agent = config.get("hebbian_local_synapses", 32)
        
        # Фиксированные цели синапсов
        self.synapse_targets = torch.randint(0, self.max_agents, 
                                           (self.max_agents, self.local_synapses_per_agent),
                                           device=self.device, dtype=torch.int32)
        
        # Предаллоцированные веса
        self.weights = torch.zeros((self.max_agents, self.local_synapses_per_agent), 
                                 dtype=torch.int64, device=self.device)
        
        # Scratch
        self.delta_scratch = torch.zeros((self.max_agents, self.local_synapses_per_agent), 
                                       dtype=torch.int64, device=self.device)

    def set_rng_generator(self, rng_generator: torch.Generator):
        self.rng_generator = rng_generator

    def update_synapses(self, mental_actions: torch.Tensor, physical_indices: torch.Tensor):
        """
        Обновление синапсов по правилу Хебба с Fixed-Point и Zero-Allocation.
        """
        # Приводим входы к Fixed-Point
        mental_fp = (mental_actions.float() * self.fp.M).to(torch.int64)
        phys_fp = (physical_indices.float() * self.fp.M).to(torch.int64)
        
        # Правильное broadcasting для локальных синапсов
        mental_exp = mental_fp.unsqueeze(1)                    # [N, 1]
        phys_targets = phys_fp[self.synapse_targets]           # [N, Local_Synapses]
        
        torch.mul(mental_exp, phys_targets, out=self.delta_scratch)
        self.delta_scratch.mul_(self.eta)
        self.delta_scratch.bitwise_right_shift_(16)
        
        # In-place обновление
        self.weights.add_(self.delta_scratch)
        
        # Затухание
        self.weights.mul_(self.fp.M - self.gamma)
        self.weights.bitwise_right_shift_(16)
        
        # Clamping
        self.weights.clamp_(-self.fp.VMAX, self.fp.VMAX)
        
        max_dw = self.delta_scratch.abs().max().item() / self.fp.M
        print(f"[HEBBIAN] Synapses updated | Max ΔW: {max_dw:.4f} | Local synapses: {self.local_synapses_per_agent}")

# Global instance
hebbian: HebbianUpdater | None = None

def get_hebbian(config: Dict, views: Dict) -> HebbianUpdater:
    global hebbian
    if hebbian is None:
        hebbian = HebbianUpdater(config, views)
    return hebbian