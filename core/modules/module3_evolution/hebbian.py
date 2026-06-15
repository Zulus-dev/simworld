import torch
from typing import Dict

from utils.fixed_point import FixedPoint


class HebbianUpdater:
    """Hebbian Learning with Fixed-Point arithmetic + clipping"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_dna_pool"].device
        self.fp = FixedPoint(config.get("fixed_point_multiplier", 65536))
        
        # Synaptic weights are part of DNA or separate scratch in full impl
        self.eta = config.get("hebbian_eta", 0.01)   # learning rate
        self.gamma = config.get("hebbian_gamma", 0.05)  # decay
    
    def update_synapses(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """ΔW = clip(η * (O_i * I_j - γ * O_j * W_ij), -Vmax, Vmax)"""
        # Demo: simplified update on DNA pool (real synapses in WM stage)
        T_dna = self.views["T_dna_pool"]
        
        # Fixed-point scaling
        eta_fp = self.fp.to_fixed(self.eta)
        gamma_fp = self.fp.to_fixed(self.gamma)
        
        # Branchless Hebbian rule
        delta = eta_fp * (outputs.unsqueeze(1) * inputs.unsqueeze(0))
        decay = gamma_fp * outputs.unsqueeze(1) * T_dna[:outputs.shape[0]]
        
        update = delta - decay
        update = torch.clamp(update, -self.fp.VMAX, self.fp.VMAX)
        
        T_dna[:update.shape[0]] += update.to(torch.int32)
        
        print(f"[HEBBIAN] Synapses updated | Max ΔW: {update.abs().max().item()}")


# Global
hebbian: HebbianUpdater | None = None

def get_hebbian(config: dict, views: Dict) -> HebbianUpdater:
    global hebbian
    if hebbian is None:
        hebbian = HebbianUpdater(config, views)
    return hebbian