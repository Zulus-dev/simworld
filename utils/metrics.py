import torch
from utils.fixed_point import get_fixed_point

def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence for Matka regularization"""
    return (p * (p.log() - q.log())).sum()

def symlog(x: torch.Tensor) -> torch.Tensor:
    """SymLog transformation for stable rewards"""
    return torch.sign(x) * torch.log1p(torch.abs(x))