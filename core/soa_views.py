import torch
from typing import Dict

from .memory import MemoryManager

def create_soa_views(memory: MemoryManager) -> Dict[str, torch.Tensor]:
    """Centralized safe SoA view creation"""
    return memory.get_soa_views()