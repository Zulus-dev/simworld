import torch
from typing import Optional

def blit_walls(env: torch.Tensor, framebuffer: torch.Tensor, color: list):
    """Zero-Allocation wall blitting"""
    mask = (env & 0x01) == 0
    framebuffer[mask] = torch.tensor(color, dtype=torch.uint8, device=framebuffer.device)


def blit_pheromones(env: torch.Tensor, framebuffer: torch.Tensor):
    """In-place pheromone color overlay"""
    level_a = (env & 0x1C) >> 2
    framebuffer[..., 1] = (level_a * 32).clamp(0, 255).to(torch.uint8)