import torch
from typing import Optional

def update_pheromone(
    env: torch.Tensor,
    mask: torch.Tensor,
    level: int = 4,
    pheromone_type: str = 'A',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """In-place pheromone level update (bits 2-4 for A, 5-7 for B)"""
    if out is None:
        out = env
    
    if pheromone_type.upper() == 'A':
        shift = 2
        mask_bits = 0x1C  # bits 2,3,4
    else:
        shift = 5
        mask_bits = 0xE0  # bits 5,6,7
    
    # Clear old pheromone bits (in-place)
    torch.bitwise_and(env, ~mask_bits, out=out)
    
    # Prepare new bits
    new_bits = (level << shift) & mask_bits
    new_bits_tensor = torch.tensor(new_bits, dtype=torch.uint8, device=env.device)
    
    # Apply to masked cells
    torch.bitwise_or(out, new_bits_tensor * mask.to(torch.uint8), out=out)
    
    return out


def set_environment_flag(
    env: torch.Tensor,
    mask: torch.Tensor,
    flag_bit: int,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Set specific bit flag (0-7) in-place"""
    if out is None:
        out = env
    
    bit_mask = 1 << flag_bit
    torch.bitwise_or(env, bit_mask * mask.to(torch.uint8), out=out)
    return out


def clear_environment_flag(
    env: torch.Tensor,
    mask: torch.Tensor,
    flag_bit: int,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Clear specific bit flag in-place"""
    if out is None:
        out = env
    
    bit_mask = ~(1 << flag_bit)
    torch.bitwise_and(env, bit_mask, out=out)
    return out


def get_pheromone_level(env: torch.Tensor, pheromone_type: str = 'A') -> torch.Tensor:
    """Extract pheromone level (0-7) without allocation"""
    if pheromone_type.upper() == 'A':
        shift = 2
        mask_bits = 0x1C
    else:
        shift = 5
        mask_bits = 0xE0
    
    # Extract bits
    levels = torch.bitwise_and(env, mask_bits)
    levels = torch.bitwise_right_shift(levels, shift)
    return levels


def set_passability(env: torch.Tensor, mask: torch.Tensor, passable: bool = True):
    """Set passability flag (bit 0)"""
    if passable:
        return set_environment_flag(env, mask, 0)
    else:
        return clear_environment_flag(env, mask, 0)


def is_passable(env: torch.Tensor) -> torch.Tensor:
    """Check bit 0 (passability)"""
    return torch.bitwise_and(env, 0x01) > 0