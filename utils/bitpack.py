import torch
from typing import Optional

def update_pheromone(
    env: torch.Tensor,
    mask: torch.Tensor,
    level: int = 4,
    pheromone_type: str = 'A',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """In-place pheromone level update using int32 safe operations."""
    if out is None:
        out = env
    
    # Используем int32 для всех масок, чтобы избежать переполнения uint8
    if pheromone_type.upper() == 'A':
        shift = 2
        mask_bits = 0x1C  # bits 2,3,4
    else:
        shift = 5
        mask_bits = 0xE0  # bits 5,6,7
    
    # Очистка бит (in-place)
    torch.bitwise_and(env, ~mask_bits, out=out)
    
    # Подготовка новых бит с учетом типа данных тензора (int32/int64)
    new_bits = (level << shift) & mask_bits
    
    # Важно: применяем маску как int32/int64, а не uint8
    new_bits_tensor = torch.tensor(new_bits, dtype=out.dtype, device=env.device)
    
    # Применяем обновление через OR
    torch.bitwise_or(out, new_bits_tensor * mask.to(out.dtype), out=out)
    
    return out


def set_environment_flag(
    env: torch.Tensor,
    mask: torch.Tensor,
    flag_bit: int,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Set specific bit flag (0-7) in-place with safe type handling."""
    if out is None:
        out = env
    
    bit_mask = 1 << flag_bit
    torch.bitwise_or(env, bit_mask * mask.to(out.dtype), out=out)
    return out


def clear_environment_flag(
    env: torch.Tensor,
    mask: torch.Tensor,
    flag_bit: int,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Clear specific bit flag in-place."""
    if out is None:
        out = env
    
    bit_mask = ~(1 << flag_bit)
    torch.bitwise_and(env, bit_mask, out=out)
    return out


def get_pheromone_level(env: torch.Tensor, pheromone_type: str = 'A') -> torch.Tensor:
    shift = 2 if pheromone_type.upper() == 'A' else 5
    return extract_bits(env, shift, 3)


def set_passability(env: torch.Tensor, mask: torch.Tensor, passable: bool = True):
    if passable:
        return set_environment_flag(env, mask, 0)
    else:
        return clear_environment_flag(env, mask, 0)


def is_passable(env: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_and(env, 0x01) > 0


def extract_bits(env: torch.Tensor, shift: int, bit_count: int, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Task-Agnostic n-Dimensional Bit Extractor.
    Работает с любым типом данных (int32/int64) без переполнения.
    """
    mask_bits = ((1 << bit_count) - 1) << shift
    
    if out is None:
        levels = torch.bitwise_and(env, mask_bits)
    else:
        levels = torch.bitwise_and(env, mask_bits, out=out)
        
    return torch.bitwise_right_shift(levels, shift, out=levels)