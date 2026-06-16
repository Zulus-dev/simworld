import torch
from typing import Union


class FixedPoint:
    """Fixed-Point arithmetic M=65536 (2^16)"""
    M: int = 65536
    VMAX: int = 1 << 30  # safety limit before int64 overflow
    
    def __init__(self, multiplier: int = 65536):
        self.M = multiplier
    
    def to_fixed(self, x: Union[float, torch.Tensor]) -> torch.Tensor:
        """Convert to fixed-point int64"""
        if isinstance(x, float):
            return torch.tensor(int(x * self.M), dtype=torch.int64, device='cpu')
        return (x * self.M).to(torch.int64)
    
    def to_float(self, x: torch.Tensor) -> torch.Tensor:
        """Convert back to float"""
        return x.to(torch.float32) / self.M
    
    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fixed-point multiplication with shift"""
        return (a * b) >> 16   # right shift for M=2^16


# Global
fp: FixedPoint | None = None

def get_fixed_point(config: dict) -> FixedPoint:
    global fp
    if fp is None:
        fp = FixedPoint(config.get("fixed_point_multiplier", 65536))
    return fp