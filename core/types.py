from typing import TypedDict, Final, Dict, Any
import torch

FixedPointMultiplier: Final[int] = 65536

class FixedPoint:
    """Fixed-point arithmetic utilities (M=2^16)"""
    M: Final[int] = FixedPointMultiplier

    @staticmethod
    def to_fixed(x: torch.Tensor) -> torch.Tensor:
        """Scale to int64 with fixed multiplier"""
        return (x * FixedPoint.M).to(torch.int64)

    @staticmethod
    def from_fixed(x: torch.Tensor) -> torch.Tensor:
        """Convert back to float32"""
        return x.to(torch.float32) / FixedPoint.M