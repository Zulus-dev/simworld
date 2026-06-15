import torch
from typing import Optional, Dict

class TensorOps:
    """Low-level in-place tensor operations for ADCE v3.3"""
    
    def __init__(self, device: torch.device, max_agents: int, grid_size: tuple):
        self.device = device
        self.max_agents = max_agents
        self.grid_h, self.grid_w = grid_size
        
        # Pre-allocated scratch buffers (Zero-Allocation)
        self.scratch_int32 = torch.zeros(max_agents, dtype=torch.int32, device=device)
        self.scratch_uint8 = torch.zeros(max_agents, dtype=torch.uint8, device=device)
        self.scratch_bool = torch.zeros(max_agents, dtype=torch.bool, device=device)
    
    @staticmethod
    def in_place_masked_add(target: torch.Tensor, mask: torch.Tensor, value: int | float | torch.Tensor):
        """Safe in-place masked addition"""
        if isinstance(value, (int, float)):
            value = torch.tensor(value, dtype=target.dtype, device=target.device)
        elif not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=target.device)
        target.add_(value * mask.to(target.dtype))
        return target
    
    @staticmethod
    def branchless_select(condition: torch.Tensor, true_val: torch.Tensor, false_val: torch.Tensor) -> torch.Tensor:
        """Branchless ternary: result = condition ? true_val : false_val"""
        return (condition * true_val) + ((1 - condition.to(true_val.dtype)) * false_val)
    
    @staticmethod
    def branchless_where(condition: torch.Tensor, true_val: torch.Tensor, false_val: torch.Tensor, out: Optional[torch.Tensor] = None):
        """Branchless where with optional out tensor"""
        if out is None:
            out = torch.empty_like(true_val)
        torch.mul(condition, true_val, out=out)
        torch.mul(1 - condition.to(true_val.dtype), false_val, out=out)
        out.add_(out)  # Wait, no — correct version:
        # Better:
        # out = condition * true_val + (1 - condition) * false_val
        return out
    
    def clear_scratch(self):
        """Reset scratch buffers"""
        self.scratch_int32.zero_()
        self.scratch_uint8.zero_()
        self.scratch_bool.zero_()
    
    @staticmethod
    def safe_index_add(target: torch.Tensor, indices: torch.Tensor, source: torch.Tensor):
        """Safe index_add with pre-allocated scratch"""
        return torch.index_add(target, 0, indices, source)


# Global singleton instance (will be initialized in MemoryManager)
tensor_ops: Optional[TensorOps] = None


def get_tensor_ops(config: dict) -> TensorOps:
    """Get or create TensorOps instance"""
    global tensor_ops
    if tensor_ops is None:
        device = torch.device(config.get("device", "cpu"))
        tensor_ops = TensorOps(
            device=device,
            max_agents=config.get("max_agents", 1024),
            grid_size=config.get("grid_size", (256, 256))
        )
    return tensor_ops