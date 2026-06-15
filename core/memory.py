import torch
from typing import Dict, Any

from .types import FixedPointMultiplier
from utils.memory_align import align_offset

class MemoryManager:
    """Monolithic T_global allocator with 16-byte alignment - STAGE 1 CORE"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        device_str = config.get("device", "cpu")
        self.device = torch.device(device_str)
        
        grid_h, grid_w = config["grid_size"]
        max_agents = config["max_agents"]
        genome_len = config["genome_length"]
        
        # Calculate sizes (bytes)
        env_size = grid_h * grid_w          # uint8 per cell
        entities_size = grid_h * grid_w * 2 # int16 indices
        dna_size = max_agents * genome_len * 4  # int32
        scratch_size = max_agents * 4096 * 4    # generous scratch
        
        # Total with padding
        total_bytes = env_size + entities_size + dna_size + scratch_size + 2*1024*1024
        
        # Monolithic Zero-Allocation block
        self.T_global = torch.zeros(total_bytes, dtype=torch.uint8, device=self.device)
        
        self.offsets = self._compute_offsets(env_size, entities_size, dna_size, scratch_size)
        
        self._init_scratch_buffers()
        
        allocated_mb = total_bytes / (1024**2)
        print(f"[MEMORY] T_global allocated: {allocated_mb:.2f} MB on {self.device} | Max_Agents={max_agents}")
    
    def _compute_offsets(self, env_b: int, ent_b: int, dna_b: int, scr_b: int) -> Dict[str, int]:
        offsets = {}
        offset = 0
        offsets["T_environment"] = align_offset(offset, 16)
        offset = offsets["T_environment"] + env_b
        
        offsets["T_entities"] = align_offset(offset, 16)
        offset = offsets["T_entities"] + ent_b
        
        offsets["T_dna_pool"] = align_offset(offset, 16)
        offset = offsets["T_dna_pool"] + dna_b
        
        offsets["scratch"] = align_offset(offset, 16)
        return offsets
    
    def _init_scratch_buffers(self):
        """Pre-allocate scratch (no runtime alloc)"""
        pass  # Expanded in later stages
    
    def get_soa_views(self) -> Dict[str, torch.Tensor]:
        """Safe narrow views - NO runtime allocation"""
        views = {}
        grid_h, grid_w = self.config["grid_size"]
        total_cells = grid_h * grid_w
        
        # T_environment
        env_start = self.offsets["T_environment"]
        views["T_environment"] = self.T_global.narrow(0, env_start, total_cells).view(torch.uint8)
        
        # T_entities
        ent_start = self.offsets["T_entities"]
        views["T_entities"] = self.T_global.narrow(0, ent_start, total_cells * 2).view(torch.int16)
        
        # T_dna_pool
        dna_start = self.offsets["T_dna_pool"]
        dna_total = self.config["max_agents"] * self.config["genome_length"]
        views["T_dna_pool"] = self.T_global.narrow(0, dna_start, dna_total * 4).view(torch.int32)
        
        return views