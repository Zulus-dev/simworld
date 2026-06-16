import torch
from typing import Dict, Any
from utils.memory_align import align_offset


class MemoryManager:
    """Monolithic T_global allocator with 16-byte alignment - STAGE 1 CORE"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        device_str = config.get("device", "cpu")
        self.device = torch.device(device_str)
        
        grid_h, grid_w = config["grid_size"]
        self.max_agents = config["max_agents"]
        self.genome_len = config["genome_length"]
        
        # Calculate sizes (bytes)
        env_size = grid_h * grid_w          # uint8 per cell
        entities_size = grid_h * grid_w * 4 # int32 indices
        self.dna_size = self.max_agents * self.genome_len * 4  # int32
        scratch_size = self.max_agents * 4096 * 4    # generous scratch
        
        # Total with padding
        total_bytes = env_size + entities_size + self.dna_size + scratch_size + 2*1024*1024
        
        # Monolithic Zero-Allocation block
        self.T_global = torch.zeros(total_bytes, dtype=torch.uint8, device=self.device)
        
        self.offsets = self._compute_offsets(env_size, entities_size, self.dna_size, scratch_size)
        
        # Предаллокация общих scratch-буферов
        self._preallocate_scratches()
        
        print(f"[MEMORY] T_global allocated: {total_bytes / (1024*1024):.2f} MB | Device={self.device}")

    def _compute_offsets(self, env_b, ent_b, dna_b, scratch_b) -> Dict[str, int]:
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

    def _preallocate_scratches(self):
        """Предаллокация scratch-буферов для модулей (Zero-Allocation)"""
        self.collision_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.fitness_scratch = torch.zeros(self.max_agents, dtype=torch.int64, device=self.device)
        self.reward_scratch = torch.zeros(self.max_agents, dtype=torch.int64, device=self.device)
        self.mutation_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.index_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)

    def get_soa_views(self) -> Dict[str, torch.Tensor]:
        """Создание представлений SoA"""
        views = {}
        grid_h, grid_w = self.config["grid_size"]
        total_cells = grid_h * grid_w
        
        # T_environment
        env_start = self.offsets["T_environment"]
        views["T_environment"] = self.T_global.narrow(0, env_start, total_cells).view(torch.uint8)
        
        # T_entities
        ent_start = self.offsets["T_entities"]
        views["T_entities"] = self.T_global.narrow(0, ent_start, total_cells * 4).view(torch.int32)
        
        # T_dna_pool — 2D представление [max_agents, genome_length]
        dna_start = self.offsets["T_dna_pool"]
        dna_flat = self.T_global.narrow(0, dna_start, self.dna_size).view(torch.int32)
        views["T_dna_pool"] = dna_flat.view(self.max_agents, self.genome_len)
        
        # Scratch buffers
        views["collision_scratch"] = self.collision_scratch
        views["fitness_scratch"] = self.fitness_scratch
        views["reward_scratch"] = self.reward_scratch

        return views

    def get_preallocated_scratches(self) -> Dict[str, torch.Tensor]:
        """Доступ к предаллоцированным scratch-буферам"""
        return {
            "collision": self.collision_scratch,
            "fitness": self.fitness_scratch,
            "reward": self.reward_scratch,
            "mutation": self.mutation_scratch,
            "indices": self.index_scratch,
        }


__all__ = ["MemoryManager"]