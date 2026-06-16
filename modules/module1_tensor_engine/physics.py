import torch
from typing import Dict
from utils.tensor_ops import branchless_select
from utils.bitpack import set_environment_flag


class PhysicsEngine:
    """Branchless physics for Tensor-Engine (Module 1) — STAGE 3"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_environment"].device
        self.max_agents = config["max_agents"]
        self.grid_h, self.grid_w = config["grid_size"]
        
        # Pre-allocated scratch (Zero-Allocation)
        self.collision_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.dx_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.dy_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.pos_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.new_pos_scratch = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        
        # RNG будет устанавливаться из engine
        self.rng_generator = None

    def set_rng_generator(self, rng_generator: torch.Generator):
        """Вызывается из engine для передачи детерминированного генератора"""
        self.rng_generator = rng_generator

    def step_physics(self, actions: torch.Tensor):
        """Branchless movement + simple collision handling"""
        if self.rng_generator is None:
            self.rng_generator = torch.Generator(device=self.device)
            self.rng_generator.manual_seed(42)
        
        T_entities = self.views["T_entities"]
        
        # Генерируем смещения через предаллоцированные scratch + rng_generator
        torch.randint(-1, 2, (self.max_agents,), generator=self.rng_generator, 
                     device=self.device, dtype=torch.int32, out=self.dx_scratch)
        torch.randint(-1, 2, (self.max_agents,), generator=self.rng_generator, 
                     device=self.device, dtype=torch.int32, out=self.dy_scratch)
        
        # Получаем текущие позиции
        current_pos = T_entities[:self.max_agents].to(torch.int32)
        self.pos_scratch.copy_(current_pos)
        
        # Branchless расчет новых позиций
        torch.mul(self.dy_scratch, self.grid_w, out=self.new_pos_scratch)
        self.new_pos_scratch.add_(self.dx_scratch)
        self.new_pos_scratch.add_(self.pos_scratch)
        
        # Clamp positions (branchless)
        max_pos = (self.grid_h * self.grid_w) - 1
        self.new_pos_scratch.clamp_(0, max_pos)
        
        # Записываем обратно в SoA
        T_entities[:self.max_agents].copy_(self.new_pos_scratch)
        
        return self.new_pos_scratch.clone()  # возвращаем копию только для совместимости


__all__ = ["PhysicsEngine"]