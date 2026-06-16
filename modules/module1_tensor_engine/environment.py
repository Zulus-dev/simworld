import torch
from typing import Dict
from utils.bitpack import update_pheromone, set_environment_flag


class EnvironmentUpdater:
    """Task-Agnostic Environment Update Engine."""
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.T_environment = views["T_environment"]
        self.max_agents = config.get("max_agents", 4096)
        
        # Scratch из MemoryManager (Zero-Allocation)
        self.scratches = views.get("preallocated_scratches") or {}
        self.active_mask_scratch = torch.zeros(self.max_agents, dtype=torch.bool, device=self.T_environment.device)

    def update_environment(self, agent_positions: torch.Tensor):
        """In-place обновление среды без тотального zero_()"""
        # 1. Принудительно приводим индексы
        indices = agent_positions.long().clamp_(0, self.T_environment.numel() - 1)
        
        # 2. Устанавливаем феромон А (роя) в позициях агентов — in-place
        # Вместо полного zero_() — точечное обновление
        flat_env = self.T_environment.view(-1)
        
        # Устанавливаем бит феромона (уровень 4 из 8)
        update_pheromone(
            flat_env, 
            torch.ones_like(indices, dtype=torch.bool),  # все активные
            level=4, 
            pheromone_type='A',
            out=flat_env
        )
        
        # Дополнительно можно добавить ресурсы/спавнеры позже через config
        
        active = (flat_env > 0).sum().item()
        if active > 0 and active % 20 == 0:  # уменьшаем спам
            print(f"[ENV] Updated environment | Active cells: {active}")


__all__ = ["EnvironmentUpdater"]