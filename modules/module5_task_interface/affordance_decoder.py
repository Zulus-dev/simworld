import torch
from typing import Dict, Tuple, Any

class AffordanceDecoder:
    """
    Исправленный декодер: устранена ошибка потери данных при записи.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.map_size = config.get("task", {}).get("affordance_map_size", 32)
        self.max_agents = config.get("max_agents", 4096)
        self.device = torch.device(config.get("device", "cpu"))
        
        self.out_physical = torch.zeros(self.max_agents, dtype=torch.int64, device=self.device)
        self.out_mental = torch.zeros(self.max_agents, dtype=torch.int64, device=self.device)

    def decode(self, abstract_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямая запись данных из нейросети в выходные буферы.
        """
        # Считаем сумму действий (первые 8 бит)
        # Если здесь будет 0, значит проблема выше по пайплайну в perceiver.py
        raw_physical = torch.sum(abstract_actions[:, :8], dim=1)
        
        # ПРИНУДИТЕЛЬНАЯ ЗАПИСЬ: прямое присваивание срезом гарантирует, 
        # что данные попадут в память, которую видит PhysicsEngine
        self.out_physical[:] = raw_physical.long() % (256 * 256)
        
        # Ментальный слой
        raw_mental = torch.sum(abstract_actions[:, 8:16], dim=1)
        self.out_mental[:] = raw_mental.long()
        
        return self.out_physical, self.out_mental