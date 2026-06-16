import torch
from typing import Dict

class MatkaRegulator:
    """
    Swarm Intelligence Regulator для Матки.
    Вычисляет динамический штраф регуляризации в формате Fixed-Point Math.
    Обеспечивает Strict Task-Agnostic и Absolute Zero-Allocation.
    """
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_dna_pool"].device
        
        # Получаем множитель фиксированной точки (например, 65536)
        self.fp_multiplier = config.get("fixed_point_multiplier", 65536)
        
        # Переводим beta_max в целочисленный Fixed-Point формат
        beta_float = config.get("evolution", {}).get("matka_beta_max", 0.1)
        self.beta_int = int(beta_float * self.fp_multiplier)
        
        self.max_agents = config["max_agents"]
        
        # Предаллоцированные буферы вывода во избежание runtime-аллокаций (Zero-Allocation)
        self.out_regulated_fitness = torch.zeros(self.max_agents, dtype=torch.int64, device=self.device)
        self.scratch_buffer = torch.zeros(self.max_agents, dtype=torch.int64, device=self.device)

    def regulate(self, fitness: torch.Tensor) -> torch.Tensor:
        """Strict Zero-Allocation fitness regulation"""
        # 1. mean_fitness (используем torch.mean)
        mean_fitness = fitness.float().mean().long()
        
        # 2. Вычисление отклонения in-place в scratch_buffer
        # scratch_buffer = fitness - mean
        torch.sub(fitness, mean_fitness, out=self.scratch_buffer)
        torch.abs_(self.scratch_buffer) # In-place absolute
        
        # 3. Регуляризация
        sum_abs_deviation = self.scratch_buffer.sum()
        mad_fitness = sum_abs_deviation // self.max_agents
        
        kl_penalty = (mean_fitness - mad_fitness).abs()
        scaled_penalty = (self.beta_int * kl_penalty) >> 16
        
        # 4. Финальный расчет в предаллоцированный out_regulated_fitness
        torch.sub(fitness, scaled_penalty, out=self.out_regulated_fitness)
        
        return self.out_regulated_fitness


# Глобальный синглтон-кэш для сохранения структуры движка
_matka_instance: MatkaRegulator | None = None

def get_matka_regulator(config: dict, views: Dict[str, torch.Tensor]) -> MatkaRegulator:
    global _matka_instance
    if _matka_instance is None:
        _matka_instance = MatkaRegulator(config, views)
    return _matka_instance