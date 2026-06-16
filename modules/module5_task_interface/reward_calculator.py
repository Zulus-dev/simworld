import torch
from typing import Dict, Any
from utils.fixed_point import get_fixed_point


class RewardCalculator:
    """
    Task-Agnostic Безопасный Калькулятор Наград.
    Рассчитывает взвешенную сумму целочисленных наград на основе токенов Perceiver.
    Соблюдает Absolute Zero-Allocation и Fixed-Point Math.
    """
    def __init__(self, config: Dict[str, Any], views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_dna_pool"].device
        
        # Fixed-Point утилита
        self.fp = get_fixed_point(config)
        
        # Извлекаем параметры задачи из конфигурации
        task_cfg = config.get("task", {})
        self.reward_components = task_cfg.get("reward_components", [
            {"type": "exploration", "weight": 1.0}
        ])
        
        # Предаллокация весов в формате Fixed-Point
        self.num_agents = config["max_agents"]
        self.weights = torch.tensor(
            [int(c["weight"] * self.fp.M) for c in self.reward_components],
            dtype=torch.int64, device=self.device
        )
        
        # Монолитный скретч-буфер для промежуточных вычислений наград [Num_Components, Max_Agents]
        self.reward_scratch = torch.zeros(
            (len(self.reward_components), self.num_agents), 
            dtype=torch.int64, device=self.device
        )
        
        # Итоговый вектор наград (предаллоцирован)
        self.out_rewards = torch.zeros(self.num_agents, dtype=torch.int64, device=self.device)

    def symlog_transform(self, x_int64: torch.Tensor) -> torch.Tensor:
        """
        Математически защищенная SymLog-трансформация.
        Локальное использование float только для log (защита градиентов).
        """
        # Локальная конверсия (без глобальных аллокаций)
        x_float = x_int64.float() / self.fp.M
        sign = torch.sign(x_float)
        transformed = sign * torch.log1p(torch.abs(x_float))   # log1p = ln(|x| + 1)
        return (transformed * self.fp.M).to(torch.int64)

    def compute(self, tokens: torch.Tensor, actions_dist: torch.Tensor) -> torch.Tensor:
        """
        Универсальный обсчет наград без рантайм-аллокаций.
        Безопасно извлекает данные как из 1D, так и из 2D токенов.
        """
        self.reward_scratch.zero_()
        
        # Проверяем размерность токенов
        is_2d = (tokens.dim() == 2)
        
        # Векторизованный Task-Agnostic разбор компонент наград
        for idx, component in enumerate(self.reward_components):
            comp_type = component["type"]
            token_idx = idx if idx < tokens.shape[-1] else 0
            
            # Безопасное извлечение (минимальные аллокации)
            if is_2d:
                source_vector = tokens[:, token_idx]
            else:
                source_vector = tokens[token_idx].expand(self.num_agents)
                
            # Копируем в scratch (все компоненты обрабатываются одинаково на данном этапе)
            self.reward_scratch[idx].copy_(source_vector.to(torch.int64))
        
        # Fixed-Point умножение + сдвиг
        self.reward_scratch.mul_(self.weights.unsqueeze(1))
        self.reward_scratch.bitwise_right_shift_(16)
        
        # Агрегация в out_rewards
        torch.sum(self.reward_scratch, dim=0, out=self.out_rewards)
        
        # SymLog
        return self.symlog_transform(self.out_rewards)


__all__ = ["RewardCalculator"]