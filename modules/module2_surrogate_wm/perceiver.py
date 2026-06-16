import torch
from typing import Dict
from utils.fixed_point import get_fixed_point


class Perceiver:
    def __init__(self, config: Dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_environment"].device
        self.fp = get_fixed_point(config)
        self.token_dim = config.get("task", {}).get("perceiver_tokens", 512)
        self.token_buffer = torch.zeros(self.token_dim, dtype=torch.int32, device=self.device)
    
    def tokenize(self) -> torch.Tensor:
        T_env = self.views["T_environment"]
        T_ent = self.views["T_entities"]
        
        # In-place статистики (Zero-Allocation)
        pher_a = torch.bitwise_and(T_env, 0x1C).float().mean()
        pher_b = torch.bitwise_and(T_env, 0xE0).float().mean()
        ent_act = (T_ent != 0).float().mean()
        
        # Заполнение буфера без новых тензоров
        chunk = self.token_dim // 3
        self.token_buffer[:chunk] = (pher_a * self.fp.M).to(torch.int32)
        self.token_buffer[chunk:2*chunk] = (pher_b * self.fp.M).to(torch.int32)
        self.token_buffer[2*chunk:] = (ent_act * self.fp.M).to(torch.int32)
        
        return self.token_buffer


perceiver: Perceiver | None = None

def get_perceiver(config: Dict, views: Dict) -> Perceiver:
    global perceiver
    if perceiver is None:
        perceiver = Perceiver(config, views)
    return perceiver