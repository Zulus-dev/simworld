import torch
from typing import Dict, Any

def create_exchange_handle(engine) -> Dict[str, Any]:
    """Создаёт минимальный handle для внешнего процесса"""
    return {
        "T_global": engine.memory.T_global,
        "views": engine.views,
        "device": engine.device,
        "step_counter": 0
    }