import torch
from typing import Dict, Any
from core.ipc import get_ipc_manager

# Alias для совместимости
def get_ipc_manager_alias(engine):
    return get_ipc_manager(engine)