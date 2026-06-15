import torch
from typing import Dict, Any
import os


class IPCManager:
    """
    CUDA IPC + POSIX SHM Manager для межпроцессного обмена (Контур Б)
    Поддерживает пинг-понг двойную буферизацию
    """
    def __init__(self, engine):
        self.engine = engine
        self.device = engine.device
        self.ping_pong = 0  # 0 or 1
        
        # Pre-allocated exchange buffers
        self.exchange_buffers = [
            torch.zeros(1024 * 1024 * 4, dtype=torch.uint8, device=self.device),  # Buffer A
            torch.zeros(1024 * 1024 * 4, dtype=torch.uint8, device=self.device)   # Buffer B
        ]
        
        self.cuda_event = torch.cuda.Event(enable_timing=False)
    
    def share_handle(self, external_process_id: int) -> Dict[str, Any]:
        """Возвращает дескрипторы для внешнего процесса"""
        handle = self.engine.get_api_exchange_handle(external_process_id)
        handle["ping_pong_buffer"] = self.exchange_buffers[self.ping_pong]
        handle["cuda_event"] = self.cuda_event
        return handle
    
    def sync_step(self):
        """Синхронизация после шага (для внешней сети)"""
        self.cuda_event.record()
        self.ping_pong = 1 - self.ping_pong  # swap buffers
        print(f"[IPC] Step synchronized | Ping-Pong buffer: {self.ping_pong}")


# Global
ipc_manager: IPCManager | None = None

def get_ipc_manager(engine) -> IPCManager:
    global ipc_manager
    if ipc_manager is None:
        ipc_manager = IPCManager(engine)
    return ipc_manager