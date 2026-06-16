import torch
from typing import Tuple, Dict, Any

class FramebufferManager:
    """Предаллокация T_framebuffer + scratch-буферов для Zero-Allocation blitting"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        h, w = config.get("render_resolution", config.get("grid_size", [256, 256]))
        self.height = h
        self.width = w
        
        # Главный framebuffer — uint8 [H, W, 3] RGB
        self.T_framebuffer = torch.zeros((h, w, 3), dtype=torch.uint8, device=device)
        
        # Scratch-буферы (предаллоцированы один раз)
        self.env_scratch = torch.zeros((h, w), dtype=torch.uint8, device=device)
        self.mask_scratch = torch.zeros((h, w), dtype=torch.bool, device=device)
        self.color_scratch = torch.zeros((h, w, 3), dtype=torch.uint8, device=device)
        
        print(f"[RENDER] Framebuffer allocated: {h}x{w}x3 uint8 | Device={device}")
    
    def reset(self):
        """Сброс кадра без новой аллокации"""
        self.T_framebuffer.zero_()