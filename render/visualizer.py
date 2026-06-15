import torch
from typing import Any, Dict

from .framebuffer import FramebufferManager
from .blit_core import blit_frame


class ADCEVisualizer:
    """Независимый Render Plane — только чтение через SoA views"""
    
    def __init__(self, engine_instance: Any, headless: bool = True):
        self.engine = engine_instance
        self.headless = headless
        self.config = engine_instance.config
        
        self.framebuffer = FramebufferManager(self.config, engine_instance.memory.device)
        self.frame_counter = 0
        
        print(f"[RENDER] ADCEVisualizer initialized | Headless={headless}")
    
    def render(self) -> torch.Tensor:
        """Основной метод — Zero-Allocation blitting"""
        views = self.engine._create_soa_views()
        frame = blit_frame(views, self.framebuffer)
        
        self.frame_counter += 1
        
        if not self.headless and self.frame_counter % self.config.get("render_every", 100) == 0:
            # TODO: Local OpenGL backend later
            print(f"[RENDER] Frame {self.frame_counter} generated")
        
        return frame.clone()  # Только для внешнего использования, внутренне — reference
    
    def save_frame(self, path: str):
        """Для отладки — сохраняем PNG (требует torchvision или PIL)"""
        try:
            import torchvision
            torchvision.utils.save_image(self.framebuffer.T_framebuffer.permute(2,0,1).float() / 255.0, path)
            print(f"[RENDER] Saved frame to {path}")
        except ImportError:
            print("[RENDER] torchvision not available for save")