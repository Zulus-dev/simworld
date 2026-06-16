import torch
from typing import Any, Dict
from .framebuffer import FramebufferManager
from .blit_core import blit_frame

class ADCEVisualizer:
    """Независимый Render Plane — читает данные из SoA (Structure of Arrays)"""
    def __init__(self, engine_instance: Any, headless: bool = True):
        self.engine = engine_instance
        self.headless = headless
        self.config = engine_instance.config
        self.framebuffer = FramebufferManager(self.config, engine_instance.memory.device)
        self.frame_counter = 0
        print(f"[RENDER] ADCEVisualizer initialized | Headless={headless}")
    
    def render(self) -> torch.Tensor:
        """Метод Zero-Allocation blitting."""
        views = self.engine._create_soa_views()
        frame = blit_frame(views, self.framebuffer)
        
        # Логика сохранения для отладки, если включен режим вывода
        if not self.headless and self.frame_counter % self.config.get("render_every", 100) == 0:
            self.save_frame(f"outputs/frame_{self.frame_counter:06d}.png")
            
        self.frame_counter += 1
        return frame.clone()
    
    def save_frame(self, path: str):
        try:
            from torchvision.utils import save_image
            # Нормализация для сохранения
            img = self.framebuffer.T_framebuffer.permute(2, 0, 1).float() / 255.0
            save_image(img, path)
            print(f"[RENDER] Frame saved: {path}")
        except Exception as e:
            print(f"[RENDER] Save failed: {e}")