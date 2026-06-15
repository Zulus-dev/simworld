import torch
from typing import Dict

from utils.bitpack import extract_bits


def blit_environment(views: Dict[str, torch.Tensor], fb: 'FramebufferManager') -> None:
    """Branchless in-place blitting стен + феромонов. Только out= и bitwise."""
    T_env = views["T_environment"]
    h, w = fb.height, fb.width
    grid_h, grid_w = fb.config["grid_size"]
    
    # Reshape environment to 2D
    env_2d = T_env.view(grid_h, grid_w)
    
    # Стены (бит 0)
    wall_mask = torch.bitwise_and(env_2d, 0x01, out=fb.env_scratch).eq(0)
    fb.T_framebuffer[..., 0].masked_fill_(wall_mask, 80)   # тёмно-серый
    fb.T_framebuffer[..., 1].masked_fill_(wall_mask, 80)
    fb.T_framebuffer[..., 2].masked_fill_(wall_mask, 80)
    
    # Феромон А (роя) — биты 2-4 → зелёный канал
    pher_a = extract_bits(env_2d, 2, 3)  # 3 бита → 0-7
    fb.T_framebuffer[..., 1].add_(pher_a * 32, alpha=1)   # in-place scaling
    
    # Феромон Б (Матки) — биты 5-7 → синий канал
    pher_b = extract_bits(env_2d, 5, 3)
    fb.T_framebuffer[..., 2].add_(pher_b * 32, alpha=1)


def blit_entities(views: Dict[str, torch.Tensor], fb: 'FramebufferManager') -> None:
    """Blitting сущностей (ГА >0 — красный, Матки <0 — фиолетовый)"""
    T_ent = views["T_entities"].view(fb.height, fb.width)
    
    # Маски
    ga_mask = T_ent > 0
    matka_mask = T_ent < 0
    
    # Очистка под сущностями + цвет (in-place)
    fb.T_framebuffer[..., 0].masked_fill_(ga_mask, 255)   # Красный для ГА
    fb.T_framebuffer[..., 1].masked_fill_(ga_mask, 60)
    fb.T_framebuffer[..., 2].masked_fill_(ga_mask, 60)
    
    fb.T_framebuffer[..., 0].masked_fill_(matka_mask, 200)  # Фиолетовый для Матки
    fb.T_framebuffer[..., 1].masked_fill_(matka_mask, 60)
    fb.T_framebuffer[..., 2].masked_fill_(matka_mask, 255)


def blit_frame(views: Dict[str, torch.Tensor], fb: 'FramebufferManager') -> torch.Tensor:
    """Полная сборка кадра БЕЗ новых аллокаций"""
    fb.reset()
    blit_environment(views, fb)
    blit_entities(views, fb)
    return fb.T_framebuffer