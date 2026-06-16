import torch
from typing import Dict

from utils.bitpack import extract_bits


def blit_environment(views: Dict[str, torch.Tensor], fb: 'FramebufferManager') -> None:
    """Branchless multi-dimensional flat blitting стен + феромонов. Только out= и bitwise."""
    T_env = views["T_environment"]
    
    # Работаем со средой как с плоским вектором (Flat Indexing) независимо от 2D/3D/4D
    env_flat = T_env.view(-1)
    
    # Разворачиваем framebuffer в плоский массив пикселей по пространственным осям для in-place blit
    fb_flat = fb.T_framebuffer.view(-1, 3)
    
    # 1. Стены (бит 0). Используем предаллоцированный скретч-буфер (выравнивание памяти)
    flat_scratch = fb.env_scratch.view(-1)
    wall_mask = torch.bitwise_and(env_flat, 0x01, out=flat_scratch).eq(0)
    
    # Закрашиваем каналы тёмно-серым (In-place)
    fb_flat[..., 0].masked_fill_(wall_mask, 80)
    fb_flat[..., 1].masked_fill_(wall_mask, 80)
    fb_flat[..., 2].masked_fill_(wall_mask, 80)
    
    # 2. Феромон А (роя) — биты 2-4 → 3 бита → извлекаем универсально
    pher_a = extract_bits(env_flat, 2, 3, out=flat_scratch)
    fb_flat[..., 1].add_(pher_a * 32, alpha=1)   # Сдвиг интенсивности в зелёный канал
    
    # 3. Феромон Б (Матки) — биты 5-7 → синий канал
    pher_b = extract_bits(env_flat, 5, 3, out=flat_scratch)
    fb_flat[..., 2].add_(pher_b * 32, alpha=1)


def blit_entities(views: Dict[str, torch.Tensor], fb: 'FramebufferManager') -> None:
    """Multi-dimensional flat blitting сущностей (ГА >0 — красный, Матки <0 — фиолетовый)"""
    # Читаем плоские индексы сущностей (O(1) доступ согласно разделу 2.3 спецификации)
    T_ent_flat = views["T_entities"].view(-1)
    fb_flat = fb.T_framebuffer.view(-1, 3)
    
    # Маски на плоском векторе
    ga_mask = T_ent_flat > 0
    matka_mask = T_ent_flat < 0
    
    # Очистка под сущностями + цвет (in-place)
    fb_flat[..., 0].masked_fill_(ga_mask, 255)   # Красный для ГА
    fb_flat[..., 1].masked_fill_(ga_mask, 60)
    fb_flat[..., 2].masked_fill_(ga_mask, 60)
    
    fb_flat[..., 0].masked_fill_(matka_mask, 200)  # Фиолетовый для Матки
    fb_flat[..., 1].masked_fill_(matka_mask, 60)
    fb_flat[..., 2].masked_fill_(matka_mask, 255)

def blit_frame(views: Dict[str, torch.Tensor], fb: 'FramebufferManager') -> torch.Tensor:
    """Полная сборка кадра БЕЗ новых аллокаций"""
    fb.reset()
    blit_environment(views, fb)
    blit_entities(views, fb)
    return fb.T_framebuffer