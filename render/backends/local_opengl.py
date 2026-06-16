import torch
from typing import Any

class LocalOpenGLBackend:
    """ModernGL + CUDA-OpenGL Interop (Local)"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        print("[OpenGL] Local backend initialized (CUDA interop ready)")
    
    def render_loop(self):
        print("[OpenGL] Render loop started (stub)")


def get_local_backend(visualizer):
    return LocalOpenGLBackend(visualizer)