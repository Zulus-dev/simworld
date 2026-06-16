import torch
import os
from typing import Any

class HeadlessNVENC:
    """Headless backend using NVENC for .mp4 recording"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.output_path = "outputs/simulation.mp4"
        os.makedirs("outputs", exist_ok=True)
        print(f"[NVENC] Initialized headless recording to {self.output_path}")
    
    def write_frame(self, frame: torch.Tensor):
        """Stub for NVENC write (real impl uses torch + ffmpeg binding)"""
        print(f"[NVENC] Frame written | Shape={frame.shape}")


def get_headless_backend(visualizer):
    return HeadlessNVENC(visualizer)