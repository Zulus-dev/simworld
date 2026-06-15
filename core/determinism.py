import os
import torch

# Global determinism lock
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def setup_determinism(seed: int = 42):
    """Setup isolated RNG Generator"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator