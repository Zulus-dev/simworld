import torch
from core.engine import ADCEEngineV33
import json

def profile_memory():
    with open("config/base.json") as f:
        config = json.load(f)
    
    engine = ADCEEngineV33(config)
    initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for _ in range(100):
        dummy = torch.zeros(engine.config["max_agents"], device=engine.device)
        engine.step(dummy, dummy)
    
    final_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"[MEMORY PROFILE] Growth: {final_mem - initial_mem} bytes (should be near 0)")

if __name__ == "__main__":
    profile_memory()