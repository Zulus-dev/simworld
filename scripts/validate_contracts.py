from core.engine import ADCEEngineV33
import json
import torch

def validate():
    with open("config/base.json") as f:
        config = json.load(f)
    
    engine = ADCEEngineV33(config)
    views = engine._create_soa_views()
    
    assert "T_environment" in views, "SoA views contract failed"
    assert engine.memory.T_global.dtype == torch.uint8, "T_global type violation"
    
    print("[VALIDATION] All core contracts PASSED")
    print(f"   T_global size: {engine.memory.T_global.numel()} bytes")

if __name__ == "__main__":
    validate()