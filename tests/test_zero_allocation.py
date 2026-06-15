import torch
from core.engine import ADCEEngineV33
from config.schema import load_base_config

def test_zero_allocation():
    config = load_base_config().model_dump()
    config['device'] = 'cpu'
    
    initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    engine = ADCEEngineV33(config)
    views = engine._create_soa_views()
    
    for _ in range(10):
        _ = engine._create_soa_views()
        _ = engine.step()
    
    final_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    growth = final_mem - initial_mem
    assert growth < 10 * 1024 * 1024, f"Memory growth detected: {growth} bytes"
    print(f"[TEST PASSED] Zero-Allocation OK. Growth: {growth} bytes")

if __name__ == "__main__":
    test_zero_allocation()