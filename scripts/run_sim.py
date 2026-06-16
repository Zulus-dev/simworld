from core.engine import ADCEEngineV33
import json

def run_simulation(steps=1000):
    with open("config/base.json") as f:
        config = json.load(f)
    engine = ADCEEngineV33(config)
    
    for i in range(steps):
        dummy_actions = torch.zeros(config["max_agents"], device=engine.device, dtype=torch.int64)
        engine.step(dummy_actions, dummy_actions)
        if i % 200 == 0:
            print(f"[SIM] Step {i} completed")

if __name__ == "__main__":
    run_simulation()