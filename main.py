import argparse
import json
import torch
import os
from core.engine import ADCEEngineV33
from render.visualizer import ADCEVisualizer
from core.ipc import get_ipc_manager
from scripts.generate_dataset import generate_dataset

# Force CPU fallback if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # remove in production

def main():
    parser = argparse.ArgumentParser(description="ADCE Engine v3.3 CLI")
    parser.add_argument("--config", default="config/base.json", help="Path to base config")
    parser.add_argument("--mode", choices=["dataset", "interactive", "benchmark"], default="interactive")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        print("[MAIN] Starting ADCE Engine v3.3...")
        engine = ADCEEngineV33(config)
        visualizer = ADCEVisualizer(engine, headless=config.get("headless", True))
        ipc = get_ipc_manager(engine)

        if args.mode == "interactive":
            print(f"[MAIN] Starting interactive mode for {args.steps} steps")
            for step in range(args.steps):
                dummy_actions = torch.zeros(engine.config["max_agents"], 
                                          device=engine.device, dtype=torch.int64)
                obs, reward = engine.step(dummy_actions, dummy_actions)
                
                if step % config.get("render_every", 100) == 0 or step == 0:
                    frame = visualizer.render()
                    print(f"[MAIN] Step {step:04d} | Frame rendered | Shape={frame.shape}")
            
            print("[MAIN] Interactive simulation completed.")

        elif args.mode == "dataset":
            print("[MAIN] Dataset generation mode")
            generate_dataset(args.config, num_steps=args.steps)

        elif args.mode == "benchmark":
            print("[MAIN] Benchmark mode - running fast simulation")
            for step in range(args.steps):
                dummy_actions = torch.zeros(engine.config["max_agents"], 
                                          device=engine.device, dtype=torch.int64)
                engine.step(dummy_actions, dummy_actions)
        
        print("[MAIN] Session finished successfully.")
    
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()