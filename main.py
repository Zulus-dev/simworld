import argparse
import json
import torch
import os
from core.engine import ADCEEngineV33
from render.visualizer import ADCEVisualizer
from core.ipc import get_ipc_manager
from scripts.generate_dataset import generate_dataset

# Force CPU fallback if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Раскомментируйте при необходимости

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

        # Динамически извлекаем параметры из неизменяемого JSON-конфига задачи
        affordance_map_size = engine.config.get("task", {}).get("affordance_map_size", 32)
        max_agents = engine.config["max_agents"]

        if args.mode == "interactive":
            print(f"[MAIN] Starting interactive mode for {args.steps} steps")
            os.makedirs("outputs", exist_ok=True)
            
            for step in range(args.steps):
                # Создаем единый абстрактной тензор действий [Max_Agents, Affordance_Map_Size] 
                # согласно Task-Agnostic инварианту
                dummy_abstract_actions = torch.zeros(
                    (max_agents, affordance_map_size), 
                    device=engine.device, 
                    dtype=torch.int64
                )
                
                # Исполняем шаг через обновленный интерфейс ядра
                obs, reward = engine.step(dummy_abstract_actions)
                
                if step % config.get("render_every", 100) == 0:
                    frame = visualizer.render()
                    print(f"[MAIN] Step {step:04d} | Frame rendered | Shape={frame.shape}")
                    
                    # СОХРАНЕНИЕ КАДРА ЧЕРЕЗ TORCHVISION С ОПТИМИЗАЦИЕЙ ПАМЯТИ
                    import torchvision.utils as tv_utils
                    
                    img_tensor = frame.permute(2, 0, 1).float() / 255.0
                    tv_utils.save_image(img_tensor, f"outputs/frame_{step:04d}.png")

                    avg_reward = reward.float().mean().item()
    
                    print(f"[METRIC] Step {step:04d} | Avg Fitness: {avg_reward:.4f} | Goal: {engine.config.get('task', {}).get('goal_metric')}")
                    
                    frame = visualizer.render()

        elif args.mode == "dataset":
            print("[MAIN] Dataset generation mode")
            # Запуск высокопроизводительного статического сборщика датасетов
            generate_dataset(args.config, num_steps=args.steps)

        elif args.mode == "benchmark":
            print("[MAIN] Benchmark mode - running fast simulation")
            for step in range(args.steps):
                dummy_abstract_actions = torch.zeros(
                    (max_agents, affordance_map_size), 
                    device=engine.device, 
                    dtype=torch.int64
                )
                engine.step(dummy_abstract_actions)
        
        print("[MAIN] Session finished successfully.")
    
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()