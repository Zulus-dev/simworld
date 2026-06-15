import torch
import os
import json
from pathlib import Path
from tqdm import tqdm

from core.engine import ADCEEngineV33


def generate_dataset(config_path: str = "config/base.json", 
                    num_steps: int = 10000,
                    batch_size: int = 2048,
                    output_dir: str = "outputs/datasets"):
    """
    Генерация чистого датасета для обучения внешних нейросетей
    Формат: obs, action_dist, action_mental, reward, next_obs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    engine = ADCEEngineV33(config)
    ipc = get_ipc_manager(engine)  # для будущей IPC интеграции
    
    dataset = {
        "observations": [],
        "actions_dist": [],
        "actions_mental": [],
        "rewards": [],
        "next_observations": []
    }
    
    print(f"[DATASET] Starting generation for {num_steps} steps...")
    
    for step in tqdm(range(num_steps)):
        # Dummy actions (в реальности — от внешней сети)
        actions_dist = torch.randint(0, 8, (engine.config["max_agents"],), 
                                   device=engine.device, dtype=torch.int64)
        actions_mental = torch.zeros_like(actions_dist, dtype=torch.int64)
        
        obs, reward = engine.step(actions_dist, actions_mental)
        
        # Next observation (after step)
        next_obs = engine.perceiver.tokenize().float() if hasattr(engine, 'perceiver') else obs
        
        # Collect batch
        if step % batch_size == 0 and step > 0:
            # Save batch
            batch_idx = step // batch_size
            batch_path = Path(output_dir) / f"batch_{batch_idx:06d}.pt"
            torch.save({
                "obs": torch.stack(dataset["observations"][-batch_size:]),
                "action_dist": torch.stack(dataset["actions_dist"][-batch_size:]),
                "action_mental": torch.stack(dataset["actions_mental"][-batch_size:]),
                "reward": torch.stack(dataset["rewards"][-batch_size:]),
                "next_obs": torch.stack(dataset["next_observations"][-batch_size:])
            }, batch_path)
            
            # Clear batch memory
            dataset["observations"].clear()
            dataset["actions_dist"].clear()
            dataset["actions_mental"].clear()
            dataset["rewards"].clear()
            dataset["next_observations"].clear()
        
        dataset["observations"].append(obs)
        dataset["actions_dist"].append(actions_dist)
        dataset["actions_mental"].append(actions_mental)
        dataset["rewards"].append(reward)
        dataset["next_observations"].append(next_obs)
    
    print(f"[DATASET] Generation completed. Saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    generate_dataset(num_steps=5000)