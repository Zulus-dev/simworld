import torch
import os
import json
from pathlib import Path
from tqdm import tqdm
from core.engine import ADCEEngineV33
from core.ipc import get_ipc_manager

def generate_dataset(config_path: str = "config/base.json", 
                    num_steps: int = 1000,
                    batch_size: int = 64,
                    output_dir: str = "outputs/datasets"):
    """
    Генерация датасета для обучения внешних нейросетей.
    Соответствует новому инварианту: один абстрактный тензор на вход step().
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    engine = ADCEEngineV33(config)
    
    # Параметры из конфига
    max_agents = engine.config["max_agents"]
    affordance_map_size = engine.config["task"]["affordance_map_size"]
    
    dataset = {
        "observations": [],
        "actions_dist": [],
        "rewards": [],
        "next_observations": []
    }
    
    print(f"[DATASET] Starting generation for {num_steps} steps with batch_size={batch_size}...")
    
    for step in tqdm(range(num_steps)):
        # 1. Генерируем случайные абстрактные действия [Max_Agents, Affordance_Map_Size]
        abstract_actions = torch.randint(
            0, 8, 
            (max_agents, affordance_map_size), 
            device=engine.device, 
            dtype=torch.int64
        )
        
        # 2. Выполняем шаг через единый интерфейс
        obs, reward = engine.step(abstract_actions)
        
        # 3. Получаем следующее состояние
        next_obs = engine.perceiver.tokenize().float()
        
        # 4. Сохраняем в буфер
        dataset["observations"].append(obs.clone())
        dataset["actions_dist"].append(abstract_actions.clone())
        dataset["rewards"].append(reward.clone())
        dataset["next_observations"].append(next_obs.clone())
        
        # 5. Периодическое сохранение батчей на диск
        if (step + 1) % batch_size == 0:
            batch_idx = (step + 1) // batch_size
            batch_path = Path(output_dir) / f"batch_{batch_idx:06d}.pt"
            
            torch.save({
                "obs": torch.stack(dataset["observations"]),
                "action_dist": torch.stack(dataset["actions_dist"]),
                "reward": torch.stack(dataset["rewards"]),
                "next_obs": torch.stack(dataset["next_observations"])
            }, batch_path)
            
            # Очистка памяти
            for key in dataset:
                dataset[key].clear()
    
    print(f"[DATASET] Generation completed. Saved to {output_dir}")

if __name__ == "__main__":
    # Для автономного запуска
    generate_dataset()