import os
import torch
from typing import Tuple, Dict, Any

from .determinism import setup_determinism
from .memory import MemoryManager
from .soa_views import create_soa_views
from config.schema import BaseConfig

# === MODULE 1 (ЭТАП 3) ===
from modules.module1_tensor_engine.physics import PhysicsEngine
from modules.module1_tensor_engine.environment import EnvironmentUpdater

# === MODULE 3 (ЭТАП 4) ===
from modules.module3_evolution.ga_no_sort import get_ga_engine
from modules.module3_evolution.hebbian import get_hebbian

# === MODULE 2, 4, 5 (ЭТАП 5) ===
from modules.module2_surrogate_wm.perceiver import get_perceiver
from modules.module2_surrogate_wm.rssm_vit import get_rssm
from modules.module2_surrogate_wm.hdc_memory import get_hdc_memory
from modules.module4_hippocampus.cognitive_dropout import get_cognitive_dropout
from modules.module4_hippocampus.matkaregulator import get_matka_regulator
from modules.module5_task_interface.affordance_decoder import AffordanceDecoder
from modules.module5_task_interface.reward_calculator import RewardCalculator

# ====================== GLOBAL DETERMINISM LOCK ======================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ADCEEngineV33:
    """
    Главный класс ADCE Engine v3.3
    Соблюдает все инварианты: Zero-Allocation, Fixed-Point, Branchless, Task-Agnostic
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = BaseConfig(**config).model_dump()
        
        device_str = self.config.get("device", "cpu")
        if "cuda" in device_str and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            self.config["device"] = "cpu"
        self.device = torch.device(self.config["device"])
        
        # Determinism
        self.rng_generator = setup_determinism(self.config.get("seed", 42))
        
        # Core Memory
        self.memory = MemoryManager(self.config)
        self.views = create_soa_views(self.memory)
        
        # === MODULE 1: Tensor Engine + Physics ===
        self.physics = PhysicsEngine(self.config, self.views)
        self.physics.set_rng_generator(self.rng_generator)          # ← Синхронизация
        
        self.env_updater = EnvironmentUpdater(self.config, self.views)
        
        # === MODULE 3: Evolution + Hebbian ===
        self.ga = get_ga_engine(self.config, self.views)
        self.ga.set_rng_generator(self.rng_generator)               # ← Синхронизация
        
        self.hebbian = get_hebbian(self.config, self.views)
        
        # === MODULE 2, 4, 5 ===
        self.perceiver = get_perceiver(self.config, self.views)
        self.rssm = get_rssm(self.config, self.views)
        self.hdc = get_hdc_memory(self.config, self.views)
        self.dropout = get_cognitive_dropout(self.config, self.views)
        self.matka = get_matka_regulator(self.config, self.views)
        
        self.affordance_decoder = AffordanceDecoder(self.config)
        self.reward_calc = RewardCalculator(self.config, self.views)
        
        # Scratch buffers
        self.out_obs_scratch = torch.zeros(
            (self.config["max_agents"], self.config["task"]["perceiver_tokens"]),
            dtype=torch.float32, device=self.device
        )
        self._action_fallback_scratch = torch.zeros(
            (self.config["max_agents"], self.config.get("task", {}).get("affordance_map_size", 32)),
            dtype=torch.int64, device=self.device
        )
        
        print(f"[ADCE v3.3] Engine initialized successfully")
        
        print(f"   Device          : {self.device}")
        print(f"   Grid size       : {self.config.get('grid_size')}")
        print(f"   Max agents      : {self.config.get('max_agents')}")
        print(f"   Modules loaded  : Physics + GA + Hebbian + Surrogate-WM + Matka + Task-Interface")

    def _create_soa_views(self) -> Dict[str, torch.Tensor]:
        """Создание представлений SoA (Zero-Allocation)"""
        return create_soa_views(self.memory)

    def get_api_exchange_handle(self, external_process_id: int = 0) -> Dict[str, Any]:
        """API для внешних процессов (IPC / пинг-понг буферизация)"""
        return {
            "T_global": self.memory.T_global,
            "views": self.views,
            "device": self.device
        }

    def step(self, abstract_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(abstract_actions, torch.Tensor) or abstract_actions.shape[0] == 0:
            raise ValueError("[ENGINE] Invalid abstract_actions tensor")

        # Safe fallback
        if abstract_actions.sum() == 0:
            torch.randint(0, 5, abstract_actions.shape, 
                         generator=self.rng_generator, 
                         out=self._action_fallback_scratch)
            abstract_actions = self._action_fallback_scratch

        # 1. Decode
        actions_distribution_indices, actions_mental = self.affordance_decoder.decode(abstract_actions)
        
        # 2. Physics
        updated_positions = self.physics.step_physics(actions_distribution_indices)
        
        # 3. Environment
        self.env_updater.update_environment(updated_positions)
        
        # 4. WM Pipeline
        tokens = self.perceiver.tokenize()
        next_state_pred = self.rssm.predict_next(tokens, actions_mental)
        self.hdc.bind(tokens)
        
        # 5. Reward
        reward = self.reward_calc.compute(tokens, actions_distribution_indices)
        fitness = self.matka.regulate(reward)
        
        # 6. Learning
        self.hebbian.update_synapses(actions_mental, actions_distribution_indices)
        self.ga.evolve(fitness)
        self.dropout.apply_dropout(self.hdc)
        
        # 7. Output
        self.out_obs_scratch.copy_(tokens.float())
        return self.out_obs_scratch, reward


__all__ = ["ADCEEngineV33"]