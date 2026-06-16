# Публичный доступ к модулям
from .module1_tensor_engine.physics import PhysicsEngine
from .module1_tensor_engine.environment import EnvironmentUpdater

from .module2_surrogate_wm.perceiver import Perceiver, get_perceiver
from .module2_surrogate_wm.rssm_vit import RSSM_ViT, get_rssm
from .module2_surrogate_wm.hdc_memory import HDCMemory, get_hdc_memory

from .module3_evolution.ga_no_sort import GANoSort, get_ga_engine
from .module3_evolution.hebbian import HebbianUpdater, get_hebbian

from .module4_hippocampus.cognitive_dropout import CognitiveDropout, get_cognitive_dropout
from .module4_hippocampus.matkaregulator import MatkaRegulator, get_matka_regulator

from .module5_task_interface.affordance_decoder import AffordanceDecoder
from .module5_task_interface.reward_calculator import RewardCalculator

__all__ = [
    "PhysicsEngine", "EnvironmentUpdater",
    "Perceiver", "RSSM_ViT", "HDCMemory",
    "GANoSort", "HebbianUpdater",
    "CognitiveDropout", "MatkaRegulator",
    "AffordanceDecoder", "RewardCalculator"
]