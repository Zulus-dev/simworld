from pydantic import BaseModel, Field, field_validator
from typing import List

class EnvironmentConfig(BaseModel):
    pheromone_levels: int = Field(8, ge=2, le=16)
    resource_spawn_rate: float = Field(0.02, ge=0.0, le=1.0)
    block_sparse_block_size: int = Field(8, ge=4, le=32)

class EvolutionConfig(BaseModel):
    mutation_rate: float = Field(0.01, ge=0.0, le=1.0)
    crossover_rate: float = Field(0.7, ge=0.0, le=1.0)
    tournament_size: int = Field(4, ge=2)
    hebbian_eta: float = Field(0.05, ge=0.0)
    hebbian_gamma: float = Field(0.1, ge=0.0)

class TaskConfig(BaseModel):
    name: str = "default_resource_collection"
    perceiver_tokens: int = Field(512, ge=64)
    affordance_map_size: int = Field(32, ge=8)

class BaseConfig(BaseModel):
    """Global configuration schema for ADCE Engine v3.3"""
    
    fixed_point_multiplier: int = Field(65536, le=65536)
    seed: int = 42
    
    max_agents: int = Field(4096, ge=64, le=65536)
    grid_size: List[int] = Field([256, 256], min_items=2, max_items=2)
    genome_length: int = Field(512, ge=64, le=4096)
    
    device: str = "cuda:0"
    headless: bool = True
    render_every: int = Field(100, ge=1)
    
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    
    @field_validator('grid_size')
    @classmethod
    def validate_grid_size(cls, v: List[int]) -> List[int]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("Grid dimensions must be positive")
        return v
    
    class Config:
        extra = "allow"

# Convenience loader
def load_base_config(path: str = "config/base.json") -> BaseConfig:
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return BaseConfig(**data)