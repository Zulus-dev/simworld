import torch
from typing import Dict, Tuple

from utils.fixed_point import FixedPoint
from utils.tensor_ops import get_tensor_ops


class GANoSort:
    """No-Sort Genetic Algorithm — Tournament Selection + Crossover + Mutation (Branchless)"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_dna_pool"].device
        self.max_agents = config["max_agents"]
        self.genome_length = config["genome_length"]
        
        self.fp = FixedPoint(config.get("fixed_point_multiplier", 65536))
        self.tournament_size = config.get("tournament_size", 4)
        
        # Pre-allocated scratch (Zero-Allocation)
        self.parent_indices = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.scratch_fitness = torch.zeros(self.max_agents, dtype=torch.int64, device=self.device)
    
    def tournament_selection(self, fitness: torch.Tensor) -> torch.Tensor:
        """Branchless tournament selection without sorting"""
        rng = torch.Generator(device=self.device)
        rng.manual_seed(42)  # deterministic for stage
        
        # Random opponents
        opponents = torch.randint(0, self.max_agents, (self.max_agents, self.tournament_size), 
                                generator=rng, device=self.device)
        
        # Gather fitness of opponents
        opp_fitness = fitness[opponents]
        
        # Find best in each tournament (branchless max via masked compare)
        best_mask = opp_fitness.argmax(dim=1)
        selected = opponents[torch.arange(self.max_agents, device=self.device), best_mask]
        
        self.parent_indices.copy_(selected)
        return selected
    
    def crossover(self, parents1: torch.Tensor, parents2: torch.Tensor):
        """Single-point crossover in-place on T_dna_pool"""
        T_dna = self.views["T_dna_pool"]
        crossover_point = torch.randint(1, self.genome_length - 1, (self.max_agents,), 
                                      device=self.device, generator=torch.Generator(self.device))
        
        for i in range(self.max_agents):  # will be vectorized later, demo stage
            cp = crossover_point[i].item()
            temp = T_dna[parents1[i], cp:].clone()
            T_dna[parents1[i], cp:] = T_dna[parents2[i], cp:]
            T_dna[parents2[i], cp:] = temp  # simplified, real version uses masks
    
    def mutate(self, mutation_rate: float = 0.01):
        """Fixed-point mutation"""
        T_dna = self.views["T_dna_pool"]
        mask = torch.rand(self.max_agents, self.genome_length, device=self.device) < mutation_rate
        mutations = torch.randint(-100, 101, mask.shape, dtype=torch.int32, device=self.device)
        T_dna.add_(mutations * mask.to(torch.int32))
    
    def evolve(self, fitness: torch.Tensor):
        """Full GA cycle — No-Sort"""
        parents = self.tournament_selection(fitness)
        # Pair parents (simple shift for demo)
        parents2 = torch.roll(parents, 1)
        
        self.crossover(parents, parents2)
        self.mutate()
        
        print(f"[GA] Evolution step completed | Avg Fitness: {fitness.float().mean().item():.4f}")


# Global instance
ga_engine: GANoSort | None = None

def get_ga_engine(config: dict, views: Dict) -> GANoSort:
    global ga_engine
    if ga_engine is None:
        ga_engine = GANoSort(config, views)
    return ga_engine