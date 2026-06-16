import torch
from typing import Dict

from utils.fixed_point import FixedPoint


class GANoSort:
    """No-Sort Genetic Algorithm"""
    
    def __init__(self, config: dict, views: Dict[str, torch.Tensor]):
        self.config = config
        self.views = views
        self.device = views["T_dna_pool"].device
        self.max_agents = config["max_agents"]
        self.genome_length = config["genome_length"]
        
        self.fp = FixedPoint(config.get("fixed_point_multiplier", 65536))
        self.tournament_size = config.get("tournament_size", 4)
        self.mutation_rate = config.get("evolution", {}).get("mutation_rate", 0.01)
        
        # Pre-allocated scratch
        self.parent_indices = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.scratch_parents2 = torch.zeros(self.max_agents, dtype=torch.int32, device=self.device)
        self.mutation_mask = torch.zeros(self.max_agents, dtype=torch.bool, device=self.device)
        self.mutation_scratch = torch.zeros((self.max_agents, self.genome_length), 
                                          dtype=torch.int32, device=self.device)
        
        self.rng_generator = None

    def set_rng_generator(self, rng_generator: torch.Generator):
        self.rng_generator = rng_generator

    def tournament_selection(self, fitness: torch.Tensor) -> torch.Tensor:
        if self.rng_generator is None:
            self.rng_generator = torch.Generator(device=self.device)
            self.rng_generator.manual_seed(42)
        
        opponents = torch.randint(0, self.max_agents, (self.max_agents, self.tournament_size),
                                generator=self.rng_generator, device=self.device)
        
        opp_fitness = fitness[opponents]
        best_mask = opp_fitness.argmax(dim=1)
        selected = opponents[torch.arange(self.max_agents, device=self.device), best_mask]
        
        self.parent_indices.copy_(selected)
        return selected

    def crossover(self, parents1: torch.Tensor, parents2: torch.Tensor):
        """Простой и надёжный crossover"""
        T_dna = self.views["T_dna_pool"]  # [N, L]
        
        parents2.copy_(torch.roll(parents1, shifts=1))
        
        half = self.genome_length // 2
        with torch.no_grad():
            temp = T_dna.clone()
            T_dna[:, half:] = temp[parents2, half:]
            T_dna[parents2, :half] = temp[parents1, :half]

    def mutate(self):
        """Branchless mutation"""
        if self.rng_generator is None:
            self.rng_generator = torch.Generator(device=self.device)
            self.rng_generator.manual_seed(42)
        
        T_dna = self.views["T_dna_pool"]
        
        # Mutation mask
        rand_vals = torch.rand(self.max_agents, generator=self.rng_generator, device=self.device)
        self.mutation_mask.copy_(rand_vals < self.mutation_rate)
        
        # Generate mutations
        torch.randint(-50, 51, (self.max_agents, self.genome_length),
                     generator=self.rng_generator,
                     device=self.device,
                     dtype=torch.int32,
                     out=self.mutation_scratch)
        
        # Apply only to masked agents
        mask_exp = self.mutation_mask.unsqueeze(1).expand(-1, self.genome_length)
        T_dna.add_(self.mutation_scratch * mask_exp.to(torch.int32))

    def evolve(self, fitness: torch.Tensor):
        parents = self.tournament_selection(fitness)
        self.crossover(parents, self.scratch_parents2)
        self.mutate()
        print(f"[GA] Evolution step completed | Avg Fitness: {fitness.float().mean().item():.4f}")


# Global
ga_engine: GANoSort | None = None

def get_ga_engine(config: dict, views: Dict) -> GANoSort:
    global ga_engine
    if ga_engine is None:
        ga_engine = GANoSort(config, views)
    return ga_engine