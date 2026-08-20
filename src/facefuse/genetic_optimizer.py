"""Real-valued genetic algorithm for gradient-free hyper-parameter search."""

from __future__ import annotations

import numpy as np


class GeneticOptimizer:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.2, seed=None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

    def optimize(self, fitness_func, bounds, verbose=True):
      
        n_params = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        population = self.rng.uniform(lower, upper, (self.population_size, n_params))

        best_fitness = -np.inf
        best_individual = None

        for gen in range(self.generations):
            fitnesses = np.array([fitness_func(ind) for ind in population])
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_individual = population[best_idx].copy()

            if verbose:
                print(f"Generation {gen + 1}/{self.generations} | Best fitness: {best_fitness:.4f}")

            sorted_idx = np.argsort(fitnesses)[::-1]
            parents = population[sorted_idx[:self.population_size // 2]]

            offspring = []
            for _ in range(self.population_size // 2):
                p1, p2 = parents[self.rng.integers(0, len(parents), 2)]
                child = (p1 + p2) / 2.0
                offspring.append(child)
            offspring = np.array(offspring)

            population = np.vstack((parents, offspring))

            for i in range(len(population)):
                if self.rng.random() < self.mutation_rate:
                    mutation = self.rng.normal(0, 0.1 * (upper - lower))
                    population[i] += mutation
                    population[i] = np.clip(population[i], lower, upper)

        return best_individual, best_fitness