import numpy as np


def suggest_ga_params(bounds, step_sizes,max_population=1000, max_generations=100):
        # Calculate number of possible values for each parameter
        num_values = [
            int((high - low) / step) + 1
            for (low, high), step in zip(bounds, step_sizes)
        ]
        total_combinations = np.prod(num_values)
        # Heuristic: population size is sqrt of total combinations, capped
        population_size = min(int(np.sqrt(total_combinations)), max_population)
        # Heuristic: generations is 2 * population size, capped
        generations = min(int(2 * np.sqrt(population_size)), max_generations)
        return population_size, generations