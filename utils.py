import numpy as np

def suggest_ga_params(bounds, step_sizes, max_population=1000, max_generations=100):
    """
    Population is set to the number of unique combinations possible given bounds and step sizes.
    Generations is set to 2 * sqrt(population).
    """
    num_values = [
        int((bounds[key][1] - bounds[key][0]) / step_sizes[key]) + 1
        for key in bounds
    ]
    population_size = min(int(np.prod(num_values) / 10), max_population)
    generations = min(int(2 * np.sqrt(population_size)), max_generations)
    return population_size, generations