import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from io import BytesIO
import tempfile

class HBERGeneticAlgorithm:
    def __init__(
        self,
        df_lookup,
        bounds={'PRE': (-100, 100), 'MAIN': (350, 800), 'POST': (-100, 100)},
        step_sizes={'PRE': 50, 'MAIN': 50, 'POST': 50},
        fitness_weights=(0.7, 0.3),  # (avg_HBER_weight, worst_HBER_weight)
        population_size=50,
        generations=30,
        cxpb=0.5,
        mutpb=0.2,
        seed=None,
        early_stopping=True
    ):
        self.df_lookup = df_lookup
        self.bounds = bounds
        self.step_sizes = step_sizes
        self.fitness_weights = fitness_weights
        self.population_size = population_size
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.early_stopping = early_stopping
        self.seed = seed
        self.mean_avg_HBER = df_lookup['avg_HBER'].mean()
        self.mean_worst_HBER = df_lookup['worst_HBER'].mean()
        self.population_snapshots = []
        self.fitness_history = []
        self.best_history = []
        self.converged_generation = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._setup_deap()

    def _fitness_function(self, individual):
        pre, main, post = individual
        row = self.df_lookup[
            (self.df_lookup['PRE'] == pre) &
            (self.df_lookup['MAIN'] == main) &
            (self.df_lookup['POST'] == post)
        ]
        if not row.empty:
            avg_hber = row['avg_HBER'].values[0]
            worst_hber = row['worst_HBER'].values[0]
        else:
            avg_hber = self.mean_avg_HBER
            worst_hber = self.mean_worst_HBER
        # Lower HBER is better, so fitness is negative weighted sum
        fitness = -(self.fitness_weights[0] * avg_hber + self.fitness_weights[1] * worst_hber)
        return (fitness,)

    def _make_grid_attr(self, key):
        low, high = self.bounds[key]
        step = self.step_sizes[key]
        def attr():
            return random.randrange(low, high + 1, step)
        return attr

    @staticmethod
    def grid_mutation(individual, step_sizes, bounds):
        for i, key in enumerate(['PRE', 'MAIN', 'POST']):
            if random.random() < 0.2:
                direction = random.choice([-1, 1])
                new_val = individual[i] + direction * step_sizes[key]
                # Clamp to bounds and snap to grid
                new_val = max(bounds[key][0], min(bounds[key][1], new_val))
                # Snap to grid
                new_val = round((new_val - bounds[key][0]) / step_sizes[key]) * step_sizes[key] + bounds[key][0]
                individual[i] = new_val
        return (individual,)

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMinHBER"):
            creator.create("FitnessMinHBER", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "IndividualHBER"):
            creator.create("IndividualHBER", list, fitness=creator.FitnessMinHBER)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_pre", self._make_grid_attr('PRE'))
        self.toolbox.register("attr_main", self._make_grid_attr('MAIN'))
        self.toolbox.register("attr_post", self._make_grid_attr('POST'))
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.IndividualHBER,
            (self.toolbox.attr_pre, self.toolbox.attr_main, self.toolbox.attr_post),
            n=1
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._fitness_function)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register(
            "mutate",
            self.grid_mutation,
            step_sizes=self.step_sizes,
            bounds=self.bounds
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self):
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        self.population_snapshots = []
        self.fitness_history = []
        self.best_history = []
        converged_generation = None
        for gen in range(self.generations):
            offspring = algorithms.varAnd(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb)
            fits = list(map(self.toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            pop = self.toolbox.select(offspring, k=len(pop))
            hof.update(pop)
            fitnesses = [ind.fitness.values[0] for ind in pop]
            self.fitness_history.append(np.mean(fitnesses))
            self.best_history.append(np.min(fitnesses))
            self.population_snapshots.append(np.array([ind[:] for ind in pop]))
            if self.early_stopping and np.std(fitnesses) < 1e-3:
                converged_generation = gen + 1
                break
        self.best_individual = hof[0]
        self.converged_generation = converged_generation if converged_generation is not None else self.generations
        return hof[0]

    def evaluate_point(self, pre, main, post):
        row = self.df_lookup[
            (self.df_lookup['PRE'] == pre) &
            (self.df_lookup['MAIN'] == main) &
            (self.df_lookup['POST'] == post)
        ]
        if not row.empty:
            avg_hber = row['avg_HBER'].values[0]
            worst_hber = row['worst_HBER'].values[0]
        else:
            avg_hber = self.mean_avg_HBER
            worst_hber = self.mean_worst_HBER
        return self.fitness_weights[0] * avg_hber + self.fitness_weights[1] * worst_hber

    def animate_population(self, save_path=None):
        """
        Animates the population evolution on the 3D scatter plot (PRE, MAIN, POST).
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # Compute fitness for all points in df_new
        w1, w2 = self.fitness_weights
        self.df_lookup['fitness'] = -(w1 * self.df_lookup['avg_HBER'] + w2 * self.df_lookup['worst_HBER'])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all possible points as background
        sc_bg = ax.scatter(
            self.df_lookup['PRE'], self.df_lookup['MAIN'], self.df_lookup['POST'],
            c=self.df_lookup['fitness'], cmap='viridis_r', alpha=0.5, s=20, label='All Points'
        )
        plt.colorbar(sc_bg, ax=ax, label='Fitness (lower is better)')

        scat = ax.scatter([], [], [], color='red', s=30, marker='x', label='Population')
        ax.set_xlabel('PRE')
        ax.set_ylabel('MAIN')
        ax.set_zlabel('POST')
        ax.set_title('Population Evolution')
        ax.legend()

        ax.set_xlim(self.bounds['PRE'][0], self.bounds['PRE'][1])
        ax.set_ylim(self.bounds['MAIN'][0], self.bounds['MAIN'][1])
        ax.set_zlim(self.bounds['POST'][0], self.bounds['POST'][1])

        def update(frame):
            data = np.array(self.population_snapshots[frame])
            if data.size > 0:
                scat._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
            ax.set_title(f'Generation {frame+1}')
            return scat,

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.population_snapshots), interval=300, blit=False
        )

        if save_path:
            ani.save(save_path, writer='pillow', fps=2)
        else:
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpfile:
                ani.save(tmpfile.name, writer=PillowWriter(fps=2))
                tmpfile.seek(0)
                gif_bytes = tmpfile.read()
            plt.close(fig)
            return BytesIO(gif_bytes)

    def animate_fitness_trend(self, save_path=None):
        """
        Animates the fitness score trend over generations.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, len(self.fitness_history))
        ax.set_ylim(min(self.best_history) * 1.1, max(self.fitness_history) * 1.1)
        line_avg, = ax.plot([], [], 'b-', label='Average Fitness')
        line_best, = ax.plot([], [], 'r-', label='Best Fitness')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Trend Over Generations (lower is better)')
        ax.legend()
        ax.grid(True)

        def update(frame):
            line_avg.set_data(range(frame+1), self.fitness_history[:frame+1])
            line_best.set_data(range(frame+1), self.best_history[:frame+1])
            return line_avg, line_best

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.fitness_history), interval=300, blit=True
        )

        if save_path:
            ani.save(save_path, writer='pillow', fps=2)
        else:
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpfile:
                ani.save(tmpfile.name, writer=PillowWriter(fps=2))
                tmpfile.seek(0)
                gif_bytes = tmpfile.read()
            plt.close(fig)
            return BytesIO(gif_bytes)