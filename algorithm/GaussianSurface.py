import numpy as np

class GaussianSurface:
    def __init__(self, num_peaks, xy_bounds={'x': (-20, 20), 'y': (-20, 20)}, amplitude_bounds=(1.0, 3.0), sigma=6.0, seed=None):
        self.num_peaks = num_peaks
        self.xy_bounds = xy_bounds  # expects dict: {'x': (xmin, xmax), 'y': (ymin, ymax)}
        self.amplitude_bounds = amplitude_bounds
        self.sigma = sigma
        self.seed = seed
        self.peaks, self.amplitudes = self._generate_peaks()

    def _generate_peaks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        x_peaks = np.random.uniform(self.xy_bounds['x'][0], self.xy_bounds['x'][1], self.num_peaks)
        y_peaks = np.random.uniform(self.xy_bounds['y'][0], self.xy_bounds['y'][1], self.num_peaks)
        amplitudes = np.random.uniform(self.amplitude_bounds[0], self.amplitude_bounds[1], self.num_peaks)
        return np.column_stack((x_peaks, y_peaks)), amplitudes

    def evaluate(self, X, Y):
        Z = np.zeros_like(X)
        for (px, py), amp in zip(self.peaks, self.amplitudes):
            Z += amp * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * self.sigma ** 2))
        return Z

    def evaluate_point(self, x, y):
        z = 0
        for (px, py), amp in zip(self.peaks, self.amplitudes):
            z += amp * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * self.sigma ** 2))
        return z