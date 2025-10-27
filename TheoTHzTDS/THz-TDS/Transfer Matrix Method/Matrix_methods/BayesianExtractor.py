import torch
import numpy as np
from skopt import gp_minimize
from .Simulate import simulate_parallel


# Define loss function for model
def gen_loss_function(y_simulated, y_exp, alpha:float=1):
    return alpha * torch.sqrt(torch.nn.functional.mse_loss(y_simulated, y_exp))



## Bayesian optimization based on gaussian process regression is implemented with gp_minimize
class BayesianLayeredExtractor():
    def __init__(self, reference_pulse, experimental_pulse, deltat, layers_init, optimize_mask=None, optimization_bounds=[0.1, 0.01, 0.15e-3]):
        super().__init__()
        self.reference_pulse = reference_pulse.clone().detach()
        self.experimental_pulse = experimental_pulse.clone().detach()
        self.deltat = deltat

        self.n_init = [np.real(layer[0]) for layer in layers_init]
        self.k_init = [np.imag(layer[0]) for layer in layers_init]
        self.D_init = [layer[1] for layer in layers_init]
        
        self.num_layers = len(layers_init)

        if optimize_mask is None:
            # Default: optimize everything
            self.optimize_mask = [(True, True, True)] * self.num_layers
        else:
            self.optimize_mask = optimize_mask
        
        # Define optimization bounds
        self.optimization_bounds = optimization_bounds

    def loss_function(self, y_simulated, alpha):
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)

    def bayesian_optimization(self, n_calls=30, alpha=1):
        print("Starting Bayesian Optimization with masks...")

        # Generate index map for optimization vector
        optimization_indices = []
        for i, (opt_n, opt_k, opt_D) in enumerate(self.optimize_mask):
            if opt_n: optimization_indices.append(('n', i))
            if opt_k: optimization_indices.append(('k', i))
            if opt_D: optimization_indices.append(('D', i))

        def objective(nkd_values):
            # Reconstruct full layer parameters from nkd_values and fixed init values
            full_n = self.n_init.copy()
            full_k = self.k_init.copy()
            full_D = self.D_init.copy()

            idx = 0
            for kind, i in optimization_indices:
                if kind == 'n':
                    full_n[i] = nkd_values[idx]
                elif kind == 'k':
                    full_k[i] = nkd_values[idx]
                elif kind == 'D':
                    full_D[i] = nkd_values[idx]
                idx += 1

            layers = [(n + 1j * k, D) for n, k, D in zip(full_n, full_k, full_D)]
            y_simulated = simulate_parallel(self.reference_pulse, layers, self.deltat, noise_level=0)[1][:len(self.experimental_pulse)]
            return gen_loss_function(y_simulated, self.experimental_pulse, alpha).item()

        # Define bounds for only the parameters being optimized
        bounds = []
        for kind, i in optimization_indices:
            if kind == 'n':
                bounds.append((self.n_init[i] - self.optimization_bounds[0], self.n_init[i] + self.optimization_bounds[0]))
            elif kind == 'k':
                bounds.append((self.k_init[i] - self.optimization_bounds[1], self.k_init[i] + self.optimization_bounds[1]))
            elif kind == 'D':
                bounds.append((self.D_init[i] - self.optimization_bounds[2], self.D_init[i] + self.optimization_bounds[2]))

        print("Search Boundaries for Optimized Parameters:")
        for (kind, i), b in zip(optimization_indices, bounds):
            print(f"Layer {i+1} - {kind} ∈ {b}")

        result = gp_minimize(objective, bounds, n_calls=n_calls, random_state=42)
        best_values = result.x

        # Reconstruct final layers
        full_n = self.n_init.copy()
        full_k = self.k_init.copy()
        full_D = self.D_init.copy()

        idx = 0
        for kind, i in optimization_indices:
            if kind == 'n':
                full_n[i] = best_values[idx]
            elif kind == 'k':
                full_k[i] = best_values[idx]
            elif kind == 'D':
                full_D[i] = best_values[idx]
            idx += 1

        return [(n + 1j * k, D) for n, k, D in zip(full_n, full_k, full_D)]



