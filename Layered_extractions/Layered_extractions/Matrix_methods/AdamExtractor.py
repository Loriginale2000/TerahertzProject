import torch
import torch.nn as nn
import torch.optim as optim
from .Simulate import simulate_parallel
from time import perf_counter
import numpy as np

# TODO: Change return method at the end to replicate the one in bayes model.

# Define loss function for model
def gen_loss_function(y_simulated, y_exp, alpha: float):
    """
    Computes the loss function.
    """
    return alpha * torch.sqrt(torch.nn.functional.mse_loss(y_simulated, y_exp))



## General optimizer to find n_j, k_j, and D_j (thickness) in the time domain
class LayeredExtractor(nn.Module):
    def __init__(self, reference_pulse, experimental_pulse, deltat, layers_init, optimize_mask=None, lr=1e-3):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach()
        self.experimental_pulse = experimental_pulse.clone().detach()
        self.deltat = deltat
        
        self.n_params = nn.ParameterList()
        self.k_params = nn.ParameterList()
        self.log_D_params = nn.ParameterList()
        
        self.fixed_n = []
        self.fixed_k = []
        self.fixed_log_D = []
        
        self.optimize_mask = optimize_mask or [(True, True, True)] * len(layers_init)
        
        for (layer, mask) in zip(layers_init, self.optimize_mask):
            n_val = np.real(layer[0])
            k_val = np.imag(layer[0])
            D_val = layer[1]
            log_D_val = np.log(D_val)

            opt_n, opt_k, opt_D = mask
            
            # n
            if opt_n:
                self.n_params.append(nn.Parameter(torch.tensor(n_val, dtype=torch.float32)))
                self.fixed_n.append(None)
            else:
                self.fixed_n.append(torch.tensor(n_val, dtype=torch.float32))

            # k
            if opt_k:
                self.k_params.append(nn.Parameter(torch.tensor(k_val, dtype=torch.float32)))
                self.fixed_k.append(None)
            else:
                self.fixed_k.append(torch.tensor(k_val, dtype=torch.float32))

            # D (log space)
            if opt_D:
                self.log_D_params.append(nn.Parameter(torch.tensor(log_D_val, dtype=torch.float32)))
                self.fixed_log_D.append(None)
            else:
                self.fixed_log_D.append(torch.tensor(log_D_val, dtype=torch.float32))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss_history = []
        self.best_loss = float('inf')
        self.best_n_params = [n.clone().detach() if n is not None else None for n in self.n_params]
        self.best_k_params = [k.clone().detach() if k is not None else None for k in self.k_params]
        self.best_log_D_params = [log_D.clone().detach() if log_D is not None else None for log_D in self.log_D_params]

    def forward(self):
        """
        Simulates time-domain pulse from current layer params.
        """
        layers = []
        n_idx = k_idx = D_idx = 0

        for i, mask in enumerate(self.optimize_mask):
            opt_n, opt_k, opt_D = mask

            n = self.n_params[n_idx] if opt_n else self.fixed_n[i]
            k = self.k_params[k_idx] if opt_k else self.fixed_k[i]
            log_D = self.log_D_params[D_idx] if opt_D else self.fixed_log_D[i]

            if opt_n: n_idx += 1
            if opt_k: k_idx += 1
            if opt_D: D_idx += 1

            layers.append((n + 1j * k, log_D.exp()))

        y_simulated = simulate_parallel(self.reference_pulse, layers, self.deltat, noise_level=0)[1]
        return y_simulated[:len(self.reference_pulse)]

    def loss_function(self, y_simulated, alpha):
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)

    def optimize(self, num_iterations=100, verbose=True, updates=10, alpha=1):
        print(f'Fine-tuning {sum(m.count(True) for m in self.optimize_mask)} parameters for {num_iterations} iterations.')

        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            start = perf_counter() # perf_counter for timing sections of optimization
            y_simulated = self.forward()
            self.forward_time = perf_counter() - start

            loss = self.loss_function(y_simulated, alpha=alpha)
            start = perf_counter()
            loss.backward()
            self.backwards_time = perf_counter() - start

            start = perf_counter()
            self.optimizer.step()
            self.optimizer_time = perf_counter() - start

            self.loss_history.append(loss.item())

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_n_params = [n.clone().detach() if n is not None else None for n in self.n_params]
                self.best_k_params = [k.clone().detach() if k is not None else None for k in self.k_params]
                self.best_log_D_params = [log_D.clone().detach() if log_D is not None else None for log_D in self.log_D_params]

            if verbose and (iteration + 1) % updates == 0:
                n_idx = k_idx = D_idx = 0
                layer_info = []
                for i, (opt_n, opt_k, opt_D) in enumerate(self.optimize_mask):
                    n = self.n_params[n_idx] if opt_n else self.fixed_n[i]
                    k = self.k_params[k_idx] if opt_k else self.fixed_k[i]
                    D = self.log_D_params[D_idx].exp() if opt_D else self.fixed_log_D[i].exp()
                    if opt_n: n_idx += 1
                    if opt_k: k_idx += 1
                    if opt_D: D_idx += 1
                    layer_info.append(f"Layer {i}: n={n.item():.4f}, k={k.item():.5f}, D={D.item()*1e6:.2f} µm")
                print(f"Iteration {iteration}, Loss: {loss.item():.6e}, " + " | ".join(layer_info))

        # Reconstruct best final layers
        results = []
        n_idx = k_idx = D_idx = 0
        for i, (opt_n, opt_k, opt_D) in enumerate(self.optimize_mask):
            n = self.best_n_params[n_idx] if opt_n else self.fixed_n[i]
            k = self.best_k_params[k_idx] if opt_k else self.fixed_k[i]
            D = self.best_log_D_params[D_idx].exp() if opt_D else self.fixed_log_D[i].exp()
            if opt_n: n_idx += 1
            if opt_k: k_idx += 1
            if opt_D: D_idx += 1
            complex_n = n + 1j * k
            results.append((complex_n.item(), D.item()))
        return results
            
