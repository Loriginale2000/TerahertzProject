import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .back_prop_utils import c

## TODO: 
# - Add Scheduler  

class FrequencyDependentModel(torch.nn.Module):
    def __init__(self, w_tensor, d, ICs_n, ICs_k):
        super().__init__()
        self.w_tensor = w_tensor  # Frequency tensor
        self.d = d  # Thickness of the material
        
        # Initialize n and k as trainable parameters (one per frequency point)
        self.n = torch.nn.Parameter(torch.tensor(ICs_n, dtype=torch.float32).repeat(len(w_tensor)))
        self.k = torch.nn.Parameter(torch.tensor(ICs_k, dtype=torch.float32).repeat(len(w_tensor)))
        
        # Store loss history for plotting
        self.loss_history = []
        
        # Track best parameters
        self.best_n = self.n.clone().detach()
        self.best_k = self.k.clone().detach()
        self.best_loss = float('inf')
    
    def forward(self, Physical_model):
        """
        Compute the theoretical transfer function given the current n, k values.
        """
        n_complex = self.n + 1j * self.k
        H_pred = Physical_model(n_complex, self.w_tensor, self.d)
        return H_pred
    
    def train_model(self, H_exp, Physical_model, loss_fn, 
                    optimizer=None, epochs=10000, lr=1e-4, verbose=True, updates=500, **loss_kwargs):
    
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute theoretical transfer function
            H_pred = self.forward(Physical_model)
            
            # Check if the loss function expects n and k for loss functions with smoothing terms
            loss_kwargs_dynamic = loss_kwargs.copy()  # Avoid modifying original kwargs
            if "n" in loss_fn.__code__.co_varnames and "k" in loss_fn.__code__.co_varnames:
                loss_kwargs_dynamic["n"] = self.n
                loss_kwargs_dynamic["k"] = self.k
            
            # Compute loss with additional parameters
            loss = loss_fn(H_exp, H_pred, **loss_kwargs_dynamic)
            self.loss_history.append(loss.item())
            
            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_n = self.n.clone().detach()
                self.best_k = self.k.clone().detach()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % updates == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()} (Best Loss = {self.best_loss})")
        
        return self.loss_history
