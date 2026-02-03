### Loss functions for physics informed machine learning 

import numpy as np
import torch


def complex_real_imag_loss(H_pred, H_exp):
    """
    Computes loss using real and imaginary parts separately.
    """

    real_loss = torch.nn.functional.mse_loss(H_pred.real, H_exp.real)
    imag_loss = torch.nn.functional.mse_loss(H_pred.imag, H_exp.imag)

    return real_loss + imag_loss



def complex_real_imag_loss_smoothed(H_exp, H_pred, n, k, smoothing_weight):
    """
    Computes loss using real and imaginary parts separately,
    with an added penalty for discontinuities in `n` and `k`.
    """
    real_loss = torch.nn.functional.mse_loss(H_pred.real, H_exp.real)
    imag_loss = torch.nn.functional.mse_loss(H_pred.imag, H_exp.imag)

    # Convert ParameterList to tensor
    n_tensor = torch.stack([param for param in n])
    k_tensor = torch.stack([param for param in k])

    smoothness_penalty = 0
    if len(n_tensor) > 1:  # Apply only if we have multiple frequency points
        smoothness_penalty = torch.sum((n_tensor[1:] - n_tensor[:-1])**2) + \
                             torch.sum((k_tensor[1:] - k_tensor[:-1])**2)

    return real_loss + imag_loss + smoothing_weight * smoothness_penalty



def log_complex_loss(H_exp, H_pred):
    """Computes loss in the log domain to avoid phase wrapping issues."""
    log_H_exp = torch.log(H_exp)  # Compute log of experimental transfer function
    log_H_pred = torch.log(H_pred)  # Compute log of predicted transfer function

    # Compute MSE loss in the log space
    return torch.nn.functional.mse_loss(log_H_pred.real, log_H_exp.real) + torch.nn.functional.mse_loss(log_H_pred.imag, log_H_exp.imag)



# Define a loss function that punishes discontinuities 
def log_complex_loss_smooth(H_exp, H_pred, n, k, lambda_smooth=1e-3):
    """Computes log-space loss with a smoothing constraint."""
    log_H_exp = torch.log(H_exp)
    log_H_pred = torch.log(H_pred)

    # Compute MSE loss in the log space
    loss_mse = torch.nn.functional.mse_loss(log_H_pred.real, log_H_exp.real) + \
               torch.nn.functional.mse_loss(log_H_pred.imag, log_H_exp.imag)

    # Smoothness penalty (finite differences)
    smoothness_penalty = torch.sum((n[:-2] - 2 * n[1:-1] + n[2:]) ** 2) + \
                         torch.sum((k[:-2] - 2 * k[1:-1] + k[2:]) ** 2)

    # Total loss
    loss = loss_mse + lambda_smooth * smoothness_penalty
    return loss