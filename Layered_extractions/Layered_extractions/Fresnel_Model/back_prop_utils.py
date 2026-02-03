### Utility files for back-propagation code

## Loss functions

import torch
import numpy as np

c = 299792458   # Speed of light in m/s

## Transfer function
def H_th_function(n_complex, w, length):
    '''
    Inputs
    ------
    n: complex refractive index
    w: frequency of light being propagated
    length: length of the sample we are modelling
    
    outputs
    -------
    returns: output for the transfer function

    Method
    ------
    Equation for transfer function derived from [Input source here]

    '''

    
    return (4 * n_complex) / ((n_complex + 1) ** 2) * torch.exp((-1j * (n_complex - 1) * w * length) / c)



# Define analytical derivative of H_theoretical
def H_prime_function(n, w, Length):
    return (1 / n) - (2 / (n + 1)) - 1j * w * Length / c



## Grid search function

def grid_search(n0, k0, d0, H_values, phi_values, freqs_ang, H_th_function, loss, verbose=False):
    """
    Performs a grid search to optimize n, k, and d parameters by minimizing the loss function.

    Parameters:
    - n0 (float): Initial guess for n.
    - k0 (float): Initial guess for k.
    - d0 (float): Initial guess for d.
    - H_values (list/array): Measured amplitude values.
    - phi_values (list/array): Measured phase values.
    - freqs_ang (list/array): Frequency values (angular).
    - H_th_function (function): Function to compute theoretical transfer function.
    - loss (function): Loss function to compare predicted and actual values.

    Returns:
    - best_params (dict): Dictionary containing optimal values for n, k, and d.
    - min_loss (float): Minimum loss achieved.
    """

    min_loss = np.inf
    best_params = {'n': 0, 'k': 0, 'd': 0}

    for ii in range(3):
        n_pred = n0 + ii * 0.01
        for ij in range(3):
            k_pred = k0 + ij * 0.005
            for ik in range(3):
                d_pred = d0 + ik * 0.0001

                # Compute the theoretical transfer function
                tf_values_pred = [H_th_function((n_pred + k_pred * 1j), f, d_pred) for f in freqs_ang]
                H_values_pred = np.abs(tf_values_pred)
                phi_values_pred = np.unwrap(np.angle(tf_values_pred))

                # Compute loss
                l = loss(H_values, H_values_pred, phi_values, phi_values_pred)

                if l < min_loss:
                    min_loss = l
                    best_params = {'n': n_pred, 'k': k_pred, 'd': d_pred}
                
                if verbose:
                    print(f"{n_pred=:.2f}, {k_pred=:.3f}, {d_pred=:.4f}, Loss: {l:.6f}")

    return best_params, min_loss



# Define NR function to test and compare methods
import numpy as np

def newton_raphson_fitting(H_th_function, H_prime_function, f_interp, A_exp, ph_extrapolated, Length, n_0=3.7 + 0.1j, max_iterations=10):
    """
    Perform Newton-Raphson fitting to extract refractive indices.
    
    Parameters:
    - H_th_function: Function computing the theoretical transfer function.
    - H_prime_function: Function computing the derivative of the transfer function.
    - f_interp: Interpolated frequency array (in THz).
    - A_exp: Experimental amplitude array.
    - ph_extrapolated: Experimental phase array.
    - Length: Sample thickness.
    - c: Speed of light.
    - n_0: Initial guess for the refractive index.
    - max_iterations: Maximum number of Newton-Raphson iterations.
    
    Returns:
    - n_extracted: Extracted complex refractive indices.
    """
    frequencies = len(f_interp)
    n_extracted = np.zeros(frequencies, dtype=complex)
    
    for f_index in range(frequencies):
        n_next = n_0  # Reset n_next for each frequency index
        for _ in range(max_iterations):
            w = 2 * np.pi * f_interp * 1e12  # Convert to angular frequency in radians/sec
            H_th = H_th_function(n_next, w, length=Length)
            A_th = np.abs(H_th)
            ph_th = np.unwrap(np.angle(H_th))
            
            # Function to optimize: Compute TF values in the log space
            fun = np.log(A_th[f_index]) - np.log(A_exp[f_index]) + 1j * ph_th[f_index] - 1j * ph_extrapolated[f_index]
            fun_prime = H_prime_function(n_next, 2 * np.pi * f_interp[f_index] * 1e12, Length=Length)
            
            # Newton-Raphson update step
            n_next = n_next - fun / fun_prime
        
        n_extracted[f_index] = n_next
    
    return n_extracted



# Function to round to significant figures
def round_to_sig_figs(value, sig_figs):
    if value == 0:
        return 0
    return round(value, sig_figs - int(f"{value:.1e}".split('e')[1]) - 1)


# Function to unwrap tensor angles and return as a tensor
def torch_unwrap(phase:torch.tensor) -> torch.tensor:
    phi = np.unwrap(phase)
    phi = torch.tensor(phi, dtype=torch.float32)
    return phi
