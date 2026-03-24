import torch
import numpy as np
import matplotlib.pyplot as plt
from .Simulate import simulate_parallel
from .AdamExtractor import gen_loss_function

def generate_landscape_multi_layer(reference_pulse, experimental_pulse, deltat, layers_init, param_ranges, num_samples):
    """
    Here, I provide info about the context of the function.
    Sweeps parameters and calculates loss, can be used for one or multiple layers.
    layers_init: List of [(complex_n, d)]. We can also use init_guess from the extraction itself as this parameter.
    param_ranges: Dictionary, mapping (layer_index, 'param_type') -> (min, max)
    num_samples: Number of random samples that are generated. 

    The function returns a list of dictionaries, with each dictionary containing the parameters and the corresponding loss.
    """
    results_data = [] # LIST OF FINAL RESULTS TO BE APPENDED TO AND RETURNED
    
    # CREATE A MULTI-DIMENSIONAL MATRIX OF SAMPLES, DEPENDING ON THE NUMBER OF PARAMETERS WE WANT TO SWEEP.
    # ALSO GENERATE RANDOM VALUES FOR EACH PARAMETER WITHIN GIVEN RANGES.
    # We generate 'num_samples' versions of the ENTIRE layer stack
    for _ in range(num_samples):
        current_stack = []
        sample_record = {}
        
        for i, (n_k_base, d_base) in enumerate(layers_init):

            # SAMPLING PARAMETERS FOR EACH LAYER 
            # NOTE: IF A PARAMETER IS NOT SPECIFIED IN param_ranges, IT WILL DEFAULT TO THE INITIAL VALUE 

            n_val = np.random.uniform(*param_ranges.get((i, 'n'), (n_k_base.real, n_k_base.real)))
            k_val = np.random.uniform(*param_ranges.get((i, 'k'), (n_k_base.imag, n_k_base.imag)))
            d_val = np.random.uniform(*param_ranges.get((i, 'd'), (d_base, d_base)))
            
            current_stack.append((n_val + 1j * k_val, d_val))
            
            # WE STORE VALUES FOR ANALYSIS AND PLOTTING
            sample_record[f'n{i}'] = n_val
            sample_record[f'k{i}'] = k_val
            sample_record[f'd{i}'] = d_val

        # SIMULATE THE PULSE FOR THE CURRENT STACK OF LAYERS AND CALCULATE LOSS
        # Note: simulate_parallel expects the full list of layers
        _, y_sim = simulate_parallel(reference_pulse, current_stack, deltat, noise_level=0)
        y_sim = y_sim[:len(experimental_pulse)]
        
        # 3. Calculate and store loss
        loss = gen_loss_function(y_sim, experimental_pulse, alpha=1).item()
        sample_record['loss'] = loss
        results_data.append(sample_record)

    return results_data

def plot_results_multi_layer(data, layer_idx):
    """Plots landscapes for a specific layer index."""
    n = [d[f'n{layer_idx}'] for d in data]
    k = [d[f'k{layer_idx}'] for d in data]
    thickness = [d[f'd{layer_idx}'] * 1e6 for d in data] # to µm
    loss = [d['loss'] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    label = "Quartz" if layer_idx == 0 else "Si"
    
    # Plot n vs k
    img1 = ax1.scatter(n, k, c=loss, cmap='viridis', s=80, edgecolors='black', linewidths=0.2)
    ax1.set_title(f'Loss Landscape: {label} (n vs k)')
    ax1.set_xlabel('n')
    ax1.set_ylabel('k')
    plt.colorbar(img1, ax=ax1, label='Loss')

    # Plot n vs d
    img2 = ax2.scatter(n, thickness, c=loss, cmap='plasma', s=80, edgecolors='black', linewidths=0.2)
    ax2.set_title(f'Loss Landscape: {label} (n vs d)')
    ax2.set_xlabel('n')
    ax2.set_ylabel('d (µm)')
    plt.colorbar(img2, ax=ax2, label='Loss')

    plt.tight_layout()
    plt.show()

