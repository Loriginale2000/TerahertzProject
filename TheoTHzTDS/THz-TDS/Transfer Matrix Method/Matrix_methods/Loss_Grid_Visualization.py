import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from .Simulate import simulate_parallel
from .AdamExtractor import gen_loss_function


# This function performs a sweep of three material parameters (n,k and d) across the layers to explore
# how the loss changes
# when these specific parameters are varied. It is designed to build a
# loss landscape of the optimization space to demonstrate the behavior of the objective function - 
# aka the root mean square error.
def generate_landscape_multi_layer(reference_pulse, experimental_pulse, deltat, layers_init, param_ranges, num_samples):
    """
    Sweep a set of random layer parameters and evaluate the resulting simulated pulse against the
    experimental signal.

    Parameters
    ----------
    reference_pulse : array-like
        Reference pulse, which we use to generate the simulated pulse of the layer stack.
    experimental_pulse : array-like
        Measured pulse that acts as the target for the loss calculation.
    deltat : float
        Time step used in the simulation. It is important that we use the correct units (seconds)
    layers_init : list of tuples
        Initial layer stack of the form [(n + k*1j, thickness)].
    param_ranges : dict
        Mapping of (layer_index, 'param_type') -> (min_value, max_value).
        Supported param types are 'n', 'k', and 'd'. We can modify it for other parameters.
    num_samples : int
        Number of random layer stacks to generate and evaluate. The bigger the number the more it will take 
        to run.

    Returns
    -------
    list of dict
        Each dictionary contains the sampled parameter values for one candidate stack and the associated loss.
    """

    # STEP 1: Initialize the output container that will hold every sampled solution and its loss.
    # This list is returned at the end so the caller can inspect the full parameter sweep results.
    results_data = []

    # STEP 2: Generate multiple random layer configurations across the specified parameter ranges.
    # Each iteration creates one complete candidate stack, which is then simulated and scored.
    for _ in range(num_samples):
        # A single candidate stack contains all sampled layer parameters for this trial.
        current_stack = []

        # Store the sampled values in a dictionary so they can be examined later or plotted.
        sample_record = {}

        # STEP 3: Loop through every layer in the current stack and sample its parameters.
        # For each layer, we choose a value for refractive index n, extinction coefficient k,
        # and thickness d. If a parameter was not explicitly provided in param_ranges, we keep
        # the original starting value for that layer.
        for i, (n_k_base, d_base) in enumerate(layers_init):

            # A random n value is drawn uniformly between the min and max limits for this layer.
            # If no range is provided for n, the initial real part is used as a fixed value.
            n_val = np.random.uniform(*param_ranges.get((i, 'n'), (n_k_base.real, n_k_base.real)))

            # The extinction coefficient k is the imaginary part of the complex refractive index.
            # We sample k in the same way, falling back to the initial value when not specified.
            k_val = np.random.uniform(*param_ranges.get((i, 'k'), (n_k_base.imag, n_k_base.imag)))

            # The layer thickness d is sampled in the same manner. If thickness is not included in
            # the range map, the original thickness from layers_init is retained.
            d_val = np.random.uniform(*param_ranges.get((i, 'd'), (d_base, d_base)))

            # Build the new complex refractive index for this layer as n + i*k and keep the sampled thickness.
            current_stack.append((n_val + 1j * k_val, d_val))

            # Save each sampled parameter into the record so the result can later be used for plotting
            # and for understanding which parameter combination produced the observed loss.
            sample_record[f'n{i}'] = n_val
            sample_record[f'k{i}'] = k_val
            sample_record[f'd{i}'] = d_val

        # STEP 4: Simulate the time-domain response of the newly sampled layer stack.
        # The transfer matrix code expects the complete list of layers and the time step.
        _, y_sim = simulate_parallel(reference_pulse, current_stack, deltat, noise_level=0)

        # Align the simulated pulse length to the experimental data so the loss can be computed properly.
        y_sim = y_sim[:len(experimental_pulse)]

        # STEP 5: Compute the objective function value for this candidate stack.
        # The loss function compares the simulated pulse to the experimental pulse and returns a scalar 
        # indicating how closely the model matches the data.
        loss = gen_loss_function(y_sim, experimental_pulse, alpha=1).item()

        # Attach the final scalar loss to the sampled parameter record and save the full result.
        sample_record['loss'] = loss
        results_data.append(sample_record)

    # Return the full set of parameter/loss combinations so the caller can visualize or inspect the landscape.
    return results_data


# This plotting function extracts the sampled values for a chosen layer and creates a visual map of the
# loss as a function of the explored parameters. The goal is to identify regions of the parameter space
# where the model matches the experimental data most closely.
def plot_results_multi_layer(data, layer_idx):
    """Plots landscapes for a specific layer index using both scatter and contour."""

    # STEP 1: Pull the sampled values for the selected layer from each result entry.
    # We do this for the refractive index n, extinction coefficient k, thickness d, and the final loss.
    n = np.array([d[f'n{layer_idx}'] for d in data])
    k = np.array([d[f'k{layer_idx}'] for d in data])
    thickness = np.array([d[f'd{layer_idx}'] * 1e6 for d in data])  # convert from meters to micrometers for plotting
    loss = np.array([d['loss'] for d in data])

    # Give a simple label for the material being visualized.
    label = "Si" if layer_idx == 0 else "Si"

    # STEP 2: Create a side-by-side figure to show two different projections of the same loss landscape.
    # The left panel shows the relationship between k and n, while the right panel shows how thickness
    # interacts with n. This helps identify parameter combinations that yield favorable loss values.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: loss as a function of k and n.
    # Here, k is on the x-axis and n is on the y-axis. The contour map reveals which combinations of
    # extinction and real refractive index produce low loss values.
    cnt1 = ax1.tricontourf(k, n, loss, levels=40, cmap='magma')
    ax1.scatter(k, n, c='white', s=5, alpha=0.2)
    ax1.set_title(f'Loss Landscape: {label} (n vs k)')
    ax1.set_xlabel('k')
    ax1.set_ylabel('n')
    plt.colorbar(cnt1, ax=ax1, label='Loss')

    # Right panel: loss as a function of thickness and n.
    # The thickness axis is converted to micrometers, which is a more natural unit for optical layer thicknesses.
    cnt2 = ax2.tricontourf(thickness, n, loss, levels=40, cmap='magma')
    ax2.set_title(f'Loss Landscape: {label} (n vs d)')
    ax2.set_xlabel('d (µm)')
    ax2.set_ylabel('n')
    plt.colorbar(cnt2, ax=ax2, label='Loss')

    # STEP 3: Finalize the layout so the plots render clearly and avoid overlap of labels or colorbars.
    plt.tight_layout()
    plt.show()

