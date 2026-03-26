import torch
import numpy as np
import matplotlib.pyplot as plt
from .Simulate import simulate_parallel
from .AdamExtractor import gen_loss_function


''' DISCLAIMER: ALL THE COMMENTS ARE MADE BY ME, TRYING TO UNDERSTAND BETTER AND FURTHER BE ABLE TO EXPLAIN, 
    HOW THIS CODE WORKS AND WHY IT IS USEFUL FOR US, WHILE WE ARE CREATING LOSS PROFILES'''

def generate_landscape_grid(reference_pulse, experimental_pulse, deltat, layers_init, param_ranges, num_points):
    
    '''Creates a structured grid, searching for only two parameters each time we run it.
    param_ranges: A dictionary we use, containing EXACTLY TWO parameters we want to sweep.'''
                  
    # DESIDE WHICH PARAMETERS TO SWEEP THROUGH
    sweep_keys = list(param_ranges.keys())
    if len(sweep_keys) != 2:
        raise ValueError("You need to provide exactly two parameters for the Grid Search.")
    
    key1, key2 = sweep_keys
    axis1 = np.linspace(*param_ranges[key1], num_points) 
    axis2 = np.linspace(*param_ranges[key2], num_points)

    ''' The asterisk * before each param_ranges[keyi] serves as an unpacking operator. Basically it
    helps Python unpack the parameter ranges, so using linspace() it sees the start and finish points 
    properly'''
    
    results_data = []

    # CREATE A NESTED LOOP TO SHAPE THE GRID
    for v1 in axis1:
        for v2 in axis2:
            ''' axis1 represents the first set of parameter values and axis2 the second. We want to create 
            a matrix with all possible combinations of the parameter values we want to sweep.
            For example,
            if we wanted to sweep over 30 n and 30 d values, we would create a grid of 900 combinations. So,
            the code would run the loop containing the first value of n and all the possible given values of d
            {that would give us the first row of the formed matrx}. After that we move to the second row and 
            so on and so forth... '''


            current_stack = []
            sample_record = {f'n{i}': l[0].real for i, l in enumerate(layers_init)}
 
            for i, (n_k_base, d_base) in enumerate(layers_init):
                sample_record[f'n{i}'] = n_k_base.real
                sample_record[f'k{i}'] = n_k_base.imag
                sample_record[f'd{i}'] = d_base

            ''' enumerate() function returns both index and object after each loop
            We initialize by creating a default sample, with it's layers coming straight from 
            the initial guesses that we input during optimization. This helps the process become
            slightly faster, becuase we don't need to input new base values.'''

            
            sample_record[f"{key1[1]}{key1[0]}"] = v1
            sample_record[f"{key2[1]}{key2[0]}"] = v2

            '''This line overwrites the default value in sample_record with the specific values
              (v1, v2) currently selected by the loops.
              If key[1]='n' and key[0]=0 the string shall become n0'''

            
            for i in range(len(layers_init)):
                n_stack = sample_record[f'n{i}'] + 1j * sample_record[f'k{i}']
                d_stack = sample_record[f'd{i}']
                current_stack.append((n_stack, d_stack))

            '''This part is pretty easy to understand. simulate_parallel needs to have tuples instead
            of dictionaries to function. That's why in this loop, we create n_stack [complex refrctive index]
            and d_stack [thickness] using the values from the sample_record dict and append them as tuples
            in current_stack list.'''

            
            _, y_sim = simulate_parallel(reference_pulse, current_stack, deltat, noise_level=0)
            y_sim = y_sim[:len(experimental_pulse)]
            loss = gen_loss_function(y_sim, experimental_pulse, alpha=1).item()
            
            sample_record['loss'] = loss
            results_data.append(sample_record)

            '''This final part simulates loss for every possible combination, using the current_stack values.
               The results_data then is used to store the final results, inlcuding loss values'''

    return results_data, axis1, axis2, sweep_keys

def plot_grid_landscape(data, axis1, axis2, keys):
    """
    This function plots a smooth contour with dynamic labels for every combination we 
    shall be plotting. Key parameters:
    axis1 are the values for the y-axis (Outer loop)
    axis2 are the values for the x-axis (Inner loop)
    keys is a list of tuples [(layer_idx, 'type'), (layer_idx, 'type')]
    """
    
    label_map = {
        'n': 'Refractive Index (n)',
        'k': 'Extinction Coefficient (k)',
        'd': 'Thickness (d) [µm]'
    }
    
    layer_names = {0: "Quartz", 1: "Si"} # Change this depending on our sample [layers, materials used etc.] 
    '''We use label_map and layer_names as dictionaries to be achieve more
       efficient plotting. '''
    
    
    y_idx, y_type = keys[0]
    x_idx, x_type = keys[1]

    '''We identify which parameter is on which axis.
       Instead of hardcoding the names and patameters, we extract these information from keys
       For example, if keys[0]=(0,'n'), then y_idx --> 0 and y_type --> 'n'.
    '''

    ' Reshape the loss data for the contour plot (grid) '
    num_points = len(axis1)
    # Ensure reshaping matches the nested loop order (axis1 is outer/Rows, axis2 is inner/Cols)
    losses = np.array([d['loss'] for d in data]).reshape(num_points, num_points)

    
    plot_x = axis2 * 1e6 if x_type == 'd' else axis2
    plot_y = axis1 * 1e6 if y_type == 'd' else axis1
    '''We convert data to μm if parameter d is selected in either y or x axis.
       If not, then we keep the order of magnitude as is.'''

    X, Y = np.meshgrid(plot_x, plot_y)

    'We create a figure to initialize plotting'
    plt.figure(figsize=(12, 6.5))
    cp = plt.contourf(X, Y, losses, levels=100, cmap='magma')
    plt.colorbar(cp, label='Loss (MSE)')


    x_label = f"{layer_names.get(x_idx, f'Layer {x_idx}')} {label_map[x_type]}"
    y_label = f"{layer_names.get(y_idx, f'Layer {y_idx}')} {label_map[y_type]}"
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Loss Landscape: {y_label} vs {x_label}")

    # WE MARK THE GLOBAL MINIMUM ON THE GRID 
    min_idx = np.argmin(losses)
    min_y, min_x = np.unravel_index(min_idx, losses.shape)
    plt.plot(plot_x[min_x], plot_y[min_y], 'rx', markersize=14, markeredgewidth=2, label='Global Minimum')
    plt.legend()


    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# p = { (0, 'n'): (1.8, 2.2), (0, 'd'): (900e-6, 1100e-6) }
# data, a1, a2, keys = generate_landscape_grid(ref, exp, dt, layers_init [could be initial guess]
# ,parameter range, num_points[we can set it ourselves])
# plot_grid_landscape(data, a1, a2, keys)