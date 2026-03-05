import numpy as np
import matplotlib.pyplot as plt
import torch
from Simulate import *
from BayesianExtractor import *
from AdamExtractor import *
from data_quartzsi import t_r_raw as t_r, A_r
from data_quartz import A_s

#Global Variables
L = 2**12
deltat = (t_r[1]-t_r[0])*1e-12  # Time step from data
x_exp = A_r
true_x_exp = A_s


#Inintial simulation
layers= [(2 - 1j*0.01, 1e-3)]

T_exp, y_exp = simulate_from_signal(x_exp, layers, deltat)

#Plotting
y_exp = y_exp[:L].detach().cpu().numpy()
plt.figure(figsize=(10,4))
plt.title('Time Domain of THz Pulse through single layered sample (from experimental signal).')
plt.plot(x_exp, label='Reference Pulse (Experimental)')
plt.plot(true_x_exp, label='Sample Pulse (Experimental)')
plt.plot(y_exp, label='Sample Pulse (Simulated)', color = 'black', linestyle = '--')
plt.xlim(0,2000)
plt.legend()
plt.grid()
#plt.savefig("Quartzsi plots /Quartz plot.png")
plt.show()

#Bayesin Extraction
x_exp = torch.from_numpy(A_r)
true_x_exp = torch.from_numpy(A_s)

extractor = BayesianLayeredExtractor(reference_pulse=x_exp, 
                                     experimental_pulse=true_x_exp, 
                                     deltat=deltat, 
                                     layers_init= layers, 
                                     optimize_mask=[(True, True, True)], 
                                     optimization_bounds=[0.2, 0.1, 50e-6])

best_layers = extractor.bayesian_optimization(n_calls=40)

#Finding induvidual values
n_complex_bays, D_bays = best_layers[0]
n_bays = n_complex_bays.real
k_bays = n_complex_bays.imag


#Bayesian results
T_bays, y_bays = simulate_from_signal(x_exp, best_layers, deltat)
y_bays = y_bays[:L].detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.plot(x_exp, label='Reference Pulse (Experimental)')
plt.plot(true_x_exp, label='Sample Pulse (Experimental)')
plt.plot(y_bays, label='Sample Pulse (Simulated Post Bayesian)', color = 'black', linestyle = '--')
plt.title('Time Domain of THz Pulse through single layered sample (Bayesian Optimization).')
plt.xlim(0,2000)
plt.legend()
plt.grid()
#plt.savefig("Quartzsi plots /Quartz Bayesian.png")
plt.show()


#Adam Extraction post Bayesian
#Converting data into floats to work with Adam Extractor
x_exp = torch.from_numpy(A_r).float()
true_x_exp = torch.from_numpy(A_s).float()

adam_extractor = LayeredExtractor(x_exp, true_x_exp, deltat, best_layers, 
                                  optimize_mask=[(True, True, True)])

adam_best_layers = adam_extractor.optimize(num_iterations=300, 
                                           verbose=True, updates=20, alpha=1)

n_complex_adam, D_adam = adam_best_layers[0]
n_adam = n_complex_adam.real
k_adam = n_complex_adam.imag


# Adam results
T_adam, y_adam = simulate_from_signal(x_exp, adam_best_layers, deltat)
y_adam = y_adam[:L].detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.plot(x_exp, label='Reference Pulse (Experimental)')
plt.plot(true_x_exp, label='Sample Pulse (Experimental)') 
plt.plot(y_adam, label='Sample Pulse (Simulated Post Adam)', color = 'black', linestyle = '--') 
plt.title('Time Domain of THz Pulse through single layered sample (Adam Optimization post Bayesian).')
plt.xlim(0,2000) 
plt.legend() 
plt.grid() 
#plt.savefig("Quartzsi plots /Quartz Adam.png") 
plt.show()


#Loss landscape 
from loss_landscape_visualization import LossLandscapeVisualizer

# Create loss function
loss_func = create_tmm_loss_function(x_exp, true_x_exp, deltat, L)

# Define ranges (adjust for your material!)
param_ranges = {
    'n': (1.5, 3.0),
    'k': (0.0, 0.3),
    'd': (0.5e-3, 2e-3)
}

# Create visualizer
viz = LossLandscapeVisualizer(loss_func, param_ranges)

# Final optimizer values
final_values = {
    'Bayesian': {'n': float(n_bays), 'k': float(k_bays), 'd': float(D_bays)},
    'ADAM': {'n': float(n_adam), 'k': float(k_adam), 'd': float(D_adam)}
}


# 1D slices (fast, very informative)
viz.plot_1d_slices(optimizer_values=final_values)
plt.show()

# 2D landscape n vs k (most important)
viz.plot_2d_slice('n', 'k', 
                  fixed_params={'d': (D_bays +   D_adam)/2},
                  resolution=40)
plt.show()

# Local minima test
viz.analyze_local_minima(final_values, n_samples=2000)
plt.show()

