import numpy as np
import matplotlib.pyplot as plt
import torch
from loss_landscape_visualization import LossLandscapeVisualizer
from Matrix_methods.AdamExtractor import gen_loss_function
from Matrix_methods.AdamExtractor import simulate_parallel


#================================================================================
# STEP 1: Create Multi-Layer Loss Function
#================================================================================

def create_multilayer_tmm_loss_function(x_exp, true_x_exp, deltat, L, num_layers=3):
    def loss_function(params):
        layers = []
        for i in range(1, num_layers + 1):
            n = params[f'n_{i}']
            k = params[f'k_{i}']
            d = params[f'd_{i}']
            layers.append((n - 1j*k, d))
        
        try:
        
            T_sim, y_sim = simulate_parallel(x_exp, layers, deltat)
            y_sim = y_sim[:L]
            loss = torch.mean((y_sim - true_x_exp[:L])**2).item()
        except Exception as e:
            print(f"Warning: Simulation failed: {e}")
            loss = 1e10
        
        return loss
    
    return loss_function


def create_single_layer_loss_function(x_exp, true_x_exp, deltat, L, 
                                      layer_index, fixed_layers):
    def loss_function(params):
        layers = fixed_layers.copy()
        n = params['n']
        k = params['k']
        d = params['d']
        layers[layer_index - 1] = (n - 1j*k, d)
        
        try:
            
            T_sim, y_sim = simulate_parallel(x_exp, layers, deltat)
            y_sim = y_sim[:L]
            loss = torch.mean((y_sim - true_x_exp[:L])**2).item()
        except Exception as e:
            print(f"Warning: Simulation failed for layer {layer_index}: {e}")
            loss = 1e10
        
        return loss
    
    return loss_function


#================================================================================
# STEP 2: Visualize Each Layer's Landscape
#================================================================================

def visualize_multilayer_landscape(x_exp, true_x_exp, deltat, L,
                                   bayesian_layers, adam_layers,
                                   param_ranges_per_layer,
                                   bayesian_history=None,
                                   adam_history=None,
                                   resolution=40):
    """
    Create loss landscapes for each layer of a multi-layer system
    
    Parameters:
    -----------
    bayesian_layers : list of tuples
        [(n1-ik1, d1), (n2-ik2, d2), (n3-ik3, d3)] from Bayesian optimization
    adam_layers : list of tuples
        [(n1-ik1, d1), (n2-ik2, d2), (n3-ik3, d3)] from ADAM optimization
    param_ranges_per_layer : list of dicts
        [{'n': (min, max), 'k': (min, max), 'd': (min, max)}, ...] for each layer
    bayesian_history : list of dicts, optional
        extractor.iteration_history from BayesianLayeredExtractor
        Each entry is {'n': [n_l1, n_l2, n_l3], 'k': [...], 'd': [...]}
    adam_history : list of dicts, optional
        adam_extractor.iteration_history from LayeredExtractor
        Each entry is {'n': [n_l1, n_l2, n_l3], 'k': [...], 'd': [...]}
    """
    
    num_layers = len(bayesian_layers)
    
    print("\n" + "="*70)
    print(f"MULTI-LAYER LOSS LANDSCAPE ANALYSIS ({num_layers} layers)")
    print("="*70)
    
    # Extract final values for each layer
    bayesian_params = []
    adam_params = []
    
    for i, (bays_layer, adam_layer) in enumerate(zip(bayesian_layers, adam_layers)):
        n_complex_bays, d_bays = bays_layer
        n_complex_adam, d_adam = adam_layer
        
        bayesian_params.append({
            'n': float(abs(n_complex_bays.real)),
            'k': float(abs(n_complex_bays.imag)),
            'd': float(abs(d_bays))
        })
        
        adam_params.append({
            'n': float(abs(n_complex_adam.real)),
            'k': float(abs(n_complex_adam.imag)),
            'd': float(abs(d_adam))
        })
    
    # Print summary
    print("\nOptimizer Results:")
    print("-" * 70)
    for i in range(num_layers):
        print(f"\nLayer {i+1}:")
        print(f"  Bayesian: n={bayesian_params[i]['n']:.4f}, "
              f"k={bayesian_params[i]['k']:.4f}, "
              f"d={bayesian_params[i]['d']*1e6:.2f}μm")
        print(f"  ADAM:     n={adam_params[i]['n']:.4f}, "
              f"k={adam_params[i]['k']:.4f}, "
              f"d={adam_params[i]['d']*1e6:.2f}μm")
    
    # Build per-layer paths from iteration histories
    # Each path is a list of (n, k) tuples — one per iteration
    bayesian_paths_per_layer = None
    adam_paths_per_layer = None

    if bayesian_history is not None:
        bayesian_paths_per_layer = []
        for layer_idx in range(num_layers):
            path = [(abs(p['n'][layer_idx]), abs(p['k'][layer_idx])) 
                    for p in bayesian_history]
            bayesian_paths_per_layer.append(path)

    if adam_history is not None:
        adam_paths_per_layer = []
        for layer_idx in range(num_layers):
            path = [(abs(p['n'][layer_idx]), abs(p['k'][layer_idx])) 
                    for p in adam_history]
            adam_paths_per_layer.append(path)

    #========================================================================
    # Analyze each layer individually
    #========================================================================
    
    all_analyses = {}

    for layer_idx in range(1, num_layers + 1):
        print("\n" + "="*70)
        print(f"ANALYZING LAYER {layer_idx}")
        print("="*70)
        
        # Use Bayesian results as fixed values for other layers
        fixed_layers = [
            (bayesian_params[i]['n'] - 1j*bayesian_params[i]['k'], 
             bayesian_params[i]['d'])
            for i in range(num_layers)
        ]
        
        loss_func = create_single_layer_loss_function(
            x_exp, true_x_exp, deltat, L,
            layer_index=layer_idx,
            fixed_layers=fixed_layers
        )
        
        param_ranges = param_ranges_per_layer[layer_idx - 1]
        viz = LossLandscapeVisualizer(loss_func, param_ranges)
        
        final_values = {
            'Bayesian': bayesian_params[layer_idx - 1],
            'ADAM': adam_params[layer_idx - 1]
        }
        
        #====================================================================
        # Visualization 1: 1D slices
        #====================================================================
        print(f"\n[Layer {layer_idx}] Generating 1D parameter slices...")
        
        fig1, axes = viz.plot_1d_slices(
            optimizer_values=final_values,
            resolution=100
        )
        plt.suptitle(f'Layer {layer_idx} - 1D Parameter Slices', 
                    fontsize=14, y=1.02)
        plt.savefig(f'layer_{layer_idx}_1d_slices.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Saved: layer_{layer_idx}_1d_slices.png")
        plt.show()
        
        #====================================================================
        # Visualization 2: n vs k landscape with optimizer paths
        #====================================================================
        print(f"\n[Layer {layer_idx}] Generating n vs k landscape...")
        
        d_fixed = (final_values['Bayesian']['d'] + 
                   final_values['ADAM']['d']) / 2

        # Build optimizer_paths dict for this layer
        optimizer_paths = {}
        if bayesian_paths_per_layer is not None:
            optimizer_paths['Bayesian'] = bayesian_paths_per_layer[layer_idx - 1]
        if adam_paths_per_layer is not None:
            optimizer_paths['ADAM'] = adam_paths_per_layer[layer_idx - 1]

        fig2, (ax_contour, ax_3d) = viz.plot_2d_slice(
            'n', 'k',
            fixed_params={'d': d_fixed},
            resolution=resolution,
            optimizer_paths=optimizer_paths if optimizer_paths else None
        )

        fig2.suptitle(f'Layer {layer_idx} - n vs k Loss Landscape', 
                     fontsize=14, y=0.98)
        plt.savefig(f'layer_{layer_idx}_n_vs_k.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Saved: layer_{layer_idx}_n_vs_k.png")
        plt.show()
        
        #====================================================================
        # Visualization 3: Local minima analysis
        #====================================================================
        print(f"\n[Layer {layer_idx}] Statistical local minima analysis...")
        
        minima_analysis = viz.analyze_local_minima(
            final_values,
            n_samples=2000
        )
        
        all_analyses[f'Layer_{layer_idx}'] = minima_analysis
    
    #========================================================================
    # Overall Summary
    #========================================================================
    print("\n" + "="*70)
    print("SUMMARY: WHICH LAYERS ARE STUCK IN LOCAL MINIMA?")
    print("="*70)
    
    for layer_idx in range(1, num_layers + 1):
        analysis = all_analyses[f'Layer_{layer_idx}']
        
        bays_pct = analysis['Bayesian']['percentage']
        adam_pct = analysis['ADAM']['percentage']
        
        print(f"\nLayer {layer_idx}:")
        
        if bays_pct < 1 and adam_pct < 1:
            print("  Both optimizers found good solutions")
        elif bays_pct < adam_pct:
            print(f"  Bayesian better ({bays_pct:.1f}% vs {adam_pct:.1f}%)")
            if adam_pct > 5:
                print("  WARNING: ADAM stuck in local minimum!")
        elif adam_pct < bays_pct:
            print(f"  ADAM better ({adam_pct:.1f}% vs {bays_pct:.1f}%)")
            if bays_pct > 5:
                print("  WARNING: Bayesian stuck in local minimum!")
        else:
            if bays_pct > 5:
                print(f"  WARNING: BOTH stuck in local minima! ({bays_pct:.1f}%)")
            else:
                print("  Similar solutions found")
    
    print("\n" + "="*70)
    print(f"Generated {num_layers * 2} visualization files")
    print("="*70)
    
    return all_analyses


#================================================================================
# STEP 3: Compare All Layers on Same Plot
#================================================================================

def compare_all_layers_1d(x_exp, true_x_exp, deltat, L,
                         bayesian_layers, adam_layers,
                         param_ranges_per_layer):
    num_layers = len(bayesian_layers)
    
    fig, axes = plt.subplots(num_layers, 3, figsize=(18, 5*num_layers))
    
    for layer_idx in range(num_layers):
        fixed_layers = [
            (bayesian_layers[i][0], bayesian_layers[i][1])
            for i in range(num_layers)
        ]
        
        loss_func = create_single_layer_loss_function(
            x_exp, true_x_exp, deltat, L,
            layer_index=layer_idx + 1,
            fixed_layers=fixed_layers
        )
        
        n_complex_bays, d_bays = bayesian_layers[layer_idx]
        n_complex_adam, d_adam = adam_layers[layer_idx]
        
        params_bays = {
            'n': float(n_complex_bays.real),
            'k': float(-n_complex_bays.imag),
            'd': float(d_bays)
        }
        params_adam = {
            'n': float(n_complex_adam.real),
            'k': float(-n_complex_adam.imag),
            'd': float(d_adam)
        }
        
        param_ranges = param_ranges_per_layer[layer_idx]
        
        for param_idx, param_name in enumerate(['n', 'k', 'd']):
            ax = axes[layer_idx, param_idx] if num_layers > 1 else axes[param_idx]
            
            fixed = params_bays.copy()
            p_min, p_max = param_ranges[param_name]
            p_vals = np.linspace(p_min, p_max, 100)
            losses = []
            
            for p_val in p_vals:
                fixed[param_name] = p_val
                losses.append(loss_func(fixed))
            
            ax.plot(p_vals, losses, 'b-', linewidth=2)
            ax.axvline(params_bays[param_name], color='cyan', 
                      linestyle='--', linewidth=2, label='Bayesian')
            ax.axvline(params_adam[param_name], color='red',
                      linestyle='--', linewidth=2, label='ADAM')
            
            ax.set_xlabel(param_name, fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'Layer {layer_idx+1} - {param_name}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('all_layers_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: all_layers_comparison.png")
    plt.show()


#================================================================================
# STEP 4: Pairwise Layer Interaction Analysis
#================================================================================

def analyze_layer_interaction(x_exp, true_x_exp, deltat, L,
                              bayesian_layers, adam_layers,
                              layer1_idx, layer2_idx,
                              param1='n', param2='n',
                              resolution=40):
    print(f"\nAnalyzing interaction between Layer {layer1_idx} ({param1}) "
          f"and Layer {layer2_idx} ({param2})...")
    
    ref_layers = bayesian_layers.copy()
    
    n1_bays = float(bayesian_layers[layer1_idx-1][0].real)
    n1_adam = float(adam_layers[layer1_idx-1][0].real)
    k1_bays = float(-bayesian_layers[layer1_idx-1][0].imag)
    k1_adam = float(-adam_layers[layer1_idx-1][0].imag)
    
    n2_bays = float(bayesian_layers[layer2_idx-1][0].real)
    n2_adam = float(adam_layers[layer2_idx-1][0].real)
    k2_bays = float(-bayesian_layers[layer2_idx-1][0].imag)
    k2_adam = float(-adam_layers[layer2_idx-1][0].imag)
    
    if param1 == 'n':
        p1_range = (min(n1_bays, n1_adam) - 0.5, max(n1_bays, n1_adam) + 0.5)
    elif param1 == 'k':
        p1_range = (max(0, min(k1_bays, k1_adam) - 0.1), 
                   max(k1_bays, k1_adam) + 0.1)
    
    if param2 == 'n':
        p2_range = (min(n2_bays, n2_adam) - 0.5, max(n2_bays, n2_adam) + 0.5)
    elif param2 == 'k':
        p2_range = (max(0, min(k2_bays, k2_adam) - 0.1),
                   max(k2_bays, k2_adam) + 0.1)
    
    p1_vals = np.linspace(p1_range[0], p1_range[1], resolution)
    p2_vals = np.linspace(p2_range[0], p2_range[1], resolution)
    P1, P2 = np.meshgrid(p1_vals, p2_vals)
    Z = np.zeros_like(P1)
    
    from Matrix_methods.AdamExtractor import simulate_parallel
    
    for i in range(resolution):
        for j in range(resolution):
            layers = ref_layers.copy()
            
            if param1 == 'n':
                n1 = P1[i, j]; k1 = k1_bays
            else:
                n1 = n1_bays;  k1 = P1[i, j]
            
            if param2 == 'n':
                n2 = P2[i, j]; k2 = k2_bays
            else:
                n2 = n2_bays;  k2 = P2[i, j]
            
            layers[layer1_idx-1] = (n1 - 1j*k1, bayesian_layers[layer1_idx-1][1])
            layers[layer2_idx-1] = (n2 - 1j*k2, bayesian_layers[layer2_idx-1][1])
            
            try:
                T_sim, y_sim = simulate_parallel(x_exp, layers, deltat)
                y_sim = y_sim[:L]
                Z[i, j] = torch.mean((y_sim - true_x_exp[:L])**2).item()
            except:
                Z[i, j] = 1e10
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    contour = ax1.contourf(P1, P2, Z, levels=20, cmap='viridis')
    ax1.contour(P1, P2, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    if param1 == 'n':
        p1_bays, p1_adam = n1_bays, n1_adam
    else:
        p1_bays, p1_adam = k1_bays, k1_adam
    
    if param2 == 'n':
        p2_bays, p2_adam = n2_bays, n2_adam
    else:
        p2_bays, p2_adam = k2_bays, k2_adam
    
    ax1.scatter(p1_bays, p2_bays, color='cyan', marker='s', s=300,
               edgecolors='white', linewidths=3, label='Bayesian', zorder=10)
    ax1.scatter(p1_adam, p2_adam, color='red', marker='o', s=300,
               edgecolors='white', linewidths=3, label='ADAM', zorder=10)
    
    ax1.set_xlabel(f'Layer {layer1_idx} {param1}', fontsize=12)
    ax1.set_ylabel(f'Layer {layer2_idx} {param2}', fontsize=12)
    ax1.set_title('Layer Interaction Loss Landscape', fontsize=14)
    ax1.legend()
    plt.colorbar(contour, ax=ax1, label='Loss')
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(P1, P2, Z, cmap='viridis', alpha=0.8)
    ax2.set_xlabel(f'Layer {layer1_idx} {param1}')
    ax2.set_ylabel(f'Layer {layer2_idx} {param2}')
    ax2.set_zlabel('Loss')
    ax2.set_title('3D View')
    plt.colorbar(surf, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig(f'layer_interaction_{layer1_idx}_{layer2_idx}.png',
               dpi=300, bbox_inches='tight')
    print(f"Saved: layer_interaction_{layer1_idx}_{layer2_idx}.png")
    plt.show()


print("\nMulti-layer loss landscape visualization module loaded!")
print("Use visualize_multilayer_landscape() to analyze each layer.")