import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class LossLandscapeVisualizer:
    """
    Visualize loss landscapes for optical parameter optimization
    """
    
    def __init__(self, loss_function, param_ranges):
        """
        Initialize the visualizer
        
        Parameters:
        -----------
        loss_function : callable
            Function that takes parameters and returns loss
            Should accept dict with keys like 'n', 'k', 'd'
        param_ranges : dict
            Dictionary with parameter names as keys and (min, max) tuples as values
            e.g., {'n': (1.0, 3.0), 'k': (0.0, 1.0), 'd': (50, 200)}
        """
        self.loss_function = loss_function
        self.param_ranges = param_ranges
        
    def plot_2d_slice(self, param1, param2, fixed_params=None, 
                      resolution=50, optimizer_paths=None, figsize=(10, 8)):
        """
        Plot 2D slice of loss landscape varying two parameters
        
        Parameters:
        -----------
        param1, param2 : str
            Names of parameters to vary
        fixed_params : dict
            Fixed values for other parameters
        resolution : int
            Grid resolution for plotting
        optimizer_paths : dict
            Dictionary with optimizer names as keys and arrays of parameter 
            trajectories as values
            e.g., {'ADAM': [(n1, k1), (n2, k2), ...], 
                   'Bayesian': [(n1, k1), (n2, k2), ...]}
        """
        if fixed_params is None:
            fixed_params = {}
        
        # Create grid
        p1_min, p1_max = self.param_ranges[param1]
        p2_min, p2_max = self.param_ranges[param2]
        
        p1_vals = np.linspace(p1_min, p1_max, resolution)
        p2_vals = np.linspace(p2_min, p2_max, resolution)
        
        P1, P2 = np.meshgrid(p1_vals, p2_vals)
        Z = np.zeros_like(P1)
        
        # Compute loss at each grid point
        print(f"Computing loss landscape for {param1} vs {param2}...")
        for i in range(resolution):
            for j in range(resolution):
                params = fixed_params.copy()
                params[param1] = P1[i, j]
                params[param2] = P2[i, j]
                Z[i, j] = self.loss_function(params)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Contour plot
        levels = 20
        contour = ax1.contourf(P1, P2, Z, levels=levels, cmap='viridis')
        contour_lines = ax1.contour(P1, P2, Z, levels=levels, colors='white', 
                                     alpha=0.3, linewidths=0.5)
        ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2e')
        
        # Plot optimizer paths
        if optimizer_paths:
            colors = {'ADAM': 'red', 'Bayesian': 'cyan', 'adam': 'red', 
                     'bayesian': 'cyan', 'BO': 'cyan'}
            markers = {'ADAM': 'o', 'Bayesian': 's', 'adam': 'o', 
                      'bayesian': 's', 'BO': 's'}
            
            for opt_name, path in optimizer_paths.items():
                if len(path) > 0:
                    path_array = np.array(path)
                    color = colors.get(opt_name, 'white')
                    marker = markers.get(opt_name, 'o')
                    
                    # Plot trajectory
                    ax1.plot(path_array[:, 0], path_array[:, 1], 
                            color=color, alpha=0.7, linewidth=2, 
                            label=f'{opt_name} path')
                    
                    # Mark start and end
                    ax1.scatter(path_array[0, 0], path_array[0, 1], 
                               color=color, marker=marker, s=150, 
                               edgecolors='white', linewidths=2, 
                               label=f'{opt_name} start', zorder=5)
                    ax1.scatter(path_array[-1, 0], path_array[-1, 1], 
                               color=color, marker='*', s=300, 
                               edgecolors='white', linewidths=2,
                               label=f'{opt_name} end', zorder=5)
        
        ax1.set_xlabel(param1, fontsize=12)
        ax1.set_ylabel(param2, fontsize=12)
        ax1.set_title(f'Loss Landscape: {param1} vs {param2}', fontsize=14)
        
        # Only create legend if there are items to display
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(loc='best', fontsize=9)
        
        plt.colorbar(contour, ax=ax1, label='Loss')
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(P1, P2, Z, cmap='viridis', alpha=0.8,
                                edgecolor='none', antialiased=True)
        
        ax2.set_xlabel(param1, fontsize=10)
        ax2.set_ylabel(param2, fontsize=10)
        ax2.set_zlabel('Loss', fontsize=10)
        ax2.set_title('3D Loss Surface', fontsize=12)
        plt.colorbar(surf, ax=ax2, shrink=0.5, label='Loss')
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_1d_slices(self, fixed_params=None, resolution=100, 
                       optimizer_values=None, figsize=(15, 4)):
        """
        Plot 1D slices varying each parameter individually
        
        Parameters:
        -----------
        fixed_params : dict
            Fixed values for parameters (will vary one at a time)
        resolution : int
            Number of points for each slice
        optimizer_values : dict
            Final values found by each optimizer
            e.g., {'ADAM': {'n': 1.5, 'k': 0.1, 'd': 100},
                   'Bayesian': {'n': 1.6, 'k': 0.15, 'd': 105}}
        """
        if fixed_params is None:
            # Use middle of ranges as default
            fixed_params = {k: (v[0] + v[1])/2 
                          for k, v in self.param_ranges.items()}
        
        n_params = len(self.param_ranges)
        fig, axes = plt.subplots(1, n_params, figsize=figsize)
        
        if n_params == 1:
            axes = [axes]
        
        for idx, (param_name, (p_min, p_max)) in enumerate(self.param_ranges.items()):
            p_vals = np.linspace(p_min, p_max, resolution)
            losses = np.zeros(resolution)
            
            # Compute loss along this dimension
            for i, p_val in enumerate(p_vals):
                params = fixed_params.copy()
                params[param_name] = p_val
                losses[i] = self.loss_function(params)
            
            # Plot
            axes[idx].plot(p_vals, losses, 'b-', linewidth=2, label='Loss')
            axes[idx].set_xlabel(param_name, fontsize=12)
            axes[idx].set_ylabel('Loss', fontsize=12)
            axes[idx].set_title(f'Loss vs {param_name}', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            
            # Mark optimizer solutions
            if optimizer_values:
                for opt_name, opt_params in optimizer_values.items():
                    if param_name in opt_params:
                        # Get loss at this point
                        params = fixed_params.copy()
                        params[param_name] = opt_params[param_name]
                        loss = self.loss_function(params)
                        
                        color = 'red' if 'ADAM' in opt_name or 'adam' in opt_name else 'cyan'
                        axes[idx].axvline(opt_params[param_name], 
                                        color=color, linestyle='--', 
                                        linewidth=2, alpha=0.7,
                                        label=f'{opt_name}')
                        axes[idx].scatter([opt_params[param_name]], [loss],
                                        color=color, s=100, zorder=5,
                                        edgecolors='white', linewidths=2)
            
            axes[idx].legend(fontsize=9)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_convergence_comparison(self, optimizer_histories, figsize=(12, 5)):
        """
        Plot loss convergence over iterations for multiple optimizers
        
        Parameters:
        -----------
        optimizer_histories : dict
            Dictionary with optimizer names as keys and loss histories as values
            e.g., {'ADAM': [loss1, loss2, ...], 'Bayesian': [loss1, loss2, ...]}
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = {'ADAM': 'red', 'Bayesian': 'cyan', 'adam': 'red', 
                 'bayesian': 'cyan', 'BO': 'cyan'}
        
        # Linear scale
        for opt_name, history in optimizer_histories.items():
            color = colors.get(opt_name, 'blue')
            ax1.plot(history, color=color, linewidth=2, 
                    label=opt_name, alpha=0.8)
            ax1.scatter(len(history)-1, history[-1], color=color, 
                       s=100, zorder=5, edgecolors='white', linewidths=2)
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Convergence (Linear Scale)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Log scale
        for opt_name, history in optimizer_histories.items():
            color = colors.get(opt_name, 'blue')
            ax2.semilogy(history, color=color, linewidth=2, 
                        label=opt_name, alpha=0.8)
            ax2.scatter(len(history)-1, history[-1], color=color, 
                       s=100, zorder=5, edgecolors='white', linewidths=2)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss (log scale)', fontsize=12)
        ax2.set_title('Loss Convergence (Log Scale)', fontsize=14)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def analyze_local_minima(self, optimizer_values, n_samples=1000):
        """
        Analyze whether optimizer solutions are likely local minima
        by random sampling around their solutions
        
        Parameters:
        -----------
        optimizer_values : dict
            Final parameter values from each optimizer
        n_samples : int
            Number of random samples to test
        """
        results = {}
        
        for opt_name, opt_params in optimizer_values.items():
            opt_loss = self.loss_function(opt_params)
            
            # Sample around this point
            lower_losses = 0
            sample_losses = []
            
            for _ in range(n_samples):
                # Perturb parameters
                perturbed = {}
                for param, value in opt_params.items():
                    p_min, p_max = self.param_ranges[param]
                    # Sample within ±20% of current value, clipped to valid range
                    delta = 0.2 * (p_max - p_min)
                    perturbed[param] = np.clip(
                        value + np.random.uniform(-delta, delta),
                        p_min, p_max
                    )
                
                sample_loss = self.loss_function(perturbed)
                sample_losses.append(sample_loss)
                
                if sample_loss < opt_loss:
                    lower_losses += 1
            
            results[opt_name] = {
                'optimizer_loss': opt_loss,
                'samples_with_lower_loss': lower_losses,
                'percentage': 100 * lower_losses / n_samples,
                'min_sample_loss': min(sample_losses),
                'mean_sample_loss': np.mean(sample_losses)
            }
        
        # Print results
        print("\n" + "="*70)
        print("LOCAL MINIMA ANALYSIS")
        print("="*70)
        for opt_name, res in results.items():
            print(f"\n{opt_name}:")
            print(f"  Loss at solution: {res['optimizer_loss']:.6e}")
            print(f"  Random samples with lower loss: {res['samples_with_lower_loss']}/{n_samples} ({res['percentage']:.1f}%)")
            print(f"  Min loss in samples: {res['min_sample_loss']:.6e}")
            print(f"  Mean loss in samples: {res['mean_sample_loss']:.6e}")
            
            if res['percentage'] > 5:
                print(f"WARNING: Likely stuck in local minimum!")
            else:
                print(f"Appears to be near global minimum")
        print("="*70 + "\n")
        
        return results


# Example usage with mock data
def example_usage():
    """
    Example showing how to use the visualizer
    """
    
    # Define a mock loss function (replace with your actual TMM loss function)
    def mock_loss_function(params):
        """
        Mock loss function with multiple local minima
        This simulates a complex optimization landscape
        """
        n = params.get('n', 1.5)
        k = params.get('k', 0.1)
        
        # Create a landscape with multiple minima
        loss = (np.sin(3*n) * np.cos(5*k) + 
                0.1 * (n - 1.5)**2 + 
                0.5 * (k - 0.2)**2 + 
                2.0)
        return loss
    
    # Define parameter ranges
    param_ranges = {
        'n': (1.0, 2.0),  # refractive index
        'k': (0.0, 0.5),  # extinction coefficient
        # 'd': (50, 200)   # thickness (nm) - can add more parameters
    }
    
    # Create visualizer
    visualizer = LossLandscapeVisualizer(mock_loss_function, param_ranges)
    
    # Mock optimizer trajectories (replace with your actual optimization paths)
    adam_path = [
        (1.1, 0.1), (1.15, 0.12), (1.2, 0.15), (1.25, 0.18),
        (1.28, 0.19), (1.30, 0.20), (1.32, 0.21) 
    ]
    
    bayesian_path = [
        (1.8, 0.4), (1.7, 0.35), (1.6, 0.3), (1.55, 0.25),
        (1.52, 0.22), (1.50, 0.20)  
    ]
    
    optimizer_paths = {
        'ADAM': adam_path,
        'Bayesian': bayesian_path
    }
    
    # Plot 2D landscape
    print("Generating 2D loss landscape...")
    fig1, _ = visualizer.plot_2d_slice('n', 'k', 
                                       optimizer_paths=optimizer_paths,
                                       resolution=100)
    print("Saved: loss_landscape_2d.png")
    
    # Plot 1D slices
    print("\nGenerating 1D slices...")
    optimizer_final_values = {
        'ADAM': {'n': adam_path[-1][0], 'k': adam_path[-1][1]},
        'Bayesian': {'n': bayesian_path[-1][0], 'k': bayesian_path[-1][1]}
    }
    
    fig2, _ = visualizer.plot_1d_slices(
        optimizer_values=optimizer_final_values,
        resolution=200
    )
    print("Saved: loss_landscape_1d.png")
    
    # Plot convergence
    print("\nGenerating convergence plots...")
    adam_losses = [mock_loss_function({'n': p[0], 'k': p[1]}) for p in adam_path]
    bayesian_losses = [mock_loss_function({'n': p[0], 'k': p[1]}) for p in bayesian_path]
    
    optimizer_histories = {
        'ADAM': adam_losses,
        'Bayesian': bayesian_losses
    }
    
    fig3, _ = visualizer.plot_convergence_comparison(optimizer_histories)
    print("Saved: convergence_comparison.png")
    
    # Analyze local minima
    print("\nAnalyzing local minima...")
    results = visualizer.analyze_local_minima(optimizer_final_values, n_samples=2000)
    
    plt.show()
    
    return visualizer, results


if __name__ == "__main__":
    example_usage()