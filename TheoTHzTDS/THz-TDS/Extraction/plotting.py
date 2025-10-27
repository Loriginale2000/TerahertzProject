import matplotlib.pyplot as plt
import numpy as np



def plot_time_domain(time_ref, signal_ref, time_sample, signal_sample):
    # Plot the time signals
    plt.figure(figsize=(8,4), dpi=150)
    plt.plot(time_ref, signal_ref, label="Reference Signal")
    plt.plot(time_sample, signal_sample, label="Sample Signal")
    plt.title('Signals in time domain')
    plt.xlabel('Time [ps]')
    plt.ylabel('Signal [nA]')
    plt.legend()
    plt.show()



def plot_frequency_domain(f_interp, A_signal_ref, ph_signal_ref, A_signal_sample, ph_signal_sample, x_lims=[0, 4]):
    # Create a tiled layout for plotting amplitude and phase
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot amplitude spectrum
    axs[0].plot(f_interp, A_signal_ref, label='Reference Amplitude')
    axs[0].plot(f_interp, A_signal_sample, label='Sample Amplitude')
    axs[0].set_xlim(x_lims)
    axs[0].set_title('Fourier transform of Signals')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()

    # Filter indices within the x_lims range
    valid_indices = (f_interp >= x_lims[0]) & (f_interp <= x_lims[1])

    # Get min and max phase values in the valid range
    ph_min = min(np.min(ph_signal_ref[valid_indices]), np.min(ph_signal_sample[valid_indices]))
    ph_max = max(np.max(ph_signal_ref[valid_indices]), np.max(ph_signal_sample[valid_indices]))

    # Plot phase spectrum
    axs[1].plot(f_interp, ph_signal_ref, label='Reference Phase')
    axs[1].plot(f_interp, ph_signal_sample, label='Sample Phase')
    axs[1].set_xlim(x_lims)
    axs[1].set_ylim(ph_min, ph_max)  # Dynamically set y limits
    axs[1].set_xlabel('Frequency [THz]')
    axs[1].set_ylabel('Phase [Radians]')
    axs[1].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()



def plot_refractive_index(f_interp, n_extracted):
    """Plot the real and imaginary parts of the refractive index."""
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 6))

    # Plot real part of refractive index
    axs2[0].plot(f_interp, np.real(n_extracted))
    axs2[0].set_xlim([0, 4])
    #axs[0].set_ylim([3.45, 3.47])
    axs2[0].set_xlabel("Frequency (THz)")
    axs2[0].set_ylabel("Real refractive index n")

    # Plot imaginary part of refractive index (extinction coefficient)
    axs2[1].plot(f_interp, np.imag(n_extracted))
    axs2[1].set_xlim([0, 4])
    #axs[1].set_ylim([-0.5, 0.5]) 
    axs2[1].set_xlabel("Frequency (THz)")
    axs2[1].set_ylabel("Extinction coefficient k")

    plt.tight_layout()
    plt.show() 