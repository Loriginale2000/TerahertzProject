import numpy as np
import torch

def get_frequency_domain(pulse: torch.Tensor, deltat: float, pad_to: int = None):
    """
    Converts a time-domain THz pulse to the frequency domain.
    
    Args:
        pulse (torch.Tensor): 1D time-domain pulse (shape: [T])
        deltat (float): Time step between samples (in seconds)
        pad_to (int, optional): Zero-pad the signal to this length (must be >= len(pulse)).
                                If None, no zero-padding is applied.
    
    Returns:
        freqs (np.ndarray): Frequency axis in Hz (shape: [N])
        spectrum (torch.Tensor): Complex spectrum (FFT result) (shape: [N])
    """
    # Convert to 1D tensor if needed
    pulse = pulse.flatten()

    N = pulse.shape[0]
    if pad_to is not None and pad_to > N:
        pad_size = pad_to - N
        pulse = torch.nn.functional.pad(pulse, (0, pad_size))
        N = pad_to
    
    # Compute FFT
    spectrum = torch.fft.fft(pulse)
    
    # Frequency axis
    freqs = np.fft.fftfreq(N, deltat)  # in Hz

    return freqs, spectrum