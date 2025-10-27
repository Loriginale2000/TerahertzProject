import numpy as np
from numpy.fft import fft



def fft_signals(signal_ref, signal_sample, interpolation: int, f_interp, unwrapping_regression_range):
        '''
        Performs FFT on time domain signals and unwraps the phase with the correct offset.
        '''
        # Compute FFTs of both signals with interpolation
        fft_ref = fft(signal_ref, interpolation)
        fft_sample = fft(signal_sample, interpolation)

        # Calculate amplitude and phase for both signals
        A_signal_ref = np.abs(fft_ref)
        ph_signal_ref = np.unwrap(np.angle(fft_ref))

        A_signal_sample = np.abs(fft_sample)
        ph_signal_sample = np.unwrap(np.angle(fft_sample))

        # Remove phase offset
        print("Reference:")
        ph_signal_ref = remove_phase_offset(f_interp, ph_signal_ref, unwrapping_regression_range)
        print("Sample:")
        ph_signal_sample = remove_phase_offset(f_interp, ph_signal_sample, unwrapping_regression_range) 
        
        return A_signal_ref, ph_signal_ref, A_signal_sample, ph_signal_sample


def remove_phase_offset(f_interp, ph_signal, unwrapping_regression_range, verbose=True):
        '''
        Adjusts the phase offset such that it follows y = mx for a given range.
        '''
        # Fit a linear model y=mx+b to the phase curve in a given range
        f_indices = np.arange(*unwrapping_regression_range)
        coef = np.polyfit(f_interp[f_indices], ph_signal[f_indices], 1)


        # Remove the offset b
        ph_signal -= coef[1]

        # Print out statistics as a sanity check
        mse = np.mean(np.square(ph_signal-f_interp*coef[0])[f_indices])

        if verbose:
                print("Phase offset fit frequency range: ", f_interp[unwrapping_regression_range])
                print("Phase slope: ", coef[0])
                print("Mean squared error: ", mse)
                print("(Should be ~< 1.)")
                print("--------------------")

        return ph_signal 