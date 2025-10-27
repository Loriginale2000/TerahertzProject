import numpy as np
import pandas as pd
from scipy.signal.windows import tukey
from .transfer_functions import *
from .plotting import *
from .transformations import *
from .constants import c



# Create extractor class
class Extractor:
    def __init__(self, reference: np.ndarray, sample: np.ndarray, thickness: float, dc_offset_range: int = 50, window=False) -> None:

        ### Preprocess time domain data and extract frequency range when the class is first called

        # define physical constants
        self.Length = thickness  # Ensure it is in meters

        self.unwrapping_regression_range = [205, 820]

        # Extract the signals and time data
        self.signal_ref = reference[:, 1]
        self.signal_sample = sample[:, 1]

        self.time_ref = reference[:, 0]
        self.time_sample = sample[:, 0]

        # Calculate offset and padding
        offset = self.time_sample[0] - self.time_ref[0]
        time_step = self.time_ref[1] - self.time_ref[0]
        n_padding = int(offset / time_step)

        # Adjust time and signal arrays by padding
        self.time_ref = np.concatenate([self.time_ref, np.linspace(self.time_ref[-1], self.time_sample[-1], n_padding)])
        self.time_sample = np.concatenate([np.linspace(self.time_ref[0], self.time_sample[0], n_padding), self.time_sample])
        self.signal_ref = np.concatenate([self.signal_ref, np.zeros(n_padding)])
        self.signal_sample = np.concatenate([np.zeros(n_padding), self.signal_sample])

        # Time and frequency domain parameters
        T = time_step
        Fs = 1 / T
        L = len(self.signal_ref)
        t = np.arange(0, L) * T
        self.f = Fs / L * np.arange(0, L)  # Frequency values

        # Remove DC offset -- const error due to Lock-in amplifier
        self.signal_ref -= np.mean(self.signal_ref[:dc_offset_range])
        self.signal_sample -= np.mean(self.signal_sample[:dc_offset_range])

        if window:
            # Perform the automated windowing
            self.window_tukey_trivial()

        # Perform the Fourier transforms on reference and sample data
        self.fft_signals()

    ###--------------------------------------------------------------------------------------------------------
    # Plots and returns time domain data

    def plot_time_domain(self):
        '''
        Plots the time domain data
        '''
        plot_time_domain(self.time_ref, self.signal_ref, self.time_sample, self.signal_sample)



    def get_processed_data(self):
        '''
        Returns the time domain processed data as a pandas data frame.
        '''
        data = {
            'frequency': self.f,
            'signal_ref': self.signal_ref,
            'time_ref': self.time_ref,
            'signal_sample': self.signal_sample,
            'time_sample': self.time_sample
        }
        processed_data = pd.DataFrame(data)
        return processed_data


    ###--------------------------------------------------------------------------------------------------------
    def window_tukey_trivial(self, window_width_ref = 150, window_width_sample = 150, parameter = 0.4):
        '''
        Performs Tukey windowing on time domain data using a trivial peak-finding method and fixed window width.
        '''

        # Find the peak locations 
        window_centre_index_ref = np.argmax(np.abs(self.signal_ref))
        window_centre_index_sample = np.argmax(np.abs(self.signal_sample))

        # Construct the window arrays — we need to pad the elements outside the window with zeros
        window_ref = tukey(2*window_width_ref, parameter)
        window_sample = tukey(2*window_width_sample, parameter)

        length = self.signal_ref.size
        full_window_ref = np.zeros(length)
        full_window_sample = np.zeros(length)

        full_window_ref[window_centre_index_ref-window_width_ref:
                        window_centre_index_ref+window_width_ref] = window_ref
        
        full_window_sample[window_centre_index_sample-window_width_sample:
                            window_centre_index_sample+window_width_sample] = window_sample

        # Multiply the signal by the window array
        self.signal_ref *= full_window_ref
        self.signal_sample *= full_window_sample

        return full_window_ref, full_window_sample



    ###--------------------------------------------------------------------------------------------------------
    # Handles frequency domain data 

    def fft_signals(self, interpolation: int = 2**12) -> None:
        '''
        Transforms data using numpy fft. Calculates the transfer function and unwraps its phase removing any offset. Calculates the refractive index.
        '''
        # Make interpolation an attribute of the class
        self.interpolation = interpolation

        # Adjust frequency array for all possible values from fft
        self.f_interp = np.linspace(self.f[0], self.f[-1], interpolation)

        # Transform data using numpy fft
        self.A_signal_ref, self.ph_signal_ref, self.A_signal_sample, self.ph_signal_sample = fft_signals(
            self.signal_ref, 
            self.signal_sample, 
            interpolation,
            self.f_interp,
            self.unwrapping_regression_range
            )
        
        # Calculate the transfer function, separate magnitude and phase, and begin unwrapping
        H_exp_general = (self.A_signal_sample * np.exp(1j * self.ph_signal_sample)) / (self.A_signal_ref * np.exp(1j * self.ph_signal_ref))
        self.A_transfer = np.abs(H_exp_general)
        self.ph_transfer = np.unwrap(np.angle(H_exp_general))

        # Remove the offset in transfer unwrapped phase like 
        print("Transfer function:")
        self.ph_transfer = remove_phase_offset(self.f_interp, self.ph_transfer, self.unwrapping_regression_range)

        # Calculate fast refractive index using n = 1 - phi*c / omega*L; 1e-16 prevents div. by zero
        self.fast_n = 1 - (self.ph_transfer * c) / (2*np.pi*self.f_interp*1e12 * self.Length + 1e-16)




    def get_fft_data(self):
    # Create a DataFrame for amplitude, phase, and frequency data
        data = {
        'interpolated frequency': self.f_interp,
        'amplitude_signal_ref': self.A_signal_ref,
        'amplitude_signal_sample': self.A_signal_sample,
        'phase_signal_ref': self.ph_signal_ref,
        'phase_signal_sample': self.ph_signal_sample,
        'amplitude_transfer': self.A_transfer,
        'phase_transfer': self.ph_transfer,
        'fast_n':self.fast_n
        }
    
        fft_data = pd.DataFrame(data)
        return fft_data



    def plot_frequency_domain(self, x_lims=[0,4]):
        '''
        Plots Frequency domain
        '''
        plot_frequency_domain(self.f_interp, self.A_signal_ref, self.ph_signal_ref, self.A_signal_sample, self.ph_signal_sample, x_lims=x_lims)



    ###--------------------------------------------------------------------------------------------------------
    # fitting method for the refractive index
    def calculate_refractive_index(self, n_0: complex, frequency_stop = 4.0):

        """Calculate refractive index using the Newton-Raphson method.
            Inputs:
            -------
            n_0: Initial guess for complex refractive index.
            self: Allows the method to access atributes of the class such as amplitude and phase of signals in frequency domain.

            Outputs:
            --------
            None: Results are appended to n_extracted attribute of the class.
        """
        
        # define experimental transfer function
        #H_exp_general = (self.A_signal_sample * np.exp(1j * self.ph_signal_sample)) / (self.A_signal_ref * np.exp(1j * self.ph_signal_ref))
        
        #define components of experimental transfer function
        self.A_exp = self.A_transfer
        self.ph_exp = self.ph_transfer
        
        # Initialize extracted array to be complex and the correct size
        self.n_extracted = np.zeros(self.interpolation, dtype=complex)
        
        # Limit the frequency range (usually no need to calculate > 4 THz)
        if frequency_stop != None:
            frequency = self.f_interp[self.f_interp <= frequency_stop]
        else:
            frequency = self.f_interp

        # Iterate over frequencies
        w = 2 * np.pi * frequency * 1e12  # Angular frequency in radians/sec

        for f_index in range(len(frequency)):
            n_next = n_0  # Reset for each frequency
            for _ in range(10):  # Arbitrary number of iterations for Newton-Raphson
                H_th = H_th_function(n_next, self.f_interp*1e12*2*np.pi, self.Length)
                A_th = np.abs(H_th)
                ph_th = np.unwrap(np.angle(H_th))
                ph_th = remove_phase_offset(self.f_interp, ph_th, self.unwrapping_regression_range, verbose=False)

                # Function to optimize
                fun = np.log(A_th[f_index]) - np.log(self.A_exp[f_index]) + 1j*(ph_th[f_index] - self.ph_exp[f_index])
                fun_prime = H_prime_function(n_next, w[f_index], self.Length)

                # Update refractive index using Newton-Raphson
                n_next = n_next - fun / fun_prime

            # Store extracted refractive index
            self.n_extracted[f_index] = n_next



    # Return refractive index data as dataframe

    def get_refractive_index_data(self):
        '''
        Return refractive index as dataframe.
        '''
        # Organise data as dict
        data = {
        'real_part': np.real(self.n_extracted),
        'imaginary_part': np.imag(self.n_extracted)
        }
    
        # Convert dict to df
        refractive_index_data = pd.DataFrame(data)

        return refractive_index_data



    def plot_refractive_index(self):
        '''
        Plot the real and imaginary parts of the refractive index.
        '''
        # call externally defined plotting fucntion
        plot_refractive_index(self.f_interp, self.n_extracted)





###--------------------------------------------------------------------------------------------------------
# Test the functionality
if __name__ == "__main__":
    ref_tab = pd.read_csv("Data_sets/simple_data/ref.pulse.csv").to_numpy()
    sample_tab = pd.read_csv("Data_sets/simple_data/Si.pulse.csv").to_numpy()

    extractor = Extractor(ref_tab, sample_tab, thickness=3*1e-3)

    print(f'length of output from get_data: {len(extractor.get_processed_data())}')
    extractor.plot_time_domain()
    print()
    print('Time domain plotted successfully')

    extractor.fft_signals()
    print(f'length of output from get_fft_data: {len(extractor.get_fft_data())}')
    print()
    extractor.plot_frequency_domain()
    print('Frequency domain plotted successfully')
    print()

    extractor.calculate_refractive_index(n_0=3.7 + 0.1j)
    extractor.plot_refractive_index()
    print('Successfully extracted and plotted refractive index of sample')