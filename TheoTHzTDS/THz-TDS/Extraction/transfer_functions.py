import numpy as np
from .constants import c

## Define transfer function for no reflection model

def H_th_function(n, w, length):
    '''
    Inputs
    ------
    n: refractive index
    w: frequency of light being propagated
    length: length of the sample we are modelling
    
    outputs
    -------
    returns: output for the transfer function

    Method
    ------
    Equation for transfer function derived from [Input source here]

    '''
    return (4 * n) / ((n + 1) ** 2) * np.exp(-1j * (n - 1) * w * length/ c)



def H_prime_function(n, w, length):
    '''
    Inputs
    ------
    n: refractive index
    w: frequency of light being propagated
    length: length of the sample we are modelling
    
    outputs
    -------
    returns: derivative of transfer function, calculated analytically

    Method
    ------
    Equation for derivative of transfer function formulated by hand 
    
    '''
    return ((1 / n) - (2 / (n + 1)) - 1j * w * length / c)