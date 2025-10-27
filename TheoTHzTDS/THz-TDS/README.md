# THz-TDS

A collection of classes, scripts, and notebooks for material parameter extraction with Terahertz Time-Domain Spectroscopy (THz-TDS).

## Table of Contents

- [About the Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## About the Project

This project uses gradient decent (optimized with adam) to fit a frequency dependent complex refractive index for an experimental transfer function. 

Trainable parameters n & k are torch.tensors with the same length as the provided frequency range. 

These are optimized to reconstruct the experimental transfer function provided.


### Built With

- [Python](https://www.python.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

## Files and Components
`Extraction:`  
A class designed to treat experimental data in the time domain. Check notebook for usage example.  
`PINNs:`  
- `Model:` Classes and utility files for back-propagation driven fitting.
- `data:` Contains two pulses through 3mm silicon, data from NR extraction on this experiment and an example of a sequential version of the back-propagation model.  

`Notebooks:` Shows the treatment of experimental data on the Si datasets, and uses an NR numerical fit to find n and k.  

`Transfer Matrix Methods:` Uses Transfer matrix methods to simulate time domain pulses for media with multiple layers of different materials and thicknesses.


## Getting Started

Instructions on setting up the project locally.

### Prerequisites

- `Python 3.11`
- `Numpy`
- `PyTorch 2.6`

### Installation

Step-by-step guide on how to install and set up the project.

Clone the repository:
```bash
git clone https://github.com/tls54/THz-TDS.git
```

### Usage
Look at Experimental_freq_dep.ipynb for implementation on experimental data.


## Contact
Contact the developer for enquiries and contributions.  
Developer: Theo Smith - tls1g21@soton.ac.uk

