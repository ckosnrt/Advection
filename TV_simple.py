# MTMW12 Introduction to Numerical Modelling
# Assignment 5: Advection
# Basic & Alternative Schemes: Test TV
# Student number: 31832278

import matplotlib.pyplot as plt
import numpy as np
from Schemes import *

def total_var(scheme, IC, t):
    """
    Calculate the total variation of a 1D signal.

    Parameters
    ----------
    phi : array. representing signal.

    Returns
    -------
    var : float. total variation.

    """
    nx = len(IC)

    TV = np.zeros(len(t))

    # Calculate Total variation
    for it in range(len(t)):
        
        phi = eval(f'advection_{scheme}')(IC, t[it])
        TV[it] = np.sum(np.abs(np.roll(phi, -1) - phi))
                        
    return TV

def main():
    """
    Execute simulations for various advection schemes and 
    compare their total variation.

    Returns
    -------
    Total variation Graphs.

    """
    
    schemes = ['FTBS', 'FTCS', 'CTCS','Warming_beam', 'TVD', 'Semi_lagrangian']

    initial_conditions = ['1', 'cos']
    
    # Parameters
    xmin = 0.        # Start of model domain (m)
    xmax = 1.        # End of model domain (m)a
    nx = 40        # Number of grid points, including both ends
    nt = 300          # Number of time steps taken to get to the endTime
    c = 0.4          # The Courant number
    u = 0.1          # Wind speed(m/s)
    
    a = 0.1
    b = 0.5
    
    dx = (xmax - xmin)/(nx-1)   
    dt = np.round(c*dx/u, decimals=12)                  
    t = dt * nt

    x = np.linspace(xmin, xmax-dx, nx)
    t = np.arange(0, nt, 10)
    
    for j, initial in enumerate(initial_conditions):
        
        IC = eval(f'advection_analytic_{initial}')(x)
        
        plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = 18
            
        for i, scheme in enumerate(schemes):
            plt.plot(t, total_var(scheme, IC, t), label=f'{scheme}')
            
        plt.title(f'TV with nt, {initial}')
        plt.xlabel('nt', fontsize=24)
        
        if initial == '1':
            plt.ylim(0, 10)
        else:
            plt.ylim(0, 4)
            
        plt.ylabel('Total Variation', fontsize=24)
        plt.legend()
        plt.savefig(f'TV_IC_{initial}.svg')
           
main()
    