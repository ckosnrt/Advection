# MTMW12 Introduction to Numerical Modelling
# Assignment 5: Advection
# Basic & Alternative Schemes: Test Order of Convergence
# Student number: 31837045

import numpy as np
import math

# Outer code for setting up the diffusion problem, calculate and plot.
import matplotlib.pyplot as plt
# Read in all the schemes, initial conditions and other helper code
from Schemes import *
# Calculating linear regression to obtain order of convergence
from scipy.optimize import curve_fit

def calculate_convergence(scheme, nx, dx, nt, c, xmin=0., xmax=1., u=0.1, endTime=1.):
    """
    Calculatate convergence for a given scheme.

    Parameters
    ----------
    scheme : string. The name of the advection scheme.
    IC : array. Initial condition
    x : array. The positions of points in space.
    dx : float. Size of the space grid.
    nt : array. Number of time steps.
    c : float. The Courant number.

    Returns
    -------
    l_2_error_norm : array
        $l_2$ error norm for different values of $\Delta x$.
    params : array
        Linear regression parameters for order of convergence.

    """
    
    l_2_error_norm = np.zeros(len(dx))
        
    for i in range(len(dx)):
        
        x = np.linspace(xmin, xmax - dx[i], nx[i])
        IC = advection_analytic_cos(x, 0, xmin, xmax, u)
        
        phi = eval(f'advection_{scheme}')(IC, nt[i], c)

        phi_analy = advection_analytic_cos(x, endTime, xmin, xmax, u)
            
        A = np.sqrt(np.sum(dx[i] * (phi - phi_analy) ** 2))
        B = np.sqrt(np.sum(dx[i] * phi_analy ** 2))
        l_2_error_norm[i] = A / B
        
        
    # Find linear regression line
    def func1(x, a, b):
        return a*x+b
        
    params, _ = curve_fit(func1, np.log(dx), np.log(l_2_error_norm))
        
    return l_2_error_norm, params


### The main code is inside a function to avoid global variables ###
def main():
    
    schemes = ['FTBS', 'FTCS', 'CTCS', 'Warming_beam', 'TVD', 'Semi_lagrangian']

    xmin = 0.        # Start of model domain (m)
    xmax = 1.        # End of model domain (m)
    c = 0.4          # The Courant number
    endTime = 1.
    u = 0.1          # Wind speed(m/s)

    nx = np.arange(40, 200, 40)
    dx = (xmax - xmin) / nx
    dt = c * dx / u
    nt = np.round(endTime / dt).astype(int)
    
    # Plot FTCS and analytic solution with changing dx
    plt.figure(figsize=(16, 12))
    plt.rcParams["font.size"] = 14

    for i, scheme in enumerate(schemes):
                
        l_2_error_norm, params = calculate_convergence(scheme, nx, dx, nt, c)
        
        plt.plot(dx, l_2_error_norm, label=f'{scheme} n={params[0]:.2f}', marker='o', markersize=5)

        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$\Delta x$ (m)', fontsize=24)
        plt.ylabel('$\ell_2$ Error Norm', fontsize=24)
        plt.title('$\ell_2$ Error Norm for Different Schemes', fontsize=30, y=1.02)
        plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
        
        # Print Order of time and Accuracy in space
        print(f'{scheme} Scheme:')
        print(f'Accuracy in Space is {params[0]:.2f}')

    plt.legend(fontsize=16, loc='best')
    plt.savefig('Test_order.svg')
    plt.show()

main()