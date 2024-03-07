# MTMW12 Introduction to Numerical Modelling
# Assignment 5: Advection
# Basic & Alternative Schemes: Functions & Plotting
# Student number: 31837045, 31832278

import numpy as np
import math
# Outer code for setting up the advection problem, calculate and plot.
import matplotlib.pyplot as plt
# Read in all the schemes, initial conditions and other helper code
from Schemes import *

def plot_phi_errors(x, schemes, IC, phi_analy, label_group, label_initial, xmax=1.):
    
    fig, axs = plt.subplots(2, len(schemes), figsize = (22, 15))
    plt.rcParams["font.size"] = 20
    
    x = np.append(x, xmax)
    phi_analy = np.append(phi_analy, phi_analy[0])
    
    for i, scheme in enumerate(schemes):
        phi = eval(f'advection_{scheme}')(IC)
    
        phi = np.append(phi, phi[0])
                
        axs[0, i].plot(x, phi, label=f'{scheme}')
        axs[0, i].plot(x, phi_analy, label='analytic')
        axs[1, i].plot(x, phi-phi_analy)
        
        axs[0, i].set_title(f'{scheme}')
        axs[0, i].set_ylim(-0.6, 1.6)
        axs[1, i].set_ylim(-0.6, 0.6)
        axs[0, 0].set_ylabel('$\phi$')
        axs[1, 0].set_ylabel('errors')
        axs[1, i].set_xlabel('x')
        axs[0, i].legend(loc='best')
        axs[0, i].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
        axs[1, i].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)

    plt.suptitle(f'$\phi$ and errors of each {label_group} with {label_initial}', fontsize=30)
    plt.savefig(f'phi_and_errors_{label_group}_IC_{label_initial}.svg')
    plt.show()
    
### The main code is inside a function to avoid global variables ###
def main():
    """Plot each schemes"""
    
    basic_schemes = ['FTBS', 'FTCS', 'CTCS']
    alternative_schemes = ['Warming_beam', 'TVD', 'Semi_lagrangian']
    initial_conditions = ['1', 'cos']
    
    # Parameters
    xmin = 0.        # Start of model domain (m)
    xmax = 1.       # End of model domain (m)a
    nx = 40          # Number of grid points, including both ends
    nt = 10          # Number of time steps taken to get to the endTime
    c = 0.4          # The Courant number
    u = 0.1          # Wind speed(m/s)
    
    # Parameters for initial conditions
    a = 0.1
    b = 0.4
    
    # Other derived parameters
    dx = (xmax - xmin)/(nx)    # The grid spacing
    dt = np.round(c*dx/u, decimals=14)                  # lengh of time step
    endTime = dt * nt            # Total time of the whole simulation
    print("dx =", dx, "dt =", dt, "The Courant number =", c)
        
    # x points
    x = np.linspace(xmin, xmax-dx, nx)

    for j, initial in enumerate(initial_conditions):
        
        IC = eval(f'advection_analytic_{initial}')(x)
        phi_analy = eval(f'advection_analytic_{initial}')(x, endTime)
    
        plot_phi_errors(x, basic_schemes, IC, phi_analy, f'basic_schemes', f'{initial}')
        plot_phi_errors(x, alternative_schemes, IC, phi_analy, f'alternative_schemes', f'{initial}')
        
main()