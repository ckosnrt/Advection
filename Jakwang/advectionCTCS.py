# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math

# Outer code for setting up the diffusion problem, calculate and plot.
import matplotlib.pyplot as plt

# Read in all the schemes, initial conditions and other helper code
from BasicSchemes import *

### The main code is inside a function to avoid global variables ###
def main():
    """Compare analytic solution with FTCS"""
    
    # Parameters
    xmin = 0.        # Start of model domain (m)
    xmax = 1.        # End of model domain (m)a
    nx = 40        # Number of grid points, including both ends
    nt = 30          # Number of time steps taken to get to the endTime
    c = 0.4          # The Courant number
    u = 0.1          # Wind speed(m/s)
    
    a = 0.1
    b = 0.5
    
    # Other derived parameters
    dx = (xmax - xmin)/(nx)    # The grid spacing
    dt = np.round(c*dx/u, decimals=12)                  # lengh of time step
    t = dt * nt            # Total time of the whole simulation
    print("dx =", dx, "dt =", dt, "The Courant number =", c)
        
    # x points
    x = np.linspace(xmin, xmax-dx, nx)

    # Initial condition
    IC_1 = initial_condition_1(x, a, b, u)
    IC_cos = initial_condition_cos(x, a, b, u)
    
    # Diffusion using FTCS and analytic solution, Plot the 
    plt.figure(figsize=(16,12))
    plt.plot(x,IC_1, label='Initial Condition 1')
    plt.plot(x, advection_analytic_cos(x, a, b, u, t), label='analytic for 1')
    plt.plot(x, advection_CTCS(IC_cos, nt, c), label='Semi-Lagrangian for 1')
    plt.legend()
    plt.show()
    
main()