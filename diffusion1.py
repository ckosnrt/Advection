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
    xmax = 1.        # End of model domain (m)
    nx = 40+1        # Number of grid points, including both ends
    nt = 10          # Number of time steps taken to get to the endTime
    c = 0.4          # The Courant number
    u = 0.1          # Wind speed(m/s)
    
    # Other derived parameters
    dx = (xmax - xmin)/(nx-1)    # The grid spacing
    dt = np.round(c*dx/u, decimals=15)                  # lengh of time step
    endTime = dt * nt            # Total time of the whole simulation
    print("dx =", dx, "dt =", dt, "The Courant number =", c)
        
    # x points
    x = np.linspace(xmin, xmax, nx)
    print(x)
    print(x[nx-1])

    # Initial condition
    # for i in range(nx):
        
    # Diffusion using FTCS and analytic solution, Plot the solutions
    
main()