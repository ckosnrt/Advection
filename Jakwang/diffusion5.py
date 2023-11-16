# -*- coding: utf-8 -*-

import numpy as np
import math

# Outer code for setting up the diffusion problem, calculate and plot.
import matplotlib.pyplot as plt
# Read in all the schemes, initial conditions and other helper code
from diffusionSchemes2 import *
# Calculating linear regression to obtain order of convergence
from scipy.optimize import curve_fit

### The main code is inside a function to avoid global variables ###
def main():
    """Find Order of Convergence"""
    # Parameters
    xmin = 0.        # Start of model domain (m)
    xmax = 10.       # End of model domain (m)
    d = 0.25
    endTime = 100
    K = 0.01
    k = np.arange(1,10+1)
    
    # Other derived parameters
    nx = np.arange(30, 200, 5)+1
    dx = (xmax - xmin)/(nx-1)
    dt = d*dx**2/K
    nt = np.round(endTime/dt).astype(int)
    
    #nt = (endTime/dt).astype(int)
    
    # Set initial value of error norm
    l_2_error_norm = np.zeros(len(nx))
    
    # Plot FTCS and analytic solution with changing dx
    plt.figure(figsize=(16, 12))
    plt.rcParams["font.size"] = 14
    
    for i in range(0, len(nx)):
        
        # Set x points
        x = np.linspace(xmin, xmax, nx[i])
        
        # Initial condition
        T = np.zeros(len(x))
        for j in range(1, len(k), 2):
            T += 4/(j*np.pi)*np.sin(0.1*j*np.pi*x)
        
        T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, dx[i], dt[i], nt[i])
        
        T_analy = np.zeros(len(x))
        for j in range(1, len(k), 2):
            T_analy += 4/(j*np.pi)*np.exp(-K*(0.1*j*np.pi)**2*endTime)*np.sin(0.1*j*np.pi*x)
              
        # Calculate l_2 error norm
        A = np.sqrt(np.sum(dx[i]*(T_FTCS - T_analy)**2))
        B = np.sqrt(np.sum(dx[i]*T_analy**2))
        l_2_error_norm[i] = A/B 
    
    # Find linear regression line
    def func1(x, a, b):
        return a*x+b
    
    paramsx, _ = curve_fit(func1, np.log(dx), np.log(l_2_error_norm))
    paramst, _ = curve_fit(func1, np.log(dt), np.log(l_2_error_norm))
    
    # Set text of linear regression line
    equation_dx = f'$log(Error) = {paramsx[0]:.2f}log(\Delta x){paramsx[1]:.2f}$'
    slope_dx_text = f'Accuracy in Space = {paramsx[0]:.2f}'
    combined_dx_text = f'{equation_dx}\n{slope_dx_text}'
    equation_dt = f'$log(Error) = {paramst[0]:.2f}log(\Delta t){paramst[1]:.2f}$'
    slope_dt_text = f'Order of time = {paramst[0]:.2f}'
    combined_dt_text = f'{equation_dt}\n{slope_dt_text}'
    
    # Plot settings
    plt.plot(dx, l_2_error_norm, label='$l_2$ Error Norm and $\Delta x$', color='blue', marker='o', markersize = 5)
    plt.plot(dt, l_2_error_norm, label='$l_2$ Error Norm and $\Delta t$', color='red', marker='s', markersize = 5)
    plt.legend(fontsize=20, loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$\Delta x$ (m): blue, $\Delta t$ (s): red', fontsize=24)
    plt.ylabel('$l_2$ Error Norm', fontsize=24)
    plt.title('$l_2$ Error Norm', fontsize=30, y=1.02)
    plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
    bbox_props = dict(boxstyle='round', facecolor='lightblue', edgecolor='black', alpha=0.5)
    plt.text(0.56, 0.48, combined_dt_text, transform=plt.gca().transAxes, fontsize = 20, \
             bbox=bbox_props, multialignment='center')
    plt.text(0.06, 0.68, combined_dx_text, transform=plt.gca().transAxes, fontsize = 20, \
             bbox=bbox_props, multialignment='center')
    plt.savefig('diffusion 5.png')
    plt.show()
    
    # Print Order of time and Accuracy in space
    print(f'Order of Time is {paramst[0]:.2f}, Accuracy in Space is {paramsx[0]:.2f}')
    
main()

    # nt_ori = endTime/dt

    # plt.figure(figsize=(16, 12))
    # plt.plot(dx, nt_ori-(nt_ori).astype(int), label='l_2 error norm', color='blue', marker='o', markersize = 7)
    # plt.plot(dx, nt_ori-nt, label='l_2 error norm', color='red', marker='o', markersize = 7)

    # plt.xscale('log')
    # plt.show()

        # plt.plot(x, T_FTCS, label='FTCS', color='blue', marker='+')
        # plt.plot(x, T_analy, label='Analytic', color='r', linestyle='dashed')
