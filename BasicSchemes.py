# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:14:17 2023

@author: nowli
"""

# Numerical schemes for simulating diffusion for outer code diffusion.py
import numpy as np
import math

def advection_analytic_1(x, a, b, u, t):
    phi = np.where((a<=(x-u*t)%1)&((x-u*t)%1<b), 1., 0.)
    return phi

def advection_analytic_cos(x, a, b, u, t):
    phi = np.where((a<=(x-u*t)%1)&((x-u*t)%1<b), 0.5*(1 - np.cos(2*np.pi*((x-u*t)%1-a)/(b-a))), 0.)
    return phi

def initial_condition_1(x, a, b, u):
    phi = advection_analytic_1(x, a, b, u, 0)
    return phi

def initial_condition_cos(x, a, b, u):
    phi = advection_analytic_cos(x, a, b, u, 0)
    return phi

def advection_FTBS(phi, nt, c):
    
    for it in range(nt):
        phiNew = phi - c*(phi - np.roll(phi, 1))
        
        phi = phiNew.copy()
    
    return phi

def advection_FTCS(phi, nt, c):
    
    for it in range(nt):
        phiNew = phi - c/2 * (np.roll(phi, -1) - np.roll(phi, 1))
        
        phi = phiNew.copy()
    
    return phi

def advection_CTCS(phi, nt, c):

    # new time-step array for phi
    phiOld = phi.copy()
    
    # CTCS for all time steps
    for it in range(nt):
        
        # FTCS for first step over all internal points
        if it == 0:
            phiNew = phi - c/2*(np.roll(phi, -1) - np.roll(phi, 1))
                    
        # CTCS for rest all time step and over all internal points
        phiNew = phiOld - c*(np.roll(phi, -1) - np.roll(phi, 1))
            
        # Update phi for next time-step
        phiOld, phi = phi.copy(), phiNew.copy()

    return phi
    
def lax_wendroff(phi, nt, c):
        
    for it in range(nt):
        phiNew = phi - c*(1/2*(1+c)*phi + 1/2*(1-c)*np.roll(phi, -1)\
                          - (1/2*(1+c)*np.roll(phi, 1) + 1/2*(1-c)*phi))
        phi = phiNew.copy()
    
    return phi

def warming_beam(phi, nt, c):
        
    for it in range(nt):
        phiNew = phi - c*(1/2*(3-c)*phi - 1/2*(1-c)*np.roll(phi, 1)\
                          - (1/2*(3-c)*np.roll(phi, 1) - 1/2*(1-c)*np.roll(phi, 2)))
        phi = phiNew.copy()
        
    return phi

def advection_TVD(phi, nt, c, u):
        
    for it in range(nt):
                
        r_plushalf = ((phi - np.roll(phi, 1))/(np.roll(phi, -1) - phi) if np.all((np.roll(phi, -1) - phi) != 0.) else 0.) # for j+1/2
        r_minushalf = ((np.roll(phi, 1) - np.roll(phi, 2))/(phi - np.roll(phi, 1)) if np.all((phi - np.roll(phi, 1)) != 0.) else 0.) # for j-1/2
                
        Psi_plushalf = (r_plushalf+abs(r_plushalf))/(1+abs(r_plushalf)) # for j+1/2
        Psi_minushalf = (r_minushalf+abs(r_minushalf))/(1+abs(r_minushalf)) # for j-1/2
                
        phi_H_plushalf = 1/2*(1+c)*phi + 1/2*(1-c)*np.roll(phi, -1) # for j+1/2
        phi_H_minushalf = 1/2*(1+c)*np.roll(phi, 1) + 1/2*(1-c)*phi # for j-1/2
                
        phi_L_plushalf = (phi if u>=0 else np.roll(phi, -1)) # for j+1/2
        phi_L_minushalf = (np.roll(phi, 1) if u>=0 else phi) # for j-1/2
                
        phi_plushalf = Psi_plushalf*phi_H_plushalf + (1 - Psi_plushalf)*phi_L_plushalf # for j+1/2
        phi_minushalf = Psi_minushalf*phi_H_minushalf + (1 - Psi_minushalf)*phi_L_minushalf # for j-1/2
                
        phiNew = phi - c*(phi_plushalf-phi_minushalf)
        
        phi = phiNew.copy()
    
    return phi

def semi_lagrangian(phi, x, dx, nt, c):
    
    nx = len(phi)
    phiNew = phi.copy()
    
    for it in range(nt):
        for j in range(nx):
            k = np.floor(x[j]/dx - c).astype(int)
            b = np.round(x[j]/dx - k - c, decimals=12)
                    
            phiNew[j] = -1/6*b*(1-b)*(2-b)*phi[(k-1)%nx] + 1/2*(1+b)*(1-b)*(2-b)*phi[k%nx] +\
                        1/2*(1+b)*b*(2-b)*phi[(k+1)%nx] - 1/6*(1+b)*b*(1-b)*phi[(k+2)%nx]
                                    
        phi = phiNew.copy()
        
    return phi

def advection_CTCS(phi, nt, c):

    # new time-step array for phi
    phiOld = phi.copy()
    
    # CTCS for all time steps
    for it in range(nt):
        
        # FTCS for first step over all internal points
        if it == 0:
            phiNew = phi - c/2*(np.roll(phi, -1) - np.roll(phi, 1))
                    
        # CTCS for rest all time step and over all internal points
        phiNew = phiOld - c*(np.roll(phi, -1) - np.roll(phi, 1))
            
        # Update phi for next time-step
        phiOld, phi = phi.copy(), phiNew.copy()

    return phi

# def artificial_diffusion(phi, nt, c):
    
#     # new time-step array for phi
#     phiOld = phi.copy()
    
#     for it in range(nt):
        
#         if it == 0:
#             phiNew = 