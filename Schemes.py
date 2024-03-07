# MTMW12 Introduction to Numerical Modelling
# Assignment 5: Advection
# Basic & Alternative Schemes: Define Function
# Student number: 31837045, 31832278

# Numerical schemes for simulating diffusion for outer code diffusion.py
import numpy as np
import math

def advection_analytic_1(x, t=0., a=0.1, b=0.5, u=0.1):
    """
    Calculate analytic solution with initial condition 1    
    Parameters
    −−−−−−−−−−
    x : array. The positions of points in space.
    a : float. integer. Initial condition constant. 
    b : float. integer. Initial condition constant.
    u : float. integer. Wind speed constant. 
    t : float. target time
    Returns
    −−−−−−−−−−
    phi : array : analytic solution of phi. 
    """
    L = (x[-1]-x[0]) + (x[1]-x[0])
    
    phi = np.where((a<=(x-u*t)%L)&((x-u*t)%L<b), 1., 0.)
    return phi

def advection_analytic_cos(x, t=0., a=0.1, b=0.5, u=0.1):
    """
    Calculate analytic solution with initial condition cos    
    Parameters
    −−−−−−−−−−
    x : array. The positions of points in space.
    a : float. integer. Initial condition constant. 
    b : float. integer. Initial condition constant.
    u : float. integer. Wind speed constant. 
    t : float. target time
    Returns
    −−−−−−−−−−
    phi : array : analytic solution of phi. 
    """
    
    L = (x[-1]-x[0]) + (x[1]-x[0])
    
    phi = np.where((a<=(x-u*t)%L)&((x-u*t)%L<b), 0.5*(1 - np.cos(2*np.pi*((x-u*t)%L-a)/(b-a))), 0.)
    
    return phi

def advection_FTBS(phi, nt=10, c=0.4):
    """
    Calculate analytic solution with initial condition phi    
    Parameters
    −−−−−−−−−−
    phi : array. Initial condition.
    nt : float. integer. Number of time steps.
    c : float. The Courant number
        
    Returns
    −−−−−−−−−−
    phi : array : FTBS solution of phi. 
    """

    # FTBS for all time steps
    for it in range(nt):
        phiNew = phi - c*(phi - np.roll(phi, 1))
        
        phi = phiNew.copy()
    
    return phi

def advection_FTCS(phi, nt=10, c=0.4):
    """
    Calculate analytic solution with initial condition phi  
    Parameters
    −−−−−−−−−−
    phi : array. Initial condition.
    nt : float. integer. Number of time steps.
    c : float. The Courant number
        
    Returns
    −−−−−−−−−−
    phi : array : FTCS solution of phi. 
    """
    
    # FTCS for all time steps
    for it in range(nt):
        phiNew = phi - c/2*(np.roll(phi, -1) - np.roll(phi, 1))
        
        phi = phiNew.copy()
    
    return phi

def advection_CTCS(phi,nt=10, c=0.4):
    """
    Calculate analytic solution with initial condition phi 
    Parameters
    −−−−−−−−−−
    phi : array. Initial condition.
    nt : float. integer. Number of time steps.
    c : float. The Courant number
        
    Returns
    −−−−−−−−−−
    phi : array : CTCS solution of phi. 
    """

    # new time-step array for phi
    phiOld = phi.copy()
    
    # FTCS for first step over all internal points
    if nt > 0:
        phiNew = phi - c/2*(np.roll(phi, -1) - np.roll(phi, 1))
        phi = phiNew.copy()
    
    # CTCS for all time steps
    for it in range(1, nt):
                
        # CTCS for rest all time step and over all internal points
        phiNew = phiOld - c*(np.roll(phi, -1) - np.roll(phi, 1))
           
        # Update phi for next time-step
        phiOld = phi.copy()
        phi = phiNew.copy()
                
    return phi
    
def advection_lax_wendroff(phi, nt=10, c=0.4):
    """
    Calculate analytic solution with initial condition phi   
    Parameters
    −−−−−−−−−−
    phi : array. Initial condition.
    nt : float. integer. Number of time steps.
    c : float. The Courant number
        
    Returns
    −−−−−−−−−−
    phi : array : Lax wendroff solution of phi. 
    """
    
    # Lax-Wendroff for all time steps
    for it in range(nt):
        phiNew = phi - c*(1/2*(1+c)*phi + 1/2*(1-c)*np.roll(phi, -1)\
                          - (1/2*(1+c)*np.roll(phi, 1) + 1/2*(1-c)*phi))
        phi = phiNew.copy()
    
    return phi

def advection_Warming_beam(phi, nt=10, c=0.4):
    """
    Calculate analytic solution with initial condition phi   
    Parameters
    −−−−−−−−−−
    phi : array. Initial condition.
    nt : float. integer. Number of time steps.
    c : float. The Courant number
        
    Returns
    −−−−−−−−−−
    phi : array : Warming Beam solution of phi. 
    """
    # Warming-beam for all time steps
    for it in range(nt):
        phiNew = phi - c*(1/2*(3-c)*phi - 1/2*(1-c)*np.roll(phi, 1)\
                          - (1/2*(3-c)*np.roll(phi, 1) - 1/2*(1-c)*np.roll(phi, 2)))
        phi = phiNew.copy()
        
    return phi

def advection_TVD(phi, nt=10, c=0.4):
    """
    Calculate analytic solution with initial condition phi   
    Parameters
    −−−−−−−−−−
    phi : array. Initial condition.
    nt : float. integer. Number of time steps.
    c : float. The Courant number
    u : float. integer. Wind speed constant. 
        
    Returns
    −−−−−−−−−−
    phi : array : TVD solution of phi. 
    """
    # TVD for all time steps
    
    nx = len(phi)
    r_plushalf, r_minushalf = np.zeros(nx), np.zeros(nx)
    
    for it in range(nt):
        
        for ix in range(nx):
            if phi[(ix+1)%nx] - phi[ix%nx] == 0.:
                r_plushalf[ix] = 0.
            else:
                r_plushalf[ix] = (phi[ix%nx] - phi[(ix-1)%nx])/(phi[(ix+1)%nx]-phi[ix%nx])

            if phi[ix%nx] - phi[(ix-1)%nx] == 0.:
                r_minushalf[ix] = 0.
            else:
                r_minushalf[ix] = (phi[(ix-1)%nx] - phi[(ix-2)%nx])/(phi[(ix)%nx]-phi[(ix-1)%nx])
                       
        Psi_plushalf = (r_plushalf+abs(r_plushalf))/(1+abs(r_plushalf)) # for j+1/2
        Psi_minushalf = (r_minushalf+abs(r_minushalf))/(1+abs(r_minushalf)) # for j-1/2
                
        phi_H_plushalf = 1/2*(1+c)*phi + 1/2*(1-c)*np.roll(phi, -1) # for j+1/2
        phi_H_minushalf = 1/2*(1+c)*np.roll(phi, 1) + 1/2*(1-c)*phi # for j-1/2
                
        phi_L_plushalf = (phi if c>=0 else np.roll(phi, -1)) # for j+1/2
        phi_L_minushalf = (np.roll(phi, 1) if c>=0 else phi) # for j-1/2
                
        phi_plushalf = Psi_plushalf*phi_H_plushalf + (1 - Psi_plushalf)*phi_L_plushalf # for j+1/2
        phi_minushalf = Psi_minushalf*phi_H_minushalf + (1 - Psi_minushalf)*phi_L_minushalf # for j-1/2
                
        phiNew = phi - c*(phi_plushalf-phi_minushalf)
        
        phi = phiNew.copy()
    
    return phi

def advection_Semi_lagrangian(phi, nt=10, c=0.4):
    """
    Calculate analytic solution with initial condition phi   
    Parameters
    −−−−−−−−−−
    phi : array. Initial condition.
    nt : float. integer. Number of time steps.
    c : float. The Courant number
        
    Returns
    −−−−−−−−−−
    phi : array : Semi-lagrangian solution of phi. 
    """
    nx = len(phi)
    phiNew = phi.copy()
    
    # Semi-lagrangian for all time steps
    for it in range(nt):
        for j in range(nx):
            
            # Find index at previous time step.
            k = np.floor(j - c).astype(int)
            b = j - k - c
                                
            # Cubic lagrangian interpolation
            phiNew[j] = -1/6*b*(1-b)*(2-b)*phi[(k-1)%nx] + 1/2*(1+b)*(1-b)*(2-b)*phi[k%nx] +\
                        1/2*(1+b)*b*(2-b)*phi[(k+1)%nx] - 1/6*(1+b)*b*(1-b)*phi[(k+2)%nx]
                                    
        phi = phiNew.copy()
        
    return phi

def conservation(phi, dx):
    nx = len(phi)
    cons = 0. 
    
    # Calculate Conservation
    for j in range(1, nx):
        cons += dx*phi[j]**2
        
    return cons