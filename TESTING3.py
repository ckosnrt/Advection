# MTMW12 Introduction to Numerical Modelling
# Assignment 4: Advection
# Basic Schemes Function & Plotting
# Student number: 31827822

import numpy as np
import matplotlib.pyplot as plt
import math

# Implement FTBS scheme, according to 2 possible sets of initial conditions.
def BasicScheme(x, nx, dx, nt, dt, a, b, c, u, ICnumber):
    """
    Solves the one-dimensional linear advection equation analytically, as well 
    as with a Forward-Time Backwards-Space scheme, both for one of two possible
    sets of initial conditions.    
    Parameters
    −−−−−−−−−−
    x : array. The positions of points in space.
    nx : float. integer. Number of points in space. 
    nt : float. integer. Number of time steps.
    dt : float. integer. Length of time steps.
    a : float. integer. Initial condition constant. 
    b : float. integer. Initial condition constant.
    c : float. integer. Courant number.
    u : float. integer. Wind speed constant. 
    ICnumber: integer = 1 or 2. Number corresponding to the desired set of
                        initial conditions.
    Returns
    −−−−−−−−−−
    phi : array : FTCS solution of phi. 
    theory : array : analytic solution of phi.
    initial: array : initial condition of phi.
    """
    
    phi = np.zeros(nx)
    
    
    phiWB = phi.copy()
    phiFTCS = phi.copy()
    
    phiNewFTCS = phi.copy()
    phiNewWM = phi.copy()
    
    theory = np.zeros(nx) 
    theoryNew = theory.copy()
    initial = np.zeros(nx)

    for n in range(nt): # Loop time.
        for j in range(nx):  # Loop space.
            # At time=0, apply initial conditions where applicable.
            if n==0 and a <= x[j] and x[j] < b:
                if ICnumber == 1: # Apply initial condition 1.
                    phi[j] = 0.5*(1-np.cos(2*np.pi*(x[j]-a)/(b-a)))

                elif ICnumber == 2: # Apply initial condition 1.
                    phi[j] = 1
                    
                else:
                    print('Error: Choose boundary condition 1 or 2.')
                    
                initial[j] = phi[j] # Save the initial phi.
                
            # Determine FTCS scheme.
            phiNewFTCS[j] = phiFTCS[j] -0.5*c*(phiFTCS[(j+1)%nx] - phiFTCS[(j-1)%nx])
            
            # Determine the Warming and Beam Scheme.
            phi_p_wm = 0.5*(3-c)*phiWB[(j)%nx] - 0.5*(1-c)*phiWB[(j-1)%nx]
            phi_m_wm = 0.5*(3-c)*phiWB[(j-1)%nx] - 0.5*(1-c)*phiWB[(j-2)%nx]
            phiNewWM[j] = phiWB[j] - (dt*u/dx)*(phi_p_wm - phi_m_wm)
        
            # Determine analytical scheme. 
            theoryNew[j] = initial[math.floor(j-c*n)]
            
        #phi = phiNew.copy()
        phiFTCS = phiNewFTCS.copy()
        phiWB = phiNewWB.copy()
        theory = theoryNew.copy()

    return initial, phiFTCS, phiWB, theory

def plots(nt, ICnumber):
    xmin = 0
    xmax = 1
    a = 0.1
    b = 0.5
    c = 0.4
    nx = 40
    u = 1
    dx = (xmax - xmin)/(nx-1) # Vertical spacing.
    dt = c*dx/u
    x = np.linspace(xmin, xmax, nx)     
        
    initial, phiFTCS, phiWB, analytic = BasicScheme(x, nx, dx, nt, dt, a, b, c, u, ICnumber)
    plt.plot(x, phiFTCS, color='b', label='FTCS')
    plt.plot(x, phiWB, color='p', label='WB')
    plt.plot(x, analytic, color='r', label='Theory')
    #plt.plot(x, initial, color='g', label='Initial')
    plt.legend()
    plt.title(u"Advection for {} timesteps for initial condition {}"
              .format(nt,ICnumber))
    plt.xlabel('X')
    plt.ylabel('$\Phi$')
    plt.show()

plots(nt=10, ICnumber=1)
#plots(nt=10, ICnumber=2)




