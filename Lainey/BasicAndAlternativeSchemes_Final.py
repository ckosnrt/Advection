# MTMW12 Introduction to Numerical Modelling
# Assignment 4: Advection
# Basic & Alternative Schemes: Functions & Plotting
# Student number: 31827822

import numpy as np
import matplotlib.pyplot as plt
import math

# Implement basic and alternative scheme, according to 2 possible sets of initial conditions.
def Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber):
    """
    Solves the one-dimensional linear advection equation analytically, for one
    of two possible sets of initial conditions, for a Forward-Time
    Backwards-Space scheme or one of five possible alternative schemes.
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
    initial: array : initial condition of phi.
    analytic : array : analytic solution of phi.
    phi : array : solution of phi using chosen scheme. 
    """
    phi = np.zeros(nx)
    phiNew = phi.copy()
    analytic = phi.copy()
    analyticNew = phi.copy()
    phiOld = phi.copy()     
    
    for n in range(nt): # Loop time.
        for j in range(nx):  # Loop space.
            if n==0 and a <= x[j] and x[j] < b: # Apply initial conditions.
                if ICnumber == 1: # Apply initial condition 1.
                    phi[j] = 0.5*(1-np.cos(2*np.pi*(x[j]-a)/(b-a)))
                    
                elif ICnumber == 2: # Apply initial condition 2.
                    phi[j] = 1
                    
                else:
                    print('Error: Choose boundary condition = 1 or 2.')
                                                          
            analyticx = x[j] - u*dt*n%nt # Analytical scheme. 
            # Note: the LHS conditon is different to the RHS in order to work for initial condition 2?
            if (a + u*dt*nt) <= x[j] and x[j] < (b + u*dt*n%nt):
                if ICnumber == 1: 
                    analyticNew[j] = 0.5*(1 - np.cos(2*np.pi*(analyticx - a)/(b - a)))
                if ICnumber == 2:
                    analyticNew[j] = 1
              
            if SNumber == 'FTCS': # FTCS scheme.
                phiNew[j] = phi[j] -0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx])  
 
            elif SNumber == 'CTCS': # CTCS scheme.
                if n==0:
                    phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx])
                else: 
                    phiNew[j] = phiOld[j] - c*(phi[(j+1)%nx] - phi[(j-1)%nx])
                    
            elif SNumber == 'FTBS': # FTBS scheme.
                phiNew[j] = phi[j] - c*(phi[j] - phi[(j-1)%nx])  

            elif SNumber == 'WB': # Warming and Beam scheme.
                phi_p = 0.5*(3 - c)*phi[(j)%nx] - 0.5*(1 - c)*phi[(j-1)%nx]
                phi_m = 0.5*(3 - c)*phi[(j-1)%nx] - 0.5*(1 - c)*phi[(j-2)%nx]
                phiNew[j] = phi[j] - (dt*u/dx)*(phi_p - phi_m)
            
            elif SNumber == 'LW': # Lax-Wendroff scheme.
                phi_p = 0.5*(1 + c)*phi[(j)%nx] + 0.5*(1 - c)*phi[(j+1)%nx]
                phi_m = 0.5*(1 + c)*phi[(j-1)%nx] + 0.5*(1 - c)*phi[(j)%nx]
                phiNew[j] = phi[j] - (dt*u/dx)*(phi_p - phi_m)
                
            elif SNumber == 'TVD': # Total Variation Diminishing scheme.                             
                r_plushalf = ((phi[j] - phi[(j-1)%nx])/(phi[(j+1)%nx] - phi[j]) if np.all((phi[(j+1)%nx] - phi[j]) != 0.) else 0.) 
                r_minushalf = ((phi[(j-1)%nx] - phi[(j-2)%nx])/(phi[j] - phi[(j-1)%nx]) if np.all((phi[j] - phi[(j-1)%nx]) != 0.) else 0.)            
                           
                Psi_plushalf = (r_plushalf + abs(r_plushalf))/(1 + abs(r_plushalf))
                Psi_minushalf = (r_minushalf + abs(r_minushalf))/(1 + abs(r_minushalf)) 
                
                phi_H_plushalf = 1/2*(1 + c)*phi[j] + 1/2*(1 - c)*phi[(j+1)%nx]
                phi_H_minushalf = 1/2*(1 + c)*phi[(j-1)%nx] + 1/2*(1 - c)*phi[j]
                
                phi_L_plushalf = phi[j] if u>=0 else phi[(j+1)%nx] 
                phi_L_minushalf = phi[(j-1)%nx] if u>=0 else phi[j]
                                              
                phi_plushalf = Psi_plushalf*phi_H_plushalf + (1 - Psi_plushalf)*phi_L_plushalf
                phi_minushalf = Psi_minushalf*phi_H_minushalf + (1 - Psi_minushalf)*phi_L_minushalf 
                
                phiNew[j] = phi[j] - c*(phi_plushalf - phi_minushalf)
                
            #elif SNumber == 'SL': # Semi-Langragian scheme.
                # YET TO IMPLEMENT
            
            else: 
                print('Enter desired alternative scheme: FTCS/CTCS/FTBS/WB/LW/TVD/SL.')
                       
        analytic = analyticNew.copy()
        phiOld = phi.copy()
        phi = phiNew.copy()
        
    return analytic, phi

def plots(nt, ICnumber):
    xmin = 0
    xmax = 1
    a = 0.1
    b = 0.5
    c = 0.4
    nx = 41
    u = 0.1
    dx = (xmax - xmin)/(nx-1) 
    dt = c*dx/u
    x = np.linspace(xmin, xmax, nx)     
        
    analytic, FTCS = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='FTCS')
    analytic, WB = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='WB')
    analytic, LW = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='LW')
    ianalytic, TVD = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='TVD')
    #analytic, SL = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='SL')

    plt.plot(x, analytic, color='b', label='Analytic')
    plt.plot(x, FTCS, color='r', label='FTCS')
    plt.plot(x, WB, color='purple', label='WB')
    plt.plot(x, LW, color='orange', label='LW')
    plt.plot(x, TVD, color='green', label='TVD')    
    #plt.plot(x, SL, color='pink', label='SL')

    plt.legend()
    plt.title(u"Advection for {} Timesteps for Initial Condition {}"
              .format(nt,ICnumber))
    plt.xlabel('X')
    plt.ylabel('$\Phi$')
    plt.show()
    return dx, FTCS, analytic

dx, FTCS, analytic = plots(nt=8, ICnumber=1)

################ IGNORE ########################
def L2Error(dx, approx, analytic):
    num = np.sqrt(np.sum(dx * (approx - analytic)**2))
    denom = np.sqrt(np.sum(dx * analytic**2))
    L2Err = num/denom
    return L2Err

l2err_FTCS = L2Error(dx, FTCS, analytic)
#print(l2err_FTCS)
    

    
