# MTMW12 Introduction to Numerical Modelling
# Assignment 4: Advection
# Basic & Alternative Schemes: Functions & Plotting
# Student number: 31827822

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    phiOld = phi.copy()   
        
    cons = np.zeros(nt)
    tv = np.zeros(nt)

    for j in range(nx):
        if a <= x[j] and  b > x[j]:
            if ICnumber == 1: # Apply initial condition 1.
                phi[j] = 0.5*(1 - np.cos(2*np.pi*(x[j] - a)/(b - a))) 
            
            elif ICnumber == 2: # Apply initial condition 2.
                phi[j] = 1
                    
            else:
                print('Error: Choose boundary condition = 1 or 2.')
    
    for n in range(nt): # Loop time.
        sum1 = 0
        sum2 = 0
        for j in range(nx):  # Loop space.
            
            if SNumber == 'A':
                analyticx = (x[j] - u*dt*n)%1 # Analytical scheme. 
                d = a%1
                e = b%1
                if d <= analyticx and analyticx < e:
                    if ICnumber == 1: 
                        phiNew[j] = 0.5*(1 - np.cos(2*np.pi*(analyticx - d)/(e - d)))
                    elif ICnumber == 2:
                        phiNew[j] = 1
                else:
                    if ICnumber == 1:
                        phiNew[j] = 0
                    elif ICnumber == 2:
                        phiNew[j] = 0
                                     
            elif SNumber == 'FTCS': # FTCS scheme.
                phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx])  
 
            elif SNumber == 'CTCS': # CTCS scheme.
                if n==0:
                    phiNew[j] = phi[j] - 0.5*c*(phi[(j+1)%nx] - phi[(j-1)%nx])
                else: 
                    phiNew[j] = phiOld[j] - c*(phi[(j+1)%nx] - phi[(j-1)%nx])
                    
            elif SNumber == 'FTBS': # FTBS scheme.
                phiNew[j] = phi[j] - c*(phi[j] - phi[(j-1)%nx])  

            elif SNumber == 'WB': # Warming and Beam scheme.
                phi_p = 0.5*(3 - c)*phi[j] - 0.5*(1 - c)*phi[(j-1)%nx]
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
                
            elif SNumber == 'SL':
                # Find index at previous time step.               
                k = np.floor(j - c).astype(int)
                beta = np.round(j - k - c, decimals=20)
                        
                # Cubic lagrangian interpolation
                phiNew[j] = -1/6*beta*(1-beta)*(2-beta)*phi[(k-1)%nx] + 1/2*(1+beta)*(1-beta)*(2-beta)*phi[k%nx] +\
                            1/2*(1+beta)*beta*(2-beta)*phi[(k+1)%nx] - 1/6*(1+beta)*beta*(1-beta)*phi[(k+2)%nx]
            
            else: 
                print('Enter desired alternative scheme: FTCS/CTCS/FTBS/WB/LW/TVD/SL.')
        
            sum1 += phiNew[j]**2   
            sum2 += np.abs(phi[(j+1)%nx] - phi[j])

        cons[n] = dx*(sum1-phiNew[0]**2)
        tv[n] = sum2 + np.abs(phi[1]-phi[0])
    
        phiOld = phi.copy()
        phi = phiNew.copy()
    return phi, cons, tv


def PLOTS123(nt, ICnumber):
    xmin = 0
    xmax = 1
    a = 0.1
    b = 0.5
    c = 0.4
    nx = 40
    u = 0.1
    dx = (xmax - xmin)/(nx-1) 
    dt = c*dx/u
    x = np.linspace(xmin, xmax-dx, nx) 

    # DATA FOR PLOTS 1, 2 & 3
    FTCS, FTCScons, FTCStv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='FTCS')
    CTCS, CTCScons, CTCStv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='CTCS')
    FTBS, FTBScons, FTBStv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='FTBS')
    WB, WBcons, WBtv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='WB')
    LW,  LWcons, LWtv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='LW')
    TVD, TVDcons, TVDtv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='TVD')
    SL, SLcons, SLtv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='SL')
    A, Acons, Atv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='A')

    # PLOT 1 - Phi vs X
    plt.plot(x, FTCS, label='FTCS')
    plt.plot(x, CTCS, label='CTCS')
    plt.plot(x, FTBS, label='FTBS')
    plt.plot(x, WB, label='WB')
    plt.plot(x, LW, label='LW')
    plt.plot(x, TVD, label='TVD') 
    plt.plot(x, SL, label='SL') 
    plt.plot(x, A, label='Analytic')
    plt.legend()
    plt.title(u"Advection for {} Timesteps for Initial Condition {}"
              .format(nt,ICnumber))
    plt.xlabel('X')
    plt.ylabel('$\Phi$')
    plt.show()
    
    # PLOT 2 - Conservation vs NT
    xplot = np.linspace(0,nt-1,nt)  
    plt.plot(xplot, FTCScons, label='FTCS')
    plt.plot(xplot, CTCScons, label='CTCS')
    plt.plot(xplot, FTBScons, label='FTBS')
    plt.plot(xplot, WBcons, label='WB')
    plt.plot(xplot, LWcons, label='LW')
    plt.plot(xplot, TVDcons, label='TVD')
    plt.plot(xplot, SLcons, label='SL')
    plt.plot(xplot, Acons, label='Analytic')
    plt.legend()
    plt.title(u"Cons. of Higher Moments against Timesteps for Initial Condition {}"
              .format(ICnumber))
    plt.xlabel('Number of Timesteps')
    plt.ylabel('$V^{(j+1)}$')
    plt.show()

    # PLOT 3 - NT vs TV
    xplot = np.linspace(0,nt-1,nt)  
    plt.plot(xplot, FTCStv, label='FTCS')
    plt.plot(xplot, CTCStv, label='CTCS')
    plt.plot(xplot, FTBStv, label='FTBS')
    plt.plot(xplot, WBtv, label='WB')
    plt.plot(xplot, LWtv, label='LW')
    plt.plot(xplot, TVDtv, label='TVD')
    plt.plot(xplot, SLtv, label='SL')
    plt.plot(xplot, Atv, label='Analytic')
    plt.legend()
    plt.title(u"Total Variation against Timesteps for Initial Condition {}"
              .format(ICnumber))
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Total Variation')
    plt.show()
PLOTS123(nt=10, ICnumber=1)


def L2Error(dx, approx, analytic):   
    num = np.sqrt(np.sum(dx * (approx - analytic)**2))
    denom = np.sqrt(np.sum(dx * analytic**2))
    L2Err = np.abs(num/denom)
    return L2Err

def L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber):

    Err = np.zeros(nt)
    NT = np.arange(1, nt+1) # Start at timestep 1 up to desired number.

    for nt in NT:
        phi, cons, tv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber)
        analytic, cons, tv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='A')
        Err[nt-1] = L2Error(dx, phi, analytic)

    return NT, Err

def L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, SNumber):

    Err = np.zeros(nx)
    NX = np.arange(40, nx+40)
    DT = np.zeros(nx)

    for nx in NX:
        dx = (xmax - xmin)/(nx-1) 
        dt = c*dx/u
        x = np.linspace(xmin, xmax-dx, nx)
        phi, cons, tv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber)
        analytic, cons, tv = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='A')

        Err[nx-40] = L2Error(dx, phi, analytic)
        DT[nx-40] = dt

    return NX, Err

def linear(x, a, b):
    return a*x+b 

def PLOT4(ICnumber, nt):
    xmin = 0
    xmax = 1
    a = 0.1
    b = 0.5
    c = 0.4
    nx = 40
    u = 0.1
    dx = (xmax - xmin)/(nx-1) 
    dt = c*dx/u
    x = np.linspace(xmin, xmax-dx, nx)

    NT, FTCSErr = L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber='FTCS')
    NT, CTCSErr = L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber='CTCS')
    NT, FTBSErr = L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber='FTBS')
    NT, WBErr = L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber='WB')
    NT, LWErr = L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber='LW')
    NT, TVDErr = L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber='TVD')
    NT, SLErr = L2Err_NT(xmin, xmax, a, b, c, nx, u, dx, dt, x, nt, ICnumber, SNumber='SL')

    # PLOT 4
    plt.loglog(NT, FTCSErr, label='FTCS')
    plt.loglog(NT, CTCSErr, label='CTCS')
    plt.loglog(NT, FTBSErr, label='FTBS')
    plt.loglog(NT, WBErr, label='WB')
    plt.loglog(NT, LWErr, label='LW')
    plt.loglog(NT, TVDErr, label='TVD')
    plt.loglog(NT, SLErr, label='SL')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.title(u"L2 Error Norm Vs Number of Time Steps for Initial Condition {}"
              .format(ICnumber))
    plt.xlabel('Log(Number of Timesteps)')
    plt.ylabel('Log(L2 Error Norm)')
    plt.show()     
PLOT4(ICnumber=1, nt=85) 

def PLOT5(ICnumber, nx):
    xmin = 0
    xmax = 1
    a = 0.1
    b = 0.5
    c = 0.4
    u = 0.1
    nt = 40

    NX, FTCSErr = L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, 'FTCS') 
    NX, CTCSErr = L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, 'CTCS')    
    NX, FTBSErr = L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, 'FTBS')    
    NX, WBErr = L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, 'WB')
    NX, LWErr = L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, 'LW')
    NX, TVDErr = L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, 'TVD')
    NX, SLErr = L2Err_NX(xmin, xmax, a, b, c, nx, u, nt, ICnumber, 'SL')

    # PLOT 5
    plt.loglog(NX, FTCSErr, label='FTCS')
    plt.loglog(NX, CTCSErr, label='CTCS')
    plt.loglog(NX, FTBSErr, label='FTBS')
    plt.loglog(NX, WBErr, label='WB')
    plt.loglog(NX, LWErr, label='LW')
    plt.loglog(NX, TVDErr, label='TVD')
    plt.loglog(NX, SLErr, label='SL')
    plt.legend()
    plt.title(u"L2 Error Norm Vs Number of Space Steps for Initial Condition {}"
              .format(ICnumber))
    plt.xlabel('Log(Number of Space Steps)')
    plt.ylabel('Log(L2 Error Norm)')
    plt.show()     
    
    # SLOPES.
    paramsFTCS, _ = curve_fit(linear, np.log(NX), np.log(FTCSErr))
    paramsCTCS, _ = curve_fit(linear, np.log(NX), np.log(CTCSErr))
    paramsFTBS, _ = curve_fit(linear, np.log(NX), np.log(FTBSErr))
    paramsWB, _ = curve_fit(linear, np.log(NX), np.log(WBErr))
    paramsLW, _ = curve_fit(linear, np.log(NX), np.log(LWErr))
    paramsTVD, _ = curve_fit(linear, np.log(NX), np.log(TVDErr))
    paramsSL, _ = curve_fit(linear, np.log(NX), np.log(SLErr))

    print(f'FTCS Accuracy in Space = {paramsFTCS[0]:.2f}')
    print(f'CTCS Accuracy in Space = {paramsCTCS[0]:.2f}')
    print(f'FTBS Accuracy in Space = {paramsFTBS[0]:.2f}')
    print(f'WB Accuracy in Space = {paramsWB[0]:.2f}')
    print(f'LW Accuracy in Space = {paramsLW[0]:.2f}')
    print(f'TVD Accuracy in Space = {paramsTVD[0]:.2f}')
    print(f'SL Accuracy in Space = {paramsSL[0]:.2f}')
PLOT5(ICnumber=1, nx=10)  






