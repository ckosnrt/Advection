
# Numericla schemes for simulationg diffusion for outer code diffusion.py
import numpy as np
# The linear algebra package for BTCS (for solving the matrix equation)
import numpy.linalg as la
import matplotlib.pyplot as plt

def FTCS_fixed_zeroGrad(phi, K, Q, dx, dt, nt):
    """
    Diffuses the initial condition phi with diffusion coefficient K, source
    term Q, grid spacing dx, time step dt for nt time steps. Fixed boundary
    conditions at phi[0] and zero gradient at phi[-1].
    Diffusion is calculated using the FTCS scheme to solve
    dphi / dt = K d^2phi / dx^2 + Q

    Parameters
    ----------
    phi : 1 darray. Profile to diffuse.
    K : float. The diffusion coeficient.
    Q : float. The source tem
    dx : float. The grid spacing
    dt : float. The time step
    nt : int. The number of time steps

    Returns
    -------
    phi: 1darray. phi after solution of the diffusion equation

    """
    nx = len(phi)
    d = K * dt / dx**2    # non-dimensional diffusion coefficient
    
    # New time-step array for phi
    phiNew = phi.copy()
    
    # FTCS for all time steps
    for it in range(nt):
        # Loop over all internal points
        for i in range(1, nx-1):
            phiNew[i] = phi[i] + d * (phi[i+1] + phi[i-1] - 2 * phi[i]) + Q * dt
        
        # Define boundary condition at z = 0
        phiNew[0] = 293.
        phiNew[nx - 1] = phiNew[nx - 2]
        
        # Update phi for next time-step
        phi = phiNew.copy()

    return phi

def BTCS_fixed_zeroGrad(phi, K, Q, dx, dt, nt):
    """
    Diffuses the initial condition phi with diffusion coefficient K, source
    term Q, grid spacing dx, time step dt for nt time steps. Fixed boundary
    conditions at phi[0] and zero gradient at phi[-1].
    Diffusion is calculated using the BTCS scheme to solve
    dphi / dt = K d^2phi / dx^2 = Q

    Parameters
    ----------
    phi : 1 darray. Profile to diffuse.
    K : float. The diffusion coeficient.
    Q : float. The source tem
    dx : float. The grid spacing
    dt : float. The time step
    nt : int. The number of time steps

    Returns
    -------
    phiNew: 1darray. phi after solution of the diffusion equation

    """
    nx = len(phi)
    d = K * dt / dx**2    # non-dimensional diffusion coefficient
    #d = 10
    # Array representing BTCS
    M = np.zeros([nx, nx])
    # Fixed value boundary conditions at the start
    M[0, 0] = 1
    # Zero gradient boundary conditions at the end
    M[-1, -1] = 1
    M[-1, -2] = -1
    
    # Other array elements
    for i in range(1, nx-1):
        M[i, i-1] = -d
        M[i, i] = 1 + 2 * d
        M[i, i+1] = -d
    
    # BTCS for all time steps
    for it in range(nt):
        #RHS vector
        RHS = phi + dt * Q
        #RHS for fixedd value boundary conditions at the start
        RHS[0] = phi[0]
        # RHS for zero gradient boundary conditions at end
        RHS[-1] = 0
        
        # Solve the matrix equation to update phi
        phi = la.solve(M, RHS)
    
    return phi   

#%% Q4, Q5

def steady_state_analytical(T, Q, K, zmin, zmax, nz):
    """
    Diffuses the initial condition phi with diffusion coefficient K, source
    term Q, grid spacing dx, time step dt for nt time steps. Fixed boundary
    conditions at phi[0] and zero gradient at phi[-1].
    Diffusion is calculated using the FTCS scheme to solve
    dphi / dt = K d^2phi / dx^2 + Q

    Parameters
    ----------
    phi : 1 darray. Profile to diffuse.
    K : float. The diffusion coeficient.
    Q : float. The source tem
    dx : float. The grid spacing
    dt : float. The time step
    nt : int. The number of time steps

    Returns
    -------
    phi: 1darray. phi after solution of the diffusion equation

    """
    X = np.linspace(zmin, zmax, nz)
    c1 = (Q * zmax) / K
    c2 = T + (Q * zmin**2) / (2 * K) - c1 * zmin

    Tsteady = -Q / (2 * K) * X**2 + c1 * X + c2

    return X, Tsteady


#%% Q4

def analytical_solutions(T, Q, K, Tinit, zmin, zmax, nz, dt, endTime):
    """
    Determine the heat equation's analytical solution over time for a given spatial domain and set of parameters.

    Parameters
    ----------
    T : 1 darray. Profile to diffuse.
    K : float. The diffusion coeficient.
    Q : float. The source tem
    dx : float. The grid spacing
    dt : float. The time step
    nt : int. The number of time steps
    Tinit : initial Temperature 
    zmin : bottom boundary
    zmax : top boundary
    nz : Number of grid points, including both ends
    dt : The time step
    endTime : Number of seconds of the whole simulation

    Returns
    -------
    tuple: A tuple containing time values (t) and the corresponding temperature profiles (history) over time.

    """
    history = []
    L = zmax - zmin
    t = np.arange(dt, endTime+dt, dt*100)
    T1 = 293
    T2 = 293

    X, Tsteady = steady_state_analytical(T, Q, K, zmin, zmax, nz)
    history=[]
    for ti in t:
        T = Tsteady.copy()
        for n in range(1, nz):
            T += (np.exp(-(n * np.pi/L)**2 * K * ti) * np.sin(n*np.pi*X/L))
        history.append(T)

    return t, history


#%% Q4, Q5

def FTCS_fixed_approximation(phi, K, dx, dt, nt):
    """
    To estimate the heat equation, use the fixed time-step FTCS (Forward-Time Central-Space) finite difference approach.

    Parameters
    ----------
    phi : 1 darray. Profile to diffuse.
    K : float. The diffusion coeficient.
    dx : float. The grid spacing
    dt : float. The time step
    nt : int. The number of time steps

    Returns
    -------
    phi: 1darray. phi after solution of the diffusion equation

    """
    nx = len(phi)
    d = K * dt / dx**2    # non-dimensional diffusion coefficient
    
    # New time-step array for phi
    phiNew = phi.copy()
    
    # FTCS for all time steps
    for it in range(nt):
        # Loop over all internal points
        for i in range(1, nx-1):
            phiNew[i] = phi[i] + d * (phi[i+1] + phi[i-1] - 2 * phi[i])
        
        # Define boundary condition at z = 0
        phiNew[0] = 0
        phiNew[-1] = 0
        # phiNew[nx - 1] = phiNew[nx - 2]

        phi = phiNew.copy()

    return phi


#%% Q5
def error_L2(dz, T_approx, T_analytic):
    """
    Determine the L2 norm error that exists between the analytical and approximate solutions.

    Parameters
    ----------
    dz : float. Spatial step size.
    T_approx : array. Approximate temperature profile
    T_analytic : array. Analytical temperature profile.

    Returns
    -------
    L2 : norm error between T_approx and T_analytic

    """
    L2_up = np.sqrt(np.sum(dz * (T_approx - T_analytic)**2))
    L2_dn = np.sqrt(np.sum(dz * T_analytic**2))
    L2 = L2_up / L2_dn
    return L2
    
    
    
    