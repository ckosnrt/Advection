
# Outer code for setting up the diffusion problem, calculate and plot
import matplotlib.pyplot as plt
import numpy as np
# read in all the schemes, initial conditions and other helper code
from diffusionSchemes import *

def main_nt():
    """
    Simulate and visualize the diffusion process using FTCS and BTCS schemes with fixed time steps.
    Visualizes temperature profiles and differences.

    """
    # Parameters
    zmin = 100                  # Start of model domain (m)                             
    zmax = 1e3                  # End of model doman (m)                                
    nz = 21                     # Number of grid points, including both ends           
    endTime = 1200           # Number of seconds of the whole simulation
    nt = 20                   # Number of time steps taken to get the the endTime
    K = 1.                      # The diffusion coefficient (m^2/s)                     
    Tinit = 293.                # The initial conditions
    Q = -1.5/86400              # The heating rate
    
    # Other derived parameters
    dt = endTime/nt                 # The time step
    dz = (zmax - zmin) / (nz - 1)   # The grid spacing
    d = K * dt / dz**2              # Non-dimensional diffusion coefficient
    print('dx =', dz, 'dt =', dt, '\nnon-dimensional diffusion coeficient =', d)
    
    # Height points
    z = np.linspace(zmin, zmax, nz)
    
    # Initial condition
    T = Tinit * np.ones(nz)
    
    # Diffusion using FTCS and BTCS
    T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt)
    T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt)
   
    # Plot the solutions Figure 1
    fig, axs = plt.subplots(1, 2, figsize = (12,8))
    plt.rcParams["font.size"] = 12
    
    axs[0].plot(T_FTCS - Tinit, z,'r--', linewidth='1', label='FTCS', marker='^', markersize='4')
    axs[0].plot(T_BTCS - Tinit, z, 'c:', linewidth='1', label='BTCS', marker='o', markersize='4')
    axs[0].set_title(f'Timestep = {dt}')
    axs[1].plot(abs(T_BTCS - T_FTCS), z, 'r:', linewidth='1', label='FTCS-BTCS', marker='+', markersize='4')
    axs[1].set_title(f'Difference Value in Timestep = {dt}')
    
    for i in axs:
        i.set_ylim(0, 1000)
        i.legend(loc='best')
        i.set_xlabel('$T-T_0$ (K)')
        i.set_ylabel('z (m)')
        i.grid(which='both')
    plt.savefig('1Stable.png', bbox_inches='tight')
    fig.tight_layout()
    plt.show()
    

def main_nt_vary():
    """
    Simulate and visualize the diffusion process using FTCS and BTCS schemes with varying time steps.
    Visualizes temperature profiles and differences with a color-coded legend for time steps.
    
    """
    # Parameters
    zmin = 0                        # Start of model domain (m)
    zmax = 1e3                      # End of model domain (m)
    nz = 21                         # Number of grid points, including both ends
    nt_vary = np.arange(0, 10000, 500)   # Number of time steps
    K = 1.                          # The diffusion coefficient (m^2/s)
    Tinit = 293.                    # The initial conditions
    Q = -1.5/86400                  # The heating rate
    dt = 600                        # The time step
    dz = (zmax - zmin) / (nz - 1)   # The grid spacing
    d = K * dt / dz**2              # Non-dimensional diffusion coefficient
    print('dz =', dz, 'dt =', dt, '\non-dimensional diffusion coefficient =', d)

    # Height points
    z = np.linspace(zmin, zmax, nz)

    # Initial condition
    T = Tinit * np.ones(nz)
    
    # Diffusion using FTCS and BTCS, Plot the solutions
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    plt.rcParams["font.size"] = 12
    cmap = plt.get_cmap('turbo', len(nt_vary))
    
    for i in range(len(nt_vary)):
        T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_vary[i])
        T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_vary[i])
        
        color = cmap(i / len(nt_vary))
        axs[0].plot(T_FTCS - Tinit, z, linewidth=1, color=color, marker='^', markersize=4)
        axs[0].plot(T_BTCS - Tinit, z, linewidth=1, color=color, marker='o', markersize=4)
        axs[1].plot(T_BTCS - T_FTCS, z, 'c:', linewidth=1, color=color, marker='o', markersize=4)
        
    axs[0].set_title('Temperature value with Timestep Difference')
    axs[1].set_title('(BTCS - FTCS) Difference Value')
    for i in axs:
        i.set_ylim(0, 1000)
        i.set_xlabel('$T - T_0$ (K)')
        i.set_ylabel('z (m)')
        i.grid(which='both')
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=nt_vary.min(), vmax=nt_vary.max()))
    sm.set_array([])  
    
    # Use the `ax` parameter to specify the axes for the colorbar
    plt.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    plt.colorbar(sm, cax=cax, label='Number of Time Steps (nt)', ax = plt.gcf().get_axes()[0])

    plt.savefig('1Stable_nt_Vary.png', bbox_inches='tight')
    fig.tight_layout()
    plt.show()

main_nt()    
main_nt_vary()
