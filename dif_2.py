
# Outer code for setting up the diffusion problem, calculate and plot
import matplotlib.pyplot as plt
import numpy as np
# read in all the schemes, initial conditions and other helper code
# from BasicSchemes import *

def advection_analytic_1(x, a, b, u, t):
    phi = np.where((a<=(x-u*t)%1)&((x-u*t)%1<b), 1., 0.)
    return phi

def initial_condition_1(x, a, b, u):
    phi = advection_analytic_1(x, a, b, u, 0)
    return phi

def advection_FTBS(phi, nt, c):
    for it in range(nt):
        phiNew = phi - c*(phi - np.roll(phi, 1))
        
        phi = phiNew.copy()
    
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

def total_var(phi, nt):
    nx = len(phi)
    phiNew = phi.copy()
    
    for it in range(nt):
        for j in range(nx-1):
            phi[j] = np.sum(np.abs(phi[j+1] - phi[j]))
            
        phi = phiNew.copy()
    return phi

def main():
    """Compare analytic solution with FTCS"""
    
    # Parameters
    xmin = 0.        # Start of model domain (m)
    xmax = 1.        # End of model domain (m)a
    nx = 40+1        # Number of grid points, including both ends
    nt = 30          # Number of time steps taken to get to the endTime
    c = 0.4          # The Courant number
    u = 0.1          # Wind speed(m/s)
    
    a = 0.1
    b = 0.5
    
    # Other derived parameters
    dx = (xmax - xmin)/(nx-1)    # The grid spacing
    dt = np.round(c*dx/u, decimals=12)                  # lengh of time step
    t = dt * nt            # Total time of the whole simulation
    print("dx =", dx, "dt =", dt, "The Courant number =", c)
        
    # x points
    x = np.linspace(xmin, xmax, nx)

    # Initial condition
    IC_1 = initial_condition_1(x, a, b, u)
    tv_IC1 = total_var(IC_1, nt)
    tv_FTBS = total_var(advection_FTBS(IC_1, nt, c), nt)
    tv_lax = total_var(lax_wendroff(IC_1, nt, c), nt)
    tv_wb = total_var(warming_beam(IC_1, nt, c), nt)
    tv_TVD = total_var(advection_TVD(IC_1, nt, c, u), nt)
    tv_semi = total_var(semi_lagrangian(IC_1, x, dx, nt, c), nt)
    
    
    plt.figure(figsize=(16,12))
    # plt.plot(x, IC_1, label='Initial Condition 1')
    plt.plot(x, advection_FTBS(IC_1, nt, c), label='FTBS')
    plt.plot(x, advection_TVD(IC_1, nt, c, u), label='TVD')
    plt.plot(x, tv_FTBS, label='TV_FTBS')
    plt.plot(x, tv_TVD, label='TV_TVD')
    plt.legend()
    plt.show()
    plt.savefig('TV_cko1.png', bbox_inches='tight')
    
    plt.figure(figsize=(16,12))
    # plt.plot(x, tv_IC1, label='TV_Initial Condition 1')
    # plt.plot(x, tv_wb, label='TV_WB')
    # plt.plot(x, warming_beam(IC_1, nt, c), label='Warming Beam')
    # plt.plot(x, tv_lax, label='TV_Lax')
    # plt.plot(x, lax_wendroff(IC_1, nt, c), label='Lax Wendroff')
    plt.plot(x, semi_lagrangian(IC_1, x, dx, nt, c), label='TV_Lax')
    plt.plot(x, tv_semi, label='TV_Lax')
    
    plt.legend()
    plt.show()
    plt.savefig('TV_cko2.png', bbox_inches='tight')
    
    plt.figure(figsize=(16,12))
    # plt.plot(x, tv_IC1, label='TV_Initial Condition 1')
    # plt.plot(warming_beam(IC_1, nt, c), tv_wb, label='TV_WB')
    # plt.plot(lax_wendroff(IC_1, nt, c), tv_lax, label='TV_Lax')
    # plt.plot(advection_FTBS(IC_1, nt, c), tv_FTBS, label='TV_FTBS')
    # plt.plot(advection_TVD(IC_1, nt, c, u), tv_TVD, label='TV_TVD')
    # plt.legend()
    # plt.show()
    # plt.savefig('TV_cko3.png', bbox_inches='tight')
    
    
main()




# def main():
#     """Compare analytic solution with FTCS"""
    
#     # Parameters
#     xmin = 0.        # Start of model domain (m)
#     xmax = 1.        # End of model domain (m)
#     nx = 40+1        # Number of grid points, including both ends
#     nt = 10          # Number of time steps taken to get to the endTime
#     c = 0.4          # The Courant number
#     u = 0.1          # Wind speed(m/s)
    
#     # Other derived parameters
#     dx = (xmax - xmin)/(nx-1)    # The grid spacing
#     dt = np.round(c*dx/u, decimals=15)                  # lengh of time step
#     endTime = dt * nt            # Total time of the whole simulation
#     print("dx =", dx, "dt =", dt, "The Courant number =", c)
        
#     # x points
#     x = np.linspace(xmin, xmax, nx)
#     print(x)
#     print(x[nx-1])

#     # Initial condition
#     # for i in range(nx):
        
#     # Diffusion using FTCS and analytic solution, Plot the solutions
    
# main()


# def main():
    
#     # Parameters
#     xmin = 0.        # Start of model domain (m)
#     xmax = 1.        # End of model domain (m)
#     nx = 100 + 1      # Number of grid points, including both ends
#     nt = 200          # Number of time steps taken to get to the endTime
#     c = 0.4          # The Courant number
#     u = 0.1          # Wind speed(m/s)
#     a = 0.1
#     b = 0.5
    
#     # Other derived parameters
#     dx = (xmax - xmin) / (nx - 1)  # The grid spacing
#     dt = np.round(c * dx / u, decimals=15)  # length of time step
#     endTime = dt * nt  # Total time of the whole simulation
#     print("dx =", dx, "dt =", dt, "The Courant number =", c)
    
#     # x points
#     x = np.linspace(xmin, xmax, nx)
    
#     # Initial conditions
#     phi_1 = initial_condition_1(x, a, b, u)
#     phi_cos = initial_condition_cos(x, a, b, u)
    
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(16, 12))
#     line, = ax.plot(x, advection_analytic_1(x, a, b, u, 0), label='Analytic', marker='o')
#     line_TVD, = ax.plot(x, advection_TVD(phi_1, 0, c, u), label='TVD', marker='o')
#     line_lax, = ax.plot(x, lax_wendroff(phi_1, 0, c), label='Lax-Wendroff', marker='o')
#     line_warming, = ax.plot(x, warming_beam(phi_1, 0, c), label='Warming_beam', marker='o')
#     line_semi_lag, = ax.plot(x, semi_lagrangian(phi_1, x, dx, 0, c), label='Semi-lagrangian', marker='o')
    
#     # Set plot limits
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(-0.5, 1.5)
    
#     # Add legend
#     ax.legend()
    
#     def update(frame):
#         ax.set_title(f'Time Step: {frame + 1}, spacing: {nx}')
#         line.set_ydata(advection_analytic_1(x, a, b, u, dt*(frame + 1)))
#         line_TVD.set_ydata(advection_TVD(phi_1, frame + 1, c, u))
#         line_lax.set_ydata(lax_wendroff(phi_1, frame + 1, c))
#         line_warming.set_ydata(warming_beam(phi_1, frame + 1, c))
#         line_semi_lag.set_ydata(semi_lagrangian(phi_1, x, dx, frame + 1, c))
#         return line, line_TVD, line_lax, line_warming, line_semi_lag
    
#     # Create an animation
#     animation = FuncAnimation(fig, update, frames=range(nt), interval=5)
    
#     # Display the animated plot
#     plt.show()
    
#     # Save the animation to a file (optional)
#     animation.save('TVD_Lax_Warming_semilag.gif', writer='imagemagick')
    
#     return animation

# # Store the animation object
# animation = main()




















# def main_nt():
#     """
#     Simulate and visualize the diffusion process using FTCS and BTCS schemes with fixed time steps.
#     Visualizes temperature profiles and differences.

#     """
#     # Parameters
#     zmin = 100                  # Start of model domain (m)                             
#     zmax = 1e3                  # End of model doman (m)                                
#     nz = 21                     # Number of grid points, including both ends           
#     endTime = 1200           # Number of seconds of the whole simulation
#     nt = 20                   # Number of time steps taken to get the the endTime
#     K = 1.                      # The diffusion coefficient (m^2/s)                     
#     Tinit = 293.                # The initial conditions
#     Q = -1.5/86400              # The heating rate
    
#     # Other derived parameters
#     dt = endTime/nt                 # The time step
#     dz = (zmax - zmin) / (nz - 1)   # The grid spacing
#     d = K * dt / dz**2              # Non-dimensional diffusion coefficient
#     print('dx =', dz, 'dt =', dt, '\nnon-dimensional diffusion coeficient =', d)
    
#     # Height points
#     z = np.linspace(zmin, zmax, nz)
    
#     # Initial condition
#     T = Tinit * np.ones(nz)
    
#     # Diffusion using FTCS and BTCS
#     T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt)
#     T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt)
   
#     # Plot the solutions Figure 1
#     fig, axs = plt.subplots(1, 2, figsize = (12,8))
#     plt.rcParams["font.size"] = 12
    
#     axs[0].plot(T_FTCS - Tinit, z,'r--', linewidth='1', label='FTCS', marker='^', markersize='4')
#     axs[0].plot(T_BTCS - Tinit, z, 'c:', linewidth='1', label='BTCS', marker='o', markersize='4')
#     axs[0].set_title(f'Timestep = {dt}')
#     axs[1].plot(abs(T_BTCS - T_FTCS), z, 'r:', linewidth='1', label='FTCS-BTCS', marker='+', markersize='4')
#     axs[1].set_title(f'Difference Value in Timestep = {dt}')
    
#     for i in axs:
#         i.set_ylim(0, 1000)
#         i.legend(loc='best')
#         i.set_xlabel('$T-T_0$ (K)')
#         i.set_ylabel('z (m)')
#         i.grid(which='both')
#     plt.savefig('1Stable.png', bbox_inches='tight')
#     fig.tight_layout()
#     plt.show()
    

# def main_nt_vary():
#     """
#     Simulate and visualize the diffusion process using FTCS and BTCS schemes with varying time steps.
#     Visualizes temperature profiles and differences with a color-coded legend for time steps.
    
#     """
#     # Parameters
#     zmin = 0                        # Start of model domain (m)
#     zmax = 1e3                      # End of model domain (m)
#     nz = 21                         # Number of grid points, including both ends
#     nt_vary = np.arange(0, 10000, 500)   # Number of time steps
#     K = 1.                          # The diffusion coefficient (m^2/s)
#     Tinit = 293.                    # The initial conditions
#     Q = -1.5/86400                  # The heating rate
#     dt = 600                        # The time step
#     dz = (zmax - zmin) / (nz - 1)   # The grid spacing
#     d = K * dt / dz**2              # Non-dimensional diffusion coefficient
#     print('dz =', dz, 'dt =', dt, '\non-dimensional diffusion coefficient =', d)

#     # Height points
#     z = np.linspace(zmin, zmax, nz)

#     # Initial condition
#     T = Tinit * np.ones(nz)
    
#     # Diffusion using FTCS and BTCS, Plot the solutions
#     fig, axs = plt.subplots(1, 2, figsize=(20, 8))
#     plt.rcParams["font.size"] = 12
#     cmap = plt.get_cmap('turbo', len(nt_vary))
    
#     for i in range(len(nt_vary)):
#         T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_vary[i])
#         T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_vary[i])
        
#         color = cmap(i / len(nt_vary))
#         axs[0].plot(T_FTCS - Tinit, z, linewidth=1, color=color, marker='^', markersize=4)
#         axs[0].plot(T_BTCS - Tinit, z, linewidth=1, color=color, marker='o', markersize=4)
#         axs[1].plot(T_BTCS - T_FTCS, z, 'c:', linewidth=1, color=color, marker='o', markersize=4)
        
#     axs[0].set_title('Temperature value with Timestep Difference')
#     axs[1].set_title('(BTCS - FTCS) Difference Value')
#     for i in axs:
#         i.set_ylim(0, 1000)
#         i.set_xlabel('$T - T_0$ (K)')
#         i.set_ylabel('z (m)')
#         i.grid(which='both')
    
#     # Create a colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=nt_vary.min(), vmax=nt_vary.max()))
#     sm.set_array([])  
    
#     # Use the `ax` parameter to specify the axes for the colorbar
#     plt.subplots_adjust(right=0.85)
#     cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#     plt.colorbar(sm, cax=cax, label='Number of Time Steps (nt)', ax = plt.gcf().get_axes()[0])

#     plt.savefig('1Stable_nt_Vary.png', bbox_inches='tight')
#     fig.tight_layout()
#     plt.show()

# main_nt()    
# main_nt_vary()


