import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from BasicSchemes import *

"""Compare analytic solution with FTCS"""

def main():
    
    # Parameters
    xmin = 0.        # Start of model domain (m)
    xmax = 1.        # End of model domain (m)
    nx = 100 + 1      # Number of grid points, including both ends
    nt = 200          # Number of time steps taken to get to the endTime
    c = 0.4          # The Courant number
    u = 0.1          # Wind speed(m/s)
    a = 0.1
    b = 0.5
    
    # Other derived parameters
    dx = (xmax - xmin) / (nx - 1)  # The grid spacing
    dt = np.round(c * dx / u, decimals=15)  # length of time step
    endTime = dt * nt  # Total time of the whole simulation
    print("dx =", dx, "dt =", dt, "The Courant number =", c)
    
    # x points
    x = np.linspace(xmin, xmax, nx)
    
    # Initial conditions
    phi_1 = initial_condition_1(x, a, b, u)
    phi_cos = initial_condition_cos(x, a, b, u)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))
    line, = ax.plot(x, advection_analytic_1(x, a, b, u, 0), label='Analytic', marker='o')
    line_TVD, = ax.plot(x, advection_TVD(phi_1, 0, c, u), label='TVD', marker='o')
    line_lax, = ax.plot(x, lax_wendroff(phi_1, 0, c), label='Lax-Wendroff', marker='o')
    line_warming, = ax.plot(x, warming_beam(phi_1, 0, c), label='Warming_beam', marker='o')
    line_semi_lag, = ax.plot(x, semi_lagrangian(phi_1, x, dx, 0, c), label='Semi-lagrangian', marker='o')
    
    # Set plot limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.5, 1.5)
    
    # Add legend
    ax.legend()
    
    def update(frame):
        ax.set_title(f'Time Step: {frame + 1}, spacing: {nx}')
        line.set_ydata(advection_analytic_1(x, a, b, u, dt*(frame + 1)))
        line_TVD.set_ydata(advection_TVD(phi_1, frame + 1, c, u))
        line_lax.set_ydata(lax_wendroff(phi_1, frame + 1, c))
        line_warming.set_ydata(warming_beam(phi_1, frame + 1, c))
        line_semi_lag.set_ydata(semi_lagrangian(phi_1, x, dx, frame + 1, c))
        return line, line_TVD, line_lax, line_warming, line_semi_lag
    
    # Create an animation
    animation = FuncAnimation(fig, update, frames=range(nt), interval=5)
    
    # Display the animated plot
    plt.show()
    
    # Save the animation to a file (optional)
    animation.save('TVD_Lax_Warming_semilag.gif', writer='imagemagick')
    
    return animation

# Store the animation object
animation = main()