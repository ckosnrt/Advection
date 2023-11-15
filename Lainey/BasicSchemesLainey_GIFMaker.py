# MTMW12 Introduction to Numerical Modelling
# Assignment 4: Advection
# Basic Schemes Gif Maker
# Student number: 31827822

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from BasicSchemesLainey import BasicScheme

def animate(nt, ax, x, nx, dt, a, b, c, u, ICnumber):
    ax.clear()
    result, analytic, initial = BasicScheme(x, nx, nt+1, dt, a, b, c, u,
                                            ICnumber)
    res, = plt.plot(x, result, color='b', label='FTCS')
    ana, = plt.plot(x, analytic, color='r', label='Analytic')
    init, = plt.plot(x, initial, color='g', label=f'I.C. {ICnumber}')
    plt.title(u"Advection for {} timesteps".format(nt+1))
    plt.xlabel('X')
    plt.ylabel('$\Phi$')
    plt.xlim([0,1])
    plt.ylim([-8,9])
    plt.legend(loc='upper right')
    return res, ana, init

def gifmaker(nt, ICnumber):
    fig, ax = plt.subplots()
    # The variables kept constant:
    xmin = 0
    xmax = 1
    a = 0.1
    b = 0.5
    c = 0.4
    nx = 40
    u = 1 
    dx = (xmax - xmin)/(nx-1) 
    dt = c*dx/u
    x = np.linspace(xmin, xmax, nx) 
    
    ani = FuncAnimation(fig, animate, blit=True, interval=40, repeat=True,
                        frames=nt, fargs=(ax, x, nx, dt, a, b, c, u, ICnumber,)
                        )    
    ani.save("InitialCondition{}For{}TimestepsLainey.gif".format(ICnumber, nt),
             dpi=300, writer=PillowWriter(fps=25))
   
# Make gifs for initial conditions 1 and 2. 
gifmaker(nt=100, ICnumber=1)
gifmaker(nt=100, ICnumber=2)




