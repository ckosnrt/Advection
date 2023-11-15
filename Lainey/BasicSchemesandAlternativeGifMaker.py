# MTMW12 Introduction to Numerical Modelling
# Assignment 4: Advection
# Basic Schemes Gif Maker
# Student number: 31827822

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from BasicSchemesAndAlternatives import Schemes

def animate(nt, ax, x, nx, dx, dt, a, b, c, u, ICnumber):
    ax.clear()
    
    initial, analytic, FTCS = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='FTCS')
    initial, analytic, WB = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='WB')
    initial, analytic, LW = Schemes(x, nx, dx, nt, dt, a, b, c, u, ICnumber, SNumber='LW')
          
    #init, = plt.plot(x, initial, color='g', label='Initial')
    ana, = plt.plot(x, analytic, color='b', label='Analytic')
    ftcs, = plt.plot(x, FTCS, color='r', label='FTCS')
    wb, = plt.plot(x, WB, color='purple', label='WB')
    lw, = plt.plot(x, LW, color='orange', label='LW')
    
    plt.title(u"Advection for {} timesteps and I.C.={}".format(nt, ICnumber))
    plt.xlabel('X')
    plt.ylabel('$\Phi$')
    plt.xlim([0,1])
    plt.ylim([-0.4,1.5])
    plt.legend(loc='upper right')
    return ana, ftcs, wb, lw #,init, 

def gifmaker(nt, ICnumber):
    fig, ax = plt.subplots()
    # The variables kept constant:
    xmin = 0
    xmax = 1
    a = 0.1
    b = 0.5
    c = 0.4
    nx = 101
    u = 1 
    dx = (xmax - xmin)/(nx-1) 
    dt = c*dx/u
    x = np.linspace(xmin, xmax, nx) 
    
    ani = FuncAnimation(fig, animate, blit=True, interval=40, repeat=True,
                        frames=nt, fargs=(ax, x, nx, dx, dt, a, b, c, u, ICnumber,))    
    ani.save("InitialCondition{}For{}TimestepsBasicAndAlternative.gif".format(ICnumber, nt),
             dpi=300, writer=PillowWriter(fps=25))
   
# Make gifs for initial conditions 1 and 2. 
gifmaker(nt=100, ICnumber=1)
#gifmaker(nt=150, ICnumber=2)






