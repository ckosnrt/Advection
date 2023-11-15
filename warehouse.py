def advection_analytic2(phi, nt, c):
    
    nx = len(phi)
    
    phi_analytic = phi.copy()
    
    for i in range(nx):
        phi_analytic[i] = phi[int((i-(c*nt)//1))%nx]
    
    return phi_analytic


def semi_lagrangian(phi, x, dx, nt, c):
    
    for it in range(nt):
        k = np.floor(x/dx - c)
        b = (x/dx - k - c)
        
        if it == 1:
            print(k)
            print(b)
        
        phiNew = -1/6*b*(1-b)*(2-b)*np.roll(phi, 1, axis=0) + 1/2*(1+b)*(1-b)*(2-b)*phi\
            + 1/2*(1+b)*b*(2-b)*np.roll(phi, -1, axis=0) - 1/6*(1+b)*b*(1-b)*np.roll(phi, -2, axis=0)
            
        phi = phiNew.copy()
        
    return phi
