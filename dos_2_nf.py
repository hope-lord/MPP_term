# libraries 
import numpy as np
from scipy.optimize import brentq
import matplotlib.pylab as plt
from concurrent.futures import ProcessPoolExecutor as PPE

# The value of t (hopping constant) and a (lattice constant) is taken 1

def delta(n,m): # Delta function
    return 1 if n==m else 0

def U(u1,u2,n):  # Potential term
    return u1*delta(1,n)+u2*delta(2,n)

def f(k):
    return 2*np.cos(k/2)

def find_a1(om,k,u1,u2,N,eta=1e-2):
    if (k!=np.pi):
        l = -(om+1j*eta-U(u1,u2,N))/f(k)
        i = N-1
        while (i>1):
            l = -(om+1j*eta-U(u1,u2,i))/f(k)-1/l
            i -= 1
        l = 1.0/(om+1j*eta-U(u1,u2,1)+f(k)/l)
        return l
    else:
        return 1.0/(om+1j*eta-u1)

def DOS(om,k,u1,u2,N,eta=1e-2):
    return -find_a1(om,k,u1,u2,N,eta).imag/np.pi


# print(U(1,2,2))
if __name__ == '__main__':
    U1 = []
    U2 = 10**np.linspace(-1,2,50)
    eta = 1e-2
    N = 1000
    omega = np.linspace(-7,0,1000)
    ar = np.ones_like(omega,dtype=int)


    for u2 in U2: 

        def func1(x):
            dos = []
            with PPE() as exe:
                dos = exe.map(DOS,omega,ar*0,ar*x,ar*u2,ar*N,ar*eta)
            dos = list(dos)
            index = np.argmax(dos)
            return omega[index]

        func = lambda x : func1(x) + 4
        U1.append(brentq(func,-1.8,-4,xtol=1e-2, maxiter=20))
        
    U1 = -np.array(U1)
    plt.semilogy(U1,U2,'ro-')
    plt.title('$N_f = 2$')
    plt.xlabel('$-U_1 /t$')
    plt.ylabel('$U_2 /t$')
    plt.xlim(0,4)
    plt.savefig("2-particle_phase_diagram.png")
    # plt.legend()
    # plt.show()
    
    # HM = peak_widths(dos, [index], rel_height=0.5)[0][0]
        # HM = round(HM)
        # print(f"dw = {omega[HM-1]-omega[0]}, u2= {u2}, peakat = {omega[index]} ")
        # exit(0)
