# libraries 
import numpy as np
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import spsolve,norm
import matplotlib.pylab as plt
from concurrent.futures import ProcessPoolExecutor as PPE

# The value of t (hopping constant) and a (lattice constant) is taken 1

def delta(n,m): # Delta function
    return 1 if n==m else 0

def U(u1,u2,n1,n2):  # Potential term
    return u1*(delta(n1,1)+delta(n2,1))+u2*(delta(n1,2)+delta(n2,2)+delta(n1+n2,2))

def gamma_m(om,k,M,u1,u2,eta = 1e-3):  # Gamma_m matrix
    A = []#sp.lil_matrix((M-1,M-1),dtype=complex)
    row =[]
    col =[]
    for i in range(M-1): 
        A.append(-(om+1j*eta-U(u1,u2,i+1,M-i-1)))
        row.append(i)
        col.append(i)
    for i in range(M-2): 
        A.append(-np.exp(-1j*k))
        row.append(i)
        col.append(i+1)
    for i in range(M-2): 
        A.append(-np.exp(1j*k))
        row.append(i+1)
        col.append(i)
    return sp.csc_matrix((A,(row,col)),shape=(M-1,M-1),dtype=complex)

def beta_m(M): # Beta_m matrix
    A = []#sp.lil_matrix((M-1,M-1))
    row =[]
    col =[]
    for i in range(M-2):
        A.append(1); A.append(1)
        row.append(i); row.append(i)
        col.append(i); col.append(i+1)
    return sp.csc_matrix((A,(row,col)),shape=(M-1,M-1),dtype=complex)

def alpha_m(M): # Alpha_m matrix
    A = []#sp.lil_matrix((M-1,M-1))
    row =[]
    col =[]
    for i in range(M-2):
        A.append(1); A.append(1)
        row.append(i); row.append(i+1)
        col.append(i); col.append(i)
        A = np.array(A)
        row = np.array(row)
        col = np.array(col)
    return sp.csc_matrix((A,(row,col)),shape=(M-1,M-1),dtype=complex)


def find_Am__help(om,k,Mc,u1,u2,eta=1e-3):  # Find A_3 matrix by iteration
    ident = sp.identity(Mc-1,dtype=complex)
    aa = spsolve(gamma_m(om,k,Mc,u1,u2,eta),ident)*alpha_m(Mc)
    while(Mc>3):
        aa = beta_m(Mc)*aa
        Mc -= 1
        aa. resize((Mc-1,Mc-1))
        aa = gamma_m(om,k,Mc,u1,u2,eta)-aa
        ident = sp.identity(Mc-1,dtype=complex)
        aa = spsolve(aa,ident)*alpha_m(Mc)
    return aa

def find_Am(om,k,Mc,u1,u2,eta=1e-3): # Check the convergence of A_3 matrix
    aa = find_Am__help(om,k,Mc,u1,u2,eta)
    eps = 10
    while(eps>1e-6):
        Mc+=10
        aa1 = find_Am__help(om,k,Mc,u1,u2,eta)
        eps = norm(aa1-aa)
        aa = aa1
    return aa


def DOS(om,k,Mc,u1,u2,eta=1e-3): # Calculate Density Of States at omega
    aa = find_Am__help(om,k,Mc,u1,u2,eta)
    aa=  1.0/(om+1j*eta-2*u1-u2+aa[0,0]+aa[1,0])
    return -aa.imag/np.pi





# aa = inv(gamma_m(-3,0,4,4,-3,0,1e-2))*alpha_m(4)
# aa =  beta_m(4)*aa
# aa.resize((2,2))
# aa = gamma_m(-3,0,3,3,-3,0,1e-2)-aa
# aa = inv(aa)*alpha_m(3)
# aa = beta_m(4)*aa
# aa *= alpha_m(3)
# print(aa.todense())


if __name__ == '__main__':

    omega = np.linspace(-10,5,500)#.tolist()+np.linspace(-6.95,-6,50).tolist()
    k = 0
    Mc = 50

    ll = []
    ar = np.ones_like(omega,dtype=int)
    with PPE() as exe:
        ll = exe.map(DOS,omega,ar*k,ar*Mc,-ar*4,ar*1,ar*1e-2)
    ll = list(ll)
    plt.plot(omega,ll,'r-',label = f'$\eta={0.01},M_c={Mc}$')
    plt.xlabel("$\omega/t$")
    plt.ylabel("$A_3(\omega)$")
    plt.ylim(0,2)
    # plt.xlim(-7.2,-6)
    plt.legend()
    plt.show()
