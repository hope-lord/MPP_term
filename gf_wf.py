# simulation of particle in a box with infinite walls.
# The distance between the walls change
import numpy as np
import matplotlib.pylab as plt
np.random.seed(15200)
n = 1


# For Box plot 
def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])#,axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def color():
    string = '#'
    for i in range(3):
        a = f"{hex(np.random.randint(0,256))}"
        a = a.split('x')[-1]
        if len(a)==1:
            a = '0'+a
        string += a    
    return string
a = np.array([70,100,200,500],dtype=np.float64)

# n = np.arange(1,10,1,dtype=np.float64)

colors = [color() for _ in range(len(a))]
ls = ['-','--','-.',':']


def wavefunc(X,N,L):
    arr = []
    for x in X:
        if x<-L/2 or x>L/2:
            arr.append(0)
        elif N%2==0:
            arr.append(np.sin(N*np.pi/L*x)**2)
        else:
            arr.append(np.cos(N*np.pi/L*x)**2 *2/L)
    return np.array(arr)

def dos(om,N,L,eta=1e-2):
    return - (1/(om+1j*eta-(N*np.pi/L)**2)).imag/np.pi

x = np.linspace(-a[-1],a[-1],1000)
plt.figure()
# fig,ax = plt.subplots(2,2)
for i in range(len(a)):
    # ax[i//2,i%2].set_title("a = "+str(a[i]))
    plt.plot(x,wavefunc(x,n,a[i]),color=colors[i],ls=ls[i],label=f"a={a[i]}")

plt.xlabel("x")
plt.ylabel(f"$|\psi_{n}(x)|^2$")
plt.title(f"PDF for n = {n}")
plt.xlim(-a[-1],a[-1])
plt.legend()

# fig,ax = plt.subplots(2,2)

## DOS
subpos = [0.15,0.6,0.3,0.3]
x=np.linspace(-0.5,0.5,100)
y = np.linspace(-0.02,0.02,100)
fig,ax = plt.subplots()
subax = add_subplot_axes(ax,subpos)
for i in range(len(a)):
    # ax[i//2,i%2].set_title("a = "+str(a[i]))
    ax.plot(x,dos(x,n,a[i]),color=colors[i],ls=ls[i],label=f"a={a[i]}")
    subax.plot(y,dos(y,n,a[i]),color=colors[i],ls=ls[i])

ax.set_xlabel("$\omega$")
ax.set_ylabel(r"$-\frac{1}{\pi}"+f"Im[G({n},\omega)]$")
ax.set_title(f"Density of States for n = {n}")
ax.set_xlim(x[0],x[-1])
subax.set_xlim(y[0],-y[0])

ax.legend()
plt.show()
