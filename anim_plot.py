"""
Code for plotting the output of finite_volume.jl
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers


def show_anim(field, save=False, name='anim', ext='.gif', title=''):
    '''
    Function that takes as input a list of matrices
    (which are the temporal states of the system)
    and constructs an animation.
    
    Parameters
    ----------
    field : list
        list of matrices, 
        physically speaking each matrix must be a scalar field
    save : bool, optional, defult False
        save or not the animation
    name : sting, optional, defult 'anim'
        name of the animation if we want to save it
    ext : string, optional, default '.gif'
        extension of file, can be .gif or .mp4
        for .mp4 ffmpeg is used
    '''
    
    fig = plt.figure(figsize=(7, 7))
    plt.title(title, fontsize=15)
    ax  = plt.gca()
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    F = plt.imshow(field[0], origin='lower',
                   norm=plt.Normalize(np.min(field[0]), np.max(field[0])),
                   cmap=plt.get_cmap('plasma'))
    plt.colorbar(F)

    def animate(i):

        F.set_array(field[i])

        return (F, ) # must be iterable

    anim = FuncAnimation(fig, animate, frames=len(field), interval=20, blit=True, repeat=True)
    
    if save :
        if ext == '.gif':
            anim.save(name+ext, fps=30)
        
        if ext == '.mp4':
            # setting up wrtiers object
            Writer = writers['ffmpeg']
            writer = Writer(fps=30, bitrate=1800)
            anim.save(name+ext, writer)
            
    plt.show()


def plot(field, vx, vy, title=''):
    '''
    Function for plot scalar (rho or p) and velocity field together
     
    Parameters
    ----------
    field : 2d array
        physically speaking must be a scalar field
    vx, vy : 2d array
        physically speaking component of vectorial field
    '''
    
    fig = plt.figure(figsize=(7, 7))
    plt.title(title, fontsize=15)
    ax  = plt.gca()
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
     # Plot velocity field
    N, M = vx.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    plt.streamplot(x, y, vx, vy, color="k", density=2.5,linewidth=0.3, arrowstyle='->', arrowsize=1)
    
    F = plt.imshow(field, origin='lower', 
                   norm=plt.Normalize(np.min(field), np.max(field)),
                   cmap=plt.get_cmap('plasma'))
    plt.colorbar(F)    
    plt.show()

#====================================================================
# Read data
#====================================================================   


name = "om"
print("read first")
RHO = np.loadtxt(f"data_rho_{name}.txt")

print("read second")
VX  = np.loadtxt(f"data_vx_{name}.txt")

print("read third")
VY  = np.loadtxt(f"data_vy_{name}.txt")

print("read fourth")
P  = np.loadtxt(f"data_P_{name}.txt")

M, N = RHO.shape

k = N if M % N == 0 else N + 2
rho = [RHO[i*k:(i+1)*k, :] for i in range(M//k + 1)]; rho.pop()
vx  = [ VX[i*k:(i+1)*k, :] for i in range(M//k + 1)]; vx.pop()
vy  = [ VY[i*k:(i+1)*k, :] for i in range(M//k + 1)]; vy.pop()
p   = [  P[i*k:(i+1)*k, :] for i in range(M//k + 1)]; p.pop()

k_p = len(rho) - 1
show_anim(rho, title='Rayleigh-Taylor instability')#,  save=1, name='rh')#, ext='.mp4')

plot(rho[k_p], vx[k_p], vy[k_p], title='Rayleigh-Taylor instability')
plot(  p[k_p], vx[k_p], vy[k_p], title='Rayleigh-Taylor instability')

