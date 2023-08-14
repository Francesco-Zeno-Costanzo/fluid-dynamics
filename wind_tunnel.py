"""
Code for wind tunnel with lattice boltzmann method
"""
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class WindTunnel:
    '''
    Class to simulate a wind tunnel with an obstacle in the middle.
    The equation of state is P = rho*cs**2 = rho/3
    The simulation is done with Lattice boltzmann method 2D9Q:

            (-1, 1)    (0, 1)    (1, 1)


            (-1, 0)    (0, 0)    (1, 0)


            (-1,-1)    (0,-1)    (-1,1)

    Varius shape are implemented but is also possible to pass
    a function that compute the bound of the shape.
    All method: plot, show_anim, save_for anim, must be use alone
    because all of them call evolve method, so for an entire evolution
    the system must be re-initialize.
    '''

    def __init__(self, Nx, Ny, xc, yc, R, tau, obstacle=None):
        '''
        Define parameter of simulations and the obstacle

        Parameters
        ----------
        Nx, Ny : int
            size of tunnel
        xc, yc : float
            center of obstacle
        R : float
            size of obstacle
        tau : float
            relazation time
        obstacle : None, string or function, optional
            If string can be: 'circle', 'square', 'vline', 'hline', 'semicir', 'ellipse',
            if is None the default shape is circle.
            If is a function bust take four variable and return equation of shape,
            for example for a circle must be:
            def f(x, y, xc, yc):
                R = 10
                reyurn np.sqrt((xi - xc)**2 + (yi - yc)**2) < R
        '''
        self.Nx  = Nx   # number of point on x axis
        self.Ny  = Ny   # number of point on y axis
        self.xc  = xc   # x center of the obstacle
        self.yc  = yc   # y center of the obstacle
        self.R   = R    # radius (size) of the obstacle
        self.tau = tau  # relaxation time (\propto kinematic viscosity)
        #==============================================
        # define lattice and distribution function
        #==============================================
        # lattice
        self.ex = np.array([0,    0,    1,   1,    1,   0,   -1,  -1,   -1 ])
        self.ey = np.array([0,    1,    1,   0,   -1,  -1,   -1,   0,    1 ])
        self.w  = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
        self.M = len(self.ex)

        # distribution function
        self.f = np.ones((Nx, Ny, self.M))
        self.f = self.f + 0.01*np.random.randn(Nx, Ny, self.M) # random inomogeneity
        # initial condition
        self.f[:,:,3] += 2

        #==============================================
        # Creation of obstacle as a mask
        #==============================================
        if obstacle is None:
            obstacle='circle'
        self.obst = np.full((Nx, Ny), False)

        for xi in range(Nx):
            for yi in range(Ny):

                if type(obstacle) == type('string'):
                    OBS = ['circle', 'square', 'vline', 'hline', 'semicir', 'ellipse']
                    if not obstacle in OBS:
                        err_msg = f'only {[o for o in OBS]} are implemented, but you can pass a function'
                        raise NotImplementedError(err_msg)

                    if obstacle=='circle':
                        if np.sqrt((xi - xc)**2 + (yi - yc)**2) < R:
                            self.obst[xi, yi] = True

                    if obstacle=='square':
                        if abs(xi - xc) < R and abs(yi - yc) < R :
                            self.obst[xi, yi] = True

                    if obstacle=='vline':
                        if xi == xc and yc - R < yi < yc + R :
                            self.obst[xi, yi] = True

                    if obstacle=='hline':
                        if yi == yc and xc - R < xi < xc + R :
                            self.obst[xi, yi] = True

                    if obstacle=='semicir':
                        if np.sqrt((xi - xc)**2 + (yi - yc)**2) < R and xi<xc:
                            self.obst[xi, yi] = True

                    if obstacle=='ellipse':
                        if np.sqrt((xi - xc)**2/2.5 + (yi - yc)**2/0.5) < R:
                            self.obst[xi, yi] = True
                else:
                    if obstacle(xi, yi, xc, yc):
                        self.obst[xi, yi] = True


    def evolve(self):
        '''
        Function for evolution

        Return
        ------
        vx, vy : 2darray
            velocity field
        rho : 2darray
            density
        '''
        # Move all particles by one step along their directions of motion
        for i, e_x, e_y in zip(range(self.M), self.ex, self.ey): # this lead to periodic boundary condition
            self.f[:, :, i] = np.roll(self.f[:, :, i], e_x, axis=0)  # axis 0 is east  <--> west
            self.f[:, :, i] = np.roll(self.f[:, :, i], e_y, axis=1)  # axis 1 is north <--> south

        # remove periodic conditions along x axis
        self.f[-1, :, [6,7,8]] = self.f[-2, :, [6,7,8]] # on right from left incoming  fluid
        self.f[ 0, :, [2,3,4]] = self.f[ 1, :, [2,3,4]] # on left from right outcoming fluid

        # compute macroscopic density and velocity
        rho = np.sum(self.f, 2)
        vx  = np.sum(self.f*self.ex, 2)/rho
        vy  = np.sum(self.f*self.ey, 2)/rho

        # boundary conditions on obstacle
        f_obst = self.f[self.obst, :]
        f_obst = f_obst[:, [0,5,6,7,8,1,2,3,4]]
        self.f[self.obst, :] = f_obst
        # the speed is zero on the obstacle
        vx[self.obst] = 0
        vy[self.obst] = 0

        # Collisions
        f_eq = np.zeros(self.f.shape)
        for i, e_x, e_y, wi in zip(range(self.M), self.ex, self.ey, self.w):
            v2 = vx**2 + vy**2
            ev = e_x*vx + e_y*vy
            f_eq[:, :, i] = rho*wi*(1 + 3*ev + 9*ev**2/2 - 3*v2/2)

        # update distribution function
        self.f -= (1/tau)*(self.f - f_eq)

        return vx, vy, rho


    def vort(self, vx, vy):
        '''
        Function to compute vorticity i.e. curl of v

        Parameters
        ----------
        vx, vy : 2darray
            velocity field

        Return
        vor : 2darray
            vorticity of filed
        '''

        vor = np.roll(vx, -1 ,axis=1) - np.roll(vx, 1, axis=1) - np.roll(vy, -1, axis=0) + np.roll(vy, 1, axis=0)
        vor[self.obst] = np.nan # for better plot
        return vor.T


    def plot(self, T, show=True):
        '''
        function to plot vorticity after a time T

        Parameter
        ---------
        T : int
            time of evolution

        Return
        ------
        fig : matplotlib figure
        '''
        for step in range(T):
            self.evolve()

        vx, vy, rho = self.evolve()

        # Plot vorticity
        fig = plt.figure(1, figsize=(10, 3.5))
        plt.xlabel('X', fontsize=15)
        plt.ylabel('Y', fontsize=15)
        plt.title('Wind Tunnel', fontsize=15)
        plt.xlim(0, self.Nx)
        plt.ylim(0, self.Ny)
        vor = plt.imshow(self.vort(vx, vy), origin='lower',
                         norm=plt.Normalize(-0.1, 0.1), cmap=plt.get_cmap('jet'))
        # Plot velocity field
        x, y = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny))
        plt.streamplot(x, y, vx.T, vy.T, color="k", density=2.5,linewidth=0.3, arrowstyle='->', arrowsize=1)
        ax  = plt.gca()
        ax.set_aspect('equal')
        if show: plt.show()
        return fig


    def show_anim(self, t, T, save=False, name=''):
        '''
        function to animate vorticity (or pressure)

        Parameter
        ---------
        t : int
            time between two steps of animation
        T : int
            Total time of evolution
        save : bool, optional, default False
            if True the animation will be saved
        name : string, optional, necessary if save=True
            name of file of animation
        '''
        vx, vy, rho = self.evolve()
        #F = rho.T/3 # norm=plt.Normalize(3.6, 3.8)
        F = self.vort(vx, vy)

        fig = plt.figure(figsize=(10, 3.5))
        plt.xlabel('X', fontsize=15)
        plt.ylabel('Y', fontsize=15)
        plt.title('Wind Tunnel', fontsize=15)
        vor = plt.imshow(F, origin='lower', norm=plt.Normalize(-0.1, 0.1), cmap=plt.get_cmap('jet'))
        #plt.colorbar(vor)
        ax  = plt.gca()
        ax.set_aspect('equal')

        def animate(i):

            for step in range(t):
                self.evolve()

            vx, vy, rho = self.evolve()
            #F = rho.T/3
            F = self.vort(vx, vy)
            vor.set_array(F)

            return (vor, ) # must be iterable

        anim = FuncAnimation(fig, animate, frames=T, interval=1, blit=True, repeat=False)
        if save : anim.save(name+'.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
        plt.show()


    def save_for_anim(self, t, T, path, name):
        '''
        For the stremplot is more convenient
        save figure and make a gif after.

        Parameter
        ---------
        t : int
            time between two steps of animation
        T : int
            Total time of evolution
        path : string
            absolute path for save plot
        name : string
            name of file of animation
        '''

        # make and save plot
        for step in range(T):
            for i in range(t):
                self.evolve()

            fig = self.plot(0, show=False)
            plt.savefig(path+"\\"f"{step}.png")
            plt.close(fig)

            print(f"{step/(T-1) * 100:.2f} % \r", end='')

        # make a gif
        path_in = path+'/*.'+'png'
        path_out = path+f'/{name}.gif'

        imgs = []

        file = glob.glob(path_in, recursive=True)
        file.sort(key=len)
        for im in file:
            imgs.append(imageio.imread(im))

        imageio.mimsave(path_out, imgs)


if __name__ == '__main__':

    #==============================================================================
    # Computational parameters
    #==============================================================================

    Nx  = 200   # number of point on x axis
    Ny  = 50    # number of point on y axis
    T   = 400   # simutation time
    t   = 25    # one frame each t
    xc  = Nx//5 # x center of the obstacle
    yc  = Ny//2 # y center of the obstacle
    R   = 6     # radius (size) of the obstacle
    tau = 0.57  # relaxation time (\propto kinematic viscosity)

    #==============================================================================
    # some obstacle
    #==============================================================================

    def f(x, y, xc, yc):
        '''wall with line
        '''
        return (x==xc and abs(y-yc)<R) or (y==yc and xc < x < xc+30)

    def g(x, y, xc, yc):
        '''triangle
        '''
        return abs(x - xc) < 10 and abs(y - yc) < 10 and abs(y-yc) < (x-xc)

    def h(x, y, xc, yc):
        '''simil airfoil
        '''
        semi = np.sqrt((x - xc)**2 + (y - yc)**2) < R and x<xc
        trig = xc-1 < x < xc + 8*R and yc - R < y < yc + R and -(y-yc)*4.5 + 4*R > (x-xc)
        return  semi or trig

    #==============================================================================
    # Simulation
    #==============================================================================

    wt = WindTunnel(Nx, Ny, xc, yc, R, tau)
    #wt.plot(3000)
    wt.show_anim(t, T, save=False, name='vor')
    #wt.save_for_anim(t, T, r'C:\Users\franc\Documents\codici python\gif', 'vor')