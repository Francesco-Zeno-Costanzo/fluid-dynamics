#=
This code solve euler equation using finite volume method (FVM) 
=#

using DelimitedFiles


int(x) = floor(Int32, x) # int function, e.g. int(2.3) = 2

#========================================================================================#

@doc raw"""
Function to compute conserved quantities for finite volume method

Parameters
----------
rho : matrix
    density field
vx, vy : matrix
    component of velocity field
P : matrix
    pressure field
g : float
    adiabatic coefficient
vol : float
    volume of cell
    
Returns
-------
M : matrix
    Mass field
Px, Py : matrix
    component of momentum field
E : matrix
    Energy field
"""
function evol_quantities(rho, vx, vy, P, g, vol)

    M  = rho .* vol                                                # mass of cell
    Px = M .* vx                                                   # momentum along x
    Py = M .* vy                                                   # momentum along y
    E  = (P ./(g-1) .+ 0.5 .* rho .* (vx .^2 .+ vy .^2)) .* vol    # energy
    
    return M, Px, Py, E

end

#========================================================================================#

@doc raw"""
Function to return from the stored quantities to the initial quantities

Parameters
----------
M : matrix
    Mass field
Px, Py : matrix
    component of momentum field
E : matrix
    Energy field
g : float
    adiabatic coefficient
vol : float
    volume of cell
    
Returns
-------
rho : matrix
    density field
vx, vy : matrix
    component of velocity field
P : matrix
    pressure field
"""
function old_quantities(M, Px, Py, E, g, vol)
    
    rho = M ./ vol                                                  # density of cell
    vx  = Px ./ M                                                   # speed along x
    vy  = Py ./ M                                                   # speed along y
    P   = (E ./ vol .- 0.5 .* rho .* (vx .^2 .+ vy .^2)) .* (g-1)   # Pressure
    
    return rho, vx, vy, P
end

#========================================================================================#

@doc raw"""
Function to compute gradient using symmetric difference

Parameters
----------
f : matrix
    function sampled on a grid
dx : float
    grid spacing

Returns
-------
fx : matrix
    df/dx
fy : matrix
    df/dy
"""
function grad(f, dx)
    
    fx = (circshift(f, (0, -1)) .- circshift(f, (0, 1))) ./ (2*dx)
    fy = (circshift(f, (-1, 0)) .- circshift(f, (1, 0))) ./ (2*dx)
    
    return fx, fy
end

#========================================================================================#

@doc raw"""
Function to obtain from the value on the node the values ​​on the cell border

  |     |     |
--o-----o-----o--    
  |   --|--   |      We compute the vaule on mid poin via taylor expantion:
  |  |  |  |  |      f_L = f - df/dx * dx/2
--o-----o-----o--    We do this for all directions and then, using circshift
  |  |  |  |  |      we make a translation to the left and down.
  |   --|--   |      We then finally obtain the values ​​on the initial edges
--o-----o-----o--
  |     |     |

Parameters
----------
f : matrix
    function sampled on a grid
fx : matrix
    df/dx
fy : matrix
    df/dy
dx : float
    grid spacing

Returns
-------
f_xL : matrix
    flux on left  edge on x
f_xR : matrix
    flux on right edge on x
f_yL : matrix
    flux on left  edge on y (bottom edge)
f_yR :
    flux on right edge on y (top edge)
"""
function extrapolate(f, fx, fy, dx)
    
    f_xL = f .- fx .* dx/2
    f_xL = circshift(f_xL, (0, -1))
    f_xR = f .+ fx .* dx/2

    f_yL = f .- fy .* dx/2
    f_yL = circshift(f_yL, (-1, 0))
    f_yR = f .+ fy .* dx/2
    
    return f_xL, f_xR, f_yL, f_yR

end

#========================================================================================#

@doc raw"""
Function to compute fluxes; Euler equations as coservatives laws

Parameters
----------
rho_L, rho_R : matrix
    density field on left and right edge
vx_L, vx_R : matrix
    velocity field x component on left and right edge
vy_L, vy_R : matrix
    velocity field y component on left and right edge
P_L, P_R : matrix
    Pressure field on left and right edge
g : float
    adiabatic coefficient

Returns
-------
flux_M : matrix
    flux of mass
flux_Px : matrix
    flux of momentum along x axis
flux_Py : matrix
    flux of momentum along y axis
flux_E : matrix
    flux of energy
"""
function Flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, g)
    
    # Energies on left and right
    en_L = P_L ./ (g-1) .+ 0.5 .*rho_L .* (vx_L .^2 .+ vy_L .^2)
    en_R = P_R ./ (g-1) .+ 0.5 .*rho_R .* (vx_R .^2 .+ vy_R .^2)

    # compute averaged states
    rho_avg = 0.5 .* (rho_L .+ rho_R)
    Px_avg  = 0.5 .* (rho_L .* vx_L .+ rho_R .* vx_R)
    Py_avg  = 0.5 .* (rho_L .* vy_L .+ rho_R .* vy_R)
    E_avg   = 0.5 .* (en_L  .+ en_R)
    P_avg   = (g-1) .* (E_avg .- 0.5 .* (Px_avg .^2 .+ Py_avg .^2) ./ rho_avg)
    
    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_M  = Px_avg
    flux_Px = Px_avg .^2 ./ rho_avg .+ P_avg
    flux_Py = Px_avg .* Py_avg ./rho_avg
    flux_E  = (E_avg .+ P_avg) .* Px_avg ./ rho_avg
    
    # find wavespeeds
    C_L = sqrt.(g .* P_L ./ rho_L) .+ abs.(vx_L)
    C_R = sqrt.(g .* P_R ./ rho_R) .+ abs.(vx_R)
    C   = ifelse.(C_L .>= C_R, C_L, C_R)
    
    # add stabilizing diffusive term
    flux_M  -= C .* 0.5 .* (rho_L .- rho_R)
    flux_Px -= C .* 0.5 .* (rho_L .* vx_L .- rho_R .* vx_R)
    flux_Py -= C .* 0.5 .* (rho_L .* vy_L .- rho_R .* vy_R)
    flux_E  -= C .* 0.5 .* ( en_L .- en_R )

    return flux_M, flux_Px, flux_Py, flux_E
end

#========================================================================================#
    
@doc raw"""
Function to update conserved quantities

Parameters
----------
F : matrix
    physical quantities
flux_F_x : matrix
    flux of F along x axis
flux_F_Y : matrix
    flux of F along y axis
dx : float
    grid spacing
dt : float
    time step

Return
------
F : matrix
    update quantity
"""
function update(F, flux_F_X, flux_F_Y, dx, dt)
    
    # update solution
    F += .- dt .* dx .* flux_F_X
    F +=    dt .* dx .* circshift(flux_F_X, (0, 1))
    F += .- dt .* dx .* flux_F_Y
    F +=    dt .* dx .* circshift(flux_F_Y, (1, 0))
    
    return F
end

#========================================================================================#


@doc raw"""
Function to compute a 2D meshgrid
    
Parameters
----------
x : array
    array for x range
y : array
    array for y range

Returns
-------
gridx, gridy : matrix
    meshgrid for our intervall
    cartesian product between x, y

Example
-------
julia> x = [1, 2, 3, 4, 5];
julia> gx, gy, = meshgrid(x, x);
julia> gx
5×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0

julia> gy
5×5 Matrix{Float64}:
 1.0  1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0  3.0
 4.0  4.0  4.0  4.0  4.0
 5.0  5.0  5.0  5.0  5.0

"""
function meshgrid(x, y)
    nx = length(x)
    ny = length(y)
    
    gridx = zeros(ny, nx)
    gridy = zeros(ny, nx)
    
    for j = 1:nx
        for i = 1:ny
            gridx[i, j] = x[j]
            gridy[i, j] = y[i]
        end
    end
    
    return gridx, gridy
end

#========================================================================================#

function main()
    
    #============ Simulation parameters ============#
    
    N = 200  # Number of points
    L = 1    # Size of box
    g = 5/3  # Ideal gas gamma
    t = 3    # Time of simulation
    s = 20   # Frame rate for animation in unit of dt
    
    #============= Saving data on file =============#
    
    #=
    Each file contains the temporal evolution of a
    quantity. Therefore it is recommended to delete
    any previous files to avoid additions to the old 
    file. Can be read by anim_plot.py
    =#
    file1 = open("data_rho.txt", "a") # To save data
    file2 = open("data_vx.txt",  "a") # To save data
    file3 = open("data_vy.txt",  "a") # To save data
    file4 = open("data_P.txt",   "a") # To save data
    
    #================ Mesh creation ================#
    
    dx   = L/N    # step
    vol  = dx^2   # cell volume
    x    = [0.5*dx + i*(L-dx)/(N-1) for i in 0:N-1]
    X, Y = meshgrid(x, x) # grid
    
    #============== Initial conditions =============#
    #= For Kelvin–Helmholtz instability
    Two fluid with different speed and rho:
    1   1   1   1   1   1    vx = -0.5  <-    
    1   1   1   1   1   1               <-
    ---------------------    
    2   2   2   2   2   2    vx = +0.5  ->
    2   2   2   2   2   2               ->
    ---------------------     
    1   1   1   1   1   1    vx = -0.5  <-
    1   1   1   1   1   1               <-
    =#
    
    sigma = 0.05     # Sigma for vy pertubation
    rho   = 1.   .+ (abs.(Y .- 0.5) .< 0.25) # Two different densities 
    vx    = -0.5 .+ (abs.(Y .- 0.5) .< 0.25) # Two diffetente speed
    vy    = 0.1 .* sin.(4*pi .* X) .* (exp.( .- (Y .- 0.25) .^2 ./ (sigma^2)) .+ exp.( .- ( Y .- 0.75) .^2 ./(sigma^2)))
    P     = 2.5 * ones(N, N) # Initialize pressure
    
    # Compute conserved quatities
    M, Px, Py, E = evol_quantities(rho, vx, vy, P, g, vol)

    #================= Simulation =================#
    
    count = 0
    time  = 0
    
    while time < t
      
      # Return to initial quantities
        rho, vx, vy, P = old_quantities(M, Px, Py, E, g, vol)
        
      # Time step from Courant–Friedrichs–Lewy (CFL) = dx / max signal speed
      # sqrt.( g .* P ./ rho ) is the sound of speed for ideal gas
      dt = 0.4 * minimum( dx ./ (sqrt.( g .* P ./ rho ) .+ sqrt.(vx .^2 .+ vy .^2)))
      
      # compute gradients
      drho_x, drho_y = grad(rho, dx)
      dvx_x,  dvx_y  = grad(vx,  dx)
      dvy_x,  dvy_y  = grad(vy,  dx)
      dP_x,   dP_y   = grad(P,   dx)
      
      # half-step in time, the first three equations when put together give the Euler equation
      rho_new = rho .- 0.5*dt .* ( vx .* drho_x .+ rho .* dvx_x .+ vy .* drho_y .+ rho .* dvy_y) # Conservation of particle number
      vx_new  = vx  .- 0.5*dt .* ( vx .* dvx_x  .+ vy  .* dvx_y .+ (1 ./rho) .* dP_x )           # Euler equation for vx
      vy_new  = vy  .- 0.5*dt .* ( vx .* dvy_x  .+ vy  .* dvy_y .+ (1 ./rho) .* dP_y )           # Euler equation for vy
      P_new   = P   .- 0.5*dt .* ( g .* P .* (dvx_x .+ dvy_y)  .+ vx .* dP_x .+ vy .* dP_y )     # Conservation of energy
      
      # Extrapolation to get the values ​​on the cell edges
      rho_xL, rho_xR, rho_yL, rho_yR = extrapolate(rho_new, drho_x, drho_y, dx)
      vx_xL,  vx_xR,  vx_yL,  vx_yR  = extrapolate(vx_new,  dvx_x,  dvx_y,  dx)
      vy_xL,  vy_xR,  vy_yL,  vy_yR  = extrapolate(vy_new,  dvy_x,  dvy_y,  dx)
      P_xL,   P_xR,   P_yL,   P_yR   = extrapolate(P_new,   dP_x,   dP_y,   dx)
      
      # Compute fluxes
      flux_M_X, flux_Px_X, flux_Py_X, flux_E_X = Flux(rho_xL, rho_xR, vx_xL, vx_xR, vy_xL, vy_xR, P_xL, P_xR, g)
      flux_M_Y, flux_Py_Y, flux_Px_Y, flux_E_Y = Flux(rho_yL, rho_yR, vy_yL, vy_yR, vx_yL, vx_yR, P_yL, P_yR, g)

      # update solution
      M  = update(M,  flux_M_X,  flux_M_Y,  dx, dt)
      Px = update(Px, flux_Px_X, flux_Px_Y, dx, dt)
      Py = update(Py, flux_Py_X, flux_Py_Y, dx, dt)
      E  = update(E,  flux_E_X,  flux_E_Y,  dx, dt)
                
      # update time
      time += dt
      
      if int(time/dt) % s == 0
          
          count += 1
          println(count, " ", time)
          
          writedlm(file1, rho)
          writedlm(file2, vx)
          writedlm(file3, vy)
          writedlm(file4, P)
          
      end 
    end
end

@time main()
