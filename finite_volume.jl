#=
This code solve euler equation using finite volume method (FVM)
=#

using DelimitedFiles

#========================================================================================#

int(x) = floor(Int32, x) # int function, e.g. int(2.3) = 2

#========================================================================================#

@doc raw"""
Function to compute conserved quantities for finite volume method.
You must pass also the matrices that will contain the output.

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

Output
------
M : matrix
    Mass field
Px, Py : matrix
    component of momentum field
E : matrix
    Energy field
"""
function evol_quantities(rho::Matrix, vx::Matrix, vy::Matrix, P::Matrix, g::Float64, vol::Float64, M::Matrix, Px::Matrix, Py::Matrix, E::Matrix)

    M  .= rho .* vol                                                # mass of cell
    Px .= M .* vx                                                   # momentum along x
    Py .= M .* vy                                                   # momentum along y
    E  .= (P ./(g-1) .+ 0.5 .* rho .* (vx .^2 .+ vy .^2)) .* vol    # energy

end

#========================================================================================#

@doc raw"""
Function to return from the conserved quantities to the initial quantities
You must pass also the matrices that will contain the output.

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

Output
------
rho : matrix
    density field
vx, vy : matrix
    component of velocity field
P : matrix
    pressure field
"""
function old_quantities(M::Matrix, Px::Matrix, Py::Matrix, E::Matrix, g::Float64, vol::Float64, rho::Matrix, vx::Matrix, vy::Matrix, P::Matrix, B::Int)

    rho .= M ./ vol                                                  # density of cell
    vx  .= Px ./ M                                                   # speed along x
    vy  .= Py ./ M                                                   # speed along y
    P   .= (E ./ vol .- 0.5 .* rho .* (vx .^2 .+ vy .^2)) .* (g-1)   # Pressure

    if B == 1
        # Bound cells on top and bottom
        rho[1, :] .= rho[2, :]
        vx[ 1, :] .=  vx[2, :]
        vy[ 1, :] .= -vy[2, :]
        P[  1, :] .=   P[2, :]

        rho[end, :] .= rho[end-1, :]
        vx[ end, :] .=  vx[end-1, :]
        vy[ end, :] .= -vy[end-1, :]
        P[  end, :] .=   P[end-1, :]
    end

end

#========================================================================================#

@doc raw"""
Function to compute gradient using symmetric difference
You must pass also the matrices that will contain the output.

Parameters
----------
f : matrix
    function sampled on a grid
dx : float
    grid spacing

Output
------
fx : matrix
    df/dx
fy : matrix
    df/dy
"""
function grad(f::Matrix, dx::Float64, fx::Matrix, fy::Matrix, B::Int)

    fx .= (circshift(f, (0, -1)) .- circshift(f, (0, 1))) ./ (2*dx)
    fy .= (circshift(f, (-1, 0)) .- circshift(f, (1, 0))) ./ (2*dx)

    if B == 1
        # Bound cells on top and bottom
        fy[1, :  ] .= -fy[2, :    ]
        fy[end, :] .= -fy[end-1,:]
	end

end

#========================================================================================#

@doc raw"""
Apply slope limiter.
The idea is to compare the current derivative with the finite differences
between adjacent cells, applying a restriction that prevents the derivative
from becoming too large or too small.  max.(0., min.(1, (...))) *fx is used
to have factor between 0 and 1 that multiply fx.
1.0e-8 is for avoiding zero division error.

Parameters
----------
f : matrix
    function sampled on a grid
dx : float
    grid spacing
fx : matrix
    df/dx
fy : matrix
    df/dy

Output
------
fx : matrix
    limited df/dx
fy : matrix
    limited df/dy
"""
function slope_limit(f::Matrix, dx::Float64, fx::Matrix, fy::Matrix)

    fx .= max.(0., min.(1., ( (f - circshift(f, (0,  1))) ./ dx) ./ (fx .+ 1.0e-8 .* (fx .== 0)))) .* fx
    fx .= max.(0., min.(1., (-(f - circshift(f, (0, -1))) ./ dx) ./ (fx .+ 1.0e-8 .* (fx .== 0)))) .* fx
    fy .= max.(0., min.(1., ( (f - circshift(f, (1,  0))) ./ dx) ./ (fy .+ 1.0e-8 .* (fy .== 0)))) .* fy
    fy .= max.(0., min.(1., (-(f - circshift(f, (-1, 0))) ./ dx) ./ (fy .+ 1.0e-8 .* (fy .== 0)))) .* fy
    
end

#========================================================================================#

@doc raw"""
Function to obtain from the value on the node the values on the cell border

  |     |     |
--o-----o-----o--
  |   --|--   |      We compute the vaule on mid poin via taylor expantion:
  |  |  |  |  |      f_L = f - df/dx * dx/2
--o-----o-----o--    We do this for all directions and then, using circshift
  |  |  |  |  |      we make a translation to the left and down.
  |   --|--   |      We then finally obtain the values on the initial edges
--o-----o-----o--
  |     |     |

You must pass also the matrices that will contain the output.

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

Output
------
f_xL : matrix
    value on left  edge on x
f_xR : matrix
    value on right edge on x
f_yL : matrix
    value on left  edge on y (bottom edge)
f_yR :
    value on right edge on y (top edge)
"""
function extrapolate(f::Matrix, fx::Matrix, fy::Matrix, dx::Float64, f_xL::Matrix, f_xR::Matrix, f_yL::Matrix, f_yR::Matrix)

    f_xL .= f .- fx .* dx/2
    f_xL .= circshift(f_xL, (0, -1))
    f_xR .= f .+ fx .* dx/2

    f_yL .= f .- fy .* dx/2
    f_yL .= circshift(f_yL, (-1, 0))
    f_yR .= f .+ fy .* dx/2

end

#========================================================================================#

@doc raw"""
Function to compute fluxes; we use Rusanov method.
You must pass also the matrices that will contain the output.

Parameters
----------
rho_L, rho_R : matrix
    density field on left and right edge
v1_L, v2_R : matrix
    velocity field (x or y) along axis 1 (x or y) on left and right edge
v2_L, v2_R : matrix
    velocity field (x or y) along axis 2 (x or y) on left and right edge
P_L, P_R : matrix
    Pressure field on left and right edge
g : float
    adiabatic coefficient

Output
-------
flux_M : matrix
    flux of mass
flux_P1 : matrix
    flux of momentum (x or y) along 1 axis
flux_P2 : matrix
    flux of momentum (x or y) along 2 axis
flux_E : matrix
    flux of energy
"""
function Flux(rho_L::Matrix, rho_R::Matrix, v1_L::Matrix, v1_R::Matrix, v2_L::Matrix, v2_R::Matrix, P_L::Matrix, P_R::Matrix, g::Float64,
              flux_M::Matrix, flux_P1::Matrix, flux_P2::Matrix, flux_E::Matrix)

    # Energies on left and right
    en_L = P_L ./ (g-1) .+ 0.5 .*rho_L .* (v1_L .^2 .+ v2_L .^2)
    en_R = P_R ./ (g-1) .+ 0.5 .*rho_R .* (v1_R .^2 .+ v2_R .^2)

    # compute averaged states
    rho_avg = 0.5 .* (rho_L .+ rho_R)
    P1_avg  = 0.5 .* (rho_L .* v1_L .+ rho_R .* v1_R)
    P2_avg  = 0.5 .* (rho_L .* v2_L .+ rho_R .* v2_R)
    E_avg   = 0.5 .* (en_L  .+ en_R)
    P_avg   = (g-1) .* (E_avg .- 0.5 .* (P1_avg .^2 .+ P2_avg .^2) ./ rho_avg)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_M  .= P1_avg
    flux_P1 .= P1_avg .^2 ./ rho_avg .+ P_avg
    flux_P2 .= P1_avg .* P2_avg ./rho_avg
    flux_E  .= (E_avg .+ P_avg) .* P1_avg ./ rho_avg

    # find wavespeeds
    C_L = sqrt.(g .* P_L ./ rho_L) .+ abs.(v1_L)
    C_R = sqrt.(g .* P_R ./ rho_R) .+ abs.(v1_R)
    # C_L if C_L .>= C_R is true, otherwise C_R
    C   = ifelse.(C_L .>= C_R, C_L, C_R)

    # add stabilizing diffusive term
    flux_M  .-= C .* 0.5 .* (rho_L .- rho_R)
    flux_P1 .-= C .* 0.5 .* (rho_L .* v1_L .- rho_R .* v1_R)
    flux_P2 .-= C .* 0.5 .* (rho_L .* v2_L .- rho_R .* v2_R)
    flux_E  .-= C .* 0.5 .* ( en_L .- en_R )

end

#========================================================================================#

@doc raw"""
Function to update conserved quantities
You must pass also the matrices that will contain the output.

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

Output
------
F : matrix
    update quantity
"""
function update(F::Matrix, flux_F_X::Matrix, flux_F_Y::Matrix, dx::Float64, dt::Float64)

    # update solution
    F .+= .- dt .* dx .* flux_F_X
    F .+=    dt .* dx .* circshift(flux_F_X, (0, 1))
    F .+= .- dt .* dx .* flux_F_Y
    F .+=    dt .* dx .* circshift(flux_F_Y, (1, 0))

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
5x5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0

julia> gy
5x5 Matrix{Float64}:
 1.0  1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0  3.0
 4.0  4.0  4.0  4.0  4.0
 5.0  5.0  5.0  5.0  5.0

"""
function meshgrid(x::Vector, y::Vector)
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

    N  = 200  # Number of points
    L  = 1    # Size of box
    g  = 5/3  # Ideal gas gamma
    t  = 3    # Time of simulation
    s  = 20   # Frame rate for animation in unit of dt
    G  = 0    # Gravity
    B  = 0    # Flag for bound on top ad bottom must be 1 if G != 0
    sl = 0    # Flag for splope limiting

    #============= Saving data on file =============#

    #===============================================
    Each file contains the temporal evolution of a
    quantity. First we open the file with the
    permission to write to erase what has been written,
    then with permission to append to save the data.
    We use anim_plot.py to read them ad made plot
    ===============================================#

    name = "om"

    file1 = open("data_rho_$name.txt", "w"); close(file1)
    file2 = open("data_vx_$name.txt",  "w"); close(file2)
    file3 = open("data_vy_$name.txt",  "w"); close(file3)
    file4 = open("data_P_$name.txt",   "w"); close(file4)

    file1 = open("data_rho_$name.txt", "a") # To save data
    file2 = open("data_vx_$name.txt",  "a") # To save data
    file3 = open("data_vy_$name.txt",  "a") # To save data
    file4 = open("data_P_$name.txt",   "a") # To save data

    #================ Mesh creation ================#

    dx   = L/N    # step
    vol  = dx^2   # cell volume
    x    = [0.5*dx + i*(L-dx)/(N-1) for i in 0:N-1]
    X, Y = meshgrid(x, x) # square grid

    #============== Initial conditions =============#

    #= For Kelvin–Helmholtz instability
    Two fluid with different speed and rho:

    1   1   1   1   1   1    vx = -0.5  <-
    1   1   1   1   1   1               <-
    2   2   2   2   2   2    vx = +0.5  ->
    2   2   2   2   2   2               ->
    1   1   1   1   1   1    vx = -0.5  <-
    1   1   1   1   1   1               <-
    =#

    sigma = 0.05     # Sigma for vy pertubation
    rho   = 1.   .+ (abs.(Y .- 0.5) .< 0.25) # Two different densities
    vx    = -0.5 .+ (abs.(Y .- 0.5) .< 0.25) # Two diffetent speeds
    vy    = 0.1 .* sin.(4*pi .* X) .* (exp.( .- (Y .- 0.25) .^2 ./ (sigma^2)) .+ exp.( .- ( Y .- 0.75) .^2 ./(sigma^2)))
    P     = 2.5 * ones(N, N) # Initialize pressure

    #***********************************************#

    #= For Rayleigh-Taylor instability
    Two fluid with different rho:
    ---------
    2   2   2
    2   2   2    | G
    2   2   2    V
    1   1   1
    1   1   1
    1   1   1
    ---------
    =#

    #rho = 1. .+ (Y .> 0.5) # Two different densities
    #vx  = zeros(N, N)      # No velocity along x
    #vy  = 0.0025 .* (1 .- cos.(2*pi .* X)) .* (1 .- cos.(2*pi .* Y ))
    #P   = 2.5 .+ G .* (Y .- 0.5) .* rho # Initialize pressure

    #==============================================#
    #====== Add bound cells to craete a wall ======#

    K = N # If we not specify we have squared matrices
    if B == 1
        rho = vcat(rho[1, :]', rho, rho[end, :]')
        vx  = vcat(vx[ 1, :]',  vx,  vx[end, :]')
        vy  = vcat(vy[ 1, :]',  vy,  vy[end, :]')
        P   = vcat(P[  1, :]',   P,   P[end, :]')
        K   = N + 2 # else we must consider bound cells
    end
    
    # Compute conserved quatities
    M, Px, Py, E  = zeros(K, N), zeros(K, N), zeros(K, N), zeros(K, N)
    evol_quantities(rho, vx, vy, P, g, vol, M, Px, Py, E)

    #============== Auxiliar matrix ===============#

    #===============================================
    To avoid an excessive (memory allocation) use of
    the garbage collector we define all these matrices
    that we will pass to the various functions
    which overwrite them (like a subroutine in Fortran)
    ===============================================#

    # For the gradient
    drho_x, drho_y = zeros(K, N), zeros(K, N)
    dvx_x,  dvx_y  = zeros(K, N), zeros(K, N)
    dvy_x,  dvy_y  = zeros(K, N), zeros(K, N)
    dP_x,   dP_y   = zeros(K, N), zeros(K, N)

    # For first update
    rho_new = zeros(K, N)
    vx_new  = zeros(K, N)
    vy_new  = zeros(K, N)
    P_new   = zeros(K, N)

    # For edge extrapolation
    rho_xL, rho_xR, rho_yL, rho_yR = zeros(K, N), zeros(K, N), zeros(K, N), zeros(K, N)
    vx_xL,  vx_xR,  vx_yL,  vx_yR  = zeros(K, N), zeros(K, N), zeros(K, N), zeros(K, N)
    vy_xL,  vy_xR,  vy_yL,  vy_yR  = zeros(K, N), zeros(K, N), zeros(K, N), zeros(K, N)
    P_xL,   P_xR,   P_yL,   P_yR   = zeros(K, N), zeros(K, N), zeros(K, N), zeros(K, N)

    # For flux
    flux_M_X, flux_Px_X, flux_Py_X, flux_E_X = zeros(K, N), zeros(K, N), zeros(K, N), zeros(K, N)
    flux_M_Y, flux_Py_Y, flux_Px_Y, flux_E_Y = zeros(K, N), zeros(K, N), zeros(K, N), zeros(K, N)

    #================= Simulation =================#

    count = 0
    time  = 0

    while time <= t

        # Return to initial quantities
        old_quantities(M, Px, Py, E, g, vol, rho, vx, vy, P, B)

        # Time step from Courant–Friedrichs–Lewy (CFL) = dx / max signal speed
        # sqrt.( g .* P ./ rho ) is the sound of speed for ideal gas
        dt = 0.4 * minimum( dx ./ (sqrt.( g .* P ./ rho ) .+ sqrt.(vx .^2 .+ vy .^2)))

        if B == 1
            # Add source term
            E  .+= 0.5 .* dt .* Py .* G
            Py .+= 0.5 .* dt .* M  .* G

            old_quantities(M, Px, Py, E, g, vol, rho, vx, vy, P, B)
        end

        # compute gradients
        grad(rho, dx, drho_x, drho_y, B)
        grad(vx,  dx, dvx_x,  dvx_y,  B)
        grad(vy,  dx, dvy_x,  dvy_y,  B)
        grad(P,   dx, dP_x,   dP_y,   B)
        
        if sl == 1
            slope_limit(rho, dx, drho_x, drho_y)
            slope_limit(vx , dx, dvx_x,  dvx_y )
            slope_limit(vy , dx, dvy_x,  dvy_y )
            slope_limit(P  , dx, dP_x,   dP_y  )
        end
        
        # half-step in time, the first three equations when put together give the Euler equation
        rho_new = rho .- 0.5*dt .* ( vx .* drho_x .+ rho .* dvx_x .+ vy .* drho_y .+ rho .* dvy_y) # Conservation of particle number
        vx_new  = vx  .- 0.5*dt .* ( vx .* dvx_x  .+ vy  .* dvx_y .+ (1 ./rho) .* dP_x )           # Euler equation for vx
        vy_new  = vy  .- 0.5*dt .* ( vx .* dvy_x  .+ vy  .* dvy_y .+ (1 ./rho) .* dP_y )           # Euler equation for vy
        P_new   = P   .- 0.5*dt .* ( g .* P .* (dvx_x .+ dvy_y)   .+ vx .* dP_x .+ vy .* dP_y )     # Conservation of energy

        # Extrapolation to get the values ​​on the cell edges
        extrapolate(rho_new, drho_x, drho_y, dx, rho_xL, rho_xR, rho_yL, rho_yR)
        extrapolate(vx_new,  dvx_x,  dvx_y,  dx, vx_xL,  vx_xR,  vx_yL,  vx_yR)
        extrapolate(vy_new,  dvy_x,  dvy_y,  dx, vy_xL,  vy_xR,  vy_yL,  vy_yR)
        extrapolate(P_new,   dP_x,   dP_y,   dx, P_xL,   P_xR,   P_yL,   P_yR)

        # Compute fluxes
        #============================================================
        We have computed the values of all our fields at all edges
        of the cell and we must compute the flux along the x axis
        and the y axis. Having the same functional form we only call
        the function two times one for flux along x axis
        and one for flux along y axis:

        # All quantities are computed on x edges
        Flux(rho_xL, rho_xR, vx_xL, vx_xR, vy_xL, vy_xR, ...)
        # All quantities are computed on t edges
        Flux(rho_yL, rho_yR, vy_yL, vy_yR, vx_yL, vx_yR, ...)

        For more clarity we report the equations:

        dq   df   dg
        -- + -- + -- = 0    where:
        dt   dx   dy

            | \rho   |      | \rho u       |      | \rho v       |
        q = | \rho u |; f = | \rho u^2 + P |; g = | \rho u v     |
            | \rho v |      | \rho u v     |      | \rho v^2 + P |
            | \rho E |      | u(\rhoE + P) |      | v(\rhoE + P) |
        ============================================================#
        Flux(rho_xL, rho_xR, vx_xL, vx_xR, vy_xL, vy_xR, P_xL, P_xR, g, flux_M_X, flux_Px_X, flux_Py_X, flux_E_X)
        Flux(rho_yL, rho_yR, vy_yL, vy_yR, vx_yL, vx_yR, P_yL, P_yR, g, flux_M_Y, flux_Py_Y, flux_Px_Y, flux_E_Y)

        # Update solution
        update(M,  flux_M_X,  flux_M_Y,  dx, dt)
        update(Px, flux_Px_X, flux_Px_Y, dx, dt)
        update(Py, flux_Py_X, flux_Py_Y, dx, dt)
        update(E,  flux_E_X,  flux_E_Y,  dx, dt)

        if B == 1
            # Add source term
            E  .+= 0.5 .* dt .* Py .* G
            Py .+= 0.5 .* dt .* M  .* G
        end
        # update time
        time += dt

        if int(time/dt) % s == 0

            count += 1
            print("Iteration: $count we are at time: $(round(time, digits=3)) / $t \r")

            writedlm(file1, rho)
            writedlm(file2, vx)
            writedlm(file3, vy)
            writedlm(file4, P)

        end
    end
    println()
    print("Total iteration: $(count*s) \n")

    close(file1)
    close(file2)
    close(file3)
    close(file4)
end

@time main()
