using Pkg; Pkg.activate(".")

using Oceananigans
using Oceananigans.Units

Nx, Nz = 200, 100

architecture = CPU()

H  = 1kilometers
Lx = 500kilometers

underlying_grid = RectilinearGrid(architecture,
                                  size = (Nx, Nz),
                                  x = (-Lx/2, Lx/2),
                                  z = (-H, 0),
                                  halo = (4, 4), # start with this commented out
                                  topology = (Periodic, Flat, Bounded))

h₀ = H/20
width = 5kilometers
bump(x, y) = - H + h₀ * exp(-x^2 / 2width^2)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bump))

T₂ = 12.421hours
const ω₂ = 2π / T₂

coriolis = FPlane(latitude = -45)

# excursion parameter
ε = 0.1 # U / (ω * width)
U_tidal = ε * ω₂ * width

const tidal_forcing_amplitude = U_tidal * (coriolis.f^2 - ω₂^2) / ω₂

@inline tidal_forcing(x, y, z, t) = tidal_forcing_amplitude * cos(ω₂ * t)

gravitational_acceleration = 9.81
gravity_wave_speed = sqrt(gravitational_acceleration * grid.Lz)
gravity_wave_timescale = 0.2 * minimum_xspacing(grid) / gravity_wave_speed

tidal_timescale = 1 / ω₂

Δt = 0.05 * tidal_timescale

@show prettytime(Δt)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = ImplicitFreeSurface(),
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    forcing = (u = tidal_forcing,))

simulation = Simulation(model, Δt=Δt, stop_time=7days)

# wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=Δt)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)

    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹, next Δt: %s\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

b = model.tracers.b
u, v, w = model.velocities

U = Field(Average(u))

u′ = u - U

N² = ∂z(b)

S² = @at (Center, Center, Face) ∂z(u)^2 + ∂z(v)^2

Ri = N² / S²

name = "barotropic_tide"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; Ri, N², u′, u, w, b),
                                                      schedule = TimeInterval(15minutes),
                                                      with_halos = false,
                                                      filename = name,
                                                      overwrite_existing = true)

# Initial conditions
uᵢ(x, y, z) = U_tidal

N² = 4e-5  # [s⁻²] buoyancy frequency / stratification
ϵb = 1e-7  # noise amplitude
bᵢ(x, y, z) = N² * z

set!(model, u=uᵢ, b=bᵢ)

using GLMakie
Makie.inline!(true)

fig = Figure()

ax = Axis(fig[1, 1],
          xlabel = "x [m]",
          ylabel = "depth [m]")

x, y, z = nodes(b)

hm = heatmap!(ax, x, z, interior(b, :, 1, :))
Colorbar(fig[1, 2], hm)

fig

run!(simulation)

saved_output_filename = name * ".jld2"

u_t  = FieldTimeSeries(saved_output_filename, "u")
u′_t = FieldTimeSeries(saved_output_filename, "u′")
w_t  = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")

times = u_t.times

xu,  yu,  zu  = nodes(u_t[1])
xw,  yw,  zw  = nodes(w_t[1])
xN², yN², zN² = nodes(N²_t[1])

using GLMakie
using Oceananigans.Grids: peripheral_node

n = Observable(1)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

is_oceanᶠᶜᶜ = [!peripheral_node(i, 1, k, grid, Face(), Center(), Center()) for i=1:grid.Nx, k=1:grid.Nz]
is_oceanᶜᶜᶠ = [!peripheral_node(i, 1, k, grid, Center(), Center(), Face()) for i=1:grid.Nx, k=1:grid.Nz+1]
        
u′ₙ = @lift ifelse.(is_oceanᶠᶜᶜ .== 0, NaN, interior(u′_t[$n], :, 1, :))
wₙ  = @lift ifelse.(is_oceanᶜᶜᶠ .== 0, NaN, interior( w_t[$n], :, 1, :))
N²ₙ = @lift ifelse.(is_oceanᶜᶜᶠ .== 0, NaN, interior(N²_t[$n], :, 1, :))

axis_kwargs = (xlabel = "x [m]",
               ylabel = "z [m]",
               limits = ((-Lx/2, Lx/2), (-H, 0)),
               titlesize = 20)

ulim   = maximum(abs, u_t[end])
wlim   = maximum(abs, w_t[end])
N²lims = minimum(N²_t[end]), maximum(N²_t[end])

fig = Figure(resolution = (600, 1100))

ax_u = Axis(fig[2, 1];
            title = "u′-velocity", axis_kwargs...)

ax_w = Axis(fig[3, 1];
            title = "w-velocity", axis_kwargs...)

ax_N² = Axis(fig[4, 1];
             title = "stratification", axis_kwargs...)

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hm_u = heatmap!(ax_u, xu, zu, u′ₙ;
                colorrange = (-ulim, ulim),
                colormap = :balance)
Colorbar(fig[2, 2], hm_u)

hm_w = heatmap!(ax_w, xw, zw, wₙ;
                colorrange = (-wlim, wlim),
                colormap = :balance)
Colorbar(fig[3, 2], hm_w)

hm_N² = heatmap!(ax_N², xN², zN², N²ₙ;
                 colorrange = N²lims,
                 colormap = :thermal)
Colorbar(fig[4, 2], hm_N²)

fig

@info "Making an animation from saved data..."

frames = 1:length(times)

record(fig, name * ".mp4", frames, framerate=24) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end


