#!/usr/bin/env python3
"""
p-11B Debug Simulation — Short Diagnostic Run
Runs 500 steps (~3 min) with verbose text output to verify physics setup.

Run: mpirun -n 8 python pb11_debug.py > pb11_debug.log 2>&1

This writes all diagnostics to both terminal and pb11_debug.log for review.
"""

import numpy as np
from mpi4py import MPI as mpi
from pywarpx import callbacks, picmi

constants = picmi.constants
comm = mpi.COMM_WORLD
rank = comm.Get_rank()

def log(msg):
    """Only rank 0 prints, for clean output."""
    if rank == 0:
        print(msg, flush=True)

# ============================================================================
# PHYSICAL PARAMETERS (same as production, shortened run)
# ============================================================================
LASER_ENERGY_J        = 5.0
PULSE_DURATION_NS     = 2.0
ROTATION_FREQ_HZ      = 500e6

N_SPOTS               = 8
RING_RADIUS_M         = 800e-6
SPOT_RADIUS_M         = 25e-6

AMU_KG                = 1.66054e-27
B11_MASS_KG           = 11.0093 * AMU_KG
PROTON_MASS_KG        = 1.00728 * AMU_KG

N_B11_DENSITY_M3      = 5e24
N_PROTON_DENSITY_M3   = 5e24

T_ION_INITIAL_EV      = 660.0
T_ELEC_INITIAL_EV     = 2200.0
B_BIERMANN_INITIAL_T  = 850.0

# Derived
n_total = N_B11_DENSITY_M3 + N_PROTON_DENSITY_M3
mass_avg = (N_B11_DENSITY_M3 * B11_MASS_KG +
            N_PROTON_DENSITY_M3 * PROTON_MASS_KG) / n_total
v_alfven = B_BIERMANN_INITIAL_T / np.sqrt(
    constants.mu0 * n_total * mass_avg)
omega_ci_proton = constants.q_e * B_BIERMANN_INITIAL_T / PROTON_MASS_KG
omega_pi_proton = np.sqrt(
    N_PROTON_DENSITY_M3 * constants.q_e**2 /
    (constants.ep0 * PROTON_MASS_KG))
d_i_proton = constants.c / omega_pi_proton
v_th_proton = np.sqrt(T_ION_INITIAL_EV * constants.q_e / PROTON_MASS_KG)
v_th_b11 = np.sqrt(T_ION_INITIAL_EV * constants.q_e / B11_MASS_KG)

log("="*78)
log("  p-11B DEBUG RUN — physics validation diagnostics")
log("="*78)
log(f"  Alfven speed:          {v_alfven:.3e} m/s")
log(f"  Proton skin depth:     {d_i_proton*1e6:.2f} um")
log(f"  Proton cyclotron freq: {omega_ci_proton:.3e} rad/s")
log(f"  Proton thermal speed:  {v_th_proton:.3e} m/s")
log(f"  B-11 thermal speed:    {v_th_b11:.3e} m/s")
log(f"  Initial B-field peak:  {B_BIERMANN_INITIAL_T:.1f} T")
log(f"  Proton density:        {N_PROTON_DENSITY_M3:.2e} m^-3")
log(f"  B-11 density:          {N_B11_DENSITY_M3:.2e} m^-3")
log("="*78)

# ============================================================================
# SHORT DEBUG DOMAIN
# ============================================================================
LX_DI = 60
LZ_DI = 60
LX_M = LX_DI * d_i_proton
LZ_M = LZ_DI * d_i_proton

NX, NZ = 128, 128
NPPC = 50
N_STEPS = 500  # Short debug run
DT = 1e-3
time_step_s = DT / omega_ci_proton
total_time_s = N_STEPS * time_step_s

log(f"\n  DEBUG DOMAIN:")
log(f"  Physical size:  {LX_M*1e6:.0f} x {LZ_M*1e6:.0f} um")
log(f"  Grid:           {NX} x {NZ}")
log(f"  Steps:          {N_STEPS}")
log(f"  Time per step:  {time_step_s*1e15:.2f} fs")
log(f"  Total sim time: {total_time_s*1e12:.3f} ps")
log("="*78)

# ============================================================================
# SIMULATION
# ============================================================================
simulation = picmi.Simulation(
    warpx_serialize_initial_conditions=True,
    verbose=1,
)

grid = picmi.Cartesian2DGrid(
    number_of_cells=[NX, NZ],
    lower_bound=[-LX_M/2, -LZ_M/2],
    upper_bound=[ LX_M/2,  LZ_M/2],
    lower_boundary_conditions=['periodic', 'periodic'],
    upper_boundary_conditions=['periodic', 'periodic'],
    lower_boundary_conditions_particles=['periodic', 'periodic'],
    upper_boundary_conditions_particles=['periodic', 'periodic'],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=T_ELEC_INITIAL_EV * constants.q_e / constants.kb,
    n0=n_total,
    plasma_resistivity=6e-3,
    substeps=20,
)
simulation.solver = solver

# Ring density expression
def ring_density_expr(peak_density):
    terms = []
    for k in range(N_SPOTS):
        angle = 2 * np.pi * k / N_SPOTS
        cx = RING_RADIUS_M * np.cos(angle)
        cy = RING_RADIUS_M * np.sin(angle)
        terms.append(f"exp(-(((x - {cx})**2 + (z - {cy})**2) / "
                     f"(2*{SPOT_RADIUS_M}**2)))")
    return f"{peak_density} * (" + " + ".join(terms) + ")"

# Biermann B-field expression
def biermann_By():
    terms = []
    for k in range(N_SPOTS):
        angle = 2 * np.pi * k / N_SPOTS
        cx = RING_RADIUS_M * np.cos(angle)
        cy = RING_RADIUS_M * np.sin(angle)
        sign = (-1)**k
        terms.append(f"({sign} * {B_BIERMANN_INITIAL_T} * "
                     f"sqrt((x-{cx})**2 + (z-{cy})**2) / {SPOT_RADIUS_M} * "
                     f"exp(-0.5*((x-{cx})**2 + (z-{cy})**2) / {SPOT_RADIUS_M}**2))")
    return " + ".join(terms)

proton_species = picmi.Species(
    particle_type='H',
    name='proton',
    charge='q_e',
    mass=PROTON_MASS_KG,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=ring_density_expr(N_PROTON_DENSITY_M3),
        rms_velocity=[v_th_proton, v_th_proton, v_th_proton],
        directed_velocity=[0.0, 0.0, 0.0],
    ),
)

b11_species = picmi.Species(
    name='boron11',
    charge=5 * constants.q_e,
    mass=B11_MASS_KG,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=ring_density_expr(N_B11_DENSITY_M3 / 5),
        rms_velocity=[v_th_b11, v_th_b11, v_th_b11],
        directed_velocity=[0.0, 0.0, 0.0],
    ),
)

simulation.add_species(
    proton_species,
    layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=NPPC)
)
simulation.add_species(
    b11_species,
    layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=NPPC)
)

# Apply initial B-field to grid
grid.warpx_initial_B_field_x = "0"
grid.warpx_initial_B_field_y = biermann_By()
grid.warpx_initial_B_field_z = "0"

# ============================================================================
# DIAGNOSTIC CALLBACK — text output at each key step
# ============================================================================
def diagnostic_callback():
    """Print physics diagnostics every 50 steps."""
    step = simulation.extension.warpx.getistep(lev=0)
    
    if step % 50 != 0 and step != 1:
        return
    
    if rank != 0:
        return
    
    t = step * time_step_s
    print(f"\n---- STEP {step:5d}  t={t*1e12:7.3f} ps ----", flush=True)

callbacks.installafterstep(diagnostic_callback)

# ============================================================================
# DIAGNOSTICS (minimal, for analysis after the fact)
# ============================================================================
field_diag = picmi.FieldDiagnostic(
    name='debug_fields',
    grid=grid,
    period=100,
    data_list=['B', 'E', 'J', 'rho'],
    write_dir='./pb11_debug_diags',
    warpx_format='openpmd',
)
simulation.add_diagnostic(field_diag)

particle_diag = picmi.ParticleDiagnostic(
    name='debug_particles',
    period=100,
    species=[proton_species, b11_species],
    data_list=['position', 'momentum', 'weighting'],
    write_dir='./pb11_debug_diags',
    warpx_format='openpmd',
)
simulation.add_diagnostic(particle_diag)

simulation.time_step_size = time_step_s
simulation.max_steps = N_STEPS

log("\nStarting debug simulation...\n")
simulation.step(N_STEPS)

log("\n" + "="*78)
log("  DEBUG SIMULATION COMPLETE")
log("="*78)
log("  Next: run pb11_text_analysis.py to extract physics numbers")
log("="*78)
