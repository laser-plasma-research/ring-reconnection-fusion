#!/usr/bin/env python3
"""
p-11B Text-Based Analysis — No Images, Just Numbers
Reads simulation output and prints detailed physics diagnostics to terminal.

Run:  python pb11_text_analysis.py [--dir pb11_debug_diags]
      python pb11_text_analysis.py --dir pb11_diags    # for full production run
"""

import os
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='pb11_debug_diags',
                    help='Diagnostics directory (default: pb11_debug_diags)')
args = parser.parse_args()

try:
    import openpmd_viewer as ov
except ImportError:
    print("Installing openpmd-viewer...")
    os.system("pip install openpmd-viewer --break-system-packages")
    import openpmd_viewer as ov

# Constants
PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
Q_E            = 1.602e-19
C              = 2.998e8
KEV_TO_J       = Q_E * 1e3

DIAG_DIR = args.dir
if not os.path.exists(DIAG_DIR):
    print(f"ERROR: {DIAG_DIR} not found")
    print(f"Available directories:")
    for d in os.listdir('.'):
        if os.path.isdir(d):
            print(f"  {d}")
    sys.exit(1)

print("\n" + "="*78)
print("  p-11B SIMULATION — TEXT DIAGNOSTIC REPORT")
print("="*78)
print(f"  Analyzing: {DIAG_DIR}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================
try:
    ts_fields = ov.OpenPMDTimeSeries(f'./{DIAG_DIR}/debug_fields/')
    ts_particles = ov.OpenPMDTimeSeries(f'./{DIAG_DIR}/debug_particles/')
except Exception:
    try:
        ts_fields = ov.OpenPMDTimeSeries(f'./{DIAG_DIR}/fields/')
        ts_particles = ov.OpenPMDTimeSeries(f'./{DIAG_DIR}/particles/')
    except Exception as e:
        print(f"ERROR loading: {e}")
        sys.exit(1)

field_its = ts_fields.iterations
particle_its = ts_particles.iterations
field_times = ts_fields.t
particle_times = ts_particles.t

print(f"  Field snapshots:    {len(field_its)}")
print(f"  Particle snapshots: {len(particle_its)}")
print(f"  Time range:         {field_times[0]*1e12:.2f} to {field_times[-1]*1e12:.2f} ps")
print("="*78)

# ============================================================================
# CHECK 1 — AVAILABLE FIELD COMPONENTS
# ============================================================================
print("\n" + "="*78)
print("  CHECK 1: FIELD DATA AVAILABILITY")
print("="*78)

try:
    it0 = field_its[0]
    # Get available fields
    available = ts_fields.avail_fields
    print(f"\n  Available fields: {available}")
    components = {}
    for f in available:
        try:
            comps = ts_fields.fields_metadata[f].get('axis_labels', [])
            components[f] = comps
        except Exception:
            pass
    for f, c in components.items():
        print(f"    {f}: components {c}")
except Exception as e:
    print(f"  Error reading fields: {e}")

# ============================================================================
# CHECK 2 — INITIAL B-FIELD MAGNITUDE
# ============================================================================
print("\n" + "="*78)
print("  CHECK 2: INITIAL MAGNETIC FIELD (verifies Biermann initialization)")
print("="*78)

try:
    By, info = ts_fields.get_field('B', 'y', iteration=field_its[0])
    Bx, _    = ts_fields.get_field('B', 'x', iteration=field_its[0])
    Bz, _    = ts_fields.get_field('B', 'z', iteration=field_its[0])
    B_mag    = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    print(f"\n  At t=0:")
    print(f"    Bx: min={Bx.min():.3e} T  max={Bx.max():.3e} T  rms={np.sqrt(np.mean(Bx**2)):.3e} T")
    print(f"    By: min={By.min():.3e} T  max={By.max():.3e} T  rms={np.sqrt(np.mean(By**2)):.3e} T")
    print(f"    Bz: min={Bz.min():.3e} T  max={Bz.max():.3e} T  rms={np.sqrt(np.mean(Bz**2)):.3e} T")
    print(f"    |B| max: {B_mag.max():.3e} T")
    print(f"    |B| mean: {B_mag.mean():.3e} T")
    
    # Field at centre
    nx, nz = B_mag.shape
    B_centre = B_mag[nx//2-5:nx//2+5, nz//2-5:nz//2+5].mean()
    print(f"    |B| at ring centre: {B_centre:.3e} T")
    
    # Compare to expected
    print(f"\n  Expected: Biermann field should peak around 850 T near spots")
    if By.max() > 100:
        print(f"  RESULT:   ✓ B-field was initialized (peak {By.max():.1f} T)")
    else:
        print(f"  RESULT:   ✗ B-field NOT initialized properly (max only {By.max():.3e} T)")
        print(f"            This is a CRITICAL issue — no reconnection without B-field")
except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# CHECK 3 — B-FIELD EVOLUTION
# ============================================================================
print("\n" + "="*78)
print("  CHECK 3: B-FIELD EVOLUTION OVER TIME")
print("="*78)

print(f"\n  Time (ps) | |B|_max (T) | |B|_mean (T) | |B|_centre (T)")
print(f"  ----------+-------------+--------------+----------------")
for i, it in enumerate(field_its):
    try:
        Bx, info = ts_fields.get_field('B', 'x', iteration=it)
        By, _    = ts_fields.get_field('B', 'y', iteration=it)
        Bz, _    = ts_fields.get_field('B', 'z', iteration=it)
        B_mag    = np.sqrt(Bx**2 + By**2 + Bz**2)
        nx, nz = B_mag.shape
        B_centre = B_mag[nx//2-5:nx//2+5, nz//2-5:nz//2+5].mean()
        t_ps = field_times[i] * 1e12
        print(f"  {t_ps:>8.2f} | {B_mag.max():>11.2e} | {B_mag.mean():>12.2e} | {B_centre:>14.2e}")
    except Exception:
        pass

# ============================================================================
# CHECK 4 — INITIAL PARTICLE STATE
# ============================================================================
print("\n" + "="*78)
print("  CHECK 4: INITIAL PARTICLE STATE (verifies thermal distribution)")
print("="*78)

try:
    x_p, z_p, ux_p, uy_p, uz_p, w_p = ts_particles.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz', 'w'],
        species='proton',
        iteration=particle_its[0]
    )
    
    print(f"\n  PROTONS at t=0:")
    print(f"    Count (macro):    {len(x_p):,}")
    print(f"    Physical count:   {w_p.sum():.3e}")
    print(f"    X range:          {x_p.min()*1e6:.1f} to {x_p.max()*1e6:.1f} um")
    print(f"    Z range:          {z_p.min()*1e6:.1f} to {z_p.max()*1e6:.1f} um")
    
    # Velocity in m/s (WarpX stores gamma*v in ux, not v)
    v_p = np.sqrt(ux_p**2 + uy_p**2 + uz_p**2)
    E_p_keV = 0.5 * PROTON_MASS_KG * v_p**2 * C**2 / KEV_TO_J
    
    print(f"\n    Velocity (m/s):")
    print(f"      mean |v|:       {v_p.mean()*C:.3e}")
    print(f"      std |v|:        {v_p.std()*C:.3e}")
    print(f"      max |v|:        {v_p.max()*C:.3e}")
    
    print(f"\n    Kinetic energy (keV):")
    print(f"      mean:           {E_p_keV.mean():.3f}")
    print(f"      median:         {np.median(E_p_keV):.3f}")
    print(f"      95th %ile:      {np.percentile(E_p_keV, 95):.3f}")
    print(f"      99th %ile:      {np.percentile(E_p_keV, 99):.3f}")
    print(f"      max:            {E_p_keV.max():.3f}")
    print(f"    Expected thermal: ~660 eV = 0.66 keV (initial temperature)")
    
    if E_p_keV.mean() > 5:
        print(f"    WARNING: Initial mean energy too high — distribution is not thermal")
    else:
        print(f"    ✓ Initial distribution consistent with thermal at 660 eV")
except Exception as e:
    print(f"  ERROR: {e}")

# B-11 initial state
try:
    x_b, z_b, ux_b, uy_b, uz_b, w_b = ts_particles.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz', 'w'],
        species='boron11',
        iteration=particle_its[0]
    )
    
    v_b = np.sqrt(ux_b**2 + uy_b**2 + uz_b**2)
    E_b_keV = 0.5 * B11_MASS_KG * v_b**2 * C**2 / KEV_TO_J
    
    print(f"\n  BORON-11 at t=0:")
    print(f"    Count (macro):    {len(x_b):,}")
    print(f"    Physical count:   {w_b.sum():.3e}")
    print(f"    Energy mean:      {E_b_keV.mean():.3f} keV")
    print(f"    Energy max:       {E_b_keV.max():.3f} keV")
    print(f"    Expected thermal: ~660 eV = 0.66 keV")
except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# CHECK 5 — PARTICLE EVOLUTION
# ============================================================================
print("\n" + "="*78)
print("  CHECK 5: PROTON ENERGY EVOLUTION")
print("="*78)

print(f"\n  Time (ps) | E_mean (keV) | E_95th (keV) | E_max (keV) | N_centre | N_outer")
print(f"  ----------+--------------+--------------+-------------+----------+----------")

for i, it in enumerate(particle_its):
    try:
        x, z, ux, uy, uz = ts_particles.get_particle(
            ['x', 'z', 'ux', 'uy', 'uz'],
            species='proton', iteration=it
        )
        v_sq = (ux**2 + uy**2 + uz**2) * C**2
        E_keV = 0.5 * PROTON_MASS_KG * v_sq / KEV_TO_J
        r = np.sqrt(x**2 + z**2)
        R_centre = 200e-6
        centre = r < R_centre
        n_centre = centre.sum()
        n_outer = (~centre).sum()
        t_ps = particle_times[i] * 1e12
        print(f"  {t_ps:>8.2f} | {E_keV.mean():>12.2f} | {np.percentile(E_keV, 95):>12.2f} "
              f"| {E_keV.max():>11.2f} | {n_centre:>8} | {n_outer:>8}")
    except Exception:
        pass

# ============================================================================
# CHECK 6 — CHARGE CONSERVATION (sanity check)
# ============================================================================
print("\n" + "="*78)
print("  CHECK 6: CHARGE CONSERVATION")
print("="*78)

print(f"\n  Time (ps) | Proton total | B11 total | Proton charge | B11 charge | Total charge")
print(f"  ----------+--------------+-----------+---------------+------------+-------------")
for i, it in enumerate(particle_its):
    try:
        _, _, _, _, _, w_p = ts_particles.get_particle(
            ['x', 'z', 'ux', 'uy', 'uz', 'w'],
            species='proton', iteration=it
        )
        _, _, _, _, _, w_b = ts_particles.get_particle(
            ['x', 'z', 'ux', 'uy', 'uz', 'w'],
            species='boron11', iteration=it
        )
        n_p = w_p.sum()
        n_b = w_b.sum()
        q_p = n_p * 1.0  # Z=1
        q_b = n_b * 5.0  # Z=5
        q_total = q_p + q_b  # ions only (no electrons in hybrid-PIC)
        t_ps = particle_times[i] * 1e12
        print(f"  {t_ps:>8.2f} | {n_p:>12.3e} | {n_b:>9.3e} | {q_p:>13.3e} | "
              f"{q_b:>10.3e} | {q_total:>12.3e}")
    except Exception:
        pass

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*78)
print("  DIAGNOSTIC SUMMARY")
print("="*78)

print("""
  Key questions to answer from this output:

  1. Did the B-field initialize correctly?
     → Check #2: |B|_max should be ~850 T at t=0
     → If |B|_max is near 0, the Biermann field is NOT being applied

  2. Is the initial particle distribution thermal?
     → Check #4: mean proton energy should be ~0.66 keV
     → If mean >> 1 keV at t=0, something is wrong with initial conditions

  3. Is the B-field decaying correctly?
     → Check #3: B-field should evolve, not drop to zero immediately
     → Should see reconnection signature (B drops at X-lines, stays in spots)

  4. Are protons being heated by reconnection?
     → Check #5: E_95th should grow over time but stay physically reasonable
     → Peak E should NOT exceed ~1 MeV at debug timescale
     → If E_max > 5 MeV at 500 steps, numerical artifact is likely

  5. Are particles converging to centre?
     → Check #5: N_centre should grow over time if convergence works
     → If N_centre stays flat, the convergence mechanism is missing
""")

print("="*78)
