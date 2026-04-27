#!/usr/bin/env python3
"""
Finds the correct B-field initialization method for your pywarpx version.
Run this BEFORE the next simulation to confirm the fix will work.

Run: python find_bfield_method.py
"""

import sys

print("="*70)
print("  pywarpx B-field initialization method finder")
print("="*70)

# Check pywarpx version
try:
    import pywarpx
    print(f"\n  pywarpx version: {pywarpx.__version__}")
except Exception as e:
    print(f"  pywarpx version check failed: {e}")

from pywarpx import picmi
constants = picmi.constants

# ---- Check 1: Simulation object attributes related to B-field ----
print("\n---- Simulation object: B-field related attributes ----")
sim = picmi.Simulation(verbose=0)
b_attrs = [a for a in dir(sim) if any(s in a.lower() for s in
           ['bfield', 'b_ext', 'external', 'b_init', 'bx', 'by', 'bz',
            'extra_input', 'parse'])]
for a in sorted(b_attrs):
    print(f"  sim.{a}")

# ---- Check 2: Grid object attributes related to B-field ----
print("\n---- Grid object: B-field related attributes ----")
grid = picmi.Cartesian2DGrid(
    number_of_cells=[8, 8],
    lower_bound=[-1, -1],
    upper_bound=[1, 1],
    lower_boundary_conditions=['periodic', 'periodic'],
    upper_boundary_conditions=['periodic', 'periodic'],
)
b_grid_attrs = [a for a in dir(grid) if any(s in a.lower() for s in
                ['bfield', 'b_ext', 'external', 'b_init', 'bx', 'by', 'bz',
                 'field', 'initial', 'parse', 'magnetic'])]
for a in sorted(b_grid_attrs):
    print(f"  grid.{a}")

# ---- Check 3: HybridPICSolver attributes ----
print("\n---- HybridPICSolver: B-field related attributes ----")
try:
    solver = picmi.HybridPICSolver(
        grid=grid,
        Te=1000.0,
        n0=1e24,
        plasma_resistivity=1e-3,
        substeps=10,
    )
    b_solver_attrs = [a for a in dir(solver) if any(s in a.lower() for s in
                      ['bfield', 'b_ext', 'external', 'b_init', 'bx', 'by', 'bz',
                       'field', 'initial', 'parse', 'magnetic'])]
    for a in sorted(b_solver_attrs):
        print(f"  solver.{a}")
except Exception as e:
    print(f"  ERROR creating solver: {e}")

# ---- Check 4: AnalyticAppliedField ----
print("\n---- AnalyticAppliedField: available parameters ----")
try:
    import inspect
    sig = inspect.signature(picmi.AnalyticAppliedField.__init__)
    print(f"  Parameters: {list(sig.parameters.keys())}")
except Exception as e:
    print(f"  ERROR: {e}")

# ---- Check 5: WarpX input parameter approach ----
print("\n---- Checking warpx_inputs / warpx_extra_inputs on Simulation ----")
extra_attrs = [a for a in dir(sim) if 'input' in a.lower() or 'extra' in a.lower()]
for a in sorted(extra_attrs):
    print(f"  sim.{a}")

# ---- Check 6: Look directly at pywarpx WarpX class ----
print("\n---- WarpX class direct attributes (B-field related) ----")
try:
    from pywarpx import libwarpx
    b_lib_attrs = [a for a in dir(libwarpx) if any(s in a.lower() for s in
                   ['bfield', 'b_ext', 'external', 'bx_ext', 'by_ext', 'bz_ext',
                    'b_external', 'magnetic'])]
    for a in sorted(b_lib_attrs):
        print(f"  libwarpx.{a}")
except Exception as e:
    print(f"  libwarpx check: {e}")

# ---- Check 7: Look at WarpXInputs / inputs structure ----
print("\n---- Checking pywarpx.inputs for B-field entries ----")
try:
    from pywarpx import _libwarpx
    b_input_attrs = [a for a in dir(_libwarpx) if any(s in a.lower() for s in
                     ['b_ext', 'bfield', 'by_ext', 'bx_ext'])]
    for a in sorted(b_input_attrs):
        print(f"  _libwarpx.{a}")
except Exception as e:
    print(f"  _libwarpx check: {e}")

# ---- Check 8: Check the WarpX example reconnection script ----
print("\n---- Looking at WarpX reconnection example for B-field method ----")
import os
reconnection_path = os.path.expanduser(
    "~/LaserFusionResearch/warpx/Examples/Tests/"
    "ohm_solver_magnetic_reconnection/"
    "inputs_test_2d_ohm_solver_magnetic_reconnection_picmi.py"
)
if os.path.exists(reconnection_path):
    with open(reconnection_path) as f:
        content = f.read()
    # Find B-field initialization lines
    lines = content.split('\n')
    b_lines = [l for l in lines if any(s in l.lower() for s in
               ['b_ext', 'bfield', 'by', 'bx', 'bz', 'magnetic', 'external',
                'initial', 'field_init', 'b_init'])]
    print(f"  Found {len(b_lines)} B-field related lines:")
    for l in b_lines[:30]:
        print(f"  {l}")
else:
    print(f"  Reconnection script not found at expected path")
    # Try to find it
    base = os.path.expanduser("~/LaserFusionResearch/warpx")
    for root, dirs, files in os.walk(base):
        for f in files:
            if 'reconnection' in f and f.endswith('.py'):
                print(f"  Found alternative: {os.path.join(root, f)}")

print("\n" + "="*70)
print("  PASTE THIS OUTPUT and the correct B-field method will be identified")
print("="*70)
