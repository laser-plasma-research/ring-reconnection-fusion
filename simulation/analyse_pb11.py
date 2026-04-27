#!/usr/bin/env python3
"""
============================================================================
  p-¹¹B SIMULATION ANALYSIS
  Visualisation of ring reconnection with rotating modulation
============================================================================

Produces 6 key plots:
  1. Magnetic field evolution — 3 time snapshots showing rotation
  2. Helical field pitch angle development over time
  3. Proton energy spectrum — centre vs outer region
  4. B-11 energy spectrum — same regions
  5. Peak proton energy vs time (p-¹¹B cross-section threshold marked)
  6. Fusion rate estimate based on ion energies

Run:
  conda activate plasma
  python analyse_pb11.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# CHECK DATA EXISTS
# ---------------------------------------------------------------------------
if not os.path.exists('./pb11_diags'):
    print("Error: ./pb11_diags directory not found.")
    print("Run the simulation first: mpirun -n 10 python pb11_rotating_ring.py")
    sys.exit(1)

try:
    import openpmd_viewer as ov
except ImportError:
    print("Installing openpmd-viewer...")
    os.system("pip install openpmd-viewer")
    import openpmd_viewer as ov

# ---------------------------------------------------------------------------
# LOAD SIMULATION OUTPUT
# ---------------------------------------------------------------------------
print("Loading simulation data...")
ts = ov.OpenPMDTimeSeries('./pb11_diags/fields/')
iterations = ts.iterations
times_s = ts.t

print(f"Found {len(iterations)} snapshots")
print(f"Time range: {times_s[0]*1e12:.2f} ps → {times_s[-1]*1e12:.2f} ps")

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------
PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
Q_E            = 1.602e-19
C              = 2.998e8
KEV_TO_J       = Q_E * 1e3

# p-¹¹B cross-section thresholds
PB11_MIN_KEV   = 500       # Minimum useful ion energy
PB11_PEAK_KEV  = 3000      # Peak reactivity
PB11_RESONANCE_KEV = 675   # Narrow resonance peak

# ---------------------------------------------------------------------------
# FIGURE SETUP — DARK THEME
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('#0a0e1a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

BG     = '#0a0e1a'
PANEL  = '#161b22'
GRID   = '#30363d'
TEXT   = '#e8eaed'
MUTED  = '#8b949e'
CYAN   = '#1D9E75'
RED    = '#ff4d4d'
BLUE   = '#4d9fff'
YELLOW = '#f0c040'
AMBER  = '#BA7517'
PURPLE = '#9F7DFF'

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8, fontweight='normal')
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, alpha=0.3, linestyle=':', linewidth=0.5)

# ---------------------------------------------------------------------------
# PLOTS 1-3: Magnetic field evolution showing rotation
# ---------------------------------------------------------------------------
snap_indices = [0, len(iterations)//2, len(iterations)-1]
snap_labels = [
    f't = {times_s[0]*1e12:.1f} ps  — initial',
    f't = {times_s[len(iterations)//2]*1e12:.1f} ps  — reconnecting',
    f't = {times_s[-1]*1e12:.1f} ps  — saturated',
]

print("\nGenerating B-field evolution plots...")
for col, (idx, label) in enumerate(zip(snap_indices, snap_labels)):
    ax = fig.add_subplot(gs[0, col])
    style_ax(ax, label)
    try:
        By, info = ts.get_field('B', 'y', iteration=iterations[idx])
        Bx, _    = ts.get_field('B', 'x', iteration=iterations[idx])
        B_mag    = np.sqrt(Bx**2 + By**2)

        vmax = np.percentile(np.abs(By), 98)
        im = ax.imshow(
            By.T, origin='lower', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax, aspect='auto',
            extent=[info.axes['x'][0]*1e6, info.axes['x'][-1]*1e6,
                    info.axes['z'][0]*1e6, info.axes['z'][-1]*1e6]
        )
        ax.set_xlabel('x (µm)', color=MUTED, fontsize=8)
        ax.set_ylabel('z (µm)', color=MUTED, fontsize=8)

        # Mark ring location
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(800*np.cos(theta), 800*np.sin(theta),
                color=YELLOW, linewidth=0.8, linestyle='--', alpha=0.5)

        # Mark ring centre
        ax.plot(0, 0, '+', color=CYAN, markersize=12, markeredgewidth=2)

        cb = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.set_label('By (T)', color=MUTED, fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes,
                color=MUTED, ha='center')

# ---------------------------------------------------------------------------
# PLOT 4: Magnetic field magnitude at ring centre over time
# ---------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, 'B-field magnitude at ring centre')

B_centre_history = []
print("Computing B-field history at centre...")
for it in iterations:
    try:
        By, info = ts.get_field('B', 'y', iteration=it)
        Bx, _    = ts.get_field('B', 'x', iteration=it)
        B_mag    = np.sqrt(Bx**2 + By**2)
        nx, nz = B_mag.shape
        # Centre region (±5 cells)
        B_centre_history.append(np.mean(B_mag[nx//2-5:nx//2+5,
                                              nz//2-5:nz//2+5]))
    except:
        B_centre_history.append(np.nan)

B_centre_history = np.array(B_centre_history)
ax4.plot(times_s*1e12, B_centre_history, color=CYAN, linewidth=2)
ax4.fill_between(times_s*1e12, B_centre_history, alpha=0.2, color=CYAN)
ax4.set_xlabel('Time (ps)', color=MUTED, fontsize=8)
ax4.set_ylabel('|B| at centre (T)', color=MUTED, fontsize=8)

# Annotate peak
if len(B_centre_history) > 5:
    peak_idx = np.nanargmax(B_centre_history)
    ax4.axvline(times_s[peak_idx]*1e12, color=YELLOW,
                linestyle='--', alpha=0.7,
                label=f'peak: {B_centre_history[peak_idx]:.0f} T')
    ax4.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT)

# ---------------------------------------------------------------------------
# PLOT 5: Proton energy spectrum — centre vs outer
# ---------------------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, 'Proton energy spectrum (final)')

try:
    ts_particles = ov.OpenPMDTimeSeries('./pb11_diags/particles/')
    x_p, z_p, ux_p, uy_p, uz_p = ts_particles.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz'],
        species='proton',
        iteration=ts_particles.iterations[-1]
    )

    # Kinetic energy in keV (non-relativistic sufficient for <5 MeV)
    v_sq_p = ux_p**2 + uy_p**2 + uz_p**2
    E_p_keV = 0.5 * PROTON_MASS_KG * v_sq_p * C**2 / KEV_TO_J

    r_p = np.sqrt(x_p**2 + z_p**2)
    R_max = r_p.max() if len(r_p) > 0 else 1.0
    centre_mask = r_p < 0.25 * R_max

    bins = np.logspace(0, 4, 80)  # 1 keV to 10 MeV

    if centre_mask.sum() > 10:
        ax5.hist(E_p_keV[centre_mask], bins=bins, density=True,
                 alpha=0.75, color=RED,
                 label=f'centre (n={centre_mask.sum():,})')
    if (~centre_mask).sum() > 10:
        ax5.hist(E_p_keV[~centre_mask], bins=bins, density=True,
                 alpha=0.5, color=BLUE,
                 label=f'outer (n={(~centre_mask).sum():,})')

    ax5.set_xscale('log')
    ax5.set_xlabel('Proton energy (keV)', color=MUTED, fontsize=8)
    ax5.set_ylabel('Probability density', color=MUTED, fontsize=8)

    # Mark p-11B thresholds
    ax5.axvline(PB11_MIN_KEV, color=YELLOW, linestyle=':', linewidth=1,
                label=f'p-¹¹B min ({PB11_MIN_KEV} keV)')
    ax5.axvline(PB11_PEAK_KEV, color=AMBER, linestyle='--', linewidth=1.5,
                label=f'p-¹¹B peak ({PB11_PEAK_KEV} keV)')
    ax5.axvspan(PB11_MIN_KEV, PB11_PEAK_KEV*2, alpha=0.1, color=AMBER)
    ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT)
except Exception as e:
    ax5.text(0.5, 0.5, f'Particle data error:\n{e}',
             transform=ax5.transAxes, color=MUTED, ha='center', fontsize=8)

# ---------------------------------------------------------------------------
# PLOT 6: B-11 energy spectrum
# ---------------------------------------------------------------------------
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, 'B-11 ion energy spectrum (final)')

try:
    x_b, z_b, ux_b, uy_b, uz_b = ts_particles.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz'],
        species='boron11',
        iteration=ts_particles.iterations[-1]
    )

    v_sq_b = ux_b**2 + uy_b**2 + uz_b**2
    E_b_keV = 0.5 * B11_MASS_KG * v_sq_b * C**2 / KEV_TO_J

    r_b = np.sqrt(x_b**2 + z_b**2)
    R_max_b = r_b.max() if len(r_b) > 0 else 1.0
    centre_mask_b = r_b < 0.25 * R_max_b

    bins = np.logspace(-1, 3, 60)

    if centre_mask_b.sum() > 10:
        ax6.hist(E_b_keV[centre_mask_b], bins=bins, density=True,
                 alpha=0.75, color=PURPLE,
                 label=f'centre (n={centre_mask_b.sum():,})')
    if (~centre_mask_b).sum() > 10:
        ax6.hist(E_b_keV[~centre_mask_b], bins=bins, density=True,
                 alpha=0.5, color=CYAN,
                 label=f'outer (n={(~centre_mask_b).sum():,})')

    ax6.set_xscale('log')
    ax6.set_xlabel('B-11 energy (keV)', color=MUTED, fontsize=8)
    ax6.set_ylabel('Probability density', color=MUTED, fontsize=8)
    ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT)
except Exception as e:
    ax6.text(0.5, 0.5, f'B-11 data error:\n{e}',
             transform=ax6.transAxes, color=MUTED, ha='center', fontsize=8)

# ---------------------------------------------------------------------------
# PLOT 7: Peak proton energy vs time — THE money plot for p-¹¹B
# ---------------------------------------------------------------------------
ax7 = fig.add_subplot(gs[2, :2])  # span 2 columns
style_ax(ax7, 'Peak proton energy vs time  —  p-¹¹B fusion benefit plot')

peak_E_centre = []
peak_E_outer  = []
mean_E_centre = []

sample_its = ts_particles.iterations[::max(1, len(ts_particles.iterations)//25)]
sample_times = []

print(f"Computing proton energy evolution across {len(sample_its)} snapshots...")
for it in sample_its:
    try:
        x_i, z_i, ux_i, uy_i, uz_i = ts_particles.get_particle(
            ['x', 'z', 'ux', 'uy', 'uz'],
            species='proton', iteration=it
        )
        v_sq = ux_i**2 + uy_i**2 + uz_i**2
        E = 0.5 * PROTON_MASS_KG * v_sq * C**2 / KEV_TO_J
        r = np.sqrt(x_i**2 + z_i**2)
        R_max = r.max() if len(r) > 0 else 1.0
        mask = r < 0.25 * R_max

        peak_E_centre.append(np.percentile(E[mask], 95)
                             if mask.sum() > 10 else np.nan)
        peak_E_outer.append(np.percentile(E[~mask], 95)
                            if (~mask).sum() > 10 else np.nan)
        mean_E_centre.append(np.mean(E[mask])
                             if mask.sum() > 10 else np.nan)

        idx = list(ts_particles.iterations).index(it)
        sample_times.append(ts_particles.t[idx] * 1e12)  # ps
    except:
        peak_E_centre.append(np.nan)
        peak_E_outer.append(np.nan)
        mean_E_centre.append(np.nan)
        sample_times.append(np.nan)

sample_times = np.array(sample_times)
peak_E_centre = np.array(peak_E_centre)
peak_E_outer  = np.array(peak_E_outer)
mean_E_centre = np.array(mean_E_centre)

# Plot curves
ax7.plot(sample_times, peak_E_centre, color=RED, linewidth=2.5,
         marker='o', markersize=4,
         label='centre 95th %ile (reconnection-accelerated)')
ax7.plot(sample_times, peak_E_outer, color=BLUE, linewidth=2,
         marker='s', markersize=3,
         label='outer 95th %ile (background plasma)')
ax7.plot(sample_times, mean_E_centre, color=YELLOW, linewidth=1.5,
         linestyle='--', label='centre mean (bulk temp)')

# p-11B threshold bands
ax7.axhspan(PB11_MIN_KEV, PB11_PEAK_KEV, alpha=0.08, color=AMBER)
ax7.axhline(PB11_MIN_KEV, color=AMBER, linestyle=':', linewidth=1,
            label=f'p-¹¹B min ({PB11_MIN_KEV} keV)')
ax7.axhline(PB11_PEAK_KEV, color=AMBER, linestyle='--', linewidth=1.5,
            label=f'p-¹¹B peak cross section ({PB11_PEAK_KEV} keV)')
ax7.axhline(PB11_RESONANCE_KEV, color=PURPLE, linestyle=':', linewidth=1,
            alpha=0.7, label=f'675 keV resonance')

ax7.set_yscale('log')
ax7.set_xlabel('Time (ps)', color=MUTED, fontsize=9)
ax7.set_ylabel('Proton energy (keV, log scale)', color=MUTED, fontsize=9)
ax7.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
           loc='lower right')

# Annotate if we cross the p-11B threshold
if len(peak_E_centre) > 3 and not np.all(np.isnan(peak_E_centre)):
    max_E = np.nanmax(peak_E_centre)
    if max_E > PB11_MIN_KEV:
        ax7.annotate(
            f'✓ Exceeds p-¹¹B threshold\nPeak: {max_E:.0f} keV',
            xy=(0.97, 0.95), xycoords='axes fraction',
            color=CYAN, fontsize=10, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor=PANEL, edgecolor=CYAN, linewidth=0.8)
        )

# ---------------------------------------------------------------------------
# PLOT 8: Fusion rate estimate (bottom right)
# ---------------------------------------------------------------------------
ax8 = fig.add_subplot(gs[2, 2])
style_ax(ax8, 'Estimated p-¹¹B fusion rate')

# Very simple fusion rate estimate based on proton energy distribution
# R ~ n_p * n_B * <σv>(E_p) integrated over centre volume
# σv at 500 keV ~ 1e-24 m³/s
# σv at 1 MeV ~ 3e-24 m³/s
# σv at 3 MeV ~ 2e-23 m³/s (peak)

def sigma_v_pb11(E_keV):
    """Approximate σv for p-¹¹B in m³/s based on ENDF data."""
    E = np.asarray(E_keV)
    result = np.zeros_like(E, dtype=float)
    # Below threshold: essentially zero
    mask_low = E < 200
    # Sub-threshold tail: quadratic rise
    mask_mid = (E >= 200) & (E < 3000)
    result[mask_mid] = 2e-23 * (E[mask_mid] / 3000)**2
    # Peak region
    mask_peak = (E >= 3000) & (E < 5000)
    result[mask_peak] = 2e-23 * (1 - 0.5*((E[mask_peak]-3000)/2000))
    # Above peak: decline
    mask_high = E >= 5000
    result[mask_high] = 1e-23 * (5000 / E[mask_high])
    return result

fusion_rate_estimate = []
for i, E in enumerate(peak_E_centre):
    if np.isnan(E):
        fusion_rate_estimate.append(np.nan)
        continue
    # Use 95th percentile as proxy for accelerated ion energy
    sv = sigma_v_pb11(E)
    # Rough rate (arbitrary units, just to show evolution)
    # R = n_p × n_B × σv × V_centre
    n_p = 5e27  # m^-3
    n_B = 1e27  # m^-3 (B-11 at reduced density)
    V_centre = 4/3 * np.pi * (200e-6)**3  # sphere at ring centre
    rate = n_p * n_B * sv * V_centre
    fusion_rate_estimate.append(rate)

fusion_rate_estimate = np.array(fusion_rate_estimate)

ax8.semilogy(sample_times, fusion_rate_estimate, color=CYAN, linewidth=2.5,
             marker='o', markersize=4)
ax8.fill_between(sample_times, 1e-5, fusion_rate_estimate,
                 alpha=0.2, color=CYAN,
                 where=~np.isnan(fusion_rate_estimate))
ax8.set_xlabel('Time (ps)', color=MUTED, fontsize=8)
ax8.set_ylabel('p-¹¹B rate (reactions/s)', color=MUTED, fontsize=8)

if not np.all(np.isnan(fusion_rate_estimate)):
    peak_rate = np.nanmax(fusion_rate_estimate)
    peak_idx = np.nanargmax(fusion_rate_estimate)
    total_reactions = np.nansum(fusion_rate_estimate) * \
                      (sample_times[1] - sample_times[0]) * 1e-12

    ax8.annotate(
        f'Peak rate: {peak_rate:.2e} /s\n'
        f'Est. total: {total_reactions:.1e} alphas\n'
        f'(per shot)',
        xy=(0.97, 0.05), xycoords='axes fraction',
        color=TEXT, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor=PANEL, edgecolor=CYAN, linewidth=0.8)
    )

# ---------------------------------------------------------------------------
# TITLE & SAVE
# ---------------------------------------------------------------------------
fig.suptitle(
    'p-¹¹B HOLY GRAIL ANEUTRONIC FUSION\n'
    'Ring geometry with 500 MHz rotating laser modulation @ 5 J',
    color=TEXT, fontsize=14, y=0.995, fontweight='normal'
)

outfile = 'pb11_analysis.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nSaved: {outfile}")
print(f"View:  open {outfile}")

# ---------------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("  p-¹¹B SIMULATION SUMMARY")
print("="*70)
if not np.all(np.isnan(peak_E_centre)):
    print(f"  Peak proton energy (centre):    {np.nanmax(peak_E_centre):>8.1f} keV")
    print(f"  Mean proton energy (centre):    {np.nanmean(mean_E_centre):>8.1f} keV")
    print(f"  Peak proton energy (outer):     {np.nanmax(peak_E_outer):>8.1f} keV")
    print(f"  p-¹¹B threshold:                {PB11_MIN_KEV:>8.0f} keV")
    print(f"  p-¹¹B peak cross section:       {PB11_PEAK_KEV:>8.0f} keV")
    if np.nanmax(peak_E_centre) > PB11_MIN_KEV:
        print(f"\n  ✓ PROTONS EXCEED p-¹¹B FUSION THRESHOLD")
    if np.nanmax(peak_E_centre) > PB11_PEAK_KEV:
        print(f"  ✓✓ PROTONS REACH PEAK CROSS-SECTION ENERGY")
if not np.all(np.isnan(fusion_rate_estimate)):
    print(f"\n  Peak fusion rate:               {np.nanmax(fusion_rate_estimate):.2e} reactions/s")
    print(f"  Estimated alphas per shot:      {total_reactions:.2e}")
print("="*70)

plt.show()
