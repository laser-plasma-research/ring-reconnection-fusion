#!/usr/bin/env python3
"""
gen_jitter_specs_realistic.py — generate jitter specs at experimentally-relevant
pointing precisions, plus additional samples at the existing 300 um stress-test
point.

Three parameter values, multiple random seeds each:
  - 5 um  (3 seeds): modern actively-stabilized petawatt class (BELLA-class, ~0.5 urad)
  - 20 um (3 seeds): typical unstabilized short-pulse facility
  - 300 um (2 additional seeds, extending existing seed1+seed2): stress-test
            (intentionally beyond any realistic experimental pointing budget)

Output: 8 JSON spec files in current working directory.

All specs use uniform-in-disk sampling with the parameter as the disk radius.
The resulting RMS shift is approximately parameter / sqrt(2) (e.g., 5 um param
gives ~3.5 um RMS shift).

Each spec file's "comment" field documents the parameter, the resulting RMS shift,
and the seed used. Same JSON schema as the original sensitivity matrix.

Usage:
    python gen_jitter_specs_realistic.py
    # produces files in current working directory:
    #   s_jitter_005um_a.json, s_jitter_005um_b.json, s_jitter_005um_c.json
    #   s_jitter_020um_a.json, s_jitter_020um_b.json, s_jitter_020um_c.json
    #   s_jitter_300um_seed3.json, s_jitter_300um_seed4.json
"""

import json
import math
import os
import random
import sys

N_SPOTS = 8

# (parameter_um, [seeds], [labels]) — label letters chosen to avoid collision
# with existing s_jitter_300um (seed 20260528) and s_jitter_300um_seed2 (20260530).
SETS = [
    # Realistic experimental range
    (5.0,   [20260601, 20260602, 20260603], ["a", "b", "c"]),
    (20.0,  [20260604, 20260605, 20260606], ["a", "b", "c"]),
    # Additional stress-test samples at the unrealistic 300 um level
    (300.0, [20260607, 20260608], ["seed3", "seed4"]),
]


def generate_spec(disk_radius_um, seed):
    """Return (spec_dict, stats_dict). Mirrors original gen_sensitivity_matrix sampling."""
    rng = random.Random(seed)
    spots = []
    mags = []
    for _ in range(N_SPOTS):
        # Uniform-in-disk: r = R*sqrt(u), theta = 2*pi*v
        u = rng.random()
        v = rng.random()
        r = disk_radius_um * math.sqrt(u)
        theta = 2.0 * math.pi * v
        dx = r * math.cos(theta)
        dz = r * math.sin(theta)
        mags.append(math.hypot(dx, dz))
        spots.append({
            "enabled": True,
            "amp_scale": 1.0,
            "dx_um": dx,
            "dz_um": dz,
        })
    rms = math.sqrt(sum(m * m for m in mags) / N_SPOTS)
    stats = {
        "rms_um": rms,
        "max_um": max(mags),
        "mean_um": sum(mags) / N_SPOTS,
    }
    return spots, stats


def main():
    out_dir = os.getcwd()
    print(f"Writing specs to {out_dir}")
    print()
    print(f"{'name':<32}{'param':>8}{'seed':>12}{'RMS':>10}{'max':>10}{'max/sigma':>12}")
    print("-" * 84)

    spot_sigma_um = 300.0  # for the max/sigma % reporting

    for disk_radius_um, seeds, labels in SETS:
        for seed, label in zip(seeds, labels):
            param_tag = f"{int(disk_radius_um):03d}um"
            name = f"s_jitter_{param_tag}_{label}"
            spots, stats = generate_spec(disk_radius_um, seed)
            spec = {
                "comment": (
                    f"position jitter, uniform-in-disk parameter {disk_radius_um:g} um "
                    f"(resulting RMS shift {stats['rms_um']:.1f} um, "
                    f"max {stats['max_um']:.1f} um), seed {seed}. "
                    f"Realistic-range experimental jitter sweep; see "
                    f"gen_jitter_specs_realistic.py."
                ),
                "spots": spots,
            }
            out_path = os.path.join(out_dir, f"{name}.json")
            with open(out_path, "w") as fh:
                json.dump(spec, fh, indent=2)
            print(
                f"{name:<32}{disk_radius_um:>7.1f}u{seed:>12d}"
                f"{stats['rms_um']:>9.1f}u{stats['max_um']:>9.1f}u"
                f"{100*stats['max_um']/spot_sigma_um:>10.1f}%"
            )

    print()
    print("Generated 8 specs. Push to cloud and run via run_jitter_sweep.sh.")


if __name__ == "__main__":
    main()
