#!/usr/bin/env python3
"""Read reconnection_rate_offline.csv and report rate_flux statistics."""

import csv
import numpy as np
import sys

# Path to CSV - default to LD long, can override with arg
csv_path = sys.argv[1] if len(sys.argv) > 1 else \
    "runs/p1_ld_512_100k/reconnection_rate_offline.csv"

with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

rate_flux_cols = [k for k in rows[0].keys() if k.endswith("_rate_flux")]
print(f"CSV: {csv_path}")
print(f"Total rows: {len(rows)}")
print(f"rate_flux columns: {len(rate_flux_cols)}")
print()

print(f"{'step':>8s} {'t(ps)':>10s} {'mean_flux':>14s} {'max_|flux|':>14s} {'n_valid':>8s}")
print("-" * 60)

for row in rows:
    fluxes = []
    for col in rate_flux_cols:
        try:
            v = float(row[col])
            if not np.isnan(v):
                fluxes.append(v)
        except (ValueError, TypeError):
            continue
    if fluxes:
        mean_f = np.mean(fluxes)
        max_f = np.max(np.abs(fluxes))
        n = len(fluxes)
    else:
        mean_f = float("nan")
        max_f = float("nan")
        n = 0
    print(f"{row['step']:>8s} {row['t_ps']:>10s} {mean_f:>14.4f} {max_f:>14.4f} {n:>8d}")

all_flux = []
for row in rows:
    for col in rate_flux_cols:
        try:
            v = float(row[col])
            if not np.isnan(v):
                all_flux.append(v)
        except (ValueError, TypeError):
            continue

print()
print("=" * 60)
print(f"Total valid rate_flux measurements: {len(all_flux)}")
if all_flux:
    print(f"Mean rate_flux:        {np.mean(all_flux):.4f}")
    print(f"Median rate_flux:      {np.median(all_flux):.4f}")
    print(f"Max |rate_flux|:       {np.max(np.abs(all_flux)):.4f}")
    print(f"Std rate_flux:         {np.std(all_flux):.4f}")
    print()
    # Distribution
    abs_flux = np.abs(all_flux)
    print(f"Distribution of |rate_flux|:")
    print(f"  < 0.001:             {(abs_flux < 0.001).sum()}")
    print(f"  0.001 - 0.05:        {((abs_flux >= 0.001) & (abs_flux < 0.05)).sum()}")
    print(f"  0.05 - 0.5:          {((abs_flux >= 0.05) & (abs_flux < 0.5)).sum()}")
    print(f"  0.5 - 5:             {((abs_flux >= 0.5) & (abs_flux < 5)).sum()}")
    print(f"  >= 5:                {(abs_flux >= 5).sum()}")
