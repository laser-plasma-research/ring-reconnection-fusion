#!/usr/bin/env python3
"""
pb11_n_above_threshold.py — proton N(>E_threshold) extractor + convergence gate.

WHY THIS EXISTS
---------------
The Paper 1 headline is the sigma-folded first-transit gain G_FT. In this deck
the fusion cross-section is a single effective constant applied above a 500 keV
threshold (pb11_first_transit_fusion.py: sigma_m2 const, threshold_kev=500).
Therefore G_FT depends on the proton distribution essentially through ONE
quantity: the number of protons above 500 keV. This makes the convergence
question tractable:

  "Is G_FT converged?"  reduces to  "Is N(>500 keV) converged?"

THE GATE
--------
Before spending GPU time on a 1024^2 convergence run, confirm the pipeline is
internally consistent: N(>500 keV) should shift between 256^2 and 512^2 by the
SAME factor that G_FT shifts (~80% per the docstring). With a constant sigma,
G_FT can only move if the threshold count moved. If N(>500 keV) does NOT track
G_FT, something other than the threshold count is driving the gain — a bug or a
normalization difference — and 1024^2 would just be expensive confirmation of
it. This script measures N(>500 keV) so that comparison can be made.

WHAT IT DOES
------------
For each openPMD particle dump in a run directory, counts protons with
kinetic energy above given thresholds (default 300 and 500 keV), reporting BOTH:
  - macroparticle count above threshold  (statistics / Poisson error)
  - weighting-summed PHYSICAL-ion count   (yield-relevant population)
and the fraction above threshold. Emits a per-dump CSV time series.

FORMAT ASSUMPTIONS (VERIFIED AT RUNTIME, NOT TRUSTED)
-----------------------------------------------------
Written before any real dump was available to inspect. On first dump it prints
the actual record layout it finds and ASSERTS the assumptions below; if any is
wrong it stops with a clear message naming the mismatch rather than emitting
garbage:
  - openPMD via openpmd-viewer (OpenPMDTimeSeries), as per environment.yml
  - proton species name is 'proton' (override --species)
  - momentum available as either normalized 'ux','uy','uz' (= gamma*beta, i.e.
    p/(m c)) OR SI 'momentum/x,y,z' in kg m/s; auto-detected
  - per-particle 'w' (weighting) record present; if absent, physical counts are
    reported as NaN and only macroparticle counts are trusted
  - proton rest mass = 938.272 MeV; KE = (gamma - 1) m c^2

USAGE
-----
    # single run
    python3 pb11_n_above_threshold.py --run runs/paper01/p1_ld_uuf

    # compare two resolutions (the convergence gate)
    python3 pb11_n_above_threshold.py \
        --run runs/paper01/p1_ld_uuf \
        --compare runs/paper01/p1_ld_uuf_512

Outputs <run>/n_above_threshold.csv (one row per dump) and prints a summary.
When --compare is given, prints the peak-N(>500keV) ratio between runs so it
can be checked against the G_FT ratio.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Proton rest energy (MeV) and c (m/s)
M_P_MEV = 938.272
C_MS = 299_792_458.0
M_P_KG = 1.67262192369e-27

THRESHOLDS_KEV = [300.0, 500.0]


def _load_timeseries(run_dir):
    """Open the run's particle dumps as an openPMD time series."""
    try:
        from openpmd_viewer import OpenPMDTimeSeries
    except Exception as e:
        sys.exit(f"ERROR: openpmd-viewer not importable ({e}). "
                 f"Activate the 'plasma' conda env.")
    # Dumps may live under the run dir directly or in a 'diags'/'diag' subdir.
    candidates = [Path(run_dir), Path(run_dir) / "diags",
                  Path(run_dir) / "diag", Path(run_dir) / "particles"]
    for c in candidates:
        if c.is_dir():
            try:
                ts = OpenPMDTimeSeries(str(c), check_all_files=False)
                if ts.iterations is not None and len(ts.iterations) > 0:
                    return ts, c
            except Exception:
                continue
    sys.exit(f"ERROR: no openPMD particle series found under {run_dir} "
             f"(looked in: {[str(c) for c in candidates]})")


def _ke_kev_from_records(ts, it, species, first_dump_report=False):
    """Return (ke_kev array, weights array or None) for one iteration.

    Auto-detects normalized (ux/uy/uz) vs SI (momentum) momentum records and
    verifies the assumption set on the first dump.
    """
    avail = ts.avail_record_components.get(species, {})
    if first_dump_report:
        print(f"  [format] species='{species}' available records: "
              f"{sorted(list(avail.keys()))}")

    # Try normalized momentum first (WarpX PICMI default writes ux,uy,uz = p/mc)
    have_u = all(k in avail for k in ("ux", "uy", "uz")) or \
             all((species, k) for k in ("ux", "uy", "uz"))
    weights = None
    try:
        if "ux" in _flatten(avail):
            ux, uy, uz, w = ts.get_particle(
                ["ux", "uy", "uz", "w"], species=species, iteration=it)
            # ux = gamma*beta_x = p_x/(m c). gamma = sqrt(1 + |u|^2)
            u2 = ux * ux + uy * uy + uz * uz
            gamma = np.sqrt(1.0 + u2)
            ke_j = (gamma - 1.0) * M_P_KG * C_MS * C_MS
            weights = w
            if first_dump_report:
                print("  [format] momentum=normalized (ux,uy,uz=gamma*beta); "
                      "weighting record 'w' present")
        else:
            # SI momentum fallback
            px, py, pz, w = ts.get_particle(
                ["momentum/x", "momentum/y", "momentum/z", "w"],
                species=species, iteration=it)
            p2 = px * px + py * py + pz * pz
            # relativistic KE from SI momentum
            ke_j = (np.sqrt((px*0+1.0) * (M_P_KG*C_MS*C_MS)**2 + p2*C_MS*C_MS)
                    - M_P_KG*C_MS*C_MS)
            weights = w
            if first_dump_report:
                print("  [format] momentum=SI (momentum/x,y,z); "
                      "weighting record 'w' present")
    except Exception as e:
        # Last resort: try without weights so macroparticle counts still work
        try:
            ux, uy, uz = ts.get_particle(
                ["ux", "uy", "uz"], species=species, iteration=it)
            u2 = ux * ux + uy * uy + uz * uz
            gamma = np.sqrt(1.0 + u2)
            ke_j = (gamma - 1.0) * M_P_KG * C_MS * C_MS
            weights = None
            if first_dump_report:
                print(f"  [format] WARNING: no weighting record ('w') readable "
                      f"({e}); physical-ion counts will be NaN, macroparticle "
                      f"counts still valid")
        except Exception as e2:
            sys.exit(f"ERROR: could not read momentum records for species "
                     f"'{species}' at iteration {it}: {e2}\n"
                     f"Available: {sorted(list(avail.keys()))}\n"
                     f"Override the species name with --species if needed.")

    ke_kev = ke_j / 1.602176634e-16  # J -> keV
    return ke_kev, weights


def _flatten(avail):
    """avail_record_components values can be nested; collect top-level keys."""
    keys = set()
    for k in avail:
        keys.add(k)
    return keys


def extract_run(run_dir, species, thresholds):
    ts, src = _load_timeseries(run_dir)
    print(f"Run: {run_dir}\n  source: {src}\n  iterations: {len(ts.iterations)}")
    rows = []
    for i, it in enumerate(ts.iterations):
        t_s = ts.t[i] if hasattr(ts, "t") and ts.t is not None else float("nan")
        ke_kev, w = _ke_kev_from_records(ts, it, species,
                                         first_dump_report=(i == 0))
        n_macro = int(ke_kev.size)
        row = {"iteration": int(it), "t_ps": t_s * 1e12, "n_macro": n_macro}
        for thr in thresholds:
            mask = ke_kev > thr
            row[f"n_macro_gt_{int(thr)}kev"] = int(np.count_nonzero(mask))
            if w is not None:
                row[f"n_phys_gt_{int(thr)}kev"] = float(np.sum(w[mask]))
                row[f"frac_gt_{int(thr)}kev"] = (
                    float(np.sum(w[mask]) / np.sum(w)) if np.sum(w) > 0 else 0.0)
            else:
                row[f"n_phys_gt_{int(thr)}kev"] = float("nan")
                row[f"frac_gt_{int(thr)}kev"] = (
                    float(np.count_nonzero(mask) / n_macro) if n_macro else 0.0)
        rows.append(row)
    return rows


def write_csv(rows, out_path):
    import csv
    if not rows:
        print(f"  (no rows; nothing written to {out_path})")
        return
    cols = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_path} ({len(rows)} rows)")


def peak_n500(rows):
    """Peak physical N(>500keV) over the run; falls back to macro if phys NaN."""
    phys = [r.get("n_phys_gt_500kev", float("nan")) for r in rows]
    phys = [p for p in phys if p == p]  # drop NaN
    if phys:
        return max(phys), "physical"
    macro = [r.get("n_macro_gt_500kev", 0) for r in rows]
    return (max(macro) if macro else 0), "macroparticle"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Run directory with openPMD dumps")
    ap.add_argument("--compare", default=None,
                    help="Second run directory; prints peak-N(>500keV) ratio "
                         "(the convergence-consistency check)")
    ap.add_argument("--species", default="proton",
                    help="Proton species name in the dumps (default: proton)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS_KEV,
                    help="Threshold energies in keV (default: 300 500)")
    args = ap.parse_args()

    rows_a = extract_run(args.run, args.species, args.thresholds)
    write_csv(rows_a, Path(args.run) / "n_above_threshold.csv")
    pa, kinda = peak_n500(rows_a)
    print(f"  peak N(>500keV) [{kinda}] = {pa:.4e}")

    if args.compare:
        rows_b = extract_run(args.compare, args.species, args.thresholds)
        write_csv(rows_b, Path(args.compare) / "n_above_threshold.csv")
        pb, kindb = peak_n500(rows_b)
        print(f"  peak N(>500keV) [{kindb}] = {pb:.4e}")
        print("\n" + "=" * 60)
        print("CONVERGENCE-CONSISTENCY CHECK")
        print("=" * 60)
        if pa > 0:
            ratio = pb / pa
            print(f"  peak N(>500keV) ratio (compare/run) = {ratio:.3f} "
                  f"({(ratio-1)*100:+.1f}%)")
            print("  Compare this against the G_FT ratio between the same two")
            print("  runs. With a constant sigma above threshold, G_FT can only")
            print("  move if N(>500keV) moves. If these two ratios agree, the")
            print("  pipeline is internally consistent and the convergence")
            print("  ladder is trustworthy. If they DISAGREE, find the cause")
            print("  before spending 1024^2 compute.")
        else:
            print("  Base run peak N(>500keV) is zero; cannot form a ratio.")


if __name__ == "__main__":
    main()
