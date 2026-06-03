#!/usr/bin/env python3
"""
analyze_sensitivity.py — relative-metric post-processor for the symmetry-
sensitivity sweep. For each run it extracts three quantities and expresses each
relative to the unperturbed baseline:

  1. fusion-grade convergence centroid shift at t~30 ps  (um)   [absolute, not ratio]
       weighted (x,z) centre-of-mass of 500-5000 keV protons; radial offset from
       the geometric centre. This is the convergence-coherence order parameter
       (the analog of Bonasera's directional-flux dial). Computed from particle
       dumps -> see compute_fusion_centroid(): THIS is the one function to verify
       against your dump format (species name / records / SI units).
  2. sigma-weighted first-transit gain G_FT, reported as relative dG_FT/G_FT,base
       parsed from the per-run first-transit summary (configurable label).
  3. peak B-collapse fraction (post-ramp), reported as relative change
       computed from the |B|_max(t) table in post_analysis_report.txt, skipping
       the t=0 / pre-ramp dump-0 artifact, matching the paper's definition.

Because every metric is a ratio or a centroid SHIFT, grid resolution cancels and
256^2 is sufficient (the convergence-strategy argument from Paper 1 Sec 3.7).

Usage:
  python analyze_sensitivity.py --runs-root runs/sensitivity \
      --manifest sensitivity_matrix/run_manifest.csv \
      --baseline s01_baseline --out sensitivity_results.csv
"""

import os, re, csv, glob, argparse, math

# Physical constants — matched to pb11_zone_analysis_paper3.py conventions
KEV         = 1.602176634e-16        # J per keV
C_LIGHT     = 299792458.0
PROTON_MASS_KG = 1.67262192369e-27

FUSION_LO_KEV = 500.0
FUSION_HI_KEV = 5000.0
T_CENTROID_PS = 30.0


def _kev_from_u(ux, uy, uz, np):
    """Relativistic KE (keV) from proper velocity u = p/m, matching the
    pb11_zone_analysis_paper3.kev_from_momentum convention exactly."""
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 / (C_LIGHT * C_LIGHT))
    return (gamma - 1.0) * PROTON_MASS_KG * C_LIGHT * C_LIGHT / KEV


def _dt_s_from_run_meta(run_dir):
    """Read time_step_s from run_meta.txt for dump->time mapping. None if absent."""
    mp = os.path.join(run_dir, "run_meta.txt")
    if not os.path.exists(mp):
        return None
    for line in open(mp, errors="ignore"):
        s = line.strip()
        if s.startswith("time_step_s"):
            try:
                return float(s.split("=", 1)[1].split()[0])
            except Exception:
                return None
    return None


# ----------------------------------------------------------------------------
# Metric 1: fusion-grade centroid from particle dumps
# Reader matches pb11_zone_analysis_paper3.py exactly (openPMD particles/
# openpmd_%T.{h5,bp}; momentum/mass -> u; weighting). Dump->time via run_meta.
# ----------------------------------------------------------------------------
def compute_fusion_centroid(run_dir, t_target_ps=T_CENTROID_PS, species="proton"):
    """Weighted (x,z) centre-of-mass radius of 500-5000 keV protons at ~t_target_ps.
    Returns (centroid_x_um, centroid_z_um, t_used_ps, n_used). Vector form
    preserves directional information: a symmetric distribution gives
    (0,0); asymmetry produces a directional shift. NaN on failure."""
    try:
        import openpmd_api as io
        import numpy as np
    except Exception:
        print("  [centroid] openpmd_api/numpy unavailable; centroid -> NaN for {}".format(run_dir))
        return (float("nan"), float("nan"), float("nan"), 0)

    pdir = os.path.join(run_dir, "particles")
    if os.path.isdir(pdir) and glob.glob(os.path.join(pdir, "openpmd_*.h5")):
        pattern = os.path.join(pdir, "openpmd_%T.h5")
    elif os.path.isdir(pdir) and glob.glob(os.path.join(pdir, "openpmd_*.bp")):
        pattern = os.path.join(pdir, "openpmd_%T.bp")
    else:
        print("  [centroid] no openpmd_* dumps under {}/particles; centroid -> NaN".format(run_dir))
        return (float("nan"), float("nan"), float("nan"), 0)

    try:
        series = io.Series(pattern, io.Access.read_only)
    except Exception as e:
        print("  [centroid] could not open series {}: {}".format(pattern, e))
        return (float("nan"), float("nan"), float("nan"), 0)

    iters = sorted(series.iterations)
    if not iters:
        return (float("nan"), float("nan"), float("nan"), 0)

    # dump -> time. Prefer iteration.time; fall back to run_meta dt * iter.
    dt_s = _dt_s_from_run_meta(run_dir)
    def t_ps_of(idx):
        it = series.iterations[idx]
        try:
            if it.time and it.time > 0:
                tu = it.time_unit_SI if it.time_unit_SI else 1.0
                return it.time * tu * 1e12
        except Exception:
            pass
        return (idx * dt_s * 1e12) if dt_s else float(idx)

    idx = min(iters, key=lambda i: abs(t_ps_of(i) - t_target_ps))
    t_used_ps = t_ps_of(idx)

    it = series.iterations[idx]
    if species not in list(it.particles):
        return (float("nan"), float("nan"), float(t_used_ps), 0)
    sp = it.particles[species]

    def load(record, component=None):
        rec = sp[record]
        comp = rec[component] if component is not None else rec[next(iter(rec))]
        chunk = comp.load_chunk()
        series.flush()
        try:
            unit = comp.unit_SI
            if unit and unit != 1.0:
                chunk = chunk * unit
        except Exception:
            pass
        return chunk

    try:
        x  = load("position", "x")
        z  = load("position", "z")
        ux = load("momentum", "x") / PROTON_MASS_KG
        uy = load("momentum", "y") / PROTON_MASS_KG
        uz = load("momentum", "z") / PROTON_MASS_KG
        w  = load("weighting")
    except Exception as e:
        print("  [centroid] load failed for {} iter {}: {}".format(run_dir, idx, e))
        return (float("nan"), float("nan"), float(t_used_ps), 0)

    ke = _kev_from_u(ux, uy, uz, np)
    sel = (ke >= FUSION_LO_KEV) & (ke <= FUSION_HI_KEV)
    if not np.any(sel):
        return (float("nan"), float("nan"), float(t_used_ps), 0)
    ws = w[sel]
    cx = float(np.average(x[sel], weights=ws)) * 1e6  # um
    cz = float(np.average(z[sel], weights=ws)) * 1e6  # um
    return (cx, cz, float(t_used_ps), int(sel.sum()))

# ----------------------------------------------------------------------------
# Metric 2: sigma-weighted first-transit gain (parse per-run summary)
# ----------------------------------------------------------------------------
def parse_gft(run_dir, label_patterns):
    """Return (G_FT, source_file) parsed from a first-transit summary, or (nan, '')."""
    cands = (glob.glob(os.path.join(run_dir, "*first_transit*summary*.txt")) +
             glob.glob(os.path.join(run_dir, "*first_transit*.txt")) +
             glob.glob(os.path.join(run_dir, "phase_analysis_report.txt")))
    for f in cands:
        try:
            txt = open(f, errors="ignore").read()
        except Exception:
            continue
        for pat in label_patterns:
            m = re.search(pat, txt)
            if m:
                return (float(m.group(1)), os.path.basename(f))
    return (float("nan"), "")

# ----------------------------------------------------------------------------
# Metric 3: peak B-collapse (post-ramp) from post_analysis field table
# ----------------------------------------------------------------------------
def parse_bcollapse(run_dir):
    """Compute (collapse_pct, source) = (Bmax_postramp - Bmin)/Bmax_postramp.

    Reads the 'B-FIELD EVOLUTION CHECK' |B|_max(t) table in
    post_analysis_report.txt, dropping the t~0 / |B|~0 pre-ramp dump-0 artifact.
    Matches the paper's 'peak collapse of post-ramp field' definition.
    """
    f = os.path.join(run_dir, "post_analysis_report.txt")
    if not os.path.exists(f):
        return (float("nan"), "")
    txt = open(f, errors="ignore").read()
    # rows like:  '   134.74        26.97        3.1909      4.897e+10'
    rows = re.findall(r"^\s*([\d.]+)\s+([\d.]+)\s+[\d.]+\s+[\d.eE+\-]+\s*$",
                      txt, flags=re.M)
    bmax = []
    for t_ps, b in rows:
        t_ps, b = float(t_ps), float(b)
        if t_ps <= 1e-6 or b <= 1e-6:   # skip pre-ramp dump-0 artifact
            continue
        bmax.append(b)
    if len(bmax) < 2:
        return (float("nan"), "")
    bpk, bmn = max(bmax), min(bmax)
    return ((bpk - bmn) / bpk * 100.0, "post_analysis_report.txt")

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True,
                    help="dir containing one subdir per run (named as in manifest)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline", default="s01_baseline")
    ap.add_argument("--out", default="sensitivity_results.csv")
    ap.add_argument("--gft-label", action="append", default=[],
                    help="regex with one capture group for G_FT; repeatable. "
                         "Defaults try the common Paper 1 / SR labels.")
    args = ap.parse_args()

    label_patterns = args.gft_label or [
        r"G_FT[^=\n]*=\s*([\d.]+)",
        r"G_SR[^=\n]*headline[^=\n]*=\s*([\d.]+)",
        r"sigma-weighted first-transit[^=\n]*=\s*([\d.]+)",
        r"Gain vs laser[^=\n]*=\s*([\d.]+)",
    ]

    rows = list(csv.DictReader(open(args.manifest)))
    results = {}
    for r in rows:
        name = r["name"]
        rd = os.path.join(args.runs_root, name)
        if not os.path.isdir(rd):
            print("[skip] missing run dir: {}".format(rd))
            continue
        cx, cz, t_used, n_used = compute_fusion_centroid(rd)
        gft, gsrc = parse_gft(rd, label_patterns)
        bcol, bsrc = parse_bcollapse(rd)
        results[name] = dict(axis=r["axis"], magnitude=r["magnitude"],
                             cx_um=cx, cz_um=cz,
                             t_centroid_ps=t_used, n_fusion=n_used,
                             G_FT=gft, gft_src=gsrc, Bcollapse_pct=bcol, bcol_src=bsrc)
        cstr = "({:.1f},{:.1f})".format(cx, cz) if cx == cx else "(NaN,NaN)"
        print("[ok] {:24s} centroid_xz={:>15} um  G_FT={:>6}  Bcollapse={:>6}%".format(
            name, cstr,
            "{:.3f}".format(gft) if gft == gft else "NaN",
            "{:.1f}".format(bcol) if bcol == bcol else "NaN"))

    base = results.get(args.baseline)
    if base is None:
        print("\n[FATAL] baseline '{}' not found among results; cannot make ratios.".format(args.baseline))
        return

    out_fields = ["name", "axis", "magnitude",
                  "centroid_x_um", "centroid_z_um",
                  "shift_dx_um", "shift_dz_um", "shift_mag_um",
                  "G_FT", "rel_dG_FT", "Bcollapse_pct", "rel_dBcollapse",
                  "t_centroid_ps", "n_fusion", "gft_src", "bcol_src"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        for name, m in results.items():
            def rel(v, b):
                if v != v or b != b or b == 0:
                    return float("nan")
                return (v - b) / b
            # Centroid shift vector relative to baseline. The magnitude is the
            # primary asymmetry metric; (dx,dz) preserves direction so we can
            # check whether the shift points toward the bright hemisphere.
            import math as _math
            def _f(v):
                return v if v == v else float("nan")
            dx = _f(m["cx_um"]) - _f(base["cx_um"])
            dz = _f(m["cz_um"]) - _f(base["cz_um"])
            smag = _math.hypot(dx, dz) if (dx == dx and dz == dz) else float("nan")
            def _r(v, n=2):
                return round(v, n) if (v == v) else ""
            w.writerow(dict(
                name=name, axis=m["axis"], magnitude=m["magnitude"],
                centroid_x_um=_r(m["cx_um"]),
                centroid_z_um=_r(m["cz_um"]),
                shift_dx_um=_r(dx),
                shift_dz_um=_r(dz),
                shift_mag_um=_r(smag),
                G_FT=_r(m["G_FT"], 4),
                rel_dG_FT=_r(rel(m["G_FT"], base["G_FT"]), 4),
                Bcollapse_pct=_r(m["Bcollapse_pct"]),
                rel_dBcollapse=_r(rel(m["Bcollapse_pct"], base["Bcollapse_pct"]), 4),
                t_centroid_ps=_r(m["t_centroid_ps"]),
                n_fusion=m["n_fusion"], gft_src=m["gft_src"], bcol_src=m["bcol_src"],
            ))
    print("\nWrote {}".format(args.out))
    print("Baseline centroid (x,z) = ({:.1f}, {:.1f}) um, G_FT = {}, B-collapse = {}%".format(
        base["cx_um"], base["cz_um"], base["G_FT"], base["Bcollapse_pct"]))
    print("\nReferee reading: directional centroid shift = (shift_dx_um, shift_dz_um) vs baseline.\n"
          "  Symmetric perturbations -> small shift_mag_um. Asymmetric (dropped beams) "
          "-> directional shift toward the bright side.")

if __name__ == "__main__":
    main()
