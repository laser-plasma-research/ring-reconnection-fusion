#!/usr/bin/env python3
"""
gen_sensitivity_matrix.py — build the symmetry-sensitivity run matrix for the
ring-reconnection Paper 1 referee response (safe three-axis, static-seed sweep).

Axes (all via the --spot-spec mechanism in sensitivity_seed_patch.md):
  - energy imbalance   : split-ring  +X% on spots {0,1,2,3} / -X% on {4,5,6,7}
                         (the direct analog of Bonasera's 4-vs-4 up/down split)
  - position jitter    : per-spot random offset, uniform in a disk of given RMS,
                         deterministic via a fixed seed so the matrix is reproducible
  - beam on/off        : 4+3 (drop one), and two-gap degraded rings

Each non-baseline run gets a specs/<name>.json. The TRUE baseline is run with NO
spec (byte-identical to the published 256^2). An all-unperturbed spec run
(s00_baseline_specform) is emitted only as a patch-identity sanity check.

Outputs (into --outdir, default ./sensitivity_matrix):
  specs/<name>.json        per-run perturbation spec
  run_manifest.csv         one row per run (name, axis, magnitude, spec path, cmd)
  launch_all.sh            sequential SSH launcher following the project pattern

All runs are 256^2 static-seed, b_seed=85 T, base p11b only — matched to the
Paper 1 256^2 ultrafine convergence partner.

NOTE / CONFIRM: the geometry+physics flags below are the ones I am confident map
to the published 256^2 run. nppc / sigma-scale / te-ev / eta-scale are left at
script defaults; if your archived 256^2 job set them explicitly, mirror them here
(and in the no-spec baseline) so the zero-perturbation identity check is exact.
"""

import os, json, csv, argparse
import numpy as np

# ----------------------------------------------------------------------------
# Fixed run configuration — matched to Paper 1 256^2 ultrafine
# ----------------------------------------------------------------------------
N_SPOTS        = 8
# Script lives in simulation/ ; runs are launched with cwd = repo root
# (~/laser-plasma-research), matching the existing runs/ output convention.
SIM_SCRIPT     = "simulation/pb11_ring_reconnection_v15_pulsed.py"
PY             = "python"

# Static-seed Paper 1 baseline flags (everything except --outdir / --spot-spec)
BASE_FLAGS = [
    "--b-seed", "85",
    "--base-fuel", "p11b",
    "--base-density", "5e24",
    "--nx", "256", "--nz", "256",
    "--max-steps", "1500",          # ~1158 ps at dt=0.772 ps (256^2)
    "--ring-radius-um", "2400",
    "--spot-radius-um", "300",
    "--n-spots", "8",
    "--field-mode", "applied",
    "--seed-topology", "harris",
    "--rotate-mode", "perturbative",
    "--dump-period", "20",          # ~15.4 ps cadence, ~75 dumps; step40~30.9ps
]

RUN_ROOT_REMOTE = "runs/sensitivity"     # on gpu-node
SSH_HOST        = os.environ.get("GPU_HOST", "gpu-node")
SSH_WORKDIR     = "~/laser-plasma-research"

JITTER_SEED     = 20260528               # deterministic jitter draws

# ----------------------------------------------------------------------------
def unperturbed_spots():
    return [dict(enabled=True, amp_scale=1.0, dx_um=0.0, dz_um=0.0)
            for _ in range(N_SPOTS)]

def spec_doc(spots, comment):
    return dict(comment=comment, spots=spots)

def energy_split_spec(pct):
    """+pct% on spots 0-3, -pct% on spots 4-7 (split-ring imbalance)."""
    spots = unperturbed_spots()
    for k in range(N_SPOTS):
        spots[k]["amp_scale"] = 1.0 + (pct/100.0 if k < N_SPOTS//2 else -pct/100.0)
    return spec_doc(spots, "energy split-ring +/-{}% (spots 0-3 hot, 4-7 cold)".format(pct))

def jitter_spec(rms_um, rng):
    """Per-spot offset uniform in a disk of radius rms_um (deterministic rng)."""
    spots = unperturbed_spots()
    for k in range(N_SPOTS):
        # uniform-in-disk draw
        r = rms_um * np.sqrt(rng.random())
        th = 2*np.pi*rng.random()
        spots[k]["dx_um"] = float(r*np.cos(th))
        spots[k]["dz_um"] = float(r*np.sin(th))
    return spec_doc(spots, "position jitter, uniform-in-disk RMS {} um, seed {}".format(rms_um, JITTER_SEED))

def missing_spec(drop_indices, label):
    spots = unperturbed_spots()
    for k in drop_indices:
        spots[k]["enabled"] = False
    return spec_doc(spots, "beam on/off: dropped spots {} ({})".format(drop_indices, label))

def combined_spec(pct, rms_um, rng):
    """Realistic combined case: split-ring energy + position jitter, all beams on."""
    spots = energy_split_spec(pct)["spots"]
    for k in range(N_SPOTS):
        r = rms_um * np.sqrt(rng.random())
        th = 2*np.pi*rng.random()
        spots[k]["dx_um"] = float(r*np.cos(th))
        spots[k]["dz_um"] = float(r*np.sin(th))
    return spec_doc(spots, "combined realistic: energy +/-{}% split + {} um jitter".format(pct, rms_um))

# ----------------------------------------------------------------------------
def build_matrix(rng):
    """Return list of (name, axis, magnitude, spec_or_None)."""
    M = []
    # true baseline — NO spec (byte-identical to published 256^2)
    M.append(("s01_baseline", "baseline", "0", None))
    # patch-identity check — unperturbed spec (should match baseline)
    M.append(("s00_baseline_specform", "identity", "0",
              spec_doc(unperturbed_spots(), "all-unperturbed spec; patch identity check")))
    # energy imbalance
    for pct in (5, 10, 20):
        M.append(("s_energy_{:02d}pct".format(pct), "energy_imbalance",
                  "+/-{}%".format(pct), energy_split_spec(pct)))
    # position jitter
    for rms in (50, 150, 300):
        M.append(("s_jitter_{:03d}um".format(rms), "position_jitter",
                  "{} um".format(rms), jitter_spec(rms, rng)))
    # beam on/off
    M.append(("s_drop1_4plus3", "missing_spot", "7 (4+3)",
              missing_spec([0], "4+3 single-gap, Bonasera analog")))
    M.append(("s_drop2_adjacent", "missing_spot", "6 (adjacent gap)",
              missing_spec([0, 1], "two adjacent gaps")))
    M.append(("s_drop2_opposite", "missing_spot", "6 (opposite gaps)",
              missing_spec([0, 4], "two diametrically opposite gaps")))
    # combined realistic
    M.append(("s_combined_realistic", "combined", "5% + 50um",
              combined_spec(5, 50, rng)))
    return M

def cmd_for(name, spec_path):
    parts = [PY, SIM_SCRIPT] + BASE_FLAGS + [
        "--outdir", "{}/{}".format(RUN_ROOT_REMOTE, name)]
    if spec_path is not None:
        parts += ["--spot-spec", spec_path]
    return " ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./sensitivity_matrix")
    args = ap.parse_args()
    specs_dir = os.path.join(args.outdir, "specs")
    os.makedirs(specs_dir, exist_ok=True)

    rng = np.random.default_rng(JITTER_SEED)
    matrix = build_matrix(rng)

    manifest_rows, launch_lines = [], []
    launch_lines.append("#!/usr/bin/env bash")
    launch_lines.append("# Sequential launcher for the symmetry-sensitivity sweep.")
    launch_lines.append("# Runs are independent; serialize to bound peak disk/GPU.")
    launch_lines.append("set -euo pipefail")
    launch_lines.append("")

    for name, axis, mag, spec in matrix:
        if spec is None:
            spec_rel = ""
            spec_remote = None
        else:
            spec_rel = os.path.join("specs", name + ".json")
            with open(os.path.join(specs_dir, name + ".json"), "w") as fh:
                json.dump(spec, fh, indent=2)
            # remote path (specs/ shipped alongside the script on gpu-node)
            spec_remote = "specs/{}.json".format(name)
        cmd = cmd_for(name, spec_remote)
        manifest_rows.append(dict(name=name, axis=axis, magnitude=mag,
                                  spec=spec_rel, cmd=cmd))
        launch_lines.append("echo '=== {} ({} {}) ==='".format(name, axis, mag))
        launch_lines.append("ssh {} 'cd {} && {}'".format(SSH_HOST, SSH_WORKDIR, cmd))
        launch_lines.append("")

    with open(os.path.join(args.outdir, "run_manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "axis", "magnitude", "spec", "cmd"])
        w.writeheader()
        w.writerows(manifest_rows)

    launch_path = os.path.join(args.outdir, "launch_all.sh")
    with open(launch_path, "w") as fh:
        fh.write("\n".join(launch_lines) + "\n")
    os.chmod(launch_path, 0o755)

    print("Wrote {} runs to {}".format(len(matrix), args.outdir))
    print("  specs/        per-run perturbation JSONs")
    print("  run_manifest.csv")
    print("  launch_all.sh")
    print()
    print("Baseline (s01_baseline) is run with NO --spot-spec and must reproduce")
    print("the published 256^2 numbers (G_FT 2.02 / Maxwellian 0.82 / |B|min 21.46 T).")

if __name__ == "__main__":
    main()
