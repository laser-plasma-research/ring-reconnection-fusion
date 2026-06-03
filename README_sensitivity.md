# Symmetry-sensitivity sweep — kit overview

Answers Bonasera's "are you assuming some symmetry?" for Paper 1 by relaxing the
imposed 8-fold symmetry along the three axes a static-seed model can legitimately
speak to: **amplitude** (energy imbalance), **geometry** (position jitter), and
**discrete distribution** (beams on/off — the direct analog of his 3-vs-4 test).
Timing/arrival jitter is deliberately out of scope (see "Scope statement" below).

All runs are 256^2, static `harris` seed, `b_seed = 85 T`, base p11b only — matched
to the Paper 1 256^2 ultrafine convergence partner. Because every reported metric
is a ratio or a centroid *shift*, grid resolution cancels and 256^2 is sufficient
(Paper 1 Sec 3.7 / Fig 7). No driven-field machinery is touched, so the Paper 2
energy-injection pathology cannot recur.

## Files

1. `sensitivity_seed_patch.md` — three contained edits to
   `pb11_ring_reconnection_v15_pulsed.py` adding per-spot amplitude / position /
   enable via a `--spot-spec` JSON. No-spec path is **byte-identical** to baseline.
2. `gen_sensitivity_matrix.py` — emits `specs/*.json`, `run_manifest.csv`, and
   `launch_all.sh` for the 12-run matrix.
3. `analyze_sensitivity.py` — computes the three relative metrics per run vs the
   baseline and writes `sensitivity_results.csv`.

## Workflow

1. Apply the three edits in `sensitivity_seed_patch.md` to the sim script.
2. `python gen_sensitivity_matrix.py --outdir sensitivity_matrix`
3. Ship `specs/` to <GPU_HOST> alongside the script, then run
   `bash sensitivity_matrix/launch_all.sh` (sequential SSH; serialize to bound
   peak disk). Each run produces the usual report set in `runs/sensitivity/<name>/`.
4. Run the standard analysis pipeline on each run dir so the per-run
   `post_analysis_report.txt` and first-transit summary exist.
5. `python analyze_sensitivity.py --runs-root runs/sensitivity \
        --manifest sensitivity_matrix/run_manifest.csv --baseline s01_baseline`

## The 12 runs

| name | axis | magnitude |
|---|---|---|
| s01_baseline | baseline (no spec) | 0 |
| s00_baseline_specform | identity check (unperturbed spec) | 0 |
| s_energy_05/10/20pct | energy imbalance (split-ring) | ±5/±10/±20% |
| s_jitter_050/150/300um | position jitter (uniform-in-disk) | 50/150/300 µm |
| s_drop1_4plus3 | beam off (Bonasera 4+3 analog) | drop 1 |
| s_drop2_adjacent / _opposite | beam off (degraded ring) | drop 2 |
| s_combined_realistic | combined | 5% + 50 µm |

## The two baseline anchors

- **s01_baseline** (no `--spot-spec`) is the n=0 point of the sweep and must
  reproduce the published 256^2 numbers: **G_FT 2.02, Maxwellian 0.82,
  |B|max,min 21.46 T**. Every ratio in `sensitivity_results.csv` divides by this.
- **s00_baseline_specform** (all-unperturbed spec) is the patch-identity check;
  it should match s01_baseline within run-to-run noise, proving the spec
  machinery is inert at zero perturbation. "Perturbation = 0 reproduces the
  published number" is itself the cleanest referee-facing validation.

## Metrics (in sensitivity_results.csv)

- `centroid_shift_vs_base_um` — shift of the 500-5000 keV proton centroid at
  ~30 ps. The convergence-coherence order parameter; the analog of Bonasera's
  directional-flux dial. Small vs the 2400 µm ring radius = graceful degradation.
- `rel_dG_FT` — fractional change in σ-weighted first-transit gain.
- `rel_dBcollapse` — fractional change in peak post-ramp B-collapse.

## Three things to confirm against your live setup

1. **[HIGH] Baseline flag fidelity.** `gen_sensitivity_matrix.py` sets the
   geometry+physics flags I'm confident about; `nppc`, `sigma-scale`, `te-ev`,
   `eta-scale` are left at script defaults. If your archived 256^2 job set any of
   these explicitly, add them to `BASE_FLAGS` (and they apply to the no-spec
   baseline too) so the zero-perturbation identity is exact.
2. **[HIGH] Centroid reader.** `compute_fusion_centroid()` in the analyzer assumes
   openPMD dumps under `particles/`, species `proton`, SI position/momentum,
   centre at (0,0). Verify against your dump format — it's isolated in one
   function. If your pipeline already emits a per-dump centroid CSV, drop it in
   the run dir and the analyzer prefers it automatically.
3. **[MEDIUM] G_FT label.** The analyzer's default regexes try `G_FT`, `G_SR`
   headline, "sigma-weighted first-transit", and "Gain vs laser". Point it at the
   field holding your **σ-weighted** Paper 1 G_FT with `--gft-label` if your
   summary uses a different label, so you don't accidentally read a zone-summed
   or Maxwellian number.

## Modeling assumptions for Methods / referee response

- **Energy → seed amplitude is a first-order proxy.** Biermann seed B ∝ ∇n×∇T,
  spot energy enters through T, so ±X% spot energy ≈ ±X% seed-B amplitude. State
  this; it's a proxy, not an identity.
- **Split-ring imbalance** (spots 0-3 hot / 4-7 cold) is a contiguous half-hot /
  half-cold ring — the direct analog of Bonasera's 4-vs-4 up/down split, chosen so
  the sweep answers him in his own coordinate.

## Scope statement (the one limitation, for the referee response)

The static-seed formulation models the Biermann fields **after** establishment, so
it speaks to amplitude and position jitter but not to beam **arrival-time** jitter,
which is out of scope by construction (not omitted by oversight). Position jitter is
a partial proxy: a mispointed spot's outflow misses the convergence point much as a
mistimed one would fail to pile up coherently. Bonasera's own symmetry break was
spatial/count (3-vs-4), not temporal, so the three covered axes meet his test directly.
