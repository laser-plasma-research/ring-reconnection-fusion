#!/usr/bin/env bash
: "${GPU_HOST:=gpu-node}"; : "${GPU_USER:=researcher}"  # override via env for your infra
# Sequential launcher for the symmetry-sensitivity sweep.
# Runs are independent; serialize to bound peak disk/GPU.
set -euo pipefail

echo '=== s01_baseline (baseline 0) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s01_baseline'

echo '=== s00_baseline_specform (identity 0) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s00_baseline_specform --spot-spec specs/s00_baseline_specform.json'

echo '=== s_energy_05pct (energy_imbalance +/-5%) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_energy_05pct --spot-spec specs/s_energy_05pct.json'

echo '=== s_energy_10pct (energy_imbalance +/-10%) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_energy_10pct --spot-spec specs/s_energy_10pct.json'

echo '=== s_energy_20pct (energy_imbalance +/-20%) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_energy_20pct --spot-spec specs/s_energy_20pct.json'

echo '=== s_jitter_050um (position_jitter 50 um) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_jitter_050um --spot-spec specs/s_jitter_050um.json'

echo '=== s_jitter_150um (position_jitter 150 um) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_jitter_150um --spot-spec specs/s_jitter_150um.json'

echo '=== s_jitter_300um (position_jitter 300 um) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_jitter_300um --spot-spec specs/s_jitter_300um.json'

echo '=== s_drop1_4plus3 (missing_spot 7 (4+3)) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_drop1_4plus3 --spot-spec specs/s_drop1_4plus3.json'

echo '=== s_drop2_adjacent (missing_spot 6 (adjacent gap)) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_drop2_adjacent --spot-spec specs/s_drop2_adjacent.json'

echo '=== s_drop2_opposite (missing_spot 6 (opposite gaps)) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_drop2_opposite --spot-spec specs/s_drop2_opposite.json'

echo '=== s_combined_realistic (combined 5% + 50um) ==='
ssh $GPU_HOST 'cd ~/laser-plasma-research && python pb11_ring_reconnection_v15_pulsed.py --b-seed 85 --base-fuel p11b --base-density 5e24 --nx 256 --nz 256 --max-steps 1500 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --field-mode applied --seed-topology harris --rotate-mode perturbative --dump-period 20 --outdir runs/sensitivity/s_combined_realistic --spot-spec specs/s_combined_realistic.json'

