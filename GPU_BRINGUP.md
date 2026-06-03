# Fresh GPU instance bring-up — operator checklist

Last verified: 2026-05-31

This is the routine for bringing a freshly provisioned cloud GPU instance to
"ready to launch a campaign" state.  Image and project source live on your
Mac; the cloud disk is treated as ephemeral.

## Image version history

- **v1.2.1** (current) — adds `cupy-cuda12x` via pip layer on top of v1.2.0.
  Required because the simulation driver
  (`pb11_ring_reconnection_v15_pulsed.py:2624`) reads particle data off the
  GPU via cupy and segfaults if cupy is missing.  v1.2.0 shipped without
  cupy in the conda-pack used to build the image; v1.2.1 patches that gap.
  The pip install also matches what `environment.yml` now specifies, so any
  clean rebuild from source via `setup_cloud.sh` should reproduce v1.2.1's
  contents.
- **v1.2.0** — DO NOT USE.  Sim crashes at STEP 1 due to missing cupy.

## Assumptions about the new instance

- H100 80 GB GPU class
- 8 TB volume mounted at `/mnt/vdc`
- Docker engine pre-installed
- NVIDIA Container Toolkit pre-installed (lets `docker run --gpus all` work)
- SSH access via key that's already in your local `~/.ssh/` keychain

If those last two aren't true, see "Bare-image bootstrap" at the bottom.


## Step 1 — point your SSH alias at the new IP

The cloud console gives you a new public IP every provision.  Your local
`~/.ssh/config` currently has `<GPU_HOST>` pointing at the OLD IP.  Edit
to point at the new one:

```bash
# Open in editor
vim ~/.ssh/config
# or
open -e ~/.ssh/config
```

Find the `Host <GPU_HOST>` block and update `HostName` to the new IP.
A typical block looks like:

```
Host <GPU_HOST>
    HostName 185.216.22.30          ← replace with new IP
    User <USER>
    IdentityFile ~/.ssh/your-key.pem
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

The `StrictHostKeyChecking no` + `UserKnownHostsFile /dev/null` lines
suppress the "host key changed" prompt that would otherwise fire every
time you replace the instance.  If your config doesn't already have those,
add them now — they save you from manually clearing `known_hosts` each
provision.

Verify SSH reaches the new box:

```bash
ssh <GPU_HOST> 'hostname; uptime; nvidia-smi | head -3'
```

Expected: a hostname (not yours), an uptime in minutes, and three lines of
nvidia-smi output showing the H100.  If nvidia-smi fails, the GPU driver
isn't loaded on the new image — stop here and resolve before continuing.


## Step 2 — confirm prerequisites

```bash
ssh <GPU_HOST> 'echo "=== docker ==="; docker --version 2>&1; echo "=== nvidia-container-toolkit ==="; docker run --rm --gpus all ubuntu:22.04 nvidia-smi 2>&1 | head -5; echo "=== disk ==="; df -h /mnt/vdc | tail -1; echo "=== mount ==="; ls -la /mnt/vdc'
```

Look for, in order:

1. A docker version (any 24.x or 25.x is fine)
2. nvidia-smi output from INSIDE a container — that proves the toolkit works
3. ~8 TB free on `/mnt/vdc`
4. `/mnt/vdc` either empty or with a `lost+found/` directory only

If any check fails, see "Bare-image bootstrap" below.


## Step 3 — upload the docker image

The image tarball is on your Mac at `~/Downloads/ring-reconnection-fusion-v1.2.1.tar.gz`
(7.4 GB; copy it elsewhere on your Mac if you want to keep it long-term).

```bash
# Push to new instance — takes ~10 minutes on typical residential up-link
scp ~/Downloads/ring-reconnection-fusion-v1.2.1.tar.gz <GPU_HOST>:/mnt/vdc/

# Verify it landed
ssh <GPU_HOST> 'ls -lah /mnt/vdc/ring-reconnection-fusion-v1.2.1.tar.gz'
```

Load the image into docker:

```bash
ssh <GPU_HOST> 'docker load -i /mnt/vdc/ring-reconnection-fusion-v1.2.1.tar.gz'
```

That takes another 2–3 minutes (uncompressing the layers).  Verify:

```bash
ssh <GPU_HOST> 'docker images ring-reconnection-fusion:v1.2.1'
```

You should see one image at ~25 GB.


## Step 4 — sync project source

```bash
# From Mac — push the project tree to the new instance, skipping stale outputs
rsync -avz --progress \
    --exclude='runs/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='papers/' \
    --exclude='*.log' \
    --exclude='snapshot_*.tar.gz' \
    --exclude='warpx_source_backup.tar.gz' \
    ~/LaserFusionResearch/research/laser-plasma-research/ \
    <GPU_HOST>:/mnt/vdc/laser-plasma-research/

# Create the runs output directory
ssh <GPU_HOST> 'mkdir -p /mnt/vdc/test_runs && chmod 755 /mnt/vdc/test_runs'
```


## Step 5 — sanity-check before launching anything

Verify the analysis script is the pristine v0.7 (no broken patches):

```bash
ssh <GPU_HOST> 'md5sum /mnt/vdc/laser-plasma-research/analysis_scripts/pb11_first_transit_fusion.py'
```

Expected: `53b9fdbbf6b436cb489aebc3c5cf6e1c`

Run a quick smoke test of the container, confirming pywarpx + GPU access work:

```bash
ssh <GPU_HOST> 'docker run --rm --gpus all ring-reconnection-fusion:v1.2.1 \
    python -c "import pywarpx; print(\"pywarpx OK\"); import torch_or_amrex_stand_in"' 2>&1 || true

ssh <GPU_HOST> 'docker run --rm --gpus all ring-reconnection-fusion:v1.2.1 \
    python -c "import amrex.space2d as a; print(\"amrex 2d OK:\", a.__file__)"'
```


## Step 6 — launch the campaign

From your Mac (NOT inside ssh):

```bash
cd ~/LaserFusionResearch/research/laser-plasma-research

# Detached: campaign survives terminal close
nohup python3 -u stage_a_paper01.py --tier 1 \
    > stage_a_paper01_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &

# Watch progress
tail -f stage_a_paper01_*.log
```


## Step 7 — when the campaign completes

The orchestrator pulls results back automatically to
`~/LaserFusionResearch/research/laser-plasma-research/runs/paper01/<sub_tag>/`.
Cloud-side particle dumps are cleaned by the chain analysis scripts; only the
CSVs and summary text files persist.

If you want to detach the cloud volume to save cost, the data is already on
your Mac.

## Step 8 — before destroying the instance: persist the image tarball

The docker image on the instance (`ring-reconnection-fusion:v1.2.1`) is
ephemeral once the instance is destroyed.  If you don't have the tarball on
your Mac already, save it BEFORE termination so the next bring-up doesn't
need a layered rebuild:

```bash
# On the GPU box, save the image
ssh <GPU_HOST> 'docker save ring-reconnection-fusion:v1.2.1 \
    -o /mnt/vdc/ring-reconnection-fusion-v1.2.1.tar.gz'

# Pull to Mac (~7-8 GB; takes ~10 min on residential up-link)
scp <GPU_HOST>:/mnt/vdc/ring-reconnection-fusion-v1.2.1.tar.gz ~/Downloads/

# Verify
ls -lah ~/Downloads/ring-reconnection-fusion-v1.2.1.tar.gz
```

Once you have the tarball locally, you can delete the older v1.2.0 tarball
from `~/Downloads/` if it's still there.


## Common troubleshooting

### "Permission denied (publickey)" on first ssh attempt

The new instance hasn't picked up your authorized_keys yet.  Wait 30 seconds
after provisioning completes; re-try.

### "Host key verification failed"

If your `~/.ssh/config` doesn't have `StrictHostKeyChecking no` for this host,
the key from the old IP is now blocking the new one.  Either add those lines
to the config block (see Step 1) or manually clear the old key:

```bash
ssh-keygen -R <new-IP>
```

### `docker load` reports "no space left on device"

The image needs ~25 GB free on the docker root (not `/mnt/vdc`).  Check with
`docker info | grep "Docker Root Dir"` — if it's on the root filesystem and
that's small, move it to `/mnt/vdc/docker` per provider docs.

### Sim crashes immediately with "no CUDA-capable device"

The container can't see the GPU.  Test:
```bash
ssh <GPU_HOST> 'docker run --rm --gpus all ring-reconnection-fusion:v1.2.1 nvidia-smi'
```
If that fails, the nvidia-container-toolkit isn't installed or registered.
On Ubuntu:
```bash
ssh <GPU_HOST> 'sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker'
```


## Bare-image bootstrap (if prerequisites aren't preinstalled)

Only needed if Steps 1–2 reveal that the new image lacks Docker or the
NVIDIA container toolkit.  Skip if both checks passed.

```bash
ssh <GPU_HOST> << 'BOOTSTRAP'
set -e

# Docker engine
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# NVIDIA container toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Add current user to docker group so sudo isn't required
sudo usermod -aG docker $USER
BOOTSTRAP

# Log out and back in for the group change to take effect
ssh <GPU_HOST> 'docker run --rm --gpus all ubuntu:22.04 nvidia-smi | head -3'
```
