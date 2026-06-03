# Provenance — ring-reconnection-fusion container

This document records the exact build provenance of the WarpX-CUDA simulation
container used for the p-11B ring-reconnection fusion study, for reproducibility
and for the Zenodo deposit.

## Current production image: v1.2.0

| Field | Value |
|-------|-------|
| Image | `ring-reconnection-fusion:v1.2.0` |
| Image ID | `58449f053e0d` |
| Size | 21.6 GB |
| Base | `ubuntu:22.04` (glibc 2.35) |
| Built | 2026-05-28, on H100 PCIe cloud instance |
| Verified | `pb11_ring_reconnection_v15_pulsed.py --test` runs on H100 and writes `fields/openpmd_000000.h5` + `particles/openpmd_000000.h5` |
| Saved image tarball | `ring-reconnection-fusion-v1.2.0.tar.gz` (7.5 GB, `docker save | gzip`) |
| Tarball SHA256 | `76504639aed39323e76a4060eba3df1344f1cc686eff373be7e8445c58089baa` |

## Component versions

| Component | Version / commit | Notes |
|-----------|------------------|-------|
| WarpX | BLAST-WarpX commit `7eccd51` (development) | built from source |
| AMReX | commit `b8117c772be841192859f501c6b869573647b631` | version string `26.05-56-gb8117c772be8`, fetched by WarpX |
| pywarpx wheel | `pywarpx-26.5` | |
| openPMD-api | `0.17.0` from source | **MPI + HDF5, ADIOS2 OFF**; WarpX linked via `internal=OFF` |
| HDF5 | `1.14.6` `mpi_openmpi` | parallel build, required by MPI openPMD |
| Python | `3.11.15` | conda-forge |
| CUDA toolkit | `12.6` | conda-forge |
| Cross-compiler | `x86_64-conda-linux-gnu-gcc 12.4.0` | glibc 2.17 sysroot (portable to >= 2.17) |
| MPI | OpenMPI 3.1 (conda-forge) | |
| GPU target | `AMReX_CUDA_ARCH=90` | H100 |

## Build environment

- Host: cloud H100 PCIe 80 GB, Ubuntu 22.04, NVIDIA driver 570.x / CUDA 12.8
- conda env `plasma` created from `environment_cuda_build.yml` (v4)
- WarpX compiled via `build_warpx_cuda.sh` (v3)
- Env packaged via `conda-pack`, container assembled via `Dockerfile` (v3.1) +
  `build_container.sh`

## The defect chain (why the recipe files carry version histories)

The v1.0.0 image built but did not run. Getting to a working v1.2.0 surfaced a
chain of independent defects, each now pinned/fixed in the declarative recipe:

1. **Missing conda-pack** (env v2) — build_container.sh needs it.
2. **glibc portability** (build script FIX A) — cross-compiler so binaries run
   on ubuntu:22.04, not just the build host.
3. **AMReX libs left outside env** (build script FIX B) — copy libamrex_*.so
   into the env so conda-pack captures them.
4. **Weak verification** (build script FIX C) — real one-step GPU PIC run, not
   just `import pywarpx`.
5. **Missing nvToolsExt.h** (env v3) — add `cuda-nvtx-dev`.
6. **Dockerfile smoke test imported pandas/yt** (Dockerfile v3.1) — removed;
   not pinned in env and not part of the v1.0.0-defect detection.
7. **Dockerfile imported amrex.space2d+3d in one process** (Dockerfile v3) —
   pybind11 type collision; split into separate processes.
8. **openPMD-MPI (the big one)** (env v4 + build script v3 FIX D/E) — WarpX's
   default `WarpX_openpmd_internal=ON` made it fetch and STATICALLY embed its
   own openPMD built WITHOUT MPI. The real deck then failed at the first
   diagnostic write with a misleading "openPMD-api built without support for
   backend 'HDF5'" (the true missing capability is MPI). Fix: build openPMD
   0.17.0 from source with MPI+HDF5 and configure WarpX with
   `-DWarpX_openpmd_internal=OFF -DopenPMD_DIR=...`. conda-forge has no
   mpi_openmpi openpmd-api for py311, so source build is mandatory; a conda
   openpmd-api pin also caused a conda-pack stale-metadata trap, so it was
   removed from the env file entirely.

9. **Missing cupy** (env v5) — the deck's particle-diagnostic code expects
   WarpX-CUDA particle data as cupy (GPU) arrays and converts via .get().
   Without cupy the analysis path segfaults AFTER the simulation completes
   (Errorcode 11). The run itself is fine; only post-step GPU particle
   analysis crashes. Installing cupy (14.1.0, CUDA 12.x) resolved it. The
   256^2 confirmation then ran 2500 steps, wrote 12 openPMD dumps, and
   produced sensible fusion diagnostics (fast-fraction ~0.96, fusion rate
   ~3-4e20/s).

A clean rebuild from the v4 env + v3 build script + v3.1 Dockerfile should now
reproduce v1.2.0 in a single pass with no live intervention.

## Reproduce from scratch

```
# 1. create env (v4)
mamba env create -f environment_cuda_build.yml
conda activate plasma

# 2. build WarpX + MPI openPMD (v3 build script; ~40-70 min on H100)
bash build_warpx_cuda.sh

# 3. package env
conda-pack -n plasma -o plasma_env.tar.gz --compress-level 1

# 4. build container (v3.1 Dockerfile)
bash build_container.sh v1.2.0

# 5. verify end-to-end
docker run --rm --gpus all -v $PWD/runs:/work/runs \
  ring-reconnection-fusion:v1.2.0 \
  bash -c 'cd /work && export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_TLS=self,sm && \
    mpirun --allow-run-as-root -n 1 python simulation/pb11_ring_reconnection_v15_pulsed.py \
    --test --outdir /work/runs/smoke_test'
# success = fields/openpmd_000000.h5 and particles/openpmd_000000.h5 written
```

## OpenMPI on cloud boxes

Runs prefix MPI with:
```
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_TLS=self,sm
```
This forces shared-memory transport and avoids an InfiniBand locked-memory
stall on instances with a low `ulimit -l` (8 MB). In-container, mpirun also
needs `--allow-run-as-root`.

## Known follow-ups (not blocking)

- containerd snapshotter stores layers on the root disk (`/var/lib/containerd`)
  not on `/mnt/vdc`; Docker's `data-root` setting does NOT move it. Big builds
  can fill root (94%). Permanent fix: set containerd `root = /mnt/vdc/containerd`
  in `/etc/containerd/config.toml` and restart, OR `docker builder prune -af` +
  remove stale images before each large build.
- Image tarball for Zenodo: `docker save ring-reconnection-fusion:v1.2.0 | gzip`
  then record its sha256 in SHA256SUMS.
- Multi-rank on a single GPU (`-n 8`) segfaults at the STEP 1->2 transition
  (Errorcode 11). Single-rank (`-n 1`) works. For multi-rank/multi-GPU
  production runs this needs CUDA MPS or per-rank device assignment (standard
  WarpX deployment topic). Not a blocker: `-n 1` at 256^2 on an H100 is fast.
  NOTE: the cupy fix was validated with -n 1; re-test -n 8 separately once
  MPS/device-assignment is configured.
