# Quickstart: Pd(111) + CO on bwUniCluster 3.0

## 1. Python environment (once)

```bash
micromamba create -n grace -f environment.yml     # tensorpotential, pandas<3, ...
micromamba activate grace
pip install grace-al
```

pandas 3 breaks gracemaker; `grace-al check` fails if it is installed.

## 2. A campaign folder

```bash
grace-al init surface pd_co --site bwunicluster3
```

Edit `pd_co/config.yaml`:

- `system:` the slab and the adsorbate (defaults: Pd(111) 3×3×5, CO; 38 seed structures);
- `aims.bin`, `aims.species_dir`: your FHI-aims build and species defaults
  (`light` for testing, `intermediate` for production);
- `hpc.partition_gpu`: a GPU partition **open to your group** (next step lists them).

Every key is in the [configuration reference](config.md).

## 3. Prepare on the login node

Compute nodes have no internet. `prepare` downloads the models, runs every check and lists
the partitions with your access:

```bash
grace-al prepare pd_co/config.yaml --seeds
ase gui pd_co/seeds.extxyz                         # look at the seeds
```

On the login node, `[warn]` for the GPU and the aims binary is expected; `[FAIL]` lines are
not. A partition marked `no` would fail with "User's group not permitted to use this
partition".

## 4. Submit

```bash
grace-al submit pd_co/config.yaml --dry-run        # read the job script
grace-al submit pd_co/config.yaml
grace-al status pd_co/config.yaml
```

The job loads the aims modules, saves that environment for aims, activates the Python env,
runs the checks, then `grace-al run`. 15 minutes before its time limit it stops and submits
the next job, which resumes where it stopped. Logs: `pd_co/logs/al_<jobid>.log`.

## 5. Look at the results

```bash
grace-al report pd_co/config.yaml --html           # pd_co/runs/report/index.html
```

The fine-tuned model of the last round is in `runs/round_XX/fit/seed/1/` and its uncertainty
model (for γ in production MD) in `runs/round_XX/fit/UQ/`.

## Before a long run

Test with a short interactive allocation and a small config (`explore.n_steps: 300`,
`explore.walkers_per_setting: 1`, `finetune.target_total_updates: 200`), or with the teacher
labeler (`label: {labeler: teacher, teacher_model: GRACE-2L-OMAT-large-ft-E}`), which needs
no DFT at all:

```bash
salloc -p <gpu partition> --gres=gpu:1 -n 8 -t 00:30:00
grace-al run pd_co/config.yaml --round 0
```
