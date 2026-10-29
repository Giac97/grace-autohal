# Configuration reference

Generated from `grace_al/config/schema.py` (`python -m grace_al.config.docs`); do not edit.

Paths are relative to the folder of `config.yaml`. Unknown keys are errors. Layers, later wins:
package defaults → site profile (`site:`) → `config.yaml` → environment variables
`GRACE_AL__SECTION__KEY=value` (e.g. `GRACE_AL__AIMS__NP=32`).

## `(top level)`

| key | type | default | meaning |
|---|---|---|---|
| `schema_version` | int | `1` | config format version; configs without it are read as al_workflow.py configs and migrated |
| `site` | str \| None | `null` | site profile (bwunicluster3, workstation, or a YAML path) merged under this file |
| `workdir` | Path | `runs` | campaign output folder (state.json, round_XX/) |
| `foundation_model` | str | `GRACE-2L-SMAX-OMAT-medium` | GRACE foundation model (full name): drives the first rounds, is fine-tuned every round |
| `seeds` | Path | `seeds.extxyz` | initial structures; built from `system:` if missing |
| `fm_rounds` | int | `1` | rounds driven by the foundation model; later rounds use the last fine-tuned model |
| `system` | SystemConfig \| None | `null` | what to build the seeds from (omit to use your own `seeds` file) |
| `artifact` | ArtifactConfig | (section) | CALM artifact (uncertainty model) of each round |
| `explore` | ExploreConfig | (section) | HAL exploration |
| `select` | SelectConfig | (section) | structure selection (grace_uq select) |
| `label` | LabelConfig | (section) | labeling (DFT) |
| `aims` | AimsConfig \| None | `null` | FHI-aims settings; keep them fixed for a campaign (refused if changed) |
| `finetune` | FinetuneConfig | (section) | gracemaker fine-tuning |
| `convergence` | ConvergenceConfig | (section) | stop criteria |
| `hpc` | HpcConfig \| None | `null` | batch jobs (usually from the site profile) |

## `system`

| key | type | default | meaning |
|---|---|---|---|
| `generator` | str | `surface` | entry point group grace_al.systems |
| `surface` | SurfaceConfig \| None | `null` | slab (generator surface) |
| `bulk` | BulkConfig \| None | `null` | crystal (generator bulk) |
| `adsorbate` | AdsorbateConfig \| list[AdsorbateConfig] \| None | `null` | one adsorbate, or a list of them |
| `initial` | InitialConfig | (section) | which configurations to build, relaxation, rattling |
| `path` | Path \| None | `null` | generator: file |

## `system.surface`

| key | type | default | meaning |
|---|---|---|---|
| `element` | str | required | element of the slab |
| `facet` | fcc100 \| fcc110 \| fcc111 \| fcc211 \| bcc100 \| bcc110 \| bcc111 \| hcp0001 \| hcp10m10 \| diamond100 \| diamond111 | `fcc111` | type: ignore[valid-type] |
| `size` | list[int] | `[3, 3, 5]` | lateral repeats x, y and number of layers |
| `vacuum` | float | `15.0` | Å on each side |
| `a` | float \| None | `null` | lattice constant; None = ASE reference value |
| `c` | float \| None | `null` | c lattice constant (hcp); null = ASE reference value |
| `orthogonal` | bool \| None | `null` | orthogonal cell (ASE builder option); null = its default |
| `fix_layers` | int | `2` | bottom layers fixed in the FM pre-relaxation |

## `system.bulk`

| key | type | default | meaning |
|---|---|---|---|
| `element` | str | required | or a compound for ase.build.bulk (e.g. NaCl, rocksalt) |
| `crystalstructure` | str \| None | `null` | fcc, bcc, hcp, diamond, rocksalt, ...; None: ASE's |
| `a` | float \| None | `null` | lattice constant in Å; null = ASE reference value |
| `c` | float \| None | `null` | c lattice constant (hcp); null = ASE reference value |
| `cubic` | bool | `True` | cubic cell instead of the primitive one |
| `size` | list[int] | `[2, 2, 2]` | supercell repeats |
| `vacancies` | list[int] | `[0, 1]` | vacancies per supercell (random sites) |
| `substitutions` | dict[str, float] | `{}` | element -> fraction of host sites (alloys) |

## `system.adsorbate`

| key | type | default | meaning |
|---|---|---|---|
| `species` | str | required | element or ase.build.molecule name |
| `sites` | list[str] \| None | `null` | None = all sites of the facet |
| `height` | float | `1.5` | Å above the top layer, before the pre-relaxation |
| `min_distance` | float | `2.5` | Å between adsorbates in coverage configs (lateral, minimum image) |
| `bind_index` | int \| None | `null` | default: least electronegative heavy atom |

## `system.initial`

| key | type | default | meaning |
|---|---|---|---|
| `strains` | list[float] | `[0.98, 1.0, 1.02]` | lateral (surface) or isotropic (bulk) strain factors |
| `coverages` | list[int] | `[2, 3]` | numbers of adsorbates of the random coverage configs |
| `n_per_coverage` | int | `2` | random configs per coverage (bulk: per vacancy count and strain) |
| `fm_relax` | bool | `True` | pre-relax every config with the foundation model |
| `fmax` | float | `0.05` | eV/Å, pre-relaxation force threshold |
| `relax_steps` | int | `100` | maximum BFGS steps of the pre-relaxation |
| `n_rattle` | int | `1` | rattled copies of every config |
| `rattle_std` | float | `0.05` | Å, standard deviation of the rattle displacements |
| `seed` | int | `0` | random seed (coverage sites, rattling) |
| `mix` | bool | `False` | several adsorbate species: coverages mix them |

## `artifact`

| key | type | default | meaning |
|---|---|---|---|
| `rp_dim_round0` | int | `32` | lowered automatically when an element has few atoms |
| `rp_dim` | int | `128` | feature dimension of later rounds' artifacts (capped by the atom count of the scarcest element) |
| `n_clusters` | list[int] | `[1, 2, 4, 8, 16]` | GMM cluster counts tried per element (round 0: [1]) |
| `n_workers` | int | `1` | grace_uq build worker processes |

## `explore`

| key | type | default | meaning |
|---|---|---|---|
| `ensembles` | list[nvt \| nvt_ramp \| npt \| npt_stress \| cluster] | `['nvt']` | HAL ensembles: nvt, nvt_ramp, npt, npt_stress, cluster (slabs: nvt) |
| `temperatures` | list[int \| float] | `[800, 1200]` | K |
| `tau` | list[float] | `[0.25]` | bias strength(s): too small collects nothing, too large gets rejected |
| `n_steps` | int | `2000` | MD steps per walker |
| `walkers_per_setting` | int | `4` | walkers per (ensemble, T, tau) |
| `backend` | inproc \| pool | `pool` | pool: walkers as subprocesses sharing the GPU; inproc: one by one in the main process |
| `walkers_per_gpu` | int | `8` | pool: walkers at a time per GPU (or on the CPU without one) |
| `run_hal_kwargs` | dict[str, Any] | `{}` | extra run_hal options (pool: passed as --kebab-case flags) |
| `start_weights` | dict[str, float] | `{'clean': 0.2}` | max fraction of walkers starting from frames whose info["config"] starts with the key |
| `allow_cell_change_on_slab` | bool | `False` | nvt_ramp/npt rescale the vacuum too |
| `calibrate` | bool | `False` | tau calibration (round 0): short walkers over calibrate_taus, keep the largest tau whose rejected/(collected+rejected) <= max_reject_ratio and that collects something |
| `calibrate_taus` | list[float] | `[0.1, 0.15, 0.2, 0.25, 0.35, 0.5]` | tau values tried by the calibration |
| `calibrate_steps` | int | `500` | MD steps of a calibration walker |
| `calibrate_starts` | int | `4` | calibration walkers per tau |
| `max_reject_ratio` | float | `0.2` | calibration: highest acceptable rejected / (collected + rejected) |
| `recalibrate_each_round` | bool | `False` | calibrate again with every fine-tuned driver |
| `est_steps_per_s` | float \| None | `null` | preflight time estimate (default: 25 on CPU) |

## `select`

| key | type | default | meaning |
|---|---|---|---|
| `max_per_round` | int | `100` | cap on calm-greedy's natural set |
| `n_random` | int | `5` | random picks per round (out-of-sample validation, convergence) |

## `label`

| key | type | default | meaning |
|---|---|---|---|
| `labeler` | str | `aims` | aims / teacher / manual / module:function |
| `teacher_model` | str \| None | `null` | labeler: teacher |
| `mode` | inline \| concurrent \| array | `inline` | inline: in the GPU job; concurrent: several at once; array: CPU job array |
| `concurrent_calcs` | int | `1` | concurrent mode: calculations side by side |
| `max_failure_fraction` | float | `0.2` | more failed labels than this stops the round |

## `aims`

| key | type | default | meaning |
|---|---|---|---|
| `command` | Path \| None | `null` | legacy launcher script (run_aims.sh) |
| `bin` | str \| None | `null` | aims.x; default: $AIMS_BIN |
| `np` | int | `8` | MPI ranks when AIMS_NP / SLURM_NTASKS are not set |
| `species_dir` | Path \| None | `null` | species defaults folder (light for tests, intermediate for production) |
| `kspacing` | float | `0.2` | 1/Å incl. 2π (VASP KSPACING) |
| `slab_min_vacuum` | float | `6.0` | Å; larger gap along c -> slab (k_z = 1, dipole) |
| `timeout` | float \| None | `null` | s per calculation |
| `parameters` | dict[str, Any] | (see schema.py) | control.in keywords on top of the defaults (pbe, atomic_zora, methfessel-paxton 0.1, dipole correction for slabs, ...); `key: null` removes a default |

## `finetune`

| key | type | default | meaning |
|---|---|---|---|
| `frozen` | bool | `True` | train only the reducing_ weights |
| `reference_energy` | Union[auto, float, dict[str, float]] | `auto` | auto: per-element E0 by least squares (aims totals are all-electron); or a number / {element: E0} |
| `learning_rate` | float | `0.001` | start of the cosine schedule (minimum: lr / 12) |
| `target_total_updates` | int | `10000` | optimizer steps of a fit |
| `batch_size` | int | `16` | structures per batch |
| `test_size` | float | `0.1` | fraction of the data held out as test set |
| `energy_weight` | float | `16` | loss weight |
| `force_weight` | float | `32` | loss weight |
| `stress_weight` | float | `128.0` | loss weight (only when every structure has stress) |
| `start_from` | fm \| previous | `fm` | fm: fine-tune the foundation model every round; previous: continue from the last fine-tuned model |
| `extra` | dict[str, Any] | `{}` | deep-merged into input.yaml |

## `convergence`

| key | type | default | meaning |
|---|---|---|---|
| `max_natural_set` | int | `0` | stop when calm-greedy finds at most this many frames |
| `force_rmse` | float \| None | `null` | eV/Å on the random picks |
| `max_rounds` | int | `5` | stop after this many rounds |

## `hpc`

| key | type | default | meaning |
|---|---|---|---|
| `scheduler` | slurm | `slurm` | batch system |
| `partition_gpu` | str \| None | `null` | partition of the GPU job (must be open to your group: grace-al prepare) |
| `partition_cpu` | str \| None | `null` | partition of the DFT job array |
| `account` | str \| None | `null` | SLURM account (--account) |
| `gres` | str | `gpu:1` | GPU request, e.g. gpu:1 or gpu:h100:1 |
| `ntasks` | int | `32` | tasks of the GPU job = aims ranks in inline mode |
| `cpus_per_task` | int | `1` | CPUs per task of the GPU job |
| `mem` | str | `128G` | memory of the GPU job |
| `time` | str | `48:00:00` | time limit of the GPU job (resubmits itself 15 min before) |
| `label_time` | str | `08:00:00` | per aims array task |
| `array_max_parallel` | int | `20` | array tasks running at once |
| `max_resubmit` | int | `10` | resubmissions at most |
| `modules` | list[str] | `[]` | modules FHI-aims was built with (loaded before the Python env) |
| `env` | dict[str, str] | `{}` | exported for aims (site quirks) |
| `activate` | str \| None | `null` | shell line that activates the Python env |
| `mpirun` | str \| None | `null` | MPI launcher of aims, may include options (srun --mpi=pmi2) |
| `sbatch_extra` | list[str] | `[]` | extra #SBATCH lines, e.g. --constraint=... |
