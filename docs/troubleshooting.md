# Troubleshooting

Each entry: symptom → cause → fix. `grace-al status` shows the message of a failed run; the
label stage's message already names the most common aims failure and its fix.

## Jobs

| Symptom | Cause | Fix |
|---|---|---|
| `sbatch: error: ... User's group not permitted to use this partition` | `hpc.partition_gpu` / `partition_cpu` is not open to your groups or accounts | `grace-al prepare` lists the partitions with your access; pick one marked yes |
| the job disappears, no `logs/al_<id>.log` | SLURM could not start it (e.g. the `--output` folder was missing, a wrong partition) | `grace-al submit` creates `logs/`; check `sacct -j <id> -o JobID,State,ExitCode,Reason%30,StdOut%60` |
| log ends with "preflight failed" | a `[FAIL]` line above it | fix what it names; `grace-al check config.yaml` on the login node |
| `submit` says jobs are still queued or running | a job of this campaign exists | `squeue -u $USER`; `scancel`, or `--force` |
| `submit --resubmit`: "max_resubmit" | the campaign used up `hpc.max_resubmit` resubmissions | raise it, or a longer `hpc.time` |
| a compute node tries to download a model and fails | compute nodes have no internet | `grace-al prepare` on the login node |

## Environment

| Symptom | Cause | Fix |
|---|---|---|
| gracemaker: `IndexError ... n_structures` | pandas 3 (`np.array_split` on a DataFrame) | `micromamba install -n grace "pandas<3"`; `check` tests it |
| aims: MPI / libfabric / `prterun` errors, category `mpi` | aims started with the Python env's Intel MPI, or mpirun not found | the job loads the modules **before** activating Python and saves them (`runs/.aims_env`); set `hpc.mpirun` / `hpc.env` in the site profile |
| aims: "stacksize too low", category `stacksize` | the stack limit | the launcher raises it to unlimited when the hard limit allows; check `ulimit -Hs` in a job |

## FHI-aims

| Symptom | Cause | Fix |
|---|---|---|
| "Charge integration error is too large", category `charge_integration` | bwUniCluster's Intel build: ELSI's default density-matrix builder and shared-memory MPI collectives give a wrong density | `aims.parameters.elsi_build_dm_method: pdgemm` and `I_MPI_COLL_INTRANODE=pt2pt`, `I_MPI_PIN_DOMAIN=1`, `I_MPI_PIN_CELL=core` (site profile `bwunicluster3` has them); then `grace-al redo --round K --from label` |
| "SCF cycle not converged", category `scf_not_converged` | metal slabs converge slowly (60-80 iterations with light settings) | raise `aims.parameters.sc_iter_limit`, or `charge_mix_param: 0.05`, more smearing |
| category `setup`, "No SCF steps" | aims stopped before the SCF (species file, control.in keyword, geometry) | the `***` lines in `runs/round_XX/dft/NNNN/aims.out` |
| category `crashed`, "stopped after N SCF iterations" | killed: time limit, memory | `aims.err`, the job log; more `hpc.label_time` (array) or `hpc.time` |
| the round stops: "N/M labels failed" | more than `label.max_failure_fraction` failed | the message names the most common category and its fix; details in `failures.json` |
| "the config sections it depends on changed" | a setting a finished stage used was changed (e.g. `aims.kspacing`) | restore it, or `grace-al redo --round K --from <stage>` (recomputes from there) |

## Exploration

| Symptom | Cause | Fix |
|---|---|---|
| "nothing collected at tau ..." | τ too small or T too low | the suggested τ (`grace-al status`), higher temperatures, or `explore.calibrate: true` |
| "... % of the extrapolative frames rejected ... tau too large" | τ too large (molecules break up: `short_bond`) | the suggested smaller τ |
| validation has no `random-all` frames | the natural set took every candidate, nothing left to draw from | more walkers / steps, or accept the calm-greedy errors as an upper bound |
| "explore.ensembles [...] on a slab" | cell-changing ensembles rescale the vacuum | `[nvt]`, or `explore.allow_cell_change_on_slab: true` if you mean it |
| "too few atoms per element for a CALM artifact" | an element (adsorbate) has fewer than 12 atoms in the data | more coverages, `n_per_coverage` or `n_rattle` in `system.initial` |
| "N element(s) have no UQ training data" | FM elements not in your system | harmless |
