# HPC

## Site profiles

A site profile is a YAML file merged **under** your `config.yaml` (`site: bwunicluster3`, or
`--site` on any command). It holds what is the same for everybody on a cluster: modules,
partitions, Intel MPI settings, aims quirks and the line that activates the Python env.
Packaged profiles: `bwunicluster3`, `workstation`. A path to your own YAML works too.

To add a site, copy `grace_al/sites/bwunicluster3.yaml`, adapt it and pass its path, or add it
to `grace_al/sites/` in a pull request. Keys: [`hpc`](config.md#hpc) plus anything else from
the config (typically `aims.parameters`).

## prepare

On a login node (it has internet; compute nodes usually don't):

1. downloads the FM checkpoint + SavedModel (and the teacher model);
2. runs every check (GPU and aims binary only warn: the login node may have neither);
3. lists the SLURM partitions with `AllowGroups`, `AllowAccounts`, `MaxTime`, `TRES`, marks
   the ones open to you, and fails if `hpc.partition_gpu` / `partition_cpu` is not among them;
4. with `--seeds`, builds the seeds (FM relaxation on the CPU, a few minutes).

## The GPU job

`grace-al submit` writes `runs/jobs/NNN_gpu_rN.sbatch` and submits it. In order, the job

1. loads `hpc.modules` (the modules aims was built with) and saves that environment to
   `runs/.aims_env`: the Python env ships its own Intel MPI, which must not reach aims;
2. activates the Python env (`hpc.activate`) and exports `hpc.env`;
3. runs `grace-al check --in-job` (no visible GPU is an error here), then `grace-al run`;
4. 15 minutes before `hpc.time` (SIGUSR1), stops the run cleanly and calls
   `grace-al submit --resubmit N+1` (at most `hpc.max_resubmit` times; never after
   `runs/FINISHED`).

`submit` refuses while a job of the campaign is queued or running (`--force` to override) and
creates `logs/` (SLURM drops a job silently when its `--output` folder is missing).

## Labeling modes

| `label.mode` | where aims runs | use |
|---|---|---|
| `inline` | in the GPU job, one calculation after another, all `hpc.ntasks` ranks | simple; the GPU idles during DFT |
| `concurrent` | in the GPU job, `label.concurrent_calcs` at once on separate cores | many small calculations |
| `array` | a CPU job array; the GPU job ends while DFT runs | long DFT; GPU hours matter |

**Array mode.** When a round reaches labeling, `grace-al run` prepares every calculation,
lists the ones not started in `runs/round_XX/dft/tasks.txt` and exits with code 3. The job
then calls `grace-al submit --labels`:

1. a CPU array (`hpc.partition_cpu`, `aims.np` ranks, `hpc.label_time`, at most
   `hpc.array_max_parallel` at once) over the unfinished tasks; each element runs
   `grace-al aims-run <config> <task folder>` in the module environment;
2. the GPU job again with `--dependency=afterany:<array>`: it collects the results (failed
   calculations count against `label.max_failure_fraction`) and continues the round.

You can switch `label.mode` in the middle of a campaign; only the DFT settings themselves
(`aims:`) are locked.

## Watching a campaign

```bash
grace-al status config.yaml        # rounds, the current stage, the last jobs (sacct)
grace-al report config.yaml --html # runs/report/: plots + index.html
```

The report shows HAL frames, the natural set, validation and fit errors per round, the γ of
the selected frames, aims failures by category, and the wall time per round with the share
spent on DFT inside the GPU job (the GPU idle time of inline mode).
