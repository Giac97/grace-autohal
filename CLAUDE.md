# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`grace-al`: active learning for fine-tuning a GRACE foundation model (FM) on one system.
Each round: CALM artifact (uncertainty model of the data so far) → HAL walkers (uncertainty-
biased MD) → `grace_uq select` → FHI-aims labels → validation → gracemaker fine-tune; the
fine-tuned model drives the next round. It wraps `tensorpotential` (CLIs `grace_uq`,
`gracemaker`, `grace_models`) and FHI-aims; it runs on a workstation and on bwUniCluster 3.0
(SLURM).

The design plan and per-task notes live **outside this repo** in `../AL/PLAN.md` and
`../AL/tasks/0N-*.md`. PLAN.md §5 holds the shared contracts (run-directory layout, `Stage`,
`Labeler`, `SeedGenerator`, `ModelRef`); each task file ends with a "Status" section listing
where the code deviates from the plan. Update them when changing a contract.

## Environment and commands

The Python env is the micromamba env `grace` (tensorpotential 0.6.2.x, pandas < 3). It is not
necessarily active in the shell; call its tools directly or put its `bin` on PATH:

```bash
G=~/.local/share/mamba/envs/grace/bin
$G/python -m pytest                                  # unit tests (no TF, ~10 s)
$G/python -m pytest tests/unit/test_plan.py::test_save_load   # one test
$G/ruff check src tests                              # lint (line length 100)
$G/mypy                                              # files listed in pyproject [tool.mypy]
$G/pip install -e . --no-deps                        # after changing entry points
```

Integration tests are excluded by default (`addopts` in pyproject); select them with `-m`.
They shell out to `gracemaker` / `grace_uq`, so the env's `bin` must be on PATH:

```bash
PATH=$G:$PATH python -m pytest -m grace tests/integration/test_rounds_teacher.py  # 2 real rounds, ~16 min CPU
PATH=$G:$PATH python -m pytest -m grace tests/integration/test_explore_pool.py    # pool + SIGTERM, ~7 min
PATH=$G:$PATH python -m pytest -m aims  tests/integration/test_aims_slab.py       # real aims, ~15 min
python -m pytest -m slurm tests/integration          # needs sbatch + GRACE_AL_TEST_CONFIG
```

Models come from `$GRACE_CACHE` (default `~/.cache/grace`); `tests/conftest.py` points it at an
empty tmp dir for every test, and the integration tests reset it to the real cache. The aims
integration test uses `AIMS_BIN` (default `/usr/local/bin/aims.x`, OpenMPI build) and
`AIMS_SPECIES_DIR`.

Generated file: after changing any config key, run
`python -m grace_al.config.docs > docs/config.md`. A unit test fails if it is stale or if a key
has no description (inline `# comment` in `config/schema.py`, or `config/docs.py:DESCRIPTIONS`).
Docs build: `mkdocs build --strict` (mkdocs-material is not in the grace env).

## Architecture

**TensorFlow isolation.** `_tp.py` is the only module that imports `tensorpotential` /
`tensorflow`, and only inside functions. Everything that loads a model in the workflow runs in a
subprocess (`grace_uq` CLIs, `python -m grace_al.labelers.teacher`, `python -m
grace_al.systems.relax`), except the `inproc` explore backend. A unit test imports every module
with `sys.modules["tensorflow"] = None`; keep new modules importable that way.

**Config** (`config/`): pydantic v2, `extra="forbid"` on every section. Layers: package defaults
→ site profile (`sites/*.yaml`, via `site:`) → user `config.yaml` → `GRACE_AL__SECTION__KEY`
env vars. Paths are resolved against the config's folder (`cfg.home`) at load time. Configs
without `schema_version` are the original `al_workflow.py` format and get migrated
(`migrate_legacy`).

**Rounds and resume** (`workflow.py`, `stages/base.py`, `state.py`): stages run in the order of
`state.STAGES` (`seeds` only in round 0; `validate` in round 0 too unless
`validation.include_round0: false`). `run_stage` writes
`runs/round_XX/.done/<stage>.json` with a fingerprint of the config sections in the stage's
`depends_on` (+ optional `fingerprint_extra(ctx)`). Same fingerprint → skip; different →
`FingerprintMismatch` telling the user to `grace-al redo`. Package versions are recorded but not
hashed. A stage returns `StageResult(status)`: `done`, `waiting` (loop stops, exit code 3),
`converged`, or `failed` (raises `StageFailed`, no marker). Partial work inside a stage must be
resumable on its own (walker summaries, finished `aims.out`, `gracemaker -rl`), because job
time limits interrupt mid-stage. Stage outputs are listed in `state.STAGE_OUTPUTS`, which
`redo` and `report` rely on. `runs/state.json` is written atomically.

**Driver model** (`models.driver`): the FM for `k < fm_rounds`, else round k-1's
`fit/seed/1/` checkpoint. After the artifact stage, `workflow.run_round` replaces the driver's
artifact with the round's `UQ/gmm_artifacts.npz`, which explore/select/validate use. Finetune
also builds `fit/UQ/` for the new model; the next round's artifact stage copies it when
`params.json` (rp_dim, n_clusters, dataset hash) matches.

**Plug-ins** (`plugins.py`): entry-point groups `grace_al.systems` (surface, bulk, file) and
`grace_al.labelers` (aims, teacher, manual), declared in pyproject (reinstall after editing).
`label.labeler: module:Class` imports from the config folder. Labelers split work into
`prepare → run(task, slot) → is_finished → collect`, so `run` can happen in another job (array
mode); optional `run_batch`, `started`, `summary` are used when present.

**Exploration** (`explore/`): an explicit walker plan (`hal/plan.json` + `starts.extxyz`) is
saved once and reloaded, never rebuilt, so resubmitted jobs run the same walkers. The pool
backend (`run_pool`) runs one `grace_uq hal --structure-index` subprocess per walker with
per-slot GPU/thread env and forwards SIGTERM to the children. Never use `grace_uq hal
--repeats` (it started every walker from frame r of the pool). Select drops picks identical to
a start structure, because HAL collects step 0 when the start is extrapolative.

**FHI-aims** (`labelers/aims.py`, `aims_parse.py`, `launcher.py`): aims is started by the
launcher, not by ASE's calculator. Its environment is the module env the job saved to
`runs/.aims_env` before activating Python (or the current env without the conda MPI
variables), plus `hpc.env` from the site profile: the grace env ships its own Intel MPI, which
breaks aims. Failures are categorized (`stacksize`, `mpi`, `charge_integration`,
`scf_not_converged`, `setup`, `crashed`, `parse`), and `aims_parse.HINTS` gives each a fix.
Slab handling: k_z = 1 and the dipole correction for slabs; partial-pbc frames are written as
3D-periodic.

**Evaluation** (`parity.py`, `properties.py`, `progress.py`, `export.py`): model vs DFT always
uses per-element energy offsets (`parity.element_offsets`): fine-tuned models predict
E - sum n E0, where gracemaker's E0 is in `fit/seed/1/log.txt` (not `fit/log.txt`). Plots
always come with a `.npz` of the plotted arrays. Status progress reads only files (each
walker / aims task writes `timing.json`), so it works from a login node during a job.

**HPC** (`hpc/`): Jinja templates in `hpc/templates/` render the GPU job and the aims job array.
The GPU job loads modules, saves the aims env, activates Python, runs `check --in-job`, then
`run`; its USR1 trap calls `grace-al submit --resubmit N+1`. In array mode, exit code 3 makes
the job call `submit --labels`, which submits the array plus a dependent GPU job (`afterany`).
Submitted job ids are stored in `state.json`.

## Conventions

- Subprocesses go through `utils.run(cmd, cwd, env, log_file)`, never `os.system` or a shell.
- Logging: `logging.getLogger("grace_al.<module>")`; the CLI configures rich output. Child
  process stdout goes to files (walker.log, aims.out, gracemaker.out, UQ/build.log).
- Exit codes: 0 finished/converged, 1 error, 2 invalid config, 3 waiting for labels.
- The version comes from git tags (hatch-vcs); the fallback is `0.1.0.dev0`.
