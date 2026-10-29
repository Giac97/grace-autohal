# Developer guide

## Layout

```
src/grace_al/
  cli.py            commands (typer)
  config/           schema (pydantic, extra=forbid), loader (site → user → env), docs.py
  sites/            site profiles
  workflow.py       the round loop; state.py: state.json, done markers, redo
  stages/           seeds, artifact, explore, select, label, validate, finetune
  systems/          seed generators: surface, bulk, file (+ adsorbates, relax)
  explore/          walker plan, backends (inproc, pool), tau calibration
  labelers/         aims (+ aims_parse, launcher), teacher, manual
  hpc/              SLURM job scripts, submission, partitions
  report.py         round history, plots
  _tp.py            the ONLY module that imports tensorpotential or calls its CLIs
```

The contracts between these parts (run directory layout, `Stage`, `Labeler`, `SeedGenerator`,
`ModelRef`) are in `PLAN.md` §5; each task file's "Status" section lists where the code goes
beyond it.

## Rules

- No module except `_tp.py` imports `tensorflow` / `tensorpotential`, and only inside
  functions: the CLI, the unit tests and `report` run without TF (a unit test enforces it).
- Subprocesses through `grace_al.utils.run`; logging through `grace_al.<module>` loggers.
- A stage declares the config sections it depends on (`depends_on`); everything it writes is
  listed in `state.STAGE_OUTPUTS`, which `redo` and `report` use.
- New config keys: add them to `config/schema.py` with a comment or to
  `config/docs.py:DESCRIPTIONS`, then `python -m grace_al.config.docs > docs/config.md`
  (a unit test checks both).

## Plug-ins

Labelers and seed generators are found through entry points, so another package can add them:

```toml
[project.entry-points."grace_al.labelers"]
vasp = "my_package.vasp:VaspLabeler"        # label.labeler: vasp

[project.entry-points."grace_al.systems"]
nanoparticle = "my_package.np:NPGenerator"  # system.generator: nanoparticle
```

A **labeler** is a class constructed with the `Config` that implements
`prepare(frames, workdir, system) -> [LabelTask]`, `run(task, slot=None)` (idempotent; may run
in another job), `is_finished(task)` and `collect(task) -> Atoms | LabelFailure`; optionally
`run_batch(tasks)`, `started(task)` and `summary(tasks)`. Failure categories with fixes:
`labelers/aims_parse.py:HINTS`. Without packaging, `label.labeler: module:Class` imports it
from the config folder.

A **seed generator** has `build(cfg) -> [Atoms]` (3D-periodic, `info["config"]` labels) and
`info(cfg) -> SystemInfo(kind, elements, vacuum_axis)`; `kind` (slab, bulk, cluster) drives the
HAL ensembles and the aims k-grid / dipole settings.

## Tests

```bash
pytest                                   # unit tests: no TF, no aims, < 30 s
pytest -m grace tests/integration        # tensorpotential + cached models (CPU: ~30 min)
pytest -m aims  tests/integration        # an FHI-aims binary + species (AIMS_BIN, AIMS_SPECIES_DIR)
pytest -m slurm tests/integration        # sbatch: the scripts pass sbatch --test-only
ruff check src tests && mypy
```

## Release checklist

1. the `grace` and `aims` integration tests pass on the workstation;
2. a real 30-min job on bwUniCluster in `inline` and in `array` mode;
3. `mkdocs build --strict`;
4. CHANGELOG entry, tag `vX.Y.Z` (hatch-vcs takes the version from the tag).
