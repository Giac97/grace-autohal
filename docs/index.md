# grace-al

Fine-tune a GRACE foundation model (FM) for one system with as few DFT calculations as
possible. `grace-al` repeats a round of:

1. **artifact**: an uncertainty model (CALM) of the data you already have;
2. **explore**: uncertainty-biased MD (HAL) driven by the current model;
3. **select**: a small, diverse set of the new structures (`grace_uq select`);
4. **label**: FHI-aims on them (or a teacher model for dry runs);
5. **validate**: the current model's errors on these new labels;
6. **finetune**: gracemaker on all labels so far; the new model drives the next round.

```bash
pip install grace-al                         # in the env with tensorpotential (grace)
grace-al init surface my_campaign --site bwunicluster3
grace-al prepare my_campaign/config.yaml     # login node: models, checks, partitions
grace-al submit my_campaign/config.yaml
grace-al status my_campaign/config.yaml
grace-al report my_campaign/config.yaml --html
```

| Command | What it does |
|---|---|
| `init` | copy a template config (`surface`, `bulk`, `custom_seeds`) |
| `prepare` | login node: download models, run checks, list the partitions open to you |
| `seeds` | build the initial structures from `system:` (preview them with `ase gui`) |
| `check` | preflight: tools, models, seeds, aims, GPU |
| `run` | run or resume the loop here (exit code 0 done, 3 waiting for labels, 1 error) |
| `submit` | write the job script from the config and submit it |
| `status` | round history and jobs |
| `redo` | delete a round's outputs from a stage on, so they are recomputed |
| `report` | plots and tables in `runs/report/` |
| `aims-run` | one aims calculation (used by the DFT job array) |

Start with the [quickstart](quickstart.md); [concepts](concepts.md) explains what happens in a
round.
