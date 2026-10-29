# grace-al

Foundation-model-driven active learning for GRACE fine-tuning: a GRACE foundation model
drives HAL exploration, CALM (`grace_uq select`) picks the structures, FHI-aims labels
them, gracemaker fine-tunes, repeat.

```bash
micromamba activate grace
pip install -e .                                   # in this folder
grace-al init surface my_campaign --site bwunicluster3
grace-al prepare my_campaign/config.yaml --seeds   # login node: models, checks, partitions
grace-al submit my_campaign/config.yaml
grace-al status my_campaign/config.yaml
grace-al report my_campaign/config.yaml --html
```

Documentation: `docs/` (`mkdocs serve`): quickstart, concepts, the configuration reference,
HPC (inline vs DFT job-array mode), troubleshooting, the bwUniCluster notes and the developer
guide. Examples: `examples/`. Configs of the original `al_workflow.py` still load (with a
warning).

Tests: `pytest` (unit, no TensorFlow), `pytest -m grace|aims|slurm tests/integration`.
