# Changelog

## 0.1.0 (unreleased)

First version of the package, built from the `al_workflow.py` scripts.

- `grace-al` command: `init`, `prepare`, `seeds`, `check`, `run`, `submit`, `status`, `redo`,
  `report`, `aims-run`.
- Typed configuration (pydantic, unknown keys are errors), site profiles (`bwunicluster3`,
  `workstation`), environment overrides; configs of the original scripts still load.
- Stage-based rounds with done markers and config fingerprints: a changed DFT setting after a
  round was labeled is refused instead of mixing data; `redo` recomputes from a stage.
- Seeds: `surface` (identical to `build_seeds.py`; several adsorbate species), `bulk`, `file`;
  FM pre-relaxation in a subprocess.
- Exploration: explicit walker plan (fixes `--repeats` starting every walker from a clean
  slab), pool backend with GPU slots and SIGTERM handling, optional tau calibration.
- Selection drops picks identical to a start structure (HAL step-0 frames).
- Labelers: FHI-aims (saved module environment, categorized failures with fixes; inline,
  concurrent and job-array modes), teacher model, manual.
- Validation of each driver on the next round's DFT data; stop criteria on the natural set and
  on the random-pick force RMSE.
- Fine-tuning from a template (frozen `reducing_` weights, reference energy / shift auto),
  `start_from: previous`, the artifact of every fitted model.
- Validation from round 0 (the FM itself) with per-element energy offsets; parity plots
  (`validate/parity.png` + `.npz`) every round.
- `grace-al validate`: parity of any model for energies/forces, relative, adsorption and
  cohesive energies (DFT-labeled references optional), PNG + `.npz`.
- `grace-al export`: SavedModel, UQ model, checkpoint, training set, model card.
- tau advice after every exploration (also in `status`).
- `status` shows the running stage's progress: DFT calculations / walkers done, running,
  pending, time per item and ETA; fit epochs.
- `grace-al monitor`: live, read-only web page of a campaign (login node + SSH tunnel).
- Clean interruption: SIGTERM stops every child process group (mpirun and its ranks too),
  status `interrupted`, exit code 4; interrupted aims array tasks are run again.
- SLURM: job scripts from the config, self-resubmission, DFT job-array chains, partition
  access check; `status` and `report` (plots, HTML).
