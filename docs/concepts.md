# Concepts

## A round

```
seeds (round 0: slab + adsorbate configs, FM-pre-relaxed, rattled)
 │
 ├─► artifact   grace_uq build: CALM model of the data known so far (round 0: the seeds)
 │   explore    HAL walkers driven by the current model → frames with γ > 1
 │   select     grace_uq select: calm-greedy → small, diverse "natural set" (+ random picks)
 │   label      FHI-aims (round 0: also all seeds)
 │   validate   from round 1: the current model's errors on these new labels
 └── finetune   gracemaker on ALL labels so far → the model of the next round
```

- **γ means "new to me".** The weights come from the FM, but the CALM artifact is built from
  **your** data, so γ > 1 means "not covered by my DFT yet". Round 0's artifact knows only the
  seeds; its feature size (`artifact.rp_dim_round0`) is lowered automatically when an element
  has few atoms.
- **Driver.** The first `fm_rounds` rounds are driven by the foundation model, later rounds by
  the model fine-tuned in the previous round. Each fine-tuned model gets its own artifact right
  away (`fit/UQ/`), so it can run MD with γ; the next round reuses it when the data match.
- **Walker plan.** Every walker starts from an explicit structure: last round's picks first
  (the frontier found so far), then the seeds, shuffled per round. At most 20 % start from a
  clean slab (`explore.start_weights`). The plan is saved (`hal/plan.json`), so a resubmitted
  job runs the same walkers.
- **Natural set.** calm-greedy picks frames until no remaining frame has γ > 1 under the
  updated model: the natural set. `select.max_per_round` caps it. Picks identical to a start
  structure (HAL collects step 0 when the start is new) are dropped: they are labeled already.
- **Random picks** (`select.n_random`) are an unbiased sample. The next round's validation
  error on them is the honest estimate of the current model's error. They are drawn from the
  candidates calm-greedy left; when the natural set takes every candidate, there are none.

## Validation and parity plots

After labeling, the validate stage evaluates the round's driver - which has not seen these
DFT data - on them: in round 0 the foundation model itself, later the model fine-tuned in the
previous round (`validation.include_round0`). Energies are compared after removing
per-element offsets fitted by least squares (the model's energy zero differs from aims's
all-electron totals by a constant per element). Results: `validation.json` (overall and per
selection strategy) and, with `validation.parity`, `validate/parity.png` with
`validate/parity.npz` holding the plotted arrays:

```python
import numpy as np, matplotlib.pyplot as plt
d = np.load("runs/round_02/validate/parity.npz")
plt.scatter(d["f_dft"], d["f_pred"], s=4)      # also e_dft, e_pred (eV/atom), group, f_element
```

`grace-al validate` does the same for any model and derived properties:

```bash
grace-al validate config.yaml                          # last model: energy, relative, adsorption
grace-al validate config.yaml --model fm -p adsorption --references refs.extxyz
grace-al validate config.yaml --model round:2 -p cohesive --references atoms.extxyz
```

| property | definition |
|---|---|
| `energy` | per-atom energy (per-element offsets removed) and force components |
| `relative` | E - E(DFT-lowest structure) among structures of the same composition and cell: site preference, coverage ordering; needs no references |
| `adsorption` | E(slab + adsorbates) - E(clean slab of the same cell) - sum n mu(adsorbate element) |
| `cohesive` | sum x mu(isolated atom) - E/N, for structures without vacuum |

`mu` comes from `--references`: a structure file with DFT-labeled references (isolated atoms,
gas molecules), which the model computes too. Without references, adsorption energies use mu
fitted per method: then only differences between sites and coverages are meaningful. Points
are labeled `trained on` (the model's training rounds) or `new`. Output:
`runs/report/validate/<property>_<model>.png` + `.npz`.

## Exporting a model

```bash
grace-al export config.yaml              # the last fitted round -> export/round_XX/
```

The folder holds `saved_model/` (ASE: `TPCalculator(model="saved_model")`), `uq/saved_model/`
(the same with per-atom γ), the checkpoint, the training set, the out-of-sample validation of
the next round, `MODEL_CARD.md` and `provenance.json`. The model's energies are relative to
per-element reference energies E0 (listed in the card): model energy + sum of E0 over the
atoms = DFT total energy.

## Stopping

The loop stops at the first of:

- a natural set of at most `convergence.max_natural_set` frames (from round 1);
- a random-pick force RMSE below `convergence.force_rmse` (eV/Å, validation);
- `convergence.max_rounds`.

`runs/FINISHED` gets the reason.

## Resume and safety

Every stage writes a **done marker** (`round_XX/.done/<stage>.json`) with a fingerprint of the
config sections it depends on. A finished stage is skipped; partial work inside a stage is
kept (walkers with a summary, finished aims runs, `gracemaker -rl`). If you change a setting a
finished stage depends on, for example `aims.kspacing` after round 0 was labeled, the run
stops instead of mixing data:

```
round 0, stage label: the config sections it depends on changed since it ran ...
grace-al redo --round 0 --from label
```

`redo` deletes that stage's outputs, the later stages of the round and every later round.
Changing how a stage runs (`label.mode`, `explore.backend`, thresholds) is always allowed.

`runs/state.json` holds the current round, stage and status (`running`,
`waiting_for_labels`, `interrupted` (stopped by SIGTERM; resumes), `converged`, `finished`,
`failed` + the message).

## Slabs

- **HAL ensemble `nvt` only.** `nvt_ramp`, `npt` and `npt_stress` rescale the whole cell
  isotropically, vacuum included; the plan refuses them for slabs. Lateral strain comes from
  the seeds (`system.initial.strains`).
- **τ (`explore.tau`)** is the bias strength and the most important setting: too small and
  nothing is collected, too large and most triggers are rejected (molecules break up,
  adsorbates desorb). After every exploration, the τ advice compares each τ's rejected /
  (collected + rejected) with `explore.max_reject_ratio` (0.2) and suggests the next value
  (one step along `explore.calibrate_taus`); `grace-al status` shows it. `explore.calibrate:
  true` picks τ from short walkers in round 0 by the same criterion.
- **aims:** k_z = 1 and the dipole correction (aims finds the vacuum level itself) are set for
  slabs automatically; Methfessel-Paxton smearing suits metals.

## Energies

aims totals are all-electron (keV per atom). `finetune.reference_energy: auto` fits per-element
E0 by least squares and `shift: auto` aligns the FM, so the fit sees small energies. Validation
reports the energy error after removing the mean per-atom offset for the same reason.

## Outputs

```
config.yaml, seeds.extxyz, logs/
runs/
  state.json, FINISHED, .aims_env, jobs/ (job scripts), report/
  round_XX/
    .done/<stage>.json        done markers (summary of every stage)
    artifact_data.pkl.gz, UQ/ the round's artifact
    starts.extxyz, hal/       walker plan, hal/w000/ ... (hal_summary.json, hal_diagnostics.png)
    selected.extxyz           the picks (atoms.info["strategy"])
    to_label.extxyz, dft/     one folder per calculation
    labeled.pkl.gz            ase_atoms, energy, forces[, stress], strategy, label_index
    failures.json             failed labels by category
    validation.json           errors of the round's driver on the new labels
    fit/                      input.yaml, train.pkl.gz, seed/1/ (the model), UQ/
```
