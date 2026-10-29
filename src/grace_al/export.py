"""`grace-al export`: the model of one round as a self-contained folder.

  saved_model/     plain SavedModel (gracemaker's final_model = best test checkpoint)
  uq/saved_model/  the same model with uncertainty (per-atom gamma, needs uq/gmm_artifacts.npz)
  uq/gmm_artifacts.npz
  checkpoint/      model.yaml + checkpoint.best_test_loss.* (fine-tune or rebuild artifacts)
  data/            training set, gracemaker input.yaml, test metrics
  validation/      this model on the NEXT round's DFT data (out of sample), if that ran
  MODEL_CARD.md, provenance.json
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al import __version__, _tp, data
from grace_al.models import fit_dir
from grace_al.report import fit_test_metrics, round_rows
from grace_al.state import read_marker, round_dir, rounds_on_disk

if TYPE_CHECKING:
    from grace_al.config.schema import Config


class ExportError(RuntimeError):
    pass


def last_fitted_round(cfg: Config) -> int:
    done = [k for k in rounds_on_disk(cfg.workdir) if (fit_dir(cfg, k) / "FINISHED").exists()]
    if not done:
        raise ExportError(f"no finished fit in {cfg.workdir}")
    return done[-1]


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def export(cfg: Config, k: int | None = None, out: Path | None = None,
           force: bool = False) -> Path:
    k = last_fitted_round(cfg) if k is None else k
    fit = fit_dir(cfg, k)
    if not (fit / "FINISHED").exists():
        raise ExportError(f"round {k} has no finished fit")
    out = out or cfg.home / "export" / f"round_{k:02d}"
    if out.exists():
        if not force:
            raise ExportError(f"{out} exists (--force to replace it)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    seed = fit / "seed" / "1"
    has = {
        "saved_model": _copy(seed / "final_model", out / "saved_model"),
        "uq_saved_model": _copy(fit / "UQ" / "saved_model", out / "uq" / "saved_model"),
        "uq_artifact": _copy(fit / "UQ" / "gmm_artifacts.npz", out / "uq" / "gmm_artifacts.npz"),
        "model_yaml": _copy(seed / "model.yaml", out / "checkpoint" / "model.yaml"),
    }
    for f in (seed / "checkpoints").glob("checkpoint.best_test_loss.*"):
        _copy(f, out / "checkpoint" / f.name)
    has["checkpoint"] = any((out / "checkpoint").glob("checkpoint.best_test_loss.*"))
    for name, dst in (("train.pkl.gz", "training_set.pkl.gz"), ("input.yaml", "input.yaml")):
        _copy(fit / name, out / "data" / dst)
    _copy(seed / "test_metrics.yaml", out / "data" / "test_metrics.yaml")
    nxt = round_dir(cfg.workdir, k + 1)
    for name in ("validation.json", "validate/parity.png", "validate/parity.npz"):
        _copy(nxt / name, out / "validation" / Path(name).name)
    if not has["saved_model"] and not has["uq_saved_model"]:
        raise ExportError(f"round {k}: neither {seed / 'final_model'} nor "
                          f"{fit / 'UQ' / 'saved_model'} exists")
    prov = provenance(cfg, k, has)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    (out / "MODEL_CARD.md").write_text(model_card(prov))
    return out


def provenance(cfg: Config, k: int, has: dict) -> dict:
    import pandas as pd

    fit = fit_dir(cfg, k)
    train = pd.read_pickle(fit / "train.pkl.gz") if (fit / "train.pkl.gz").exists() else None
    ft = (read_marker(round_dir(cfg.workdir, k), "finetune") or {})
    lab = read_marker(round_dir(cfg.workdir, 0), "label") or {}
    nxt = round_dir(cfg.workdir, k + 1) / "validation.json"
    return {
        "campaign": cfg.home.name, "round": k,
        "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "foundation_model": cfg.foundation_model,
        "start_from": ft.get("summary", {}).get("start_from"),
        "elements": sorted(data.element_counts(train)) if train is not None else None,
        "n_train_structures": len(train) if train is not None else None,
        "reference_energies": (ft.get("summary", {}).get("reference_energies")
                               or _tp.fit_reference_energies(fit)),
        "fit_test": fit_test_metrics(fit),
        "out_of_sample": json.loads(nxt.read_text()) if nxt.exists() else None,
        "labeler": cfg.label.labeler, "teacher_model": cfg.label.teacher_model,
        "dft": cfg.aims.model_dump(mode="json") if cfg.aims and cfg.label.labeler == "aims"
        else None,
        "versions": {"grace_al": __version__, "tensorpotential": _tp.tp_version(),
                     "aims": lab.get("versions", {}).get("aims")},
        "contents": has,
        "rounds": [{key: r[key] for key in ("round", "collected", "selected", "labeled",
                                            "val_force_rmse", "fit_done")}
                   for r in round_rows(cfg)],
        "config": cfg.to_dict(),
    }


def _num(v, scale=1.0, unit="") -> str:
    return "–" if v is None else f"{v * scale:.3g}{unit}"


def model_card(p: dict) -> str:
    oos = p["out_of_sample"] or {}
    ft = p["fit_test"] or {}
    ref = p["reference_energies"] or {}
    dft = p["dft"] or {}
    rnd = (oos.get("by_strategy") or {}).get("random-all", {})
    if p["labeler"] == "aims":
        labels = f"FHI-aims {p['versions']['aims'] or ''}".strip()
    elif p["labeler"] == "teacher":
        labels = f"the teacher model {p['teacher_model']} (dry run, not DFT)"
    else:
        labels = p["labeler"]
    L = [f"# {p['campaign']}: model of round {p['round']}", "",
         f"Fine-tuned from **{p['foundation_model']}** with grace-al "
         f"{p['versions']['grace_al']} (tensorpotential {p['versions']['tensorpotential']}), "
         f"exported {p['exported']}.", "",
         f"- Elements: {', '.join(p['elements'] or [])}",
         f"- Training data: {p['n_train_structures']} structures labeled with {labels}",
         f"- Fit test set: force RMSE {_num(ft.get('force_rmse'), unit=' eV/Å')}, energy RMSE "
         f"{_num(ft.get('energy_rmse_per_atom'), 1e3, ' meV/atom')}"]
    if oos:
        L.append(f"- Out of sample (round {p['round'] + 1}'s new DFT data, never trained on): "
                 f"force RMSE {_num(oos.get('force_rmse'), unit=' eV/Å')} (random picks: "
                 f"{_num(rnd.get('force_rmse'), unit=' eV/Å')}), energy RMSE "
                 f"{_num(oos.get('energy_rmse_per_atom'), 1e3, ' meV/atom')}; parity plot in "
                 "`validation/`")
    else:
        L.append("- Out of sample: not measured (no later round validated this model)")
    L += ["", "## Use", "", "```python", "from tensorpotential.calculator import TPCalculator",
          'calc = TPCalculator(model="saved_model")']
    if p["contents"].get("uq_saved_model"):
        L.append('calc = TPCalculator(model="uq/saved_model")  # also calc.results["gamma"], '
                 "per atom")
    refs = ", ".join(f"{el} {e:.6f} eV" for el, e in sorted(ref.items())) or "not recorded"
    L += ["```", "",
          "**Energies are relative to per-element reference energies** (`reference_energy: "
          "auto`): model energy + sum over the atoms of E0(element) = the label energy "
          f"({labels}). E0: {refs}.", ""]
    if dft:
        params = ", ".join(f"`{k} {v}`" for k, v in dft.get("parameters", {}).items())
        L += ["## DFT settings", "", f"{params}; kspacing {dft.get('kspacing')} 1/Å (k_z = 1 "
              f"for slabs); species `{dft.get('species_dir')}`. Use the same settings for any "
              "data you compare with or add.", ""]
    L += ["## Rounds", "", "| round | HAL frames collected | selected | labeled | "
          "validation F RMSE (eV/Å) |", "|---|---|---|---|---|"]
    L += [f"| {r['round']} | {_num(r['collected'])} | {_num(r['selected'])} | "
          f"{_num(r['labeled'])} | {_num(r['val_force_rmse'])} |" for r in p["rounds"]]
    L += ["", "Full provenance, including the campaign config: `provenance.json`.", ""]
    return "\n".join(L)
