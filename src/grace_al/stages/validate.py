"""validate (round k >= 1): the driver of round k - trained without this round's data -
on this round's DFT labels. One out-of-sample error point per round.

Fine-tuned models predict energies relative to fitted per-element references, so absolute
energies are not comparable with DFT totals: the energy error is reported after removing
the mean per-atom offset. Errors are also split by the `strategy` of each frame
(calm-greedy: the hardest frames; random-all: a fair sample, used for convergence).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from grace_al import _tp, data
from grace_al.stages.base import RoundContext, StageResult
from grace_al.state import write_json_atomic

log = logging.getLogger("grace_al.stages.validate")


def errors(e_dft, e_pred, n_atoms, f_dft, f_pred, offset: float | None = None) -> dict:
    """E (eV/atom, after removing the mean per-atom offset) and F (eV/Å, per component)
    RMSE/MAE. `offset`: use this per-atom offset instead of the mean of these frames."""
    f = lambda a: np.asarray(list(a), dtype=float)
    d = (f(e_pred) - f(e_dft)) / f(n_atoms)
    off = float(d.mean()) if offset is None else offset
    de = d - off
    df = np.concatenate([np.ravel(p) - np.ravel(t) for p, t in zip(f_pred, f_dft, strict=True)])
    return {"n": len(d), "energy_offset_per_atom": off,
            "energy_rmse_per_atom": float(np.sqrt(np.mean(de ** 2))),
            "energy_mae_per_atom": float(np.mean(np.abs(de))),
            "force_rmse": float(np.sqrt(np.mean(df ** 2))),
            "force_mae": float(np.mean(np.abs(df)))}


def strategies(ctx: RoundContext, labeled: pd.DataFrame) -> list[str]:
    """Strategy of each labeled frame: the `strategy` column, else matched by structure
    against selected.extxyz, else 'unknown'."""
    if "strategy" in labeled.columns:
        return [s if isinstance(s, str) and s else "unknown" for s in labeled["strategy"]]
    from ase.io import read

    sel = ctx.rdir / "selected.extxyz"
    by_hash = {}
    if sel.exists() and sel.stat().st_size:
        by_hash = {data.structure_hash(at, 4): at.info.get("strategy", "unknown")
                   for at in read(sel, ":")}
    return [by_hash.get(data.structure_hash(at, 4), "unknown") for at in labeled["ase_atoms"]]


class ValidateStage:
    name = "validate"
    depends_on: tuple[str, ...] = ()

    def applies(self, ctx: RoundContext) -> bool:
        return ctx.k >= 1

    def run(self, ctx: RoundContext) -> StageResult:
        labeled_path = ctx.rdir / "labeled.pkl.gz"
        labeled = pd.read_pickle(labeled_path)
        if not len(labeled):
            return StageResult("done", {"n": 0})
        out = ctx.rdir / "validate"
        out.mkdir(exist_ok=True)
        pred = _tp.predict(ctx.driver, labeled_path, out / "predicted.pkl.gz",
                           log_file=out / "predict.log")
        n_atoms = [len(at) for at in labeled["ase_atoms"]]
        args = (labeled["energy"], pred["energy_predicted"], n_atoms,
                labeled["forces"], pred["forces_predicted"])
        res = {"round": ctx.k, "driver": ctx.driver.kind, **errors(*args)}
        strat = np.array(strategies(ctx, labeled))
        res["by_strategy"] = {}
        for s in sorted(set(strat)):
            m = strat == s
            e_dft, e_pred, nat, f_dft, f_pred = ([x for x, keep in zip(a, m, strict=True) if keep]
                                                 for a in args)
            res["by_strategy"][s] = errors(e_dft, e_pred, nat, f_dft, f_pred,
                                           offset=res["energy_offset_per_atom"])
        if "gamma" in pred.columns:
            res["max_gamma"] = float(max(np.max(g) for g in pred["gamma"]))
        write_json_atomic(ctx.rdir / "validation.json", res)
        log.info("round %d validation (%s driver): E %.1f meV/atom, F %.3f eV/Å RMSE (n=%d)",
                 ctx.k, ctx.driver.kind, 1e3 * res["energy_rmse_per_atom"],
                 res["force_rmse"], res["n"])

        target = ctx.cfg.convergence.force_rmse
        rnd = res["by_strategy"].get("random-all")
        if target is not None:
            if rnd is None:
                log.warning("convergence.force_rmse is set, but no random-all frames were "
                            "labeled this round (select.n_random = 0?)")
            elif rnd["force_rmse"] < target:
                log.info("round %d: converged (random-all force RMSE %.3f < %.3f eV/Å)",
                         ctx.k, rnd["force_rmse"], target)
                return StageResult("converged", {**res, "converged": "force_rmse"})
        return StageResult("done", res)
