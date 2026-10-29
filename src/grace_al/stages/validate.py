"""validate: the round's driver - trained without this round's data - on this round's DFT
labels. One out-of-sample error point per round; in round 0 (validation.include_round0) it
measures the foundation model itself.

Model and DFT energies differ by a constant per element (all-electron totals, fitted
reference energies), so energies are compared after removing per-element offsets fitted by
least squares (parity.element_offsets). Errors are also split by the `strategy` of each
frame (calm-greedy: the hardest frames; random-all: a fair sample, used for convergence;
seed). With validation.parity: validate/parity.png + parity.npz (the arrays to replot).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from grace_al import _tp, data, parity
from grace_al.stages.base import RoundContext, StageResult
from grace_al.state import write_json_atomic

log = logging.getLogger("grace_al.stages.validate")


def errors(e_dft, e_pred, symbols, f_dft, f_pred, offsets: dict[str, float] | None = None
           ) -> dict:
    """E (eV/atom, per-element offsets removed) and F (eV/Å, per component) RMSE/MAE.
    `offsets`: use these instead of fitting them on these frames."""
    d = parity.energy_force_data(symbols, e_dft, e_pred, f_dft, f_pred,
                                 [""] * len(symbols), offsets)
    e, f = parity.metrics(d["e_dft"], d["e_pred"]), parity.metrics(d["f_dft"], d["f_pred"])
    return {"n": len(symbols),
            "energy_offsets": dict(zip(d["offset_elements"].tolist(),
                                       d["offset_values"].tolist(), strict=True)),
            "energy_rmse_per_atom": e["rmse"], "energy_mae_per_atom": e["mae"],
            "force_rmse": f["rmse"], "force_mae": f["mae"], "force_max": f["max"]}


def strategies(ctx: RoundContext, labeled: pd.DataFrame) -> list[str]:
    """Strategy of each labeled frame: the `strategy` column, else matched by structure
    against selected.extxyz, else 'unknown'."""
    if "strategy" in labeled.columns:
        return [s if isinstance(s, str) and s else "unknown" for s in labeled["strategy"]]
    sel = ctx.rdir / "selected.extxyz"
    by_hash = {}
    if sel.exists() and sel.stat().st_size:
        by_hash = {data.structure_hash(at, 4): at.info.get("strategy", "unknown")
                   for at in data.read_frames(sel)}
    return [by_hash.get(data.structure_hash(at, 4), "unknown") for at in labeled["ase_atoms"]]


class ValidateStage:
    name = "validate"
    depends_on: tuple[str, ...] = ()

    def applies(self, ctx: RoundContext) -> bool:
        return ctx.k >= 1 or ctx.cfg.validation.include_round0

    def run(self, ctx: RoundContext) -> StageResult:
        labeled_path = ctx.rdir / "labeled.pkl.gz"
        labeled = pd.read_pickle(labeled_path)
        if not len(labeled):
            return StageResult("done", {"n": 0})
        out = ctx.rdir / "validate"
        out.mkdir(exist_ok=True)
        pred = _tp.predict(ctx.driver, labeled_path, out / "predicted.pkl.gz",
                           log_file=out / "predict.log")
        symbols = [at.get_chemical_symbols() for at in labeled["ase_atoms"]]
        args = (list(labeled["energy"]), list(pred["energy_predicted"]), symbols,
                list(labeled["forces"]), list(pred["forces_predicted"]))
        res = {"round": ctx.k, "driver": ctx.driver.kind, **errors(*args)}
        strat = np.array([str(s) for s in strategies(ctx, labeled)])
        res["by_strategy"] = {}
        for s in sorted(set(strat.tolist())):
            m = strat == s
            ed, ep, sy, fd, fp = ([x for x, keep in zip(a, m, strict=True) if keep]
                                  for a in args)
            res["by_strategy"][s] = errors(ed, ep, sy, fd, fp, offsets=res["energy_offsets"])
        if "gamma" in pred.columns:
            res["max_gamma"] = float(max(np.max(g) for g in pred["gamma"]))
        if ctx.cfg.validation.parity:
            e_dft, e_pred, _, f_dft, f_pred = args
            d = parity.energy_force_data(symbols, e_dft, e_pred, f_dft, f_pred, strat,
                                         res["energy_offsets"])
            png = out / "parity.png"
            who = ("foundation model" if ctx.driver.kind == "fm"
                   else f"model fine-tuned in round {ctx.k - 1}")
            parity.plot_energy_forces(d, png, f"Round {ctx.k}: {who} on the new DFT data")
            parity.save(png, d)
            res["parity"] = str(png.relative_to(ctx.root))
        write_json_atomic(ctx.rdir / "validation.json", res)
        log.info("round %d validation (%s driver): E %.1f meV/atom, F %.3f eV/Å RMSE (n=%d)",
                 ctx.k, ctx.driver.kind, 1e3 * res["energy_rmse_per_atom"],
                 res["force_rmse"], res["n"])

        target = ctx.cfg.convergence.force_rmse
        rnd = res["by_strategy"].get("random-all")
        if target is not None and ctx.k >= 1:
            if rnd is None:
                log.warning("convergence.force_rmse is set, but no random-all frames were "
                            "labeled this round (select.n_random = 0, or the natural set "
                            "took every candidate)")
            elif rnd["force_rmse"] < target:
                log.info("round %d: converged (random-all force RMSE %.3f < %.3f eV/Å)",
                         ctx.k, rnd["force_rmse"], target)
                return StageResult("converged", {**res, "converged": "force_rmse"})
        return StageResult("done", res)
