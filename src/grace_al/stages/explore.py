"""explore: HAL walkers driven by the round's model + artifact (explore.plan, .backends).

Files: starts.extxyz (the start pool), hal/plan.json, hal/w<ID>/ (one per walker),
hal/walkers.txt (the walker folders with candidates, read by select), and with tau
calibration hal/calibrate/ + hal/calibration.json.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from grace_al import data
from grace_al.explore import calibrate
from grace_al.explore.backends import make_backend
from grace_al.explore.plan import (
    PlanError,
    WalkerSpec,
    build_plan,
    load_plan,
    save_plan,
    start_pool,
)
from grace_al.stages.base import RoundContext, StageResult
from grace_al.state import round_dir

log = logging.getLogger("grace_al.stages.explore")


def summarize(hal: Path, plan: list[WalkerSpec], failures: dict[int, dict]) -> dict:
    """Per walker: collected, rejected by reason, stop reason; totals and ratios."""
    walkers, with_features = [], []
    for w in plan:
        p = hal / w.dir / "hal_summary.json"
        s = json.loads(p.read_text()) if p.exists() else {}
        walkers.append({"id": w.id, "start_index": w.start_index, "ensemble": w.ensemble,
                        "temperature": w.temperature, "tau": w.tau,
                        "stop_reason": s.get("stop_reason", "failed"),
                        "collected": s.get("n_collected", 0),
                        "rejected": s.get("rejected", {}),
                        **({"error": failures[w.id]} if w.id in failures else {})})
        if (hal / w.dir / "hal_features.npz").exists():
            with_features.append(w.dir)
    (hal / "walkers.txt").write_text("".join(f"{d}\n" for d in with_features))
    rejected: Counter[str] = Counter()
    for row in walkers:
        rejected.update(row["rejected"])
    n_col, n_rej = sum(row["collected"] for row in walkers), sum(rejected.values())
    ran = [row for row in walkers if row["stop_reason"] != "failed"]
    return {"n_walkers": len(plan), "n_failed": len(plan) - len(ran),
            "n_with_candidates": len(with_features),
            "collected": n_col, "rejected": dict(rejected.most_common()),
            "zero_collected_fraction": (sum(row["collected"] == 0 for row in ran) / len(ran)
                                        if ran else None),
            "reject_ratio": n_rej / (n_col + n_rej) if n_col + n_rej else 0.0,
            "walkers": walkers}


def warnings(s: dict) -> list[str]:
    out = []
    if s["n_walkers"] > s["n_failed"] and s["collected"] == 0:
        out.append("no walker collected anything: tau too small or T too low "
                   "(explore.tau, explore.temperatures)")
    if s["reject_ratio"] > 0.5:
        reason = next(iter(s["rejected"]), "?")
        out.append(f"{100 * s['reject_ratio']:.0f} % of the extrapolative frames were "
                   f"rejected (mostly: {reason}): tau too large (explore.tau)")
    return out


class ExploreStage:
    name = "explore"
    depends_on = ("explore",)

    def fingerprint_extra(self, ctx: RoundContext) -> str:
        return ctx.driver.kind

    def applies(self, ctx: RoundContext) -> bool:
        return True

    def run(self, ctx: RoundContext) -> StageResult:
        from ase.io import write

        cfg, hal = ctx.cfg, ctx.rdir / "hal"
        hal.mkdir(exist_ok=True)
        starts_file, plan_file = ctx.rdir / "starts.extxyz", hal / "plan.json"
        backend = make_backend(cfg)
        if plan_file.exists() and starts_file.exists():
            plan = load_plan(plan_file)
        else:
            prev = round_dir(ctx.root, ctx.k - 1) / "selected.extxyz"
            picks = data.read_frames(prev) if ctx.k > 0 and prev.exists() else []
            starts = start_pool(data.read_frames(cfg.seeds), picks, ctx.k)
            write(starts_file, starts, format="extxyz")
            try:
                taus = self._taus(ctx, starts, starts_file, backend)
                plan = build_plan(cfg, ctx.k, starts, ctx.system, taus)
            except PlanError as err:
                return StageResult("failed", {"error": str(err)})
            save_plan(plan_file, plan, starts_file.name)
        n_todo = sum(not (hal / w.dir / "hal_summary.json").exists() for w in plan)
        log.info("round %d: %d walkers (%d to run), backend %s", ctx.k, len(plan), n_todo,
                 cfg.explore.backend)
        failures = backend.run(plan, starts_file, hal, ctx.driver)
        s = summarize(hal, plan, failures)
        for msg in warnings(s):
            log.warning("round %d: %s", ctx.k, msg)
        s["warnings"] = warnings(s)
        if s["n_failed"] == len(plan):
            first = next(iter(failures.values()), {})
            return StageResult("failed", {**s, "error": "every walker failed; last lines of "
                                          f"the first log:\n{first.get('log_tail', '')}"})
        if s["n_failed"]:
            log.warning("round %d: %d of %d walkers failed (see hal/w*/walker.log); the "
                        "others are used", ctx.k, s["n_failed"], len(plan))
        return StageResult("done", s)

    @staticmethod
    def _taus(ctx: RoundContext, starts, starts_file: Path, backend) -> list[float] | None:
        """[calibrated tau], or None (explore.tau)."""
        ex = ctx.cfg.explore
        if not ex.calibrate:
            return None
        again = ex.recalibrate_each_round and ctx.driver.kind == "finetuned"
        if ctx.k > 0 and not again:
            for j in range(ctx.k - 1, -1, -1):
                c = calibrate.load(round_dir(ctx.root, j) / "hal")
                if c and c.get("tau") is not None:
                    return [c["tau"]]
        hal = ctx.rdir / "hal"
        done = calibrate.load(hal)
        if done is None:
            base = build_plan(ctx.cfg, ctx.k, starts, ctx.system, [ex.tau[0]])
            walkers = calibrate.calibration_walkers(base, ex.calibrate_taus,
                                                    ex.calibrate_starts, ex.calibrate_steps)
            log.info("round %d: tau calibration, %d short walkers over tau %s", ctx.k,
                     len(walkers), ex.calibrate_taus)
            backend.run(walkers, starts_file, hal / "calibrate", ctx.driver)
            res = calibrate.tally(walkers, hal / "calibrate")
            tau = calibrate.choose(res, ex.max_reject_ratio)
            done = calibrate.save(hal, tau, res, ctx.driver.kind)
            log.info("round %d: calibrated tau = %s (%s)", ctx.k, tau, res)
        if done.get("tau") is None:
            log.warning("tau calibration: no tau collected with at most %.0f %% rejected; "
                        "using explore.tau", 100 * ex.max_reject_ratio)
            return None
        return [done["tau"]]
