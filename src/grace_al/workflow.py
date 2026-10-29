"""The active-learning loop: rounds of stages, state transitions, stop criteria.

One round (PLAN.md §5.1):
  seeds     round 0 only: the initial structures
  artifact  CALM artifact = the driver's feature space, fitted on the data we already have
  explore   HAL walkers driven by that model + artifact -> extrapolative frames
  select    grace_uq select (calm-greedy + random-all) against the same artifact
  label     DFT -> labeled.pkl.gz; the training set grows
  validate  from round 1: the driver's errors on the new labels (out of sample)
  finetune  fine-tune on everything labeled so far; artifact of the new model

The loop stops when a stage waits for labels (exit code 3: resume later), when a stop
criterion is met (natural set small, force RMSE below target) or after max_rounds. SIGTERM
(the job's time-limit trap, scancel) stops every child process and records `interrupted`
(exit code 4); the next run resumes.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from grace_al import _tp, models
from grace_al.stages.base import RoundContext, get_stage, run_stage
from grace_al.state import STAGES, Status, round_dir, update_state
from grace_al.utils import Interrupted, on_sigterm

if TYPE_CHECKING:
    from grace_al.config.schema import Config
    from grace_al.systems.base import SystemInfo

log = logging.getLogger("grace_al.workflow")

# exit codes of `grace-al run` for the job scripts
EXIT_CODES: dict[str, int] = {"finished": 0, "converged": 0, "running": 0,
                              "waiting_for_labels": 3, "failed": 1, "interrupted": 4}


def system_info(cfg: Config) -> SystemInfo:
    """From the seed generator if it is registered, else inferred from the seeds file."""
    from grace_al.data import read_frames
    from grace_al.plugins import PluginError
    from grace_al.systems.base import get_generator, infer_system

    if cfg.system is not None:
        try:
            return get_generator(cfg.system.generator).info(cfg)
        except PluginError:
            pass
    return infer_system(read_frames(cfg.seeds), cfg.aims.slab_min_vacuum if cfg.aims else 6.0)


def ensure_fm(cfg: Config) -> None:
    if not _tp.checkpoint_cached(cfg.foundation_model):
        log.info("downloading the %s checkpoint (needs internet)", cfg.foundation_model)
        _tp.download_fm(cfg.foundation_model, checkpoint=True, savedmodel=False)


def context(cfg: Config, k: int, system: SystemInfo) -> RoundContext:
    rdir = round_dir(cfg.workdir, k)
    rdir.mkdir(parents=True, exist_ok=True)
    driver = models.driver(cfg, k)
    if driver.kind == "fm":
        ensure_fm(cfg)
    return RoundContext(cfg, k, cfg.workdir, rdir, system, driver)


def run_seeds(cfg: Config) -> None:
    """The seeds stage (round 0); needs neither the system info nor a driver."""
    from grace_al.systems.base import SystemInfo

    rdir = round_dir(cfg.workdir, 0)
    rdir.mkdir(parents=True, exist_ok=True)
    update_state(cfg.workdir, round=0, stage="seeds", status="running", message="")
    ctx = RoundContext(cfg, 0, cfg.workdir, rdir, SystemInfo("unknown", ()),
                       _tp.fm_ref(cfg.foundation_model))
    run_stage(get_stage("seeds"), ctx)


def run_round(cfg: Config, k: int, system: SystemInfo) -> str:
    """The stages of round k after seeds; 'done', 'waiting' or 'converged'."""
    ctx = context(cfg, k, system)
    log.info("======== round %d  (driver: %s) ========", k, ctx.driver.kind)
    for name in STAGES[1:]:
        update_state(cfg.workdir, round=k, stage=name, status="running", message="")
        res = run_stage(get_stage(name), ctx)
        if name == "artifact":
            # explore, select and validate use the driver with this round's artifact
            ctx.driver = dataclasses.replace(ctx.driver,
                                             artifact=ctx.rdir / "UQ" / "gmm_artifacts.npz")
        if res.status in ("waiting", "converged"):
            return res.status
    return "done"


def run(cfg: Config, only_round: int | None = None) -> Status:
    """Run or resume the campaign (or only round `only_round`). Returns the final status,
    also written to runs/state.json; runs/FINISHED gets the reason when the campaign ends.
    Errors (FingerprintMismatch, StageFailed, ...) are recorded as 'failed' and re-raised."""
    root, max_rounds = cfg.workdir, cfg.convergence.max_rounds
    root.mkdir(parents=True, exist_ok=True)
    rounds = [only_round] if only_round is not None else list(range(max_rounds))
    k, result = rounds[0], "done"
    try:
        with on_sigterm():
            run_seeds(cfg)
            system = system_info(cfg)
            for k in rounds:
                result = run_round(cfg, k, system)
                log.info("round %d: %s", k, result)
                if result != "done":
                    break
    except (Interrupted, KeyboardInterrupt) as err:
        update_state(root, round=k, status="interrupted",
                     message=f"stopped ({type(err).__name__}: {err}); the next run resumes")
        raise
    except BaseException as err:
        update_state(root, round=k, status="failed", message=f"{type(err).__name__}: {err}")
        raise
    status: Status
    if result == "waiting":
        status, msg = "waiting_for_labels", f"round {k}: waiting for labels"
    elif result == "converged":
        status, msg = "converged", f"converged in round {k}"
    elif k == max_rounds - 1:
        status, msg = "finished", f"max_rounds ({max_rounds}) reached"
    else:                                   # --round k of several
        status, msg = "running", f"round {k} done"
    if status in ("converged", "finished"):
        (root / "FINISHED").write_text(f"{status}: {msg}\n")
    update_state(root, round=k, status=status, message=msg)   # stage: where it stopped
    log.info("%s", msg)
    return status
