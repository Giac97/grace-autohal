"""Exploration checks: HAL ensembles vs the kind of system, and a time estimate of one
round's walkers vs the job's time limit."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from grace_al.config.schema import Config
from grace_al.preflight import Check, register_check

CPU_STEPS_PER_S = 25.0          # one HAL walker on CPU threads (Pd slab, 2L model)


def system_kind(cfg: Config) -> str:
    from grace_al.systems.base import get_generator, infer_system

    if cfg.system is not None:
        return get_generator(cfg.system.generator).info(cfg).kind
    if cfg.seeds.exists():
        from grace_al.data import read_frames

        return infer_system(read_frames(cfg.seeds)).kind
    return "unknown"


def seconds(slurm_time: str) -> float:
    """SLURM time ('D-HH:MM:SS', 'HH:MM:SS', 'MM:SS', 'MM') in seconds."""
    days, _, rest = slurm_time.rpartition("-")
    parts = [float(p) for p in rest.split(":")]
    if len(parts) == 1:
        parts = [0, parts[0], 0]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return (float(days) if days else 0) * 86400 + h * 3600 + m * 60 + s


def n_gpus(cfg: Config) -> int:
    m = re.search(r"(\d+)$", cfg.hpc.gres) if cfg.hpc else None
    return int(m.group(1)) if m else 1


@register_check
def check_slab_ensembles(cfg: Config) -> Iterable[Check]:
    try:
        kind = system_kind(cfg)
    except Exception:            # the seeds checks report missing / broken inputs
        return
    ens = set(cfg.explore.ensembles)
    if kind == "slab":
        yield Check(f"HAL ensembles {sorted(ens)} for a slab",
                    ens <= {"nvt"} or cfg.explore.allow_cell_change_on_slab,
                    fix="use [nvt]: nvt_ramp/npt/npt_stress rescale the cell isotropically, "
                        "including the vacuum (or set explore.allow_cell_change_on_slab)",
                    group="explore")
    elif kind == "cluster" and ens != {"cluster"}:
        yield Check(f"HAL ensembles {sorted(ens)} for a cluster", True,
                    "non-periodic seeds: the plan uses the `cluster` ensemble",
                    severity="warning", group="explore")


@register_check
def check_explore_time(cfg: Config) -> Iterable[Check]:
    ex = cfg.explore
    n = (len(ex.ensembles) * len(ex.temperatures) * len(ex.tau) * ex.walkers_per_setting)
    slots = 1 if ex.backend == "inproc" else ex.walkers_per_gpu * n_gpus(cfg)
    rate = ex.est_steps_per_s or CPU_STEPS_PER_S
    est = math.ceil(n / slots) * ex.n_steps / rate
    if ex.calibrate:
        est += (math.ceil(len(ex.calibrate_taus) * ex.calibrate_starts / slots)
                * ex.calibrate_steps / rate)
    from grace_al.progress import fmt_duration

    msg = (f"{n} walkers x {ex.n_steps} steps, {slots} at a time, ~{rate:g} steps/s: "
           f"~{fmt_duration(est)} per round (explore.est_steps_per_s)")
    limit = seconds(cfg.hpc.time) if cfg.hpc else None
    ok = limit is None or est < 0.5 * limit
    yield Check(f"exploration time estimate: ~{fmt_duration(est)} per round", ok,
                msg + (f"; more than half of hpc.time {cfg.hpc.time}" if not ok and cfg.hpc
                       else ""),
                fix="fewer walkers or steps, more walkers_per_gpu, or a longer hpc.time",
                severity="warning", group="explore")
