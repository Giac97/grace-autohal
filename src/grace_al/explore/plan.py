"""Walker plan: which walker starts from which frame, with which settings.

The start pool is last round's picks (the frontier found so far) followed by the seeds,
each shuffled with default_rng(k). Walkers (ensembles x temperatures x taus x
walkers_per_setting) take the pool in order, cyclically, so the walkers of one setting start
from different frames. explore.start_weights caps the fraction of walkers starting from
frames whose info["config"] begins with a key (default: at most 20 % from clean slabs);
capped frames are skipped while their quota is used up.

The plan and the pool are saved as hal/plan.json + starts.extxyz and loaded, not rebuilt,
when they exist: a resubmitted job runs exactly the same walkers.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from grace_al.state import write_json_atomic

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config
    from grace_al.systems.base import SystemInfo

CELL_CHANGING = {"nvt_ramp", "npt", "npt_stress"}


class PlanError(ValueError):
    pass


@dataclass
class WalkerSpec:
    id: int
    start_index: int                  # frame of starts.extxyz
    ensemble: str
    temperature: float
    tau: float
    seed: int
    n_steps: int
    extra: dict = field(default_factory=dict)

    @property
    def dir(self) -> str:
        return f"w{self.id:03d}"


def start_pool(seeds: list[Atoms], picks: list[Atoms], k: int) -> list[Atoms]:
    """Last round's picks first, then the seeds; each shuffled with default_rng(k)."""
    rng = np.random.default_rng(k)
    shuffled = lambda frames: [frames[i] for i in rng.permutation(len(frames))]
    for at in picks:
        at.info.setdefault("config", "pick")
    return shuffled(picks) + shuffled(seeds)


def ensembles(cfg: Config, system: SystemInfo | None) -> list[str]:
    ens: list[str] = list(cfg.explore.ensembles)
    kind = system.kind if system else "unknown"
    if kind == "cluster":
        return ["cluster"]
    bad = sorted(set(ens) & CELL_CHANGING)
    if kind == "slab" and bad and not cfg.explore.allow_cell_change_on_slab:
        raise PlanError(f"explore.ensembles {bad} on a slab: they rescale the whole cell, "
                        "vacuum included. Use [nvt], or set explore.allow_cell_change_on_slab")
    return ens


def start_order(starts: list[Atoms], n_walkers: int, weights: dict[str, float]) -> list[int]:
    """Start frame of each walker: the pool cyclically, skipping capped frames whose quota
    (floor(fraction * n_walkers)) is used up."""
    if not starts:
        raise PlanError("empty start pool")
    quota = {key: math.floor(frac * n_walkers) for key, frac in weights.items()}
    used = dict.fromkeys(weights, 0)
    key_of = [next((key for key in weights
                    if str(at.info.get("config", "")).startswith(key)), None) for at in starts]
    if all(key is not None and quota[key] == 0 for key in key_of):
        quota = {key: n_walkers for key in weights}     # nothing else to start from
    order: list[int] = []
    i = 0
    while len(order) < n_walkers:
        for _ in range(len(starts)):
            j, i = i % len(starts), i + 1
            key = key_of[j]
            if key is None or used[key] < quota[key]:
                break
        else:       # every frame capped and its quota used: take the next one anyway
            j = (i - 1) % len(starts)
        capped = key_of[j]
        if capped is not None:
            used[capped] += 1
        order.append(j)
    return order


def build_plan(cfg: Config, k: int, starts: list[Atoms], system: SystemInfo | None,
               taus: list[float] | None = None) -> list[WalkerSpec]:
    """`taus` replaces explore.tau (tau calibration)."""
    ex = cfg.explore
    combos = [(e, T, tau) for e in ensembles(cfg, system) for T in ex.temperatures
              for tau in (taus or ex.tau) for _ in range(ex.walkers_per_setting)]
    order = start_order(starts, len(combos), ex.start_weights)
    return [WalkerSpec(i, order[i], e, T, tau, 1000 * k + i, ex.n_steps)
            for i, (e, T, tau) in enumerate(combos)]


def save_plan(path: Path, plan: list[WalkerSpec], starts_name: str = "starts.extxyz") -> None:
    write_json_atomic(path, {"starts": starts_name, "walkers": [asdict(w) for w in plan]})


def load_plan(path: Path) -> list[WalkerSpec]:
    return [WalkerSpec(**w) for w in json.loads(path.read_text())["walkers"]]
