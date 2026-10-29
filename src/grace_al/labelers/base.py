"""Labeler protocol: prepare -> run -> collect, so that `run` can happen in another job."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from grace_al import plugins

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config
    from grace_al.systems.base import SystemInfo

GROUP = "grace_al.labelers"


@dataclass
class LabelTask:
    index: int
    workdir: Path                     # runs/round_XX/dft/NNNN
    atoms: Atoms


@dataclass
class LabelFailure:
    index: int
    # stacksize | mpi | charge_integration | scf_not_converged | setup | crashed | parse
    category: str
    message: str


@dataclass
class Slot:
    """Resources of one calculation (concurrent mode: a share of the allocation)."""
    ranks: int | None = None          # None: the labeler's default
    env: dict[str, str] = field(default_factory=dict)


class Labeler(Protocol):
    name: str

    def prepare(self, frames: list[Atoms], workdir: Path,
                system: SystemInfo) -> list[LabelTask]: ...

    # idempotent; can run in another job
    def run(self, task: LabelTask, slot: Slot | None = None) -> None: ...

    def is_finished(self, task: LabelTask) -> bool: ...

    # finished: Atoms with a SinglePointCalculator; else the reason
    def collect(self, task: LabelTask) -> Atoms | LabelFailure: ...

    # optional: run_batch(tasks) (one process for many tasks), summary(tasks) -> dict


def get_labeler(cfg: Config) -> Labeler:
    """Instance of the labeler named by label.labeler (an entry point of grace_al.labelers,
    or `module:Class` importable from the config folder), constructed with the config."""
    name = cfg.label.labeler
    if ":" in name:
        import importlib
        import sys

        mod, attr = name.split(":", 1)
        if str(cfg.home) not in sys.path:
            sys.path.insert(0, str(cfg.home))
        return getattr(importlib.import_module(mod), attr)(cfg)
    return plugins.load(GROUP, name)(cfg)


def task_dirs(workdir: Path, n: int) -> list[Path]:
    return [workdir / f"{i:04d}" for i in range(n)]


def clean_atoms(at: Atoms) -> Atoms:
    """Numbers, positions, cell and pbc only: no HAL arrays/info (gamma, ...), no calculator."""
    from ase import Atoms

    return Atoms(at.numbers, at.positions, cell=at.cell, pbc=at.pbc)


def write_failures(path: Path, failures: list[LabelFailure]) -> None:
    """failures.json: {"counts": {category: n}, "failures": [{index, category, message}]}."""
    counts = Counter(f.category for f in failures)
    path.write_text(json.dumps({"counts": dict(counts.most_common()),
                                "failures": [asdict(f) for f in failures]}, indent=2) + "\n")
