"""Seed generators: build the initial structures and say what kind of system they are."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from grace_al import plugins

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config

GROUP = "grace_al.systems"


@dataclass(frozen=True)
class SystemInfo:
    kind: Literal["slab", "bulk", "cluster", "unknown"]
    elements: tuple[str, ...]
    vacuum_axis: int | None = None    # slab: cell axis along the vacuum (2 for ASE slabs)


class SeedGenerator(Protocol):
    name: str

    def build(self, cfg: Config) -> list[Atoms]: ...

    def info(self, cfg: Config) -> SystemInfo: ...


def get_generator(name: str) -> SeedGenerator:
    """Instance of the generator registered under `name` (system.generator)."""
    return plugins.load(GROUP, name)()


def vacuum_gap(atoms: Atoms, axis: int = 2) -> float:
    """Width (Å) of the largest empty gap along cell axis `axis` (atoms wrapped)."""
    import numpy as np

    length = float(np.linalg.norm(atoms.cell[axis]))
    s = np.sort(atoms.get_scaled_positions(wrap=True)[:, axis]) * length
    return float(np.diff(np.append(s, s[0] + length)).max())


def infer_system(frames: list[Atoms], min_vacuum: float = 6.0) -> SystemInfo:
    """SystemInfo from structures alone: slab if every frame has a gap > min_vacuum along
    c, cluster if none is periodic, else bulk."""
    elements = tuple(sorted({s for at in frames for s in at.get_chemical_symbols()}))
    if frames and not any(at.pbc.any() for at in frames):
        return SystemInfo("cluster", elements)
    if frames and all(vacuum_gap(at) > min_vacuum for at in frames):
        return SystemInfo("slab", elements, vacuum_axis=2)
    return SystemInfo("bulk" if frames else "unknown", elements)
