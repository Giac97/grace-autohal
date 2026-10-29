"""Your own seed structures (generator `file`): system.path, or a plain `seeds:` file
without a `system:` section.

All frames must be 3D-periodic (slabs: vacuum along c, pbc = True) or all non-periodic
(clusters). The kind is inferred: a vacuum gap > 6 Å along c in every frame -> slab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from grace_al.systems.base import SystemInfo, infer_system

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms

    from grace_al.config.schema import Config


def source(cfg: Config) -> Path:
    return cfg.system.path if cfg.system is not None and cfg.system.path else cfg.seeds


def check_pbc(frames: list[Atoms]) -> None:
    periodic = [bool(at.pbc.all()) for at in frames]
    if not frames:
        raise ValueError("no structures")
    if not (all(periodic) or not any(at.pbc.any() for at in frames)):
        bad = [i for i, p in enumerate(periodic) if not p][:5]
        raise ValueError(f"frames {bad}...: pbc must be True on all three axes (slabs: "
                         "vacuum along c), or False everywhere (clusters)")


class FileGenerator:
    name = "file"

    def build(self, cfg: Config) -> list[Atoms]:
        from grace_al.data import read_frames

        frames = read_frames(source(cfg))
        check_pbc(frames)
        for at in frames:
            at.info.setdefault("config", "file")
            at.info["generator"] = "file"
        return frames

    def info(self, cfg: Config) -> SystemInfo:
        from grace_al.data import read_frames

        return infer_system(read_frames(source(cfg)))
