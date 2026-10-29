"""FHI-aims labeler: energies, forces (and stress) of the selected frames.

One folder per structure (dft/0000, 0001, ...) with geometry.in + control.in, written by
ASE's aims template; aims itself is started by launcher.run_aims (not through ASE), so a
task can run in another job (array mode). A finished aims.out is never recomputed.

Parameters on top of aims.parameters:
  every frame   compute_forces
  periodic      (along any axis; aims gets 3D lattice vectors) k_grid from aims.kspacing
                (VASP-style, 1/Å incl. 2π), analytical stress;
                slab (SystemInfo.kind == "slab" or a gap > aims.slab_min_vacuum along c):
                k_z = 1 and use_dipole_correction (aims finds the vacuum level itself)
  otherwise     no k-grid, no stress, no dipole correction
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from grace_al.labelers import aims_parse, launcher
from grace_al.labelers.base import LabelFailure, LabelTask, Slot, clean_atoms, task_dirs
from grace_al.systems.base import vacuum_gap

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config
    from grace_al.systems.base import SystemInfo

log = logging.getLogger("grace_al.labelers.aims")


def k_grid(atoms: Atoms, kspacing: float, slab: bool = False) -> list[int]:
    """Monkhorst-Pack grid with spacing <= kspacing (1/Å, incl. 2π, as VASP KSPACING).
    Slab: k_z = 1."""
    rec = np.linalg.norm(atoms.cell.reciprocal(), axis=1) * 2 * math.pi
    k = [max(1, math.ceil(b / kspacing)) for b in rec]
    if slab:
        k[2] = 1
    return k


def is_slab(atoms: Atoms, system: SystemInfo | None, min_vacuum: float) -> bool:
    return bool(atoms.pbc.any()) and (
        (system is not None and system.kind == "slab") or vacuum_gap(atoms) > min_vacuum)


def aims_parameters(cfg: Config, atoms: Atoms, system: SystemInfo | None = None) -> dict:
    """control.in parameters for one frame."""
    assert cfg.aims is not None
    params = dict(cfg.aims.parameters)
    params["compute_forces"] = True
    if atoms.pbc.any():                 # aims is periodic in 3D or not at all
        slab = is_slab(atoms, system, cfg.aims.slab_min_vacuum)
        params["k_grid"] = k_grid(atoms, cfg.aims.kspacing, slab)
        params["compute_analytical_stress"] = True
        if not slab:
            params.pop("use_dipole_correction", None)
    else:
        for k in ("use_dipole_correction", "compute_analytical_stress", "k_grid"):
            params.pop(k, None)
    return params


def write_inputs(cfg: Config, atoms: Atoms, directory: Path,
                 system: SystemInfo | None = None) -> None:
    from ase.calculators.aims import AimsProfile, AimsTemplate

    assert cfg.aims is not None
    directory.mkdir(parents=True, exist_ok=True)
    if atoms.pbc.any() and not atoms.pbc.all():
        atoms = atoms.copy()
        atoms.pbc = True                # e.g. ASE slabs (T, T, F): vacuum along the open axis
    profile = AimsProfile(command="aims.x", default_species_directory=str(cfg.aims.species_dir))
    AimsTemplate().write_input(profile, directory, atoms, aims_parameters(cfg, atoms, system),
                               ["energy", "forces"])


class AimsLabeler:
    name = "aims"

    def __init__(self, cfg: Config):
        if cfg.aims is None:
            raise ValueError("the aims labeler needs an `aims:` section")
        self.cfg = cfg

    def prepare(self, frames: list[Atoms], workdir: Path,
                system: SystemInfo | None = None) -> list[LabelTask]:
        tasks = []
        for i, (frame, d) in enumerate(zip(frames, task_dirs(workdir, len(frames)),
                                           strict=True)):
            task = LabelTask(i, d, clean_atoms(frame))
            if not self.is_finished(task):
                write_inputs(self.cfg, task.atoms, d, system)
            tasks.append(task)
        return tasks

    def run(self, task: LabelTask, slot: Slot | None = None) -> None:
        if self.is_finished(task):
            return
        slot = slot or Slot()
        rc = launcher.run_aims(task.workdir, self.cfg, slot.ranks, slot.env)
        ok = self.is_finished(task)
        log.info("  aims %s: %s", task.workdir.name, "finished" if ok else
                 f"failed (exit code {rc}, {self._failure(task).category})")

    def started(self, task: LabelTask) -> bool:
        """aims was started on this task (array mode: don't list it again)."""
        return (task.workdir / "aims.out").exists()

    def is_finished(self, task: LabelTask) -> bool:
        return aims_parse.finished(aims_parse.read_text(task.workdir / "aims.out"))

    def collect(self, task: LabelTask) -> Atoms | LabelFailure:
        if not self.is_finished(task):
            return self._failure(task)
        try:
            return aims_parse.read_results(task.workdir / "aims.out", pbc=task.atoms.pbc)
        except Exception as err:
            return LabelFailure(task.index, "parse", f"{type(err).__name__}: {err}")

    def summary(self, tasks: list[LabelTask]) -> dict:
        """aims version, mean SCF iterations of the finished runs (for the done marker)."""
        versions: Counter[str] = Counter()
        iters = []
        for t in tasks:
            text = aims_parse.read_text(t.workdir / "aims.out")
            if aims_parse.finished(text):
                versions[aims_parse.version(text) or "unknown"] += 1
                iters.append(aims_parse.scf_iterations(text))
        out: dict = {"mean_scf_iterations": float(np.mean(iters)) if iters else None}
        if versions:
            out["versions"] = {"aims": versions.most_common(1)[0][0]}
            if len(versions) > 1:
                out["aims_versions"] = dict(versions)
        return out

    def _failure(self, task: LabelTask) -> LabelFailure:
        if not (task.workdir / "aims.out").exists():
            return LabelFailure(task.index, "crashed", "not run (no aims.out)")
        cat, msg = aims_parse.categorize(aims_parse.read_text(task.workdir / "aims.out"),
                                         aims_parse.read_text(task.workdir / "aims.err"))
        return LabelFailure(task.index, cat, msg)
