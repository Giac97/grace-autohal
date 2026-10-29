"""Teacher labeler: a GRACE foundation model plays DFT (dry runs of the whole loop, tests).

Each task folder gets structure.extxyz; `run` writes results.extxyz (energy, forces, and
stress for 3D-periodic frames). The model runs in a subprocess, so TensorFlow stays out
of the main process (`inproc = True` computes in this process instead):

    python -m grace_al.labelers.teacher MODEL TASK_DIR [TASK_DIR ...]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al.labelers.base import LabelFailure, LabelTask, Slot, clean_atoms, task_dirs
from grace_al.utils import run as run_cmd

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config
    from grace_al.systems.base import SystemInfo

log = logging.getLogger("grace_al.labelers.teacher")

INPUT, OUTPUT = "structure.extxyz", "results.extxyz"


def label_dirs(model: str, dirs: list[Path]) -> None:
    """Compute results.extxyz in each folder (loads TF)."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    from grace_al import _tp
    from grace_al.data import read_frames

    calc = _tp.fm_calculator(model)
    for d in dirs:
        at = read_frames(d / INPUT)[0]
        at.calc = calc
        res = {"energy": at.get_potential_energy(), "forces": at.get_forces()}
        if at.pbc.all():
            res["stress"] = at.get_stress()
        at.calc = SinglePointCalculator(at, **res)
        tmp = d / f".{OUTPUT}"
        write(tmp, at, format="extxyz")
        tmp.replace(d / OUTPUT)


class TeacherLabeler:
    name = "teacher"
    inproc = False

    def __init__(self, cfg: Config):
        if not cfg.label.teacher_model:
            raise ValueError("the teacher labeler needs label.teacher_model")
        self.model = cfg.label.teacher_model

    def prepare(self, frames: list[Atoms], workdir: Path,
                system: SystemInfo | None = None) -> list[LabelTask]:
        from ase.io import write

        tasks = []
        for i, (frame, d) in enumerate(zip(frames, task_dirs(workdir, len(frames)),
                                           strict=True)):
            task = LabelTask(i, d, clean_atoms(frame))
            if not self.is_finished(task):
                d.mkdir(parents=True, exist_ok=True)
                write(d / INPUT, task.atoms, format="extxyz")
            tasks.append(task)
        return tasks

    def run(self, task: LabelTask, slot: Slot | None = None) -> None:
        self.run_batch([task])

    def run_batch(self, tasks: list[LabelTask]) -> None:
        """All pending tasks in one process (one model load)."""
        dirs = [t.workdir for t in tasks if not self.is_finished(t)]
        if not dirs:
            return
        if self.inproc:
            label_dirs(self.model, dirs)
        else:
            run_cmd([sys.executable, "-m", "grace_al.labelers.teacher", self.model, *dirs],
                    log_file=dirs[0].parent / "teacher.log", check=False)

    def is_finished(self, task: LabelTask) -> bool:
        return (task.workdir / OUTPUT).exists()

    def collect(self, task: LabelTask) -> Atoms | LabelFailure:
        from grace_al.data import read_frames

        if not self.is_finished(task):
            return LabelFailure(task.index, "crashed", f"no {OUTPUT} (see teacher.log)")
        try:
            at = read_frames(task.workdir / OUTPUT)[0]
        except Exception as err:
            return LabelFailure(task.index, "parse", f"{type(err).__name__}: {err}")
        at.pbc = task.atoms.pbc
        return at


if __name__ == "__main__":
    label_dirs(sys.argv[1], [Path(p) for p in sys.argv[2:]])
