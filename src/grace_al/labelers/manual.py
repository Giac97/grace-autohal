"""Manual labeler: export the structures, compute them yourself, import the results.

prepare writes round_XX/to_label.extxyz (atoms.info["label_index"] = task index) and
round_XX/MANUAL_LABELING.txt. The round waits until round_XX/labeled.extxyz (energy, forces
[, stress] as in any ASE-written extxyz) or round_XX/labeled.pkl.gz (columns ase_atoms,
energy, forces[, stress]) exists. Results are matched to tasks by `label_index` (info key or
column); without it, positionally if the counts agree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al.labelers.base import LabelFailure, LabelTask, Slot, clean_atoms, task_dirs

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config
    from grace_al.systems.base import SystemInfo

log = logging.getLogger("grace_al.labelers.manual")

INSTRUCTIONS = """\
{n} structures to label in {to_label}.

Compute energy and forces (and stress for 3D-periodic cells) with the same DFT settings
for every round, then write ONE of
  {rdir}/labeled.extxyz    e.g. ase.io.write(path, frames) with a SinglePointCalculator
  {rdir}/labeled.pkl.gz    pandas DataFrame, columns ase_atoms, energy, forces[, stress]
Keep atoms.info["label_index"] (or a `label_index` column) so that results find their
structure; failed structures may simply be left out. Then run grace-al again.
"""


class ManualLabeler:
    name = "manual"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._results: dict[Path, dict[int, Atoms] | str] = {}

    def prepare(self, frames: list[Atoms], workdir: Path,
                system: SystemInfo | None = None) -> list[LabelTask]:
        from ase.io import write

        rdir = workdir.parent
        tasks, out = [], []
        for i, (frame, d) in enumerate(zip(frames, task_dirs(workdir, len(frames)),
                                           strict=True)):
            tasks.append(LabelTask(i, d, clean_atoms(frame)))
            at = clean_atoms(frame)
            at.info = {"label_index": i, **({"strategy": frame.info["strategy"]}
                                            if "strategy" in frame.info else {})}
            out.append(at)
        rdir.mkdir(parents=True, exist_ok=True)
        if not self._source(rdir):
            if not (rdir / "to_label.extxyz").exists():      # the label stage writes it too
                write(rdir / "to_label.extxyz", out, format="extxyz")
            (rdir / "MANUAL_LABELING.txt").write_text(INSTRUCTIONS.format(
                n=len(frames), to_label=rdir / "to_label.extxyz", rdir=rdir))
        return tasks

    def run(self, task: LabelTask, slot: Slot | None = None) -> None:
        pass

    def is_finished(self, task: LabelTask) -> bool:
        return self._source(task.workdir.parent.parent) is not None

    def collect(self, task: LabelTask) -> Atoms | LabelFailure:
        res = self._load(task.workdir.parent.parent)
        if isinstance(res, str):
            return LabelFailure(task.index, "parse", res)
        if task.index not in res:
            return LabelFailure(task.index, "crashed", "no result for this structure")
        at = res[task.index]
        at.pbc = task.atoms.pbc
        return at

    @staticmethod
    def _source(rdir: Path) -> Path | None:
        for name in ("labeled.extxyz", "labeled.pkl.gz"):
            if (rdir / name).exists():
                return rdir / name
        return None

    def _load(self, rdir: Path) -> dict[int, Atoms] | str:
        """{task index: Atoms with SinglePointCalculator}, or an error message."""
        if rdir in self._results:
            return self._results[rdir]
        src = self._source(rdir)
        n_tasks = _count(rdir)
        try:
            frames, idx = _read(src)
        except Exception as err:
            res: dict[int, Atoms] | str = f"cannot read {src}: {type(err).__name__}: {err}"
        else:
            if idx is None and len(frames) != n_tasks:
                res = (f"{src}: {len(frames)} results for {n_tasks} structures and no "
                       "label_index to match them")
            else:
                res = dict(zip(idx if idx is not None else range(len(frames)), frames,
                               strict=True))
        self._results[rdir] = res
        return res


def _count(rdir: Path) -> int:
    from grace_al.data import read_frames

    return len(read_frames(rdir / "to_label.extxyz"))


def _read(src: Path | None) -> tuple[list, list[int] | None]:
    """Frames with SinglePointCalculators and their label_index (None if absent)."""
    from ase.calculators.singlepoint import SinglePointCalculator

    assert src is not None
    if src.name.endswith(".pkl.gz"):
        import pandas as pd

        df = pd.read_pickle(src)
        frames = []
        for _, row in df.iterrows():
            at = row["ase_atoms"].copy()
            kw = {"energy": row["energy"], "forces": row["forces"]}
            if "stress" in df.columns and row["stress"] is not None:
                kw["stress"] = row["stress"]
            at.calc = SinglePointCalculator(at, **kw)
            frames.append(at)
        if "label_index" in df.columns:
            return frames, [int(i) for i in df["label_index"]]
        idx = [at.info.get("label_index") for at in frames]
        return frames, None if None in idx else [int(i) for i in idx]
    from grace_al.data import read_frames

    frames = read_frames(src)
    idx = [at.info.get("label_index") for at in frames]
    return frames, None if None in idx else [int(i) for i in idx]
