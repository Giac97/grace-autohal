"""Foundation-model pre-relaxation of seed structures (bottom slab layers fixed).

Runs in a subprocess by default, so the main process doesn't hold TensorFlow / GPU memory:

    python -m grace_al.systems.relax in.extxyz out.extxyz --model NAME [--fix-layers 2]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms


def relax_inproc(frames: list[Atoms], model: str, fix_layers: int, fmax: float,
                 steps: int) -> list[Atoms]:
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS

    from grace_al import _tp

    calc = _tp.fm_calculator(model)
    out = []
    for at in frames:
        at = at.copy()
        tags = at.get_tags()
        n_layers = tags.max()
        if fix_layers and n_layers:
            # ASE slab tags: 1 = top layer ... n_layers = bottom; adsorbates 0
            at.set_constraint(FixAtoms(mask=tags > n_layers - fix_layers))
        at.calc = calc
        BFGS(at, logfile=None).run(fmax=fmax, steps=steps)
        e = at.get_potential_energy()
        at.calc = None
        at.set_constraint()
        out.append(at)
        print(f"  relaxed {at.info.get('config', len(out) - 1)}: E = {e:.3f} eV", flush=True)
    return out


def fm_relax(frames: list[Atoms], model: str, fix_layers: int = 2, fmax: float = 0.05,
             steps: int = 100, inproc: bool = False) -> list[Atoms]:
    """Relaxed copies of `frames` (info and tags kept)."""
    if inproc:
        return relax_inproc(frames, model, fix_layers, fmax, steps)
    from ase.io import write

    from grace_al.data import read_frames
    from grace_al.utils import run

    with tempfile.TemporaryDirectory(prefix="grace_al_relax_") as tmp:
        src, dst = Path(tmp) / "in.extxyz", Path(tmp) / "out.extxyz"
        write(src, frames, format="extxyz")
        run([sys.executable, "-m", "grace_al.systems.relax", src, dst, "--model", model,
             "--fix-layers", fix_layers, "--fmax", fmax, "--steps", steps])
        return read_frames(dst)


def main(argv: list[str] | None = None) -> None:
    from ase.io import write

    from grace_al.data import read_frames

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model", required=True)
    p.add_argument("--fix-layers", type=int, default=2)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args(argv)
    frames = relax_inproc(read_frames(a.input), a.model, a.fix_layers, a.fmax, a.steps)
    write(a.output, frames, format="extxyz")


if __name__ == "__main__":
    main()
