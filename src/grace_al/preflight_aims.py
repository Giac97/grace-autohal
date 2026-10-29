"""FHI-aims checks (labeler: aims): launcher, binary and mpirun as aims will see them,
species files, saved module environment, stack limit."""

from __future__ import annotations

import resource
import shlex
import shutil
from collections.abc import Iterable
from pathlib import Path

from grace_al.config.schema import Config
from grace_al.labelers.launcher import aims_bin, aims_env, env_file
from grace_al.preflight import Check, register_check, system_elements

__all__ = ["aims_bin", "check_aims"]


def _found(exe: str, path: str | None) -> bool:
    return bool(shutil.which(exe, path=path) or Path(exe).is_file())


@register_check
def check_aims(cfg: Config) -> Iterable[Check]:
    if cfg.label.labeler != "aims":
        return
    from ase.data import atomic_numbers

    g, a = "FHI-aims", cfg.aims
    assert a is not None and a.species_dir is not None   # validated for labeler 'aims'
    env = aims_env(cfg)
    if a.command is not None:
        yield Check(f"launcher {a.command}", a.command.exists(),
                    fix="fix aims.command, or remove it (grace-al starts aims itself)", group=g)
    else:
        b = aims_bin(cfg)
        yield Check(f"aims binary {b}", _found(b, env.get("PATH")),
                    fix="set aims.bin (or export AIMS_BIN=/path/to/aims.x)", group=g)
        mpirun = shlex.split(cfg.hpc.mpirun if cfg.hpc and cfg.hpc.mpirun else "mpirun")[0]
        # with site modules and no saved env yet (login node), the job loads the MPI module
        # itself before aims runs: then only a warning
        later = bool(cfg.hpc and cfg.hpc.modules) and not env_file(cfg).exists()
        yield Check(f"{mpirun} in aims's PATH", _found(mpirun, env.get("PATH")),
                    "not in this shell; the job loads hpc.modules first" if later else "",
                    fix=(f"check: module load {' '.join(cfg.hpc.modules)} && which {mpirun}"
                         if later and cfg.hpc else
                         "set hpc.mpirun (site profile), or load the MPI module"),
                    severity="warning" if later else "error", group=g)
    sd = Path(a.species_dir)
    yield Check(f"species_dir {sd}", sd.is_dir(), fix="fix aims.species_dir", group=g)
    for el in system_elements(cfg):
        f = sd / f"{atomic_numbers[el]:02d}_{el}_default"
        yield Check(f"species file {f.name}", f.exists(), fix=f"missing in {sd}", group=g)
    saved = env_file(cfg)
    if cfg.hpc is not None and cfg.hpc.modules:
        yield Check(f"saved module env {saved}", saved.exists(),
                    "written by the job script before the Python env is activated; until "
                    "then aims gets the current env without the conda MPI variables",
                    severity="warning", group=g)
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    yield Check("stack hard limit unlimited or >= 1 GB",
                hard == resource.RLIM_INFINITY or hard >= 2**30,
                f"hard limit {hard} bytes: aims may stop with 'stacksize too low'",
                fix="ask for a higher limit (ulimit -Hs)", severity="warning", group=g)
