"""Starting FHI-aims: environment, MPI command, stack limit. Replaces run_aims.sh.

aims must run with the MPI it was built with. The grace env ships its own Intel MPI
(I_MPI_ROOT, libfabric in LD_LIBRARY_PATH, FI_PROVIDER_PATH), so the job saves the module
environment **before** activating the Python env (`save_env_snippet`, inserted by the job
template), and every aims run gets that environment back. Without a saved env (workstation)
the current env is used with the conda MPI variables removed. Then the site's `hpc.env`
(Intel MPI pinning/collectives, or a clean PATH) and OMP_NUM_THREADS=1 go on top.
"""

from __future__ import annotations

import logging
import os
import resource
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al.utils import Interrupted, wait_process

if TYPE_CHECKING:
    from grace_al.config.schema import Config

log = logging.getLogger("grace_al.labelers.launcher")

ENV_FILE = ".aims_env"                 # in runs/
# set by the conda env (Intel MPI, libfabric, compilers); never passed to aims
CONDA_MPI_VARS = ("I_MPI_ROOT", "FI_PROVIDER_PATH", "LIBRARY_PATH", "CONDA_PREFIX",
                  "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "MAMBA_ROOT_PREFIX", "CPATH")
_warned_stack = False


# ---------------------------------------------------------------- saved environment
def env_file(cfg: Config) -> Path:
    return cfg.workdir / ENV_FILE


def save_env_snippet(path: str | Path = f"runs/{ENV_FILE}") -> str:
    """Shell line for the job script: after `module load`, before activating Python."""
    return f'env -0 > "{path}"'


def load_saved_env(path: Path) -> dict[str, str]:
    """The environment written by `env -0` (NUL-separated KEY=VALUE)."""
    out = {}
    for item in path.read_bytes().decode(errors="replace").split("\0"):
        key, sep, val = item.partition("=")
        if sep and key and "\n" not in key:
            out[key] = val
    return out


def without_conda(environ: Mapping[str, str]) -> dict[str, str]:
    """`environ` minus the conda MPI variables and the conda env's bin/lib from the paths."""
    env = dict(environ)
    prefix = env.get("CONDA_PREFIX")
    for k in CONDA_MPI_VARS:
        env.pop(k, None)
    if prefix:
        for k in ("PATH", "LD_LIBRARY_PATH"):
            if k in env:
                kept = [p for p in env[k].split(os.pathsep) if p and not p.startswith(prefix)]
                env[k] = os.pathsep.join(kept)
    return env


def save_module_env(path: Path, from_current: bool = True) -> Path:
    """Workstation (no modules): write the current env, conda MPI variables removed."""
    if not from_current:
        raise ValueError("only from_current=True can be done from Python; in a job use "
                         "save_env_snippet() before activating the Python env")
    env = without_conda(os.environ)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(f"{k}={v}\0" for k, v in env.items()).encode())
    return path


def aims_env(cfg: Config, extra: Mapping[str, str] | None = None,
             environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment of an aims run: saved env (or the current one without conda MPI), then
    hpc.env, OMP/MKL threads = 1, then `extra` (per-slot pinning)."""
    saved = env_file(cfg)
    if saved.exists():
        env = load_saved_env(saved)
        env.pop("CONDA_PREFIX", None)
        env.pop("LIBRARY_PATH", None)
    else:
        env = without_conda(os.environ if environ is None else environ)
    if cfg.hpc is not None:
        env.update(cfg.hpc.env)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env.update(extra or {})
    return env


# ---------------------------------------------------------------- command
def aims_bin(cfg: Config) -> str:
    return (cfg.aims.bin if cfg.aims and cfg.aims.bin else None) \
        or os.environ.get("AIMS_BIN", "aims.x")


def total_ranks(cfg: Config) -> int:
    """MPI ranks for one calculation: $AIMS_NP, else $SLURM_NTASKS, else aims.np."""
    for var in ("AIMS_NP", "SLURM_NTASKS"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    return cfg.aims.np if cfg.aims else 1


def build_command(cfg: Config, np: int) -> list[str]:
    """[mpirun..., -n, np, aims_bin]; hpc.mpirun may hold options ("srun --mpi=pmi2").
    A legacy `aims.command` launcher script is run with bash instead (AIMS_NP set)."""
    if cfg.aims and cfg.aims.command:
        return ["bash", str(cfg.aims.command)]
    mpirun = (cfg.hpc.mpirun if cfg.hpc and cfg.hpc.mpirun else None) or "mpirun"
    return [*shlex.split(mpirun), "-n", str(np), aims_bin(cfg)]


def _unlimited_stack():
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))


def stack_unlimited_possible() -> bool:
    return resource.getrlimit(resource.RLIMIT_STACK)[1] == resource.RLIM_INFINITY


def run_aims(task_dir: Path, cfg: Config, np: int | None = None,
             extra_env: Mapping[str, str] | None = None) -> int:
    """Run aims in task_dir (control.in, geometry.in there): stdout -> aims.out,
    stderr -> aims.err. Returns the exit code (failures are judged from the output)."""
    global _warned_stack
    np = np or total_ranks(cfg)
    if not stack_unlimited_possible() and not _warned_stack:
        log.warning("stack hard limit is %s, not unlimited: aims may stop with 'stacksize "
                    "too low'", resource.getrlimit(resource.RLIMIT_STACK)[1])
        _warned_stack = True
    env = aims_env(cfg, {"AIMS_NP": str(np), "AIMS_BIN": aims_bin(cfg), **(extra_env or {})})
    cmd = build_command(cfg, np)
    timeout = cfg.aims.timeout if cfg.aims else None
    log.debug("$ %s   (in %s)", " ".join(cmd), task_dir)
    timing = start_timing(task_dir, ranks=np)
    rc: int | str = 127
    try:
        with open(task_dir / "aims.out", "wb") as out, \
                open(task_dir / "aims.err", "wb") as err:
            try:
                rc = wait_process(cmd, timeout, cwd=task_dir, env=env, stdout=out,
                                  stderr=err, preexec_fn=_unlimited_stack)
            except subprocess.TimeoutExpired:
                err.write(f"\ngrace-al: timeout after {timeout} s (aims.timeout)\n".encode())
                rc = -9
            except FileNotFoundError as e:   # mpirun not found
                err.write(f"\ngrace-al: {e} (mpirun: command not found)\n".encode())
                rc = 127
            except (Interrupted, KeyboardInterrupt):
                # aims was stopped (SIGTERM to its process group): rerun on the next start
                err.write(f"\n{INTERRUPTED_MARK} (SIGTERM): rerun on restart\n".encode())
                rc = INTERRUPTED
                raise
    finally:
        end_timing(task_dir, timing, rc)
    return int(rc)


INTERRUPTED = "interrupted"          # timing.json exit_code of a stopped run
INTERRUPTED_MARK = "grace-al: interrupted"


def interrupted(task_dir: Path) -> bool:
    """aims in task_dir was stopped by an interruption (not a failure of aims)."""
    import json

    p = task_dir / TIMING
    try:
        return p.exists() and json.loads(p.read_text()).get("exit_code") == INTERRUPTED
    except json.JSONDecodeError:
        return False


TIMING = "timing.json"


def start_timing(d: Path, **info) -> dict:
    """d/timing.json: {start, end: null, host, job, ...} (read by `grace-al status`)."""
    import json
    import socket
    import time

    t = {"start": time.time(), "end": None, "host": socket.gethostname(),
         "job": os.environ.get("SLURM_JOB_ID"), **info}
    d.mkdir(parents=True, exist_ok=True)
    (d / TIMING).write_text(json.dumps(t))
    return t


def end_timing(d: Path, t: dict, exit_code: int | str | None = None) -> None:
    import json
    import time

    (d / TIMING).write_text(json.dumps({**t, "end": time.time(), "exit_code": exit_code}))


def run_task(cfg: Config, task_dir: Path, np: int | None = None) -> int:
    """One prepared aims task (`grace-al aims-run <task_dir>`, a job-array element):
    skipped if aims.out already says it finished."""
    from grace_al.labelers import aims_parse

    if aims_parse.finished(aims_parse.read_text(task_dir / "aims.out")):
        log.info("%s: finished before", task_dir)
        return 0
    rc = run_aims(task_dir, cfg, np)
    if aims_parse.finished(aims_parse.read_text(task_dir / "aims.out")):
        return 0
    cat, msg = aims_parse.categorize(aims_parse.read_text(task_dir / "aims.out"),
                                     aims_parse.read_text(task_dir / "aims.err"))
    log.error("%s: %s: %s", task_dir, cat, msg)
    return rc or 1
