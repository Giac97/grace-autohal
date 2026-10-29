"""Exploration backends: run the walkers of a plan.

InprocBackend  run_hal in this process, one walker after another (loads TF here)
PoolBackend    one `grace_uq hal --structure-index i` subprocess per walker, at most
               walkers_per_gpu x n_gpus at once; GPUs from CUDA_VISIBLE_DEVICES (none: the
               CPU with walkers_per_gpu slots). Each process sees one GPU, grows its memory
               as needed and gets cpus // slots threads; its output goes to hal/w<ID>/walker.log

A walker counts as done when hal/w<ID>/hal_summary.json exists. On SIGTERM (job time
limit) the pool terminates its children and raises Terminated; the unfinished walkers
are run again (overwrite) next time. A LAMMPS monitoring backend can follow the same
`run(walkers, ...)` shape.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al import _tp
from grace_al.labelers.launcher import end_timing, start_timing

if TYPE_CHECKING:
    from grace_al.explore.plan import WalkerSpec
    from grace_al.models import ModelRef

log = logging.getLogger("grace_al.explore")


class Terminated(RuntimeError):
    """SIGTERM received; unfinished walkers are left to the next run."""


def done(out: Path) -> bool:
    return (out / "hal_summary.json").exists()


# ---------------------------------------------------------------- in-process
def make_backend(cfg) -> InprocBackend | PoolBackend:
    ex = cfg.explore
    if ex.backend == "inproc":
        return InprocBackend(ex.run_hal_kwargs)
    return PoolBackend(ex.walkers_per_gpu, _cli(ex.run_hal_kwargs))


class InprocBackend:
    def __init__(self, run_hal_kwargs: dict | None = None):
        self.kw = dict(run_hal_kwargs or {})

    def run(self, walkers: Sequence[WalkerSpec], starts_file: Path, hal: Path,
            model: ModelRef) -> dict[int, dict]:
        from grace_al.data import read_frames

        starts = read_frames(starts_file)
        failures: dict[int, dict] = {}
        old = signal.signal(signal.SIGTERM, _raise_terminated)
        try:
            for w in walkers:
                out = hal / w.dir
                if done(out):
                    continue
                t = start_timing(out, walker=w.id)
                s = _tp.hal_run_inproc(starts[w.start_index].copy(), model, out,
                                       ensemble=w.ensemble, temperature_K=w.temperature,
                                       n_steps=w.n_steps, tau=w.tau, seed=w.seed,
                                       walker_id=w.id, **{**self.kw, **w.extra})
                end_timing(out, t, 0)
                log.info("walker %d (%s T=%g tau=%g): %s, collected %d", w.id, w.ensemble,
                         w.temperature, w.tau, s["stop_reason"], s["n_collected"])
        finally:
            signal.signal(signal.SIGTERM, old)
        return failures


def _raise_terminated(signum, frame):
    raise Terminated("SIGTERM")


# ---------------------------------------------------------------- pool
@dataclass
class Job:
    id: int
    cmd: list[str]
    log_file: Path
    env: dict[str, str] = field(default_factory=dict)


def visible_gpus(environ=os.environ) -> list[str]:
    v = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return [g for g in v.split(",") if g and g != "-1"] if v else []


def slot_envs(gpus: list[str], per_gpu: int, cpus: int) -> list[dict[str, str]]:
    """One env per slot: GPUs round-robin (slot s -> gpus[s % n]), threads = cpus // slots."""
    n = max(1, per_gpu) * max(1, len(gpus))
    threads = str(max(1, cpus // n))
    out = []
    for s in range(n):
        env = {"TF_NUM_INTRAOP_THREADS": threads, "TF_NUM_INTEROP_THREADS": "1",
               "OMP_NUM_THREADS": threads, "TF_FORCE_GPU_ALLOW_GROWTH": "true",
               "TF_CPP_MIN_LOG_LEVEL": "2"}
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[s % len(gpus)]
        out.append(env)
    return out


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:          # not on Linux
        return os.cpu_count() or 1


def run_pool(jobs: Sequence[Job], slots: list[dict[str, str]],
             on_done: Callable[[Job, int], None] | None = None,
             poll: float = 0.2, grace: float = 10.0) -> dict[int, int]:
    """Run jobs with len(slots) at a time (slot env on top of ours); returns {id: exit code}.
    A failing job doesn't stop the others. SIGTERM: terminate the children, raise
    Terminated."""
    queue, free = list(jobs), list(range(len(slots)))
    running: dict[int, tuple[Job, subprocess.Popen, int, dict]] = {}
    codes: dict[int, int] = {}
    stop = []
    old = signal.signal(signal.SIGTERM, lambda s, f: stop.append(s))
    try:
        while queue or running:
            if stop:
                raise Terminated(f"SIGTERM: {len(running)} walkers stopped, "
                                 f"{len(queue)} not started")
            while queue and free:
                job, slot = queue.pop(0), free.pop(0)
                job.log_file.parent.mkdir(parents=True, exist_ok=True)
                fh = open(job.log_file, "ab")
                env = {**os.environ, **slots[slot], **job.env}
                timing = start_timing(job.log_file.parent, walker=job.id, slot=slot)
                proc = subprocess.Popen(job.cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                fh.close()
                running[job.id] = (job, proc, slot, timing)
            for jid, (job, proc, slot, timing) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                end_timing(job.log_file.parent, timing, rc)
                del running[jid]
                free.append(slot)
                codes[jid] = rc
                if on_done:
                    on_done(job, rc)
            time.sleep(poll)
    finally:
        signal.signal(signal.SIGTERM, old)
        _terminate([p for _, p, _, _ in running.values()], grace)
    return codes


def _terminate(procs: list[subprocess.Popen], grace: float) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + grace
    for p in procs:
        try:
            p.wait(max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


class PoolBackend:
    def __init__(self, walkers_per_gpu: int, extra_args: Sequence[str] = (),
                 gpus: list[str] | None = None, cpus: int | None = None):
        self.slots = slot_envs(visible_gpus() if gpus is None else gpus, walkers_per_gpu,
                               cpus or available_cpus())
        self.extra = list(extra_args)

    def run(self, walkers: Sequence[WalkerSpec], starts_file: Path, hal: Path,
            model: ModelRef) -> dict[int, dict]:
        """Returns {walker id: {exit_code, log_tail}} of the failed walkers."""
        todo = [w for w in walkers if not done(hal / w.dir)]
        by_id = {w.id: w for w in walkers}
        jobs = [Job(w.id, _tp.hal_cli_command(
                    model, starts_file, w.start_index, hal / w.dir, ensemble=w.ensemble,
                    temperature=w.temperature, tau=w.tau, n_steps=w.n_steps, seed=w.seed,
                    walker_id=w.id, overwrite=True, extra=[*self.extra, *_cli(w.extra)]),
                    hal / w.dir / "walker.log") for w in todo]
        failures: dict[int, dict] = {}
        n_done = [len(walkers) - len(todo)]

        def report(job: Job, rc: int) -> None:
            n_done[0] += 1
            w = by_id[job.id]
            if rc or not done(hal / w.dir):
                failures[job.id] = {"exit_code": rc, "log_tail": tail(job.log_file)}
                log.warning("walker %d failed (exit code %d, see %s) [%d/%d]", job.id, rc,
                            job.log_file, n_done[0], len(walkers))
                return
            import json

            s = json.loads((hal / w.dir / "hal_summary.json").read_text())
            log.info("walker %d (%s T=%g tau=%g): %s, collected %d [%d/%d]", w.id, w.ensemble,
                     w.temperature, w.tau, s.get("stop_reason"), s.get("n_collected", 0),
                     n_done[0], len(walkers))

        log.info("pool: %d walkers to run, %d at a time", len(jobs), len(self.slots))
        run_pool(jobs, self.slots, report)
        return failures


def _cli(extra: dict) -> list[str]:
    """{"trigger_gamma": 2} -> ["--trigger-gamma", "2"] (per-walker CLI options)."""
    out = []
    for k, v in extra.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out
