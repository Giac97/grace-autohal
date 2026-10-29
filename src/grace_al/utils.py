"""Small helpers shared by all modules: subprocesses, interruption, dict merging, hashing.

Every child process runs in its own process group. When this process is interrupted
(SIGTERM from the job's time-limit trap or `scancel`, or Ctrl-C), the whole group - e.g.
mpirun and all its aims ranks - gets SIGTERM, then SIGKILL after a grace period, so nothing
keeps running unsupervised.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import os
import signal
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger("grace_al.utils")

GRACE_S = 10.0                     # SIGTERM -> SIGKILL
_live: set[subprocess.Popen] = set()
_live_lock = threading.Lock()


class Interrupted(BaseException):
    """SIGTERM (time limit, scancel) or Ctrl-C: stop cleanly; the next run resumes.
    A BaseException, like KeyboardInterrupt, so `except Exception` doesn't swallow it."""


def _raise_interrupted(signum, frame):
    raise Interrupted(signal.Signals(signum).name)


@contextlib.contextmanager
def on_sigterm() -> Iterator[None]:
    """Within the block, SIGTERM raises Interrupted (main thread only; elsewhere a no-op)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.signal(signal.SIGTERM, _raise_interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def stop_process(proc: subprocess.Popen, grace: float = GRACE_S) -> None:
    """SIGTERM to the process group of `proc`, SIGKILL after `grace` seconds."""
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def stop_all(grace: float = GRACE_S) -> None:
    """Stop every child started through `wait_process` that is still running."""
    with _live_lock:
        procs = list(_live)
    for p in procs:
        stop_process(p, grace)


def wait_process(cmd: Sequence[str], timeout: float | None = None, **popen) -> int:
    """Popen in a new process group and wait; on any interruption (Interrupted, Ctrl-C,
    timeout) the group is stopped before the exception propagates."""
    proc = subprocess.Popen(list(cmd), start_new_session=True, **popen)
    with _live_lock:
        _live.add(proc)
    try:
        return proc.wait(timeout)
    except BaseException:
        stop_process(proc)
        raise
    finally:
        with _live_lock:
            _live.discard(proc)


def run(cmd: Sequence[Any], cwd: Path | str | None = None, env: Mapping[str, str] | None = None,
        log_file: Path | str | None = None, check: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command (no shell). stdout+stderr go to `log_file` if given, else to ours."""
    cmd = [str(c) for c in cmd]
    log.info("$ %s%s", " ".join(cmd), f"   (in {cwd})" if cwd else "")
    if log_file is None:
        rc = wait_process(cmd, timeout, cwd=cwd, env=env)
    else:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "ab") as fh:
            rc = wait_process(cmd, timeout, cwd=cwd, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    if check and rc:
        raise subprocess.CalledProcessError(rc, cmd)
    return subprocess.CompletedProcess(cmd, rc)


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """New dict: `override` on top of `base`. Dicts merge recursively, everything else
    (lists included) is replaced."""
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    """sha256 of the canonical JSON of `obj` (sorted keys): stable under key order."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()
