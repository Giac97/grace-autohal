"""Small helpers shared by all modules: subprocesses, dict merging, hashing."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger("grace_al.utils")


def run(cmd: Sequence[Any], cwd: Path | str | None = None, env: Mapping[str, str] | None = None,
        log_file: Path | str | None = None, check: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command (no shell). stdout+stderr go to `log_file` if given, else to ours."""
    cmd = [str(c) for c in cmd]
    log.info("$ %s%s", " ".join(cmd), f"   (in {cwd})" if cwd else "")
    if log_file is None:
        return subprocess.run(cmd, cwd=cwd, env=env, check=check, timeout=timeout)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as fh:
        return subprocess.run(cmd, cwd=cwd, env=env, check=check, timeout=timeout,
                              stdout=fh, stderr=subprocess.STDOUT)


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
