"""tau calibration: short walkers on the first calibrate_starts planned starts for every tau
of calibrate_taus; the largest tau whose rejected / (collected + rejected) is at most
max_reject_ratio and that collects something wins. Stored in hal/calibration.json.

Round 0 calibrates when explore.calibrate is set; later rounds reuse round 0's value, or
calibrate again with a fine-tuned driver if recalibrate_each_round is set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al.state import write_json_atomic

if TYPE_CHECKING:
    from grace_al.explore.plan import WalkerSpec

log = logging.getLogger("grace_al.explore.calibrate")

FILE = "calibration.json"


def calibration_walkers(base: list[WalkerSpec], taus: list[float], n_starts: int,
                        n_steps: int) -> list[WalkerSpec]:
    """For each tau: copies of the first n_starts walkers (ids 0, 1, ...; own seeds)."""
    out: list[WalkerSpec] = []
    for tau in taus:
        for w in base[:n_starts]:
            out.append(replace(w, id=len(out), tau=tau, n_steps=n_steps,
                               seed=w.seed + 100_000 + len(out)))
    return out


def choose(results: dict[float, dict], max_reject_ratio: float) -> float | None:
    """results: tau -> {collected, rejected}; the largest acceptable tau."""
    ok = [tau for tau, r in results.items()
          if r["collected"] > 0
          and r["rejected"] / (r["collected"] + r["rejected"]) <= max_reject_ratio]
    return max(ok) if ok else None


def tally(walkers: list[WalkerSpec], cal_dir: Path) -> dict[float, dict]:
    res: dict[float, dict] = {}
    for w in walkers:
        p = cal_dir / w.dir / "hal_summary.json"
        s = json.loads(p.read_text()) if p.exists() else {}
        r = res.setdefault(w.tau, {"collected": 0, "rejected": 0, "walkers": 0})
        r["collected"] += s.get("n_collected", 0)
        r["rejected"] += sum(s.get("rejected", {}).values())
        r["walkers"] += 1
    return res


def load(hal: Path) -> dict | None:
    p = hal / FILE
    return json.loads(p.read_text()) if p.exists() else None


def save(hal: Path, tau: float | None, results: dict[float, dict], driver: str) -> dict:
    d = {"tau": tau, "driver": driver,
         "results": {str(t): r for t, r in sorted(results.items())}}
    write_json_atomic(hal / FILE, d)
    return d
