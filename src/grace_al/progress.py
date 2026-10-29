"""Progress of the running stage (`grace-al status`), read from files only, so it works from
a login node while the job (or the DFT job array) runs.

Per item (walker, DFT calculation): done, running, failed or pending, with wall times from
the item's timing.json (written by the walker pool and the aims launcher) or, for aims runs
without one, aims's own "| Total time" line. ETA = mean time per item x the items left /
the items that run at once, minus what the running ones have already spent.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from grace_al.labelers import aims_parse
from grace_al.labelers.launcher import TIMING

if TYPE_CHECKING:
    from grace_al.config.schema import Config

STALE = 900          # s without output: a "running" item whose job is gone


def _timing(d: Path) -> dict | None:
    p = d / TIMING
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        return None


def _aims_wall(text: str) -> float | None:
    found = re.findall(r"\| Total time\s+:\s+[\d.]+ s\s+([\d.]+) s", text)
    return float(found[-1]) if found else None


def _fresh(*paths: Path, now: float) -> bool:
    return any(p.exists() and now - p.stat().st_mtime < STALE for p in paths)


def summarize(items: list[dict], parallel: int, now: float | None = None) -> dict:
    """items: {state: done|running|failed|pending, seconds, elapsed?}."""
    now = now or time.time()
    count = {s: sum(i["state"] == s for i in items)
             for s in ("done", "running", "failed", "pending")}
    secs = [i["seconds"] for i in items if i["state"] == "done" and i.get("seconds")]
    out: dict = {"total": len(items), **count,
                 "mean_s": float(np.mean(secs)) if secs else None,
                 "min_s": float(np.min(secs)) if secs else None,
                 "max_s": float(np.max(secs)) if secs else None,
                 "parallel": parallel, "eta_s": None}
    if secs:
        mean = out["mean_s"]
        running_left = sum(max(0.0, mean - i.get("elapsed", 0.0))
                           for i in items if i["state"] == "running")
        waves = math.ceil(count["pending"] / max(1, parallel))
        out["eta_s"] = waves * mean + running_left / max(1, min(parallel, count["running"] or 1))
    out["running_items"] = [{k: i[k] for k in ("name", "elapsed", "detail") if k in i}
                            for i in items if i["state"] == "running"]
    return out


# ---------------------------------------------------------------- stages
def label_items(rdir: Path, now: float | None = None) -> list[dict]:
    """One item per dft/NNNN folder (aims or teacher)."""
    now = now or time.time()
    items = []
    for d in sorted((rdir / "dft").glob("[0-9][0-9][0-9][0-9]")):
        t = _timing(d)
        item: dict = {"name": d.name, "state": "pending"}
        if (d / "results.extxyz").exists():                     # teacher
            item["state"] = "done"
        out = d / "aims.out"
        if out.exists():
            text = aims_parse.read_text(out)
            if aims_parse.finished(text):
                item["state"] = "done"
                item["seconds"] = ((t["end"] - t["start"]) if t and t.get("end")
                                   else _aims_wall(text))
            elif t and not t.get("end") and _fresh(out, now=now):
                item["state"] = "running"
                item["elapsed"] = now - t["start"]
                item["detail"] = f"SCF iteration {aims_parse.scf_iterations(text)}"
            elif t is None and _fresh(out, now=now):
                item["state"] = "running"
                item["detail"] = f"SCF iteration {aims_parse.scf_iterations(text)}"
            else:
                item["state"] = "failed"
        items.append(item)
    return items


def explore_items(hal: Path, now: float | None = None) -> list[dict]:
    now = now or time.time()
    plan = hal / "plan.json"
    if not plan.exists():
        return []
    items = []
    for w in json.loads(plan.read_text())["walkers"]:
        d = hal / f"w{w['id']:03d}"
        t = _timing(d)
        item: dict = {"name": d.name, "state": "pending"}
        if (d / "hal_summary.json").exists():
            item["state"] = "done"
            if t and t.get("end"):
                item["seconds"] = t["end"] - t["start"]
        elif t and not t.get("end"):
            fresh = _fresh(d / "walker.log", d / TIMING, now=now)
            item["state"] = "running" if fresh else "failed"
            item["elapsed"] = now - t["start"]
        elif t and t.get("end"):
            item["state"] = "failed"
        items.append(item)
    return items


def fit_progress(fit: Path, now: float | None = None) -> dict | None:
    """Epochs done / planned from gracemaker's log.txt and train_metrics.yaml."""
    now = now or time.time()
    logs = [p for p in (fit / "seed" / "1" / "log.txt", fit / "log.txt") if p.exists()]
    if not logs:
        return None
    log = logs[0]
    m = re.findall(r"maxiter resolved from target_total_updates=\d+: (\d+) epochs",
                   log.read_text(errors="ignore"))
    total = int(m[-1]) if m else None
    tm = fit / "seed" / "1" / "train_metrics.yaml"
    done = tm.read_text(errors="ignore").count("\n- ") + 1 if tm.exists() else 0
    started = (fit / "input.yaml").stat().st_mtime if (fit / "input.yaml").exists() else None
    out: dict = {"epochs_done": done, "epochs_total": total, "eta_s": None}
    if total and done and started:
        rate = (now - started) / done
        out["eta_s"] = max(0.0, (total - done) * rate)
    return out


def stage_progress(cfg: Config, k: int, stage: str | None) -> dict | None:
    """Progress of `stage` in round k, or None if there is nothing to measure."""
    from grace_al.state import round_dir

    rdir = round_dir(cfg.workdir, k)
    if stage == "label":
        mode = cfg.label.mode
        parallel = (cfg.label.concurrent_calcs if mode == "concurrent"
                    else cfg.hpc.array_max_parallel if mode == "array" and cfg.hpc else 1)
        items = label_items(rdir)
        return {"what": "DFT calculations", **summarize(items, parallel)} if items else None
    if stage == "explore":
        from grace_al.preflight_explore import n_gpus

        parallel = (1 if cfg.explore.backend == "inproc"
                    else cfg.explore.walkers_per_gpu * n_gpus(cfg))
        items = explore_items(rdir / "hal")
        return {"what": "HAL walkers", **summarize(items, parallel)} if items else None
    if stage == "finetune":
        fp = fit_progress(rdir / "fit")
        return {"what": "fit epochs", **fp} if fp else None
    return None


def fmt_duration(s: float | None) -> str:
    if s is None:
        return "–"
    s = int(round(s))
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"
